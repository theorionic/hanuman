"""Kimi Delta Attention (KDA) -- linear attention with the delta update rule.

Implements the recurrent form of KDA (Kimi-Linear, arXiv:2507.05927) using a
`jax.lax.scan` over the sequence, vectorized over batch and heads. This is the
O(N) memory / O(N) time form. The chunked parallel form (WY representation +
associative_scan across chunks) is a documented future optimization -- the
recurrent scan form is correct and trainable, and for the smoke config
(seq_len=128) it is fast enough on CPU.

Why the delta rule (not plain linear attention):
  Plain linear attention updates the state as  S += outer(k, v)  and reads
  o = S @ q. The delta rule replaces that with a *corrective* update:
      v_new = v - beta * (S @ k)        # subtract the state's current prediction
      S    += beta * outer(v_new, k)    # add the residual
  so the state learns to *overwrite* (not just accumulate) key-value
  associations. `beta = sigmoid(W_b @ x)` is a per-head, per-token gate that
  controls how aggressively the state is rewritten. This is the same delta
  rule as in DeltaNet / RetNet-variants; KDA's contribution is the
  per-channel log-space decay `g = -exp(A_log) * softplus(f + dt_bias)` that
  makes the forget gate data-dependent and per-channel, plus the ShortConv
  front-end and the low-rank output gate.

Per-head state S is [K, V] (K=head_dim, V=head_v_dim). The full per-batch
state is [B, H, K, V], kept in float32 regardless of compute dtype so the
accumulating outer products do not lose precision.

Parameter shapes (D=d_model, H=n_heads, K=head_dim, V=head_v_dim=K):
  W_q, W_k      : [D, H*K]      query / key projections
  W_v           : [D, H*V]      value projection
  q_conv,k_conv,v_conv : [H*K, 1, 4] / [H*V, 1, 4]  depthwise conv1d, kernel=4
  f_proj[0]     : [D, K]        gate bottleneck  (D -> head_v_dim)
  f_proj[1]     : [K, H*K]      gate             (head_v_dim -> H*K), NO bias
  W_b           : [D, H]        beta projection
  A_log         : [H]           per-head decay log-param, init log(Uniform(1,16))
  dt_bias       : [H*K]         per-channel decay bias (see init formula)
  g_proj[0]     : [D, K]        output-gate bottleneck (D -> head_v_dim)
  g_proj[1]     : [K, H*V] + bias [H*V]  output gate (head_v_dim -> H*V), bias
  W_o           : [H*V, D]      output projection
  o_norm        : [V]           RMSNorm weight on the readout, init ones

A_log and dt_bias receive NO weight decay (see optimizer.is_no_decay).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.nnx as nnx


def l2_norm(x, eps: float = 1e-6):
    """L2-normalize the last axis. eps=1e-6 (NOT 1e-5, which is for RMSNorm)."""
    norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / norm


class ShortConv1D(nnx.Module):
    """Depthwise 1D convolution, kernel_size=4, groups=input_dim.

    Each channel is convolved with its own 4-tap kernel independently (no
    cross-channel mixing). This is the "ShortConv" front-end from Kimi-Linear:
    it gives each token a 4-step local context before the q/k/v split, which
    matters because linear-attention states otherwise see tokens one at a time
    with no local window.

    For training (full sequence) we use `jax.lax.conv_general_dilated` with
    'CAUSAL' padding so the convolution is causal (token t only sees t-3..t).
    The kernel layout is [in_channels, 1, kernel_size] (depthwise = one
    filter per channel, spatial dim 1, 4 taps).

    A `step(x_t, state)` method is provided for recurrent inference: `state`
    is the sliding window of the last (kernel_size-1) inputs for this channel,
    shape [B, D, kernel_size-1]. It returns (out_t, new_state).
    """

    def __init__(self, dim: int, rngs: nnx.Rngs, kernel_size: int = 4):
        self.dim = dim
        self.kernel_size = kernel_size
        # Init: small std so the conv is close to identity at start. The center
        # tap (index kernel_size-1, the current token) is biased slightly up so
        # the untrained model passes the input through mostly unchanged.
        kernel = jax.random.normal(rngs.params(), (dim, 1, kernel_size)) * 0.01
        # Emphasize the last tap (current token) -> near-identity init.
        kernel = kernel.at[:, :, -1].set(kernel[:, :, -1] + 1.0)
        self.weight = nnx.Param(kernel)

    def __call__(self, x_seq):
        """x_seq: [B, S, D] -> [B, S, D] (causal depthwise conv)."""
        B, S, D = x_seq.shape
        # 1D depthwise conv via conv_general_dilated. dimension_numbers use
        # "NHC" (batch, spatial, channel) for lhs/out and "HIO" for the kernel
        # (spatial, in, out). Depthwise = feature_group_count=D, so the kernel
        # has in=1, out=D per group: shape [K, 1, D].
        x = x_seq.astype(jnp.float32)  # [B, S, D]
        w = self.weight.value  # [D, 1, K]
        # Build HIO: [K, 1, D].
        w_hio = jnp.transpose(w, (2, 1, 0))  # [K, 1, D]
        # Causal padding: pad (K-1) on the left of the sequence axis, 0 on the
        # right, so output[t] depends on input[t-K+1..t]. One spatial dim ->
        # a single (K-1, 0) entry.
        pad = [(self.kernel_size - 1, 0)]
        y = jax.lax.conv_general_dilated(
            x, w_hio,
            window_strides=(1,),
            padding=pad,
            dimension_numbers=("NHC", "HIO", "NHC"),
            feature_group_count=D,
        )
        return y.astype(x_seq.dtype)  # [B, S, D]

    def step(self, x_t, state):
        """Recurrent step. x_t: [B, D], state: [B, D, K-1] (last K-1 inputs).
        Returns (out_t [B, D], new_state [B, D, K-1])."""
        w = self.weight.value  # [D, 1, K]
        # window = [x_{t-K+1}, ..., x_{t-1}, x_t]; state holds the first K-1.
        window = jnp.concatenate([state, x_t[:, :, None]], axis=-1)  # [B, D, K]
        out = jnp.sum(window * w, axis=-1)  # [B, D]
        new_state = window[:, :, 1:]  # drop the oldest, keep last K-1
        return out, new_state


def _init_dt_bias(rng, shape):
    """dt_bias init from the Kimi-Linear spec.

        dt      = exp(rand * (log(0.1) - log(0.001)) + log(0.001)).clamp(min=1e-4)
        inv_dt  = dt + log(-expm1(-dt))
        dt_bias = inv_dt

    `dt` is a per-channel time-step sampled log-uniformly in [0.001, 0.1].
    `inv_dt` is the log-space correction so that softplus(f + dt_bias) ~= dt
    when f is near zero (the softplus inverse). This makes the initial decay
    rate well-conditioned without a long warm-up.
    """
    log_min, log_max = jnp.log(jnp.asarray(0.001)), jnp.log(jnp.asarray(0.1))
    rand = jax.random.uniform(rng, shape, minval=0.0, maxval=1.0)
    dt = jnp.exp(rand * (log_max - log_min) + log_min)
    dt = jnp.clip(dt, min=1e-4)
    # -expm1(-dt) = 1 - exp(-dt) > 0, so log(...) is finite.
    inv_dt = dt + jnp.log(-jnp.expm1(-dt))
    return inv_dt


class KDA(nnx.Module):
    """Kimi Delta Attention layer (recurrent scan form).

    `__call__(x, positions=None)` mirrors the `Attention` interface: x is
    [B, S, D] and the output is [B, S, D]. `positions` is accepted for API
    compatibility but ignored -- KDA has no positional embedding (the
    ShortConv + recurrent state provide the position signal).

    The state S is created fresh (zeros) on every forward call: we do not
    carry state across batches during training. For inference a separate
    recurrent path would be used; here we only support the training/whole-
    sequence path.
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 dtype, rngs: nnx.Rngs, chunk_size: int = 64):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim          # K
        self.head_v_dim = head_dim        # V = K (KDA uses V=K)
        self.dtype = dtype
        self.chunk_size = chunk_size      # reserved for a future chunked form

        H, K, V = n_heads, head_dim, self.head_v_dim
        D = d_model
        std = (1.0 / D) ** 0.5

        # ---- Input projections ----
        self.wq = nnx.Param(jax.random.normal(rngs.params(), (D, H * K)) * std)
        self.wk = nnx.Param(jax.random.normal(rngs.params(), (D, H * K)) * std)
        self.wv = nnx.Param(jax.random.normal(rngs.params(), (D, H * V)) * std)

        # ---- ShortConv (depthwise, kernel=4) ----
        self.q_conv = ShortConv1D(H * K, kernel_size=4, rngs=rngs)
        self.k_conv = ShortConv1D(H * K, kernel_size=4, rngs=rngs)
        self.v_conv = ShortConv1D(H * V, kernel_size=4, rngs=rngs)

        # ---- Gate (channel-wise decay) ----
        # f_proj: D -> head_v_dim(=K) -> H*K, two linears, NO bias on either.
        self.f_proj0 = nnx.Param(jax.random.normal(rngs.params(), (D, K)) * std)
        self.f_proj1 = nnx.Param(jax.random.normal(rngs.params(), (K, H * K)) * std)

        # ---- Beta (update strength) ----
        self.wb = nnx.Param(jax.random.normal(rngs.params(), (D, H)) * std)

        # ---- Per-head decay log-param ----
        # A_log init: log(Uniform(1, 16)) per head.
        a = jax.random.uniform(rngs.params(), (H,), minval=1.0, maxval=16.0)
        self.a_log = nnx.Param(jnp.log(a))

        # ---- Per-channel decay bias ----
        self.dt_bias = nnx.Param(_init_dt_bias(rngs.params(), (H * K,)))

        # ---- Output gate (low-rank) ----
        # g_proj: D -> head_v_dim(=K) -> H*V, bias on the SECOND layer only.
        self.g_proj0 = nnx.Param(jax.random.normal(rngs.params(), (D, K)) * std)
        self.g_proj1 = nnx.Param(jax.random.normal(rngs.params(), (K, H * V)) * std)
        self.g_proj1_bias = nnx.Param(jnp.zeros((H * V,), dtype=jnp.float32))

        # ---- Output projection + readout RMSNorm ----
        self.wo = nnx.Param(jax.random.normal(rngs.params(), (H * V, D)) * std)
        self.o_norm = nnx.Param(jnp.ones((V,), dtype=jnp.float32))

    # ------------------------------------------------------------------ #
    #  Forward (recurrent scan over the sequence, vectorized over B,H)   #
    # ------------------------------------------------------------------ #
    def __call__(self, x, cos=None, sin=None, positions=None, window=None):
        """x: [B, S, D] -> [B, S, D]. cos/sin/positions/window are accepted
        for interface compatibility with Attention and are ignored."""
        del cos, sin, positions, window  # KDA has no RoPE / no window
        B, S, D = x.shape
        H, K, V = self.n_heads, self.head_dim, self.head_v_dim
        dtype = self.dtype

        # ---- Input projections + ShortConv + Swish ----
        # Run the projections in compute dtype, then the depthwise conv in
        # fp32 (conv_general_dilated is tiny and precision-sensitive), then
        # back to compute dtype for the Swish.
        x_f = x.astype(jnp.float32)
        q_raw = x_f @ self.wq.value.astype(jnp.float32)   # [B, S, H*K]
        k_raw = x_f @ self.wk.value.astype(jnp.float32)
        v_raw = x_f @ self.wv.value.astype(jnp.float32)

        q_conv = self.q_conv(q_raw.astype(dtype)).astype(jnp.float32)
        k_conv = self.k_conv(k_raw.astype(dtype)).astype(jnp.float32)
        v_conv = self.v_conv(v_raw.astype(dtype)).astype(jnp.float32)

        q_raw = jax.nn.silu(q_conv)   # [B, S, H*K]
        k_raw = jax.nn.silu(k_conv)
        v_raw = jax.nn.silu(v_conv)   # [B, S, H*V]

        # Reshape to [B, S, H, K] / [B, S, H, V] and L2-normalize q,k (eps=1e-6).
        q = q_raw.reshape(B, S, H, K)
        k = k_raw.reshape(B, S, H, K)
        v = v_raw.reshape(B, S, H, V)
        q = l2_norm(q, eps=1e-6)
        k = l2_norm(k, eps=1e-6)

        # ---- Gate (log-space per-channel decay) ----
        # f = f_proj(x): D -> K -> H*K  (two linears, no bias, no activation --
        # the spec lists f_proj as two plain linear layers).
        f0 = x_f @ self.f_proj0.value                       # [B, S, K]
        f = f0 @ self.f_proj1.value                          # [B, S, H*K]
        f = f.reshape(B, S, H, K)
        # g = -exp(A_log) * softplus(f + dt_bias)   -> [B, S, H, K]
        a_log = self.a_log.value                            # [H]
        dt_bias = self.dt_bias.value.reshape(H, K)          # [H, K]
        g = -jnp.exp(a_log[None, None, :, None]) * jax.nn.softplus(f + dt_bias)

        # ---- Beta (update strength) ----
        beta = jax.nn.sigmoid(x_f @ self.wb.value)          # [B, S, H]

        # ---- Scan over the sequence ----
        # We carry the state S: [B, H, K, V] (float32) and emit o: [B, S, H, V].
        # The body computes, per token t:
        #   S     = S * exp(g_t)[:, :, None]                  # per-channel decay
        #   v_new = v_t - beta_t * (S @ k_t)                   # delta correction
        #   S     = S + beta_t * outer(v_new, k_t)             # state update
        #   o_t   = (S @ q_t) * (1/sqrt(K))                    # readout
        # All in float32 for accumulation precision.
        scale = 1.0 / jnp.sqrt(jnp.asarray(K, jnp.float32))

        # Transpose to time-major for the scan: [S, B, H, K] etc.
        q_t = jnp.transpose(q, (1, 0, 2, 3))   # [S, B, H, K]
        k_t = jnp.transpose(k, (1, 0, 2, 3))
        v_t = jnp.transpose(v, (1, 0, 2, 3))   # [S, B, H, V]
        g_t = jnp.transpose(g, (1, 0, 2, 3))   # [S, B, H, K]
        beta_t = jnp.transpose(beta, (1, 0, 2))  # [S, B, H]

        S0 = jnp.zeros((B, H, K, V), dtype=jnp.float32)

        def step_fn(carry, inputs):
            S = carry  # [B, H, K, V]
            q_i, k_i, v_i, g_i, b_i = inputs
            # q_i, k_i, g_i: [B, H, K]; v_i: [B, H, V]; b_i: [B, H]
            # Per-channel decay: S = S * exp(g) along the K axis.
            decay = jnp.exp(g_i)[:, :, :, None]   # [B, H, K, 1]
            S = S * decay                          # [B, H, K, V]
            # State's current prediction of v from k:  S @ k  -> [B, H, V]
            # einsum: for each (b,h), (K,V) @ (K) -> (V)
            sk = jnp.einsum("bhkv,bhk->bhv", S, k_i)   # [B, H, V]
            v_new = v_i - b_i[:, :, None] * sk         # [B, H, V]
            # State update: S += beta * outer(v_new, k)
            # outer(v_new, k) for each (b,h): (V, K) from v_new (V) and k (K).
            # We want S[b,h,k,v] += beta[b,h] * v_new[b,h,v] * k[b,h,k].
            outer = jnp.einsum("bhv,bhk->bhkv", v_new, k_i)  # [B, H, K, V]
            S = S + b_i[:, :, None, None] * outer
            # Readout: o = (S @ q) * scale  -> [B, H, V]
            o_i = jnp.einsum("bhkv,bhk->bhv", S, q_i) * scale  # [B, H, V]
            return S, o_i

        S_final, o_t = jax.lax.scan(step_fn, S0, (q_t, k_t, v_t, g_t, beta_t))
        # o_t: [S, B, H, V] -> [B, S, H, V]
        o = jnp.transpose(o_t, (1, 0, 2, 3))   # [B, S, H, V]

        # ---- Output gating: RMSNorm(o) * sigmoid(gate), then W_o ----
        # RMSNorm on the V axis, eps=1e-5 (this IS RMSNorm, not L2Norm).
        o_f = o.astype(jnp.float32)
        var = jnp.mean(o_f * o_f, axis=-1, keepdims=True)
        o_normed = o_f * jax.lax.rsqrt(var + 1e-5) * self.o_norm.value

        # g_proj: D -> K -> H*V, bias on the second layer only (no activation --
        # the spec lists g_proj as two plain linear layers).
        g0 = x_f @ self.g_proj0.value                         # [B, S, K]
        gate = g0 @ self.g_proj1.value + self.g_proj1_bias.value  # [B, S, H*V]
        gate = jax.nn.sigmoid(gate).reshape(B, S, H, V)

        o_gated = o_normed * gate                          # [B, S, H, V]
        o_flat = o_gated.reshape(B, S, H * V)              # [B, S, H*V]
        out = o_flat @ self.wo.value.astype(jnp.float32)   # [B, S, D]
        return out.astype(jnp.float32)