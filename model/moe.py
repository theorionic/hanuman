"""Mixture-of-Experts layer with DeepSeek-V3 style aux-loss-free routing.

Key features:
  - Sigmoid gating (not softmax)
  - Per-expert bias term adjusted each step for load balancing (gamma)
  - Top-k selection on (sigmoid(gate) + bias); gating weights use original sigmoid
  - 1 shared expert always active (weight 1.0)
  - routed_scaling_factor scales routed expert outputs
  - Returns aux dict: {router_z_loss, balance_loss, mean_entropy, expert_counts}
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.sharding import PartitionSpec as P

from .sharding import expert_shard_axis

try:
    from jax.experimental.pallas.ops.tpu import megablox as _megablox
except ImportError:  # non-TPU backends
    _megablox = None


def swiglu(x, w_gate, w_up, w_down, dtype):
    """SwiGLU FFN: down(silu(x @ gate) * (x @ up))."""
    g = jax.nn.silu((x @ w_gate.astype(dtype)))
    u = (x @ w_up.astype(dtype))
    h = g * u
    out = h @ w_down.astype(dtype)
    return out.astype(jnp.float32)


_ROW_ALIGN = 128  # TPU tile alignment for the grouped-matmul row dimension
_TILE = 128       # Pallas gmm tile size; K and N must be multiples of it


def grouped_matmul(xs, w, group_sizes):
    """One grouped matmul: row block `g` of `xs` times `w[g]`.

    Prefers the Pallas `megablox.gmm` kernel over `jax.lax.ragged_dot`. Both
    compute the same thing, but ragged_dot's TPU lowering cannot build a
    backward pass when its operand is produced by a collective -- which is
    exactly our case, since the expert weights arrive via an all_gather inside
    shard_map ("Pattern match for backwards collectives + grad_y - NYI"). gmm
    is a custom_vjp whose backward is itself a Pallas kernel (tgmm), so there is
    no ragged_dot backward for XLA to choke on.

    Falls back to ragged_dot when the shapes are not tileable (small configs)
    or the kernel is unavailable.
    """
    K, N = w.shape[-2], w.shape[-1]
    if _megablox is not None and K % _TILE == 0 and N % _TILE == 0 and xs.shape[0] % _TILE == 0:
        return _megablox.gmm(xs, w, group_sizes, preferred_element_type=jnp.float32)
    return jax.lax.ragged_dot(xs, w, group_sizes)


def ragged_dispatch(x_flat, top_idx, weights, w_gate, w_up, w_down, dtype):
    """Route each token to its top-k experts and run SwiGLU, one dot per expert.

    x_flat:  [T, D]      tokens
    top_idx: [T, K]      selected expert id per token
    weights: [T, K]      gate weight per (token, slot)
    w_*:     [N, D, F] / [N, F, D]  stacked expert weights

    Every (token, slot) pair becomes one row; rows are sorted by expert id so
    that `jax.lax.ragged_dot` can do the whole layer as three grouped matmuls.
    Cost is O(T*K*D*F) instead of the O(T*K*N*D*F) a dense-all-experts pass
    would need, and no per-token copy of the expert weights is materialized.
    """
    T, D = x_flat.shape
    K = top_idx.shape[1]
    N = w_gate.shape[0]

    slot_expert = top_idx.reshape(-1)                 # [T*K]
    order = jnp.argsort(slot_expert)                  # stable -> rows grouped by expert
    xs = x_flat[order // K].astype(dtype)             # [T*K, D] tokens in expert order
    group_sizes = jnp.bincount(slot_expert, length=N)  # [N] rows per expert

    # The TPU grouped-matmul kernels require the row count to be a multiple of
    # the tile size. Pad with rows that belong to no group (group_sizes still
    # sums to T*K, so the padding contributes nothing) and drop them after.
    rows = T * K
    padded = -(-rows // _ROW_ALIGN) * _ROW_ALIGN
    if padded != rows:
        xs = jnp.pad(xs, ((0, padded - rows), (0, 0)))

    g = grouped_matmul(xs, w_gate.astype(dtype), group_sizes)   # [rows, F]
    u = grouped_matmul(xs, w_up.astype(dtype), group_sizes)     # [rows, F]
    h = (jax.nn.silu(g) * u).astype(dtype)
    ys = grouped_matmul(h, w_down.astype(dtype), group_sizes)   # [rows, D]
    ys = ys[:rows]

    # Undo the permutation, then combine the K slots of each token.
    out = jnp.zeros((T * K, D), ys.dtype).at[order].set(ys)
    out = out.reshape(T, K, D).astype(jnp.float32)
    return jnp.sum(out * weights[..., None], axis=1)  # [T, D]


class DenseFFN(nnx.Module):
    """Dense SwiGLU FFN used in the first `dense_layers` layers."""

    def __init__(self, d_model: int, d_ff: int, dtype, rngs: nnx.Rngs):
        std = (1.0 / d_model) ** 0.5
        self.w_gate = nnx.Param(jax.random.normal(rngs.params(), (d_model, d_ff)) * std)
        self.w_up = nnx.Param(jax.random.normal(rngs.params(), (d_model, d_ff)) * std)
        self.w_down = nnx.Param(jax.random.normal(rngs.params(), (d_ff, d_model)) * std)
        self.dtype = dtype

    def __call__(self, x):
        return swiglu(x, self.w_gate, self.w_up, self.w_down, self.dtype)


class Expert(nnx.Module):
    """A single SwiGLU expert."""

    def __init__(self, d_model: int, d_ff: int, dtype, rngs: nnx.Rngs):
        std = (1.0 / d_model) ** 0.5
        self.w_gate = nnx.Param(jax.random.normal(rngs.params(), (d_model, d_ff)) * std)
        self.w_up = nnx.Param(jax.random.normal(rngs.params(), (d_model, d_ff)) * std)
        self.w_down = nnx.Param(jax.random.normal(rngs.params(), (d_ff, d_model)) * std)
        self.dtype = dtype

    def __call__(self, x):
        return swiglu(x, self.w_gate, self.w_up, self.w_down, self.dtype)


class MoE(nnx.Module):
    """Mixture of Experts with aux-loss-free sigmoid routing.

    Params:
      router: [d_model, n_experts] (init std=router_init_std)
      bias:   [n_experts] (not trained by gradient; updated manually each step)
      experts: n_experts Expert modules (routed)
      shared:  n_shared_experts Expert modules (always on)
    """

    def __init__(self, d_model: int, d_ff: int, n_experts: int, n_active: int,
                 n_shared_experts: int, router_init_std: float, routed_scaling_factor: float,
                 dtype, rngs: nnx.Rngs, mesh=None):
        self.mesh = mesh
        self.expert_axis_name = expert_shard_axis(n_experts, mesh) if mesh is not None else None
        self.n_experts = n_experts
        self.n_active = n_active
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.dtype = dtype

        # Router: near-uniform init
        self.router = nnx.Param(jax.random.normal(rngs.params(), (d_model, n_experts)) * router_init_std)
        # Bias: load-balancing bias, init 0, updated outside gradient
        self.bias = nnx.Param(jnp.zeros((n_experts,), dtype=jnp.float32))

        # Routed experts - store weights as stacked tensors for efficiency:
        # [n_experts, d_model, d_ff] etc. This also makes expert-parallel sharding easy.
        std = (1.0 / d_model) ** 0.5
        self.expert_w_gate = nnx.Param(jax.random.normal(rngs.params(), (n_experts, d_model, d_ff)) * std)
        self.expert_w_up = nnx.Param(jax.random.normal(rngs.params(), (n_experts, d_model, d_ff)) * std)
        self.expert_w_down = nnx.Param(jax.random.normal(rngs.params(), (n_experts, d_ff, d_model)) * std)

        # Shared expert(s)
        if n_shared_experts == 1:
            self.shared_w_gate = nnx.Param(jax.random.normal(rngs.params(), (d_model, d_ff)) * std)
            self.shared_w_up = nnx.Param(jax.random.normal(rngs.params(), (d_model, d_ff)) * std)
            self.shared_w_down = nnx.Param(jax.random.normal(rngs.params(), (d_ff, d_model)) * std)
        else:
            self.shared_w_gate = nnx.Param(jax.random.normal(rngs.params(), (n_shared_experts, d_model, d_ff)) * std)
            self.shared_w_up = nnx.Param(jax.random.normal(rngs.params(), (n_shared_experts, d_model, d_ff)) * std)
            self.shared_w_down = nnx.Param(jax.random.normal(rngs.params(), (n_shared_experts, d_ff, d_model)) * std)

    def _shared_ffn(self, x):
        if self.n_shared_experts == 1:
            return swiglu(x, self.shared_w_gate, self.shared_w_up, self.shared_w_down, self.dtype)
        # multiple shared: average
        out = 0.0
        for i in range(self.n_shared_experts):
            out = out + swiglu(x, self.shared_w_gate[i], self.shared_w_up[i], self.shared_w_down[i], self.dtype)
        return out / self.n_shared_experts

    def __call__(self, x):
        """x: [B, S, d_model]. Returns (output [B,S,d_model], aux dict)."""
        B, S, D = x.shape
        N = self.n_experts
        K = self.n_active
        x_f = x.astype(jnp.float32)

        # ---- Router ----
        gate_logits = x_f @ self.router.astype(jnp.float32)  # [B, S, N]
        scores = jax.nn.sigmoid(gate_logits)  # [B, S, N]

        # Selection uses scores + bias (bias not in gradient path for weights)
        effective = scores + self.bias[None, None, :].astype(jnp.float32)
        top_scores, top_idx = jax.lax.top_k(effective, K)  # [B, S, K]

        # Gating weights use ORIGINAL sigmoid scores at selected experts, normalized, scaled
        # gather original scores at top_idx
        one_hot = jax.nn.one_hot(top_idx, N, dtype=scores.dtype)  # [B, S, K, N]
        # selected_scores: gather scores at the selected expert indices -> [B, S, K]
        # one_hot * scores[..., None, :] -> [B, S, K, N]; sum over N (last) -> [B, S, K]
        selected_scores = jnp.sum(one_hot * scores[..., None, :], axis=-1)  # [B, S, K]
        # normalize top-k scores to sum 1, then scale
        denom = jnp.sum(selected_scores, axis=-1, keepdims=True) + 1e-9
        weights = (selected_scores / denom) * self.routed_scaling_factor  # [B, S, K]

        # ---- Compute routed expert outputs via sorted ragged dispatch ----
        # Flatten to [B*S, D]
        x_flat = x_f.reshape(B * S, D)
        top_idx_flat = top_idx.reshape(B * S, K)  # [BS, K]
        weights_flat = weights.reshape(B * S, K)  # [BS, K]

        # Cast before dispatch: on a mesh the expert weights are all-gathered on
        # the way in, and gathering bf16 moves half the bytes of fp32.
        wg = self.expert_w_gate.value.astype(self.dtype)
        wu = self.expert_w_up.value.astype(self.dtype)
        wd = self.expert_w_down.value.astype(self.dtype)
        if self.mesh is not None and self.mesh.devices.size > 1:
            # Pin the gather (and the cast feeding it) inside this layer. Blocks
            # run under lax.scan, and XLA is otherwise free to hoist the gather
            # out of the loop -- gathering every layer's experts at once, in
            # fp32, which is a 7.4 GiB buffer for the 7B config and instantly
            # exhausts a 16 GB v5e chip. The barrier keeps it to one bf16
            # layer's worth at a time.
            wg, wu, wd = jax.lax.optimization_barrier((wg, wu, wd))

        if self.mesh is None or self.mesh.devices.size == 1:
            routed_out = ragged_dispatch(x_flat, top_idx_flat, weights_flat,
                                         wg, wu, wd, self.dtype)
        else:
            # jax.lax.ragged_dot has no GSPMD partitioning rule, so it has to run
            # on per-device-local shapes. shard_map gives us exactly that: each
            # device dispatches its own slice of the batch over the full set of
            # experts, which it gathers on entry (FSDP).
            #
            # The gather is written out explicitly rather than left to the
            # shard_map boundary. Handing the weights in as replicated (P())
            # also works forward, but its transpose is a psum, so the backward
            # pass builds a *replicated* fp32 gradient for every layer at once
            # -- 7.4 GiB for the 7B config, which no 16 GB chip can hold.
            # Passing them in sharded makes the transpose a reduce-scatter, so
            # the gradient stays sharded and matches the parameter layout.
            axis = self.expert_axis_name
            w_spec = P() if axis is None else P(axis, None, None)

            @partial(jax.shard_map, mesh=self.mesh,
                     in_specs=(P("data", None), P("data", None), P("data", None),
                               w_spec, w_spec, w_spec),
                     out_specs=P("data", None), check_vma=False)
            def local_dispatch(xs, idx, wts, wg, wu, wd):
                if axis is not None:
                    wg, wu, wd = (jax.lax.all_gather(w, axis, axis=0, tiled=True)
                                  for w in (wg, wu, wd))
                return ragged_dispatch(xs, idx, wts, wg, wu, wd, self.dtype)

            routed_out = local_dispatch(x_flat, top_idx_flat, weights_flat, wg, wu, wd)
        routed_out = routed_out.reshape(B, S, D)

        # ---- Shared expert (always on, weight 1.0) ----
        shared_out = self._shared_ffn(x)  # [B, S, D]

        output = routed_out + shared_out

        # ---- Aux losses ----
        # z-loss: 1e-3 * mean(logsumexp(gate_logits)^2)
        lse = jax.nn.logsumexp(gate_logits, axis=-1)  # [B, S]
        router_z_loss = jnp.mean(lse ** 2)

        # balance loss: sequence-wise fraction of tokens routed to each expert
        # count per expert: for each sequence, fraction of tokens that selected expert i
        # counts: [B, N] = sum over S of one_hot(top_idx)
        # one_hot over top_idx: [B, S, K, N] -> sum over K -> [B, S, N] -> sum over S -> [B, N]
        sel = jnp.sum(one_hot, axis=-2)  # [B, S, N]
        counts = jnp.sum(sel, axis=1)  # [B, N]
        fractions = counts / (S * K)  # [B, N] fraction of (token,slot) pairs per expert
        # balance loss = N * sum(f_i^2) averaged over batch
        balance_loss = N * jnp.sum(fractions ** 2, axis=-1)  # [B]
        balance_loss = jnp.mean(balance_loss)

        # mean entropy of routing distribution (for logging; lower = more concentrated)
        probs = scores / (jnp.sum(scores, axis=-1, keepdims=True) + 1e-9)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-9), axis=-1)  # [B, S]
        mean_entropy = jnp.mean(entropy)

        # expert counts for bias update (sum over batch+seq) [N]
        expert_counts = jnp.sum(counts, axis=0)  # [N]

        aux = {
            "router_z_loss": router_z_loss,
            "balance_loss": balance_loss,
            "mean_entropy": mean_entropy,
            "expert_counts": expert_counts,
        }
        return output.astype(jnp.float32), aux