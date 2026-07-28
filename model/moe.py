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

import jax
import jax.numpy as jnp
import flax.nnx as nnx


def swiglu(x, w_gate, w_up, w_down, dtype):
    """SwiGLU FFN: down(silu(x @ gate) * (x @ up))."""
    g = jax.nn.silu((x @ w_gate.astype(dtype)))
    u = (x @ w_up.astype(dtype))
    h = g * u
    out = h @ w_down.astype(dtype)
    return out.astype(jnp.float32)


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
                 dtype, rngs: nnx.Rngs):
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

        # ---- Compute routed expert outputs via per-token dispatch ----
        # Flatten to [B*S, D]
        x_flat = x_f.reshape(B * S, D)
        top_idx_flat = top_idx.reshape(B * S, K)  # [BS, K]
        weights_flat = weights.reshape(B * S, K)  # [BS, K]

        # For each token and each selected expert, compute expert(x).
        # We do this with a vectorized gather: build [BS, K, D] by selecting expert weights.
        # expert_w_gate: [N, D, d_ff]
        # For each (token, k): x_flat[token] @ expert_w_gate[top_idx_flat[token,k]]
        # Use vmap over K.
        def expert_for_idx(token_idx, k_idx):
            e = top_idx_flat[token_idx, k_idx]
            wg = self.expert_w_gate[e]
            wu = self.expert_w_up[e]
            wd = self.expert_w_down[e]
            return swiglu(x_flat[token_idx], wg, wu, wd, self.dtype)

        token_idxs = jnp.arange(B * S)
        # vmap over tokens and k
        def per_token(ti):
            ks = jnp.arange(K)
            def per_k(ki):
                return expert_for_idx(ti, ki)
            outs = jax.vmap(per_k)(ks)  # [K, D]
            return outs
        expert_outs = jax.vmap(per_token)(token_idxs)  # [BS, K, D]
        # weighted sum
        routed_out = jnp.sum(expert_outs * weights_flat[..., None], axis=1)  # [BS, D]
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