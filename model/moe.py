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

import os
from functools import partial

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.sharding import PartitionSpec as P

from .sharding import expert_shard_axis

def swiglu(x, w_gate, w_up, w_down, dtype):
    """SwiGLU FFN: down(silu(x @ gate) * (x @ up))."""
    g = jax.nn.silu((x @ w_gate.astype(dtype)))
    u = (x @ w_up.astype(dtype))
    h = g * u
    out = h @ w_down.astype(dtype)
    return out.astype(jnp.float32)


_ROW_ALIGN = 128  # TPU tile alignment for the grouped-matmul row dimension
_BARRIER = os.environ.get("HANUMAN_MOE_BARRIER", "1") == "1"


def ragged_dispatch(x_flat, top_idx, weights, w_gate, w_up, w_down, dtype):
    """Route each token to its top-k experts and run SwiGLU, one dot per expert.

    x_flat:  [T, D]      tokens (already in compute dtype)
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
    xs = x_flat[order // K]                           # [T*K, D] tokens in expert order
    group_sizes = jnp.bincount(slot_expert, length=N)  # [N] rows per expert

    # ragged_dot wants the row count tile-aligned. Pad with rows that belong to
    # no group (group_sizes still sums to T*K, so the padding is never read) and
    # drop them after.
    rows = T * K
    padded = -(-rows // _ROW_ALIGN) * _ROW_ALIGN
    if padded != rows:
        xs = jnp.pad(xs, ((0, padded - rows), (0, 0)))

    # Named so a remat policy can keep these three specifically: recomputing
    # them in the backward means redoing the expert all-gather and the sort.
    ckpt = jax.ad_checkpoint.checkpoint_name
    g = ckpt(jax.lax.ragged_dot(xs, w_gate, group_sizes), "moe_gate")   # [rows, F]
    u = ckpt(jax.lax.ragged_dot(xs, w_up, group_sizes), "moe_up")       # [rows, F]
    h = (jax.nn.silu(g) * u).astype(dtype)
    ys = ckpt(jax.lax.ragged_dot(h, w_down, group_sizes), "moe_out")    # [rows, D]
    ys = ys[:rows]

    # Undo the permutation. Written as a gather through the inverse permutation
    # rather than `zeros.at[order].set(ys)`: a scatter of [T*K, D] costs 1.28 ms
    # on v5e against 0.17 ms for the equivalent gather, and the extra argsort
    # that builds the inverse is 0.06 ms.
    inv = jnp.argsort(order)
    out = ys[inv].reshape(T, K, D)
    # Combine the K slots. einsum with an fp32 accumulator keeps the summation
    # exact without materializing a [T, K, D] fp32 copy (200 MB at S=4096).
    return jnp.einsum("tkd,tk->td", out, weights.astype(out.dtype),
                      preferred_element_type=jnp.float32)


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

        # Gating weights use ORIGINAL sigmoid scores at selected experts, normalized, scaled.
        # take_along_axis gathers them directly; building a [B, S, K, N] one-hot
        # and contracting it costs S*K*N elements of HBM traffic per layer (10 MB
        # at S=4096, N=80) to express the same gather.
        selected_scores = jnp.take_along_axis(scores, top_idx, axis=-1)  # [B, S, K]
        # normalize top-k scores to sum 1, then scale
        denom = jnp.sum(selected_scores, axis=-1, keepdims=True) + 1e-9
        weights = (selected_scores / denom) * self.routed_scaling_factor  # [B, S, K]

        # ---- Compute routed expert outputs via sorted ragged dispatch ----
        # Flatten to [B*S, D]. Cast to compute dtype *before* the dispatch: the
        # first thing it does is gather T*K = 8x as many rows as there are
        # tokens, and gathering bf16 moves half the bytes of fp32.
        x_flat = x.astype(self.dtype).reshape(B * S, D)
        top_idx_flat = top_idx.reshape(B * S, K)  # [BS, K]
        weights_flat = weights.reshape(B * S, K)  # [BS, K]

        # Cast before dispatch: on a mesh the expert weights are all-gathered on
        # the way in, and gathering bf16 moves half the bytes of fp32.
        wg = self.expert_w_gate.value.astype(self.dtype)
        wu = self.expert_w_up.value.astype(self.dtype)
        wd = self.expert_w_down.value.astype(self.dtype)
        if _BARRIER and self.mesh is not None and self.mesh.devices.size > 1:
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

        # balance loss: sequence-wise fraction of tokens routed to each expert.
        # A bincount per sequence gives the same [B, N] histogram as summing a
        # one-hot, without materializing the [B, S, K, N] tensor.
        counts = jax.vmap(lambda t: jnp.bincount(t.reshape(-1), length=N))(top_idx)
        counts = counts.astype(jnp.float32)  # [B, N]
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