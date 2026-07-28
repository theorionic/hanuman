"""Transformer block (pre-norm, residual scale) and full model."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .attention import Attention
from .moe import MoE, DenseFFN
from .rope import RopeCache, precompute_rope


class RMSNorm(nnx.Module):
    def __init__(self, d_model: int, eps: float, dtype):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((d_model,), dtype=jnp.float32))
        self.dtype = dtype

    def __call__(self, x):
        x = x.astype(jnp.float32)
        var = jnp.mean(x ** 2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.eps)
        return (x * self.weight).astype(self.dtype)


class Block(nnx.Module):
    """Pre-norm transformer block with residual scale.

    Uses dense FFN for first `dense_layers` layers, MoE for the rest.
    """

    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int, head_dim: int,
                 d_ff: int, n_experts: int, n_active: int, n_shared_experts: int,
                 d_ff_dense: int,
                 router_init_std: float, routed_scaling_factor: float,
                 norm_eps: float, residual_scale_init: float,
                 use_moe: bool, dtype, rngs: nnx.Rngs, mesh=None):
        self.norm1 = RMSNorm(d_model, norm_eps, dtype)
        self.attn = Attention(d_model, n_q_heads, n_kv_heads, head_dim, dtype, rngs, mesh=mesh)
        self.norm2 = RMSNorm(d_model, norm_eps, dtype)
        if use_moe:
            self.ffn = MoE(d_model, d_ff, n_experts, n_active, n_shared_experts,
                           router_init_std, routed_scaling_factor, dtype, rngs,
                           mesh=mesh, d_ff_shared=d_ff_dense)
        else:
            self.ffn = DenseFFN(d_model, d_ff_dense, dtype, rngs)
        self.use_moe = use_moe
        # residual scale (DeepSeek-style learnable scalar, init 1.0)
        self.residual_scale = nnx.Param(jnp.array(residual_scale_init, dtype=jnp.float32))

    def __call__(self, x, cos, sin, positions=None):
        # pre-norm attention
        h = self.norm1(x)
        a = self.attn(h, cos, sin, positions=positions)
        x = x + self.residual_scale.astype(jnp.float32) * a
        # pre-norm FFN
        h = self.norm2(x)
        if self.use_moe:
            f, aux = self.ffn(h)
            x = x + self.residual_scale.astype(jnp.float32) * f
            return x, aux
        else:
            f = self.ffn(h)
            x = x + self.residual_scale.astype(jnp.float32) * f
            return x, {}


def remat_policy(name: str):
    """Map a config string to a `jax.checkpoint` policy.

    Which intermediates are kept for the backward pass is the single biggest
    throughput knob in this model. `full` (save nothing, recompute everything)
    is the textbook choice, but it measures at ~76 ms per MoE layer against an
    11 ms forward: the recompute drags the expert-weight all-gather and the
    routing sort onto the backward's critical path, where there is no other
    work to overlap them with. Saving just the grouped-matmul outputs removes
    most of that while keeping the big [T*K, D] tensors out of HBM.
    """
    p = jax.checkpoint_policies
    return {
        "none": None,                                   # no remat at all
        "full": p.nothing_saveable,                     # recompute everything
        "dots": p.checkpoint_dots,                      # keep every matmul output
        "dots_no_batch": p.checkpoint_dots_with_no_batch_dims,
        "experts": p.save_only_these_names("moe_gate", "moe_up", "moe_out"),
    }[name]


class BlockStack(nnx.Module):
    """`n` structurally identical blocks with their params stacked on axis 0.

    The forward pass is one `lax.scan`, so XLA compiles a single block body
    instead of `n` unrolled copies. That keeps compile time and compiler memory
    flat in depth -- unrolling 21 MoE layers (each with its own shard_map and
    remat region) produced an HLO graph that took minutes and hundreds of GB of
    host RAM to optimize.
    """

    def __init__(self, n: int, make_block, remat: bool, policy: str = "full"):
        blocks = [make_block() for _ in range(n)]
        graphdefs, states = zip(*[nnx.split(b) for b in blocks])
        # Same structure for every block, so stacking leaf-by-leaf gives one
        # Block whose parameters carry a leading layer axis.
        stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
        self.blocks = nnx.merge(graphdefs[0], stacked)
        self.n = n
        self.remat = remat
        self.policy = policy

    def __call__(self, x, cos, sin, positions=None):
        graphdef, state = nnx.split(self.blocks)

        def body(carry, layer_state):
            block = nnx.merge(graphdef, layer_state)
            return block(carry, cos, sin, positions=positions)

        pol = remat_policy(self.policy) if self.remat else None
        f = body if (not self.remat or self.policy == "none") else jax.checkpoint(body, policy=pol)
        x, aux = jax.lax.scan(f, x, state)
        return x, aux


class Transformer(nnx.Module):
    """Full MoE transformer with embedding tying."""

    def __init__(self, config, dtype, rngs: nnx.Rngs, mesh=None):
        self.config = config
        self.dtype = dtype
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

        # Token embedding (tied with output)
        std = (1.0 / config.d_model) ** 0.5
        self.wte = nnx.Param(jax.random.normal(rngs.params(), (config.vocab_size, config.d_model)) * std)

        # One RoPE table shared by every layer: it is a constant, identical
        # across blocks, so storing it per-block would replicate it `layers`
        # times in HBM for no reason.
        max_seq_len = max(config.seq_len, config.infer_seq_len)
        cos, sin = precompute_rope(config.head_dim, max_seq_len,
                                   config.rope_base, config.yarn_factor,
                                   config.yarn_beta_fast, config.yarn_beta_slow)
        self.rope_cos = RopeCache(cos.astype(dtype))
        self.rope_sin = RopeCache(sin.astype(dtype))

        # Dense and MoE blocks differ structurally, so they form two stacks.
        def make_block(use_moe):
            return lambda: Block(
                d_model=config.d_model,
                n_q_heads=config.n_q_heads,
                n_kv_heads=config.n_kv_heads,
                head_dim=config.head_dim,
                d_ff=config.d_ff,
                d_ff_dense=(config.d_ff if getattr(config, "d_ff_dense", None) is None
                            else config.d_ff_dense),
                n_experts=config.n_experts,
                n_active=config.n_active,
                n_shared_experts=config.n_shared_experts,
                router_init_std=config.router_init_std,
                routed_scaling_factor=config.routed_scaling_factor,
                norm_eps=config.norm_eps,
                residual_scale_init=config.residual_scale_init,
                use_moe=use_moe,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
            )

        remat = getattr(config, "remat", True)
        policy = getattr(config, "remat_policy", "full")
        self.n_dense = config.dense_layers
        self.n_moe = config.layers - config.dense_layers
        self.dense_stack = (BlockStack(self.n_dense, make_block(False), remat, policy)
                            if self.n_dense else None)
        self.moe_stack = (BlockStack(self.n_moe, make_block(True), remat, policy)
                          if self.n_moe else None)

        self.norm_f = RMSNorm(config.d_model, config.norm_eps, dtype)

    def __call__(self, tokens, positions=None):
        """tokens: [B, S] int. Returns logits [B, S, vocab]."""
        x = self.wte[tokens].astype(jnp.float32)  # [B, S, D]
        cos, sin = self.rope_cos.value, self.rope_sin.value

        if self.dense_stack is not None:
            x, _ = self.dense_stack(x, cos, sin, positions)
        aux_stacked = {}
        if self.moe_stack is not None:
            x, aux_stacked = self.moe_stack(x, cos, sin, positions)

        x = self.norm_f(x)
        # tied output projection: logits = x @ wte.T
        logits = x @ self.wte.T.astype(jnp.float32)  # [B, S, vocab]
        # scan stacks each MoE layer's aux on a leading axis; average over layers
        n_moe = max(1, self.n_moe)
        aux_out = {k: jnp.sum(v, axis=0) / n_moe for k, v in aux_stacked.items()}
        return logits, aux_out

    def generate(self, tokens, positions=None):
        """Forward for generation: returns logits only (no aux)."""
        logits, _ = self(tokens, positions=positions)
        return logits