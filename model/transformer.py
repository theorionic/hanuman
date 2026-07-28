"""Transformer block (pre-norm, residual scale) and full model."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .attention import Attention
from .moe import MoE, DenseFFN


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
                 router_init_std: float, routed_scaling_factor: float,
                 norm_eps: float, residual_scale_init: float,
                 rope_base: float, yarn_factor: float, max_seq_len: int,
                 use_moe: bool, dtype, rngs: nnx.Rngs):
        self.norm1 = RMSNorm(d_model, norm_eps, dtype)
        self.attn = Attention(d_model, n_q_heads, n_kv_heads, head_dim,
                              rope_base, yarn_factor, max_seq_len, dtype, rngs)
        self.norm2 = RMSNorm(d_model, norm_eps, dtype)
        if use_moe:
            self.ffn = MoE(d_model, d_ff, n_experts, n_active, n_shared_experts,
                           router_init_std, routed_scaling_factor, dtype, rngs)
        else:
            self.ffn = DenseFFN(d_model, d_ff, dtype, rngs)
        self.use_moe = use_moe
        # residual scale (DeepSeek-style learnable scalar, init 1.0)
        self.residual_scale = nnx.Param(jnp.array(residual_scale_init, dtype=jnp.float32))

    def __call__(self, x, positions=None):
        # pre-norm attention
        h = self.norm1(x)
        a = self.attn(h, positions=positions)
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


class Transformer(nnx.Module):
    """Full MoE transformer with embedding tying."""

    def __init__(self, config, dtype, rngs: nnx.Rngs):
        self.config = config
        self.dtype = dtype
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

        # Token embedding (tied with output)
        std = (1.0 / config.d_model) ** 0.5
        self.wte = nnx.Param(jax.random.normal(rngs.params(), (config.vocab_size, config.d_model)) * std)

        self.blocks: list = nnx.data([])
        for i in range(config.layers):
            use_moe = i >= config.dense_layers
            b = Block(
                d_model=config.d_model,
                n_q_heads=config.n_q_heads,
                n_kv_heads=config.n_kv_heads,
                head_dim=config.head_dim,
                d_ff=config.d_ff,
                n_experts=config.n_experts,
                n_active=config.n_active,
                n_shared_experts=config.n_shared_experts,
                router_init_std=config.router_init_std,
                routed_scaling_factor=config.routed_scaling_factor,
                norm_eps=config.norm_eps,
                residual_scale_init=config.residual_scale_init,
                rope_base=config.rope_base,
                yarn_factor=config.yarn_factor,
                max_seq_len=max(config.seq_len, config.infer_seq_len),
                use_moe=use_moe,
                dtype=dtype,
                rngs=rngs,
            )
            self.blocks.append(b)

        self.norm_f = RMSNorm(config.d_model, config.norm_eps, dtype)

    def __call__(self, tokens, positions=None):
        """tokens: [B, S] int. Returns logits [B, S, vocab]."""
        x = self.wte[tokens].astype(jnp.float32)  # [B, S, D]
        aux_all = {}
        for block in self.blocks:
            x, aux = block(x, positions=positions)
            if aux:
                # accumulate aux from MoE layers
                for k, v in aux.items():
                    if k in aux_all:
                        aux_all[k] = aux_all[k] + v
                    else:
                        aux_all[k] = v
        x = self.norm_f(x)
        # tied output projection: logits = x @ wte.T
        logits = x @ self.wte.T.astype(jnp.float32)  # [B, S, vocab]
        n_moe = max(1, len(self.blocks) - self.config.dense_layers)
        # average aux over MoE layers
        aux_out = {}
        for k, v in aux_all.items():
            aux_out[k] = v / n_moe
        return logits, aux_out

    def generate(self, tokens, positions=None):
        """Forward for generation: returns logits only (no aux)."""
        logits, _ = self(tokens, positions=positions)
        return logits