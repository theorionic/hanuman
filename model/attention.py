"""GQA attention with RoPE, using jax.nn.dot_product_attention (flash)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .rope import precompute_rope, apply_rope


class Attention(nnx.Module):
    """Grouped Query Attention.

    n_q_heads query heads, n_kv_heads key/value heads (n_q % n_kv == 0).
    head_dim is per-head dimension. Uses BTNH layout.
    """

    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int, head_dim: int,
                 dtype, rngs: nnx.Rngs):
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        qkv_dim = n_q_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        std = (1.0 / d_model) ** 0.5
        self.wq = nnx.Param(jax.random.normal(rngs.params(), (d_model, qkv_dim)) * std)
        self.wk = nnx.Param(jax.random.normal(rngs.params(), (d_model, kv_dim)) * std)
        self.wv = nnx.Param(jax.random.normal(rngs.params(), (d_model, kv_dim)) * std)
        self.wo = nnx.Param(jax.random.normal(rngs.params(), (qkv_dim, d_model)) * std)

    def __call__(self, x, cos, sin, positions=None):
        B, S, D = x.shape
        x = x.astype(self.dtype)
        q = x @ self.wq.astype(self.dtype)  # [B, S, Nq*H]
        k = x @ self.wk.astype(self.dtype)  # [B, S, Nkv*H]
        v = x @ self.wv.astype(self.dtype)
        q = q.reshape(B, S, self.n_q_heads, self.head_dim)
        k = k.reshape(B, S, self.n_kv_heads, self.head_dim)
        v = v.reshape(B, S, self.n_kv_heads, self.head_dim)

        q, k = apply_rope(q, k, cos.astype(self.dtype), sin.astype(self.dtype), positions)

        # jax.nn.dot_product_attention expects BTNH / BSKH layout, handles GQA
        out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        out = out.reshape(B, S, self.n_q_heads * self.head_dim)
        out = out @ self.wo.astype(self.dtype)
        return out.astype(jnp.float32)