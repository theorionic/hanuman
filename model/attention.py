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
                 rope_base: float, yarn_factor: float, max_seq_len: int,
                 dtype, rngs: nnx.Rngs):
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.yarn_factor = yarn_factor
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        qkv_dim = n_q_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        std = (1.0 / d_model) ** 0.5
        self.wq = nnx.Param(jax.random.normal(rngs.params(), (d_model, qkv_dim)) * std)
        self.wk = nnx.Param(jax.random.normal(rngs.params(), (d_model, kv_dim)) * std)
        self.wv = nnx.Param(jax.random.normal(rngs.params(), (d_model, kv_dim)) * std)
        self.wo = nnx.Param(jax.random.normal(rngs.params(), (qkv_dim, d_model)) * std)

        # Precompute RoPE tables (registered as a buffer-like Param so it's in state,
        # but we mark it non-trainable by not including in optimizer mask).
        cos, sin = precompute_rope(head_dim, max_seq_len, rope_base, yarn_factor)
        # store as plain arrays (not nnx.Param) -> they won't be in nnx.split state by default?
        # Actually nnx stores attributes that are arrays as part of state. Use a separate holder.
        self.cos = cos.astype(dtype)
        self.sin = sin.astype(dtype)

    def __call__(self, x, positions=None):
        B, S, D = x.shape
        x = x.astype(self.dtype)
        q = x @ self.wq.astype(self.dtype)  # [B, S, Nq*H]
        k = x @ self.wk.astype(self.dtype)  # [B, S, Nkv*H]
        v = x @ self.wv.astype(self.dtype)
        q = q.reshape(B, S, self.n_q_heads, self.head_dim)
        k = k.reshape(B, S, self.n_kv_heads, self.head_dim)
        v = v.reshape(B, S, self.n_kv_heads, self.head_dim)

        q, k = apply_rope(q, k, self.cos, self.sin, positions)

        # jax.nn.dot_product_attention expects BTNH / BSKH layout, handles GQA
        out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        out = out.reshape(B, S, self.n_q_heads * self.head_dim)
        out = out @ self.wo.astype(self.dtype)
        return out.astype(jnp.float32)