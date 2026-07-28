"""GQA attention with RoPE, using Splash (flash) attention on TPU."""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
from jax.sharding import PartitionSpec as P

from .rope import precompute_rope, apply_rope

try:
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as _spk,
        splash_attention_mask as _spm,
    )
except ImportError:  # non-TPU backends
    _spk = None


_MIN_BLOCK = 128


@functools.lru_cache(maxsize=8)
def _splash_mask(seq_len: int, n_q_heads: int):
    """Causal mask description for splash. Pure Python/numpy, so safe to cache."""
    return _spm.MultiHeadMask([_spm.CausalMask((seq_len, seq_len))] * n_q_heads)


def _splash_kernel(seq_len: int, n_q_heads: int):
    """Build a causal splash kernel for this shape.

    Deliberately *not* cached across calls. `make_splash_mha` materializes the
    block-sparse mask metadata as jnp arrays, so under jit those are tracers
    belonging to the trace that built them; reusing a cached kernel in a later
    trace leaks them (UnexpectedTracerError). Only the mask description above,
    which holds no arrays, is cached.
    """
    block = min(512, seq_len)
    kv_block = min(1024, seq_len)
    sizes = _spk.BlockSizes(
        block_q=block, block_kv=kv_block, block_kv_compute=block,
        block_q_dkv=block, block_kv_dkv=kv_block, block_kv_dkv_compute=block,
        block_q_dq=block, block_kv_dq=kv_block,
    )
    return _spk.make_splash_mha(mask=_splash_mask(seq_len, n_q_heads),
                                head_shards=1, q_seq_shards=1, block_sizes=sizes)


def _can_splash(seq_len: int, head_dim: int) -> bool:
    return (_spk is not None
            and jax.default_backend() == "tpu"
            and seq_len >= _MIN_BLOCK and seq_len % _MIN_BLOCK == 0
            and head_dim % _MIN_BLOCK == 0)


def causal_attention(q, k, v, head_dim: int, mesh=None):
    """Causal GQA. Splash on TPU, XLA elsewhere.

    q: [B, S, Nq, H], k/v: [B, S, Nkv, H] -> [B, S, Nq, H]

    `jax.nn.dot_product_attention` has no TPU flash path: it materializes the
    full [Nq, S, S] score matrix and applies the causal mask afterwards, so it
    both pays for the masked-out half of the FLOPs and moves ~400 MB per layer
    through HBM. Splash skips fully-masked blocks entirely -- measured 4.3x
    faster end to end (8.36 ms -> 1.95 ms fwd+bwd at S=4096, Nq=12, H=128).
    """
    S = q.shape[1]
    if not _can_splash(S, head_dim):
        return jax.nn.dot_product_attention(q, k, v, is_causal=True)

    n_q_heads = q.shape[2]
    # Splash takes head-major [N, S, H] and does not scale q itself.
    scale = jnp.asarray(1.0 / np.sqrt(head_dim), q.dtype)

    def run(q, k, v):
        kernel = _splash_kernel(S, n_q_heads)
        out = jax.vmap(kernel)(q.transpose(0, 2, 1, 3),
                               k.transpose(0, 2, 1, 3),
                               v.transpose(0, 2, 1, 3))
        return out.transpose(0, 2, 1, 3)

    q = q * scale
    if mesh is None or mesh.devices.size == 1:
        return run(q, k, v)
    # Mosaic kernels have no GSPMD partitioning rule, so the call has to see
    # per-device shapes. The batch is the sharded axis; heads and sequence stay
    # whole on each device, which is what head_shards=q_seq_shards=1 assumes.
    bshape = P("data", None, None, None)
    return jax.shard_map(run, mesh=mesh, in_specs=(bshape,) * 3,
                         out_specs=bshape, check_vma=False)(q, k, v)


class Attention(nnx.Module):
    """Grouped Query Attention.

    n_q_heads query heads, n_kv_heads key/value heads (n_q % n_kv == 0).
    head_dim is per-head dimension. Uses BTNH layout.
    """

    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int, head_dim: int,
                 dtype, rngs: nnx.Rngs, mesh=None):
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.mesh = mesh

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

        out = causal_attention(q, k, v, self.head_dim, self.mesh)
        out = out.reshape(B, S, self.n_q_heads * self.head_dim).astype(self.dtype)
        out = out @ self.wo.astype(self.dtype)
        return out.astype(jnp.float32)
