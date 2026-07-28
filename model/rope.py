"""RoPE (Rotary Position Embeddings) + YaRN long-context extension.

Precomputes cos/sin tables for the configured head_dim and max sequence length.
YaRN: scales the interpolation factor for frequencies to extend context.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.nnx as nnx


class RopeCache(nnx.Variable):
    """Precomputed cos/sin table.

    A distinct Variable type (not nnx.Param) so `nnx.split(model, nnx.Param, ...)`
    keeps it out of the trainable state: these tables are constants, and if they
    land in the optimizer they both get gradient updates (silently drifting the
    positional encoding) and carry a full-size momentum buffer.
    """


def _yarn_find_correction_dim(num_rotations: float, dim: int, base: float,
                              max_position_embeddings: int, beta_fast: float, beta_slow: float) -> float:
    return (dim * jnp.log(max_position_embeddings / (2 * jnp.pi))) / (2 * num_rotations * jnp.log(base))


def _yarn_linear_ramp_mask(min_value: float, max_value: float, dim: int) -> jnp.ndarray:
    if min_value == max_value:
        max_value += 0.001
    linear = jnp.arange(dim, dtype=jnp.float32) / dim
    linear = (linear - min_value) / (max_value - min_value)
    return jnp.clip(linear, 0.0, 1.0)


def compute_inv_freqs(head_dim: int, base: float, yarn_factor: float,
                      max_seq_len: int, beta_fast: float = 32.0, beta_slow: float = 1.0) -> jnp.ndarray:
    """Compute inverse frequencies, applying YaRN scaling when yarn_factor > 1."""
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    if yarn_factor != 1.0:
        # YaRN: scale inv_freq by interpolation factor
        # Compute correction range
        high_freq = _yarn_find_correction_dim(beta_fast, head_dim, base, max_seq_len, beta_fast, beta_slow)
        low_freq = _yarn_find_correction_dim(beta_slow, head_dim, base, max_seq_len, beta_fast, beta_slow)
        # interpolation factor per dim
        dim = head_dim // 2
        freqs = jnp.arange(dim, dtype=jnp.float32)
        # mask: dims below low_freq -> extrapolate (factor 1/yarn), above high -> interpolate (1/yarn)
        ramp = _yarn_linear_ramp_mask(low_freq, high_freq, dim)
        # extrapolation factor for low-freq dims, interpolation for high-freq
        inv_freq_extrapolation = 1.0 / yarn_factor
        inv_freq_interpolation = 1.0 / yarn_factor
        # blend: low freq -> extrapolate (1.0), high freq -> interpolate (1/yarn)
        factor = 1.0 - (1.0 - 1.0 / yarn_factor) * ramp
        inv_freq = inv_freq / factor
    return inv_freq


def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0,
                    yarn_factor: float = 1.0, beta_fast: float = 32.0, beta_slow: float = 1.0):
    """Precompute cos/sin tables of shape [max_seq_len, head_dim].

    Returns (cos, sin) each [max_seq_len, head_dim].
    """
    inv_freq = compute_inv_freqs(head_dim, base, yarn_factor, max_seq_len, beta_fast, beta_slow)
    # [max_seq_len, head_dim//2]
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)  # [S, D/2]
    # repeat interleave to full head_dim: cos/sin [S, D]
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # [S, D]
    cos = jnp.cos(emb)
    sin = jnp.sin(emb)
    return cos, sin


def rotate_half(x):
    """Rotate half of the hidden dims: (x1, x2) -> (-x2, x1)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(q, k, cos, sin, positions=None):
    """Apply RoPE to q and k.

    q, k: [..., S, H, D] (BTNH layout for jax.nn.dot_product_attention)
    cos, sin: [S, D] or [B, S, D] if positions given
    positions: optional [..., S] integer positions; if None use 0..S-1
    """
    # q/k shape: [B, S, N, H]
    if positions is not None:
        cos = jnp.take(cos, positions, axis=0)  # broadcast index
        sin = jnp.take(sin, positions, axis=0)
        # add head axis
        cos = cos[..., None, :]  # [B, S, 1, D]
        sin = sin[..., None, :]
    else:
        S = q.shape[-3]
        cos = cos[:S][None, :, None, :]  # [1, S, 1, D]
        sin = sin[:S][None, :, None, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot