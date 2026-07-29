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


def _is_traced(x) -> bool:
    """True if `x` is a JAX tracer (cannot branch on it with Python `if`)."""
    import jax.core as _core
    return isinstance(x, _core.Tracer)


def _normalize_window(window):
    """Normalize a window value to `None` (full causal) or an int/traced int.

    0 and None both mean "full causal attention". A concrete 0 -> None so the
    downstream `local_window_size=None` path is taken. A *traced* 0 (from a
    lax.scan per-layer array) cannot be tested with a Python `if`, so it is
    passed through as-is and `causal_attention` handles it with lax.cond /
    dynamic masking. Positive ints are returned unchanged.
    """
    if window is None:
        return None
    try:
        # Concrete int / numpy scalar.
        if int(window) == 0:
            return None
        return int(window)
    except (TypeError, ValueError):
        # Traced scalar from scan: leave as-is.
        return window


@functools.lru_cache(maxsize=16)
def _splash_mask(seq_len: int, n_q_heads: int, window: int | None):
    """Causal mask description for splash. Pure Python/numpy, so safe to cache.

    When `window` is None this is a plain causal mask. When `window` is set it
    is a *sliding-window causal* mask: token i attends to j iff
    j <= i and i - j < window (i.e. `window` tokens to the left plus itself,
    nothing to the right). We build this with `LocalMask((left, right))` where
    left=window and right=0 -- right=0 collapses to the causal constraint
    (q_ids + 0 >= kv_ids  <=>  q_ids >= kv_ids), and left=window adds the
    window bound (q_ids - window <= kv_ids). This is exactly the
    "causal + sliding window" mask used by Mistral/Gemma long-context layers.
    """
    if window is None:
        masks = [_spm.CausalMask((seq_len, seq_len))] * n_q_heads
    else:
        masks = [_spm.LocalMask((seq_len, seq_len),
                                window_size=(window, 0), offset=0)
                 for _ in range(n_q_heads)]
    return _spm.MultiHeadMask(masks)


def _splash_kernel(seq_len: int, n_q_heads: int, window: int | None):
    """Build a causal splash kernel for this shape.

    Deliberately *not* cached across calls. `make_splash_mh` materializes the
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
    return _spk.make_splash_mha(mask=_splash_mask(seq_len, n_q_heads, window),
                                head_shards=1, q_seq_shards=1, block_sizes=sizes)


def _can_splash(seq_len: int, head_dim: int, window: int | None) -> bool:
    # Splash tiles in blocks of 128, so the window must be 128-aligned too:
    # an unaligned window would cut a tile in half and the block-sparse mask
    # metadata can only drop whole tiles. Round up to the nearest 128 on TPU.
    return (_spk is not None
            and jax.default_backend() == "tpu"
            and seq_len >= _MIN_BLOCK and seq_len % _MIN_BLOCK == 0
            and head_dim % _MIN_BLOCK == 0
            and (window is None
                 or (window >= _MIN_BLOCK and window % _MIN_BLOCK == 0)))


def _splash_window(window: int | None) -> int | None:
    """Round a SWA window up to a 128-aligned value for the Splash path.

    Only matters on TPU; on CPU we never reach the Splash branch. We round up
    (never down) so the window is at least as large as requested -- rounding
    down would drop tokens the model expects to see.
    """
    if window is None:
        return None
    return ((window + _MIN_BLOCK - 1) // _MIN_BLOCK) * _MIN_BLOCK


def causal_attention(q, k, v, head_dim: int, mesh=None, window: int | None = None):
    """Causal GQA. Splash on TPU, XLA elsewhere.

    q: [B, S, Nq, H], k/v: [B, S, Nkv, H] -> [B, S, Nq, H]

    `jax.nn.dot_product_attention` has no TPU flash path: it materializes the
    full [Nq, S, S] score matrix and applies the causal mask afterwards, so it
    both pays for the masked-out half of the FLOPs and moves ~400 MB per layer
    through HBM. Splash skips fully-masked blocks entirely -- measured 4.3x
    faster end to end (8.36 ms -> 1.95 ms fwd+bwd at S=4096, Nq=12, H=128).

    When `window` is not None, attention is *sliding-window causal*: each query
    attends only to the previous `window` tokens plus itself (no future). This
    is the local-attention half of the SWA hybrid. Full-attention layers call
    this with window=None and behave exactly as before.

    XLA path: `jax.nn.dot_product_attention` accepts
    `local_window_size: int | tuple[int, int] | None` as a (left, right) pair.
    For causal SWA we pass (window, 0): `window` tokens to the left, 0 to the
    right (no future), combined with is_causal=True. When window is None we
    pass local_window_size=None and get plain causal attention.

    Splash path: we swap `CausalMask` for `LocalMask((window, 0))`, which
    encodes the same causal+window constraint as a block-sparse mask so the
    kernel still skips fully-masked tiles. The window is rounded up to a
    multiple of 128 (see `_splash_window`) because Splash tiles at 128.
    """
    S = q.shape[1]
    # Normalize: 0 / None -> full causal. A traced 0 (from a per-layer scan
    # array) can't be tested with a Python `if`, so we keep it as a traced int
    # and let the XLA path substitute the full seq_len for 0 below.
    if window is not None:
        try:
            if int(window) == 0:
                window = None
        except (TypeError, ValueError):
            pass  # traced scalar, handle in the XLA branch

    # Splash needs a *static* window to build the block-sparse mask at trace
    # time, so a traced window (per-layer scan array) cannot use Splash. In that
    # case fall through to the XLA path, which accepts a dynamic window.
    splash_window = _splash_window(window) if isinstance(window, int) else None
    use_splash = (_can_splash(S, head_dim, splash_window)
                  and not _is_traced(window))
    if not use_splash:
        # XLA fallback. local_window_size=(left, right): left=window tokens in
        # the past, right=0 (no future). is_causal=True still applies. When
        # window is None -> full causal (local_window_size=None). When window is
        # a traced scalar, 0 means full causal -- substitute the full seq_len so
        # the local window is a no-op (is_causal already bounds it).
        if window is None:
            lws = None
        elif _is_traced(window):
            eff = jnp.where(window == 0, S, window)
            lws = (eff, 0)
        else:
            lws = (int(window), 0)
        return jax.nn.dot_product_attention(q, k, v, is_causal=True,
                                            local_window_size=lws)

    n_q_heads = q.shape[2]
    # Splash takes head-major [N, S, H] and does not scale q itself.
    scale = jnp.asarray(1.0 / np.sqrt(head_dim), q.dtype)

    def run(q, k, v):
        kernel = _splash_kernel(S, n_q_heads, splash_window)
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
                 dtype, rngs: nnx.Rngs, mesh=None, window: int | None = None):
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.mesh = mesh
        # Sliding-window size for this layer. None = full causal attention.
        # Set per-layer by the transformer to implement the SWA hybrid
        # (even layers local, odd layers global, or whatever swa_period gives).
        # Stored as a plain Python int (or None) so it is static at trace time
        # -- the window shapes the attention mask, which must be known when the
        # kernel is compiled, so it cannot be a traced array.
        self.window = window

        qkv_dim = n_q_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        std = (1.0 / d_model) ** 0.5
        self.wq = nnx.Param(jax.random.normal(rngs.params(), (d_model, qkv_dim)) * std)
        self.wk = nnx.Param(jax.random.normal(rngs.params(), (d_model, kv_dim)) * std)
        self.wv = nnx.Param(jax.random.normal(rngs.params(), (d_model, kv_dim)) * std)
        self.wo = nnx.Param(jax.random.normal(rngs.params(), (qkv_dim, d_model)) * std)

    def __call__(self, x, cos, sin, positions=None, window: int | None = None):
        # `window` can be overridden at call time. This is how the BlockStack
        # scan applies a per-layer window from a `windows` array: the scan body
        # passes the per-step value (0 = full causal, >0 = SWA window). If
        # neither the call-time nor the init-time window is set, the layer is
        # full causal attention. 0 is treated as "no window" everywhere.
        w = window if window is not None else self.window
        # A traced 0 from lax.scan can't be compared to 0 with a Python `if`,
        # so normalize via a helper that handles both concrete and traced ints.
        w = _normalize_window(w)
        B, S, D = x.shape
        x = x.astype(self.dtype)
        q = x @ self.wq.astype(self.dtype)  # [B, S, Nq*H]
        k = x @ self.wk.astype(self.dtype)  # [B, S, Nkv*H]
        v = x @ self.wv.astype(self.dtype)
        q = q.reshape(B, S, self.n_q_heads, self.head_dim)
        k = k.reshape(B, S, self.n_kv_heads, self.head_dim)
        v = v.reshape(B, S, self.n_kv_heads, self.head_dim)

        q, k = apply_rope(q, k, cos.astype(self.dtype), sin.astype(self.dtype), positions)

        out = causal_attention(q, k, v, self.head_dim, self.mesh, window=w)
        out = out.reshape(B, S, self.n_q_heads * self.head_dim).astype(self.dtype)
        out = out @ self.wo.astype(self.dtype)
        return out.astype(jnp.float32)
