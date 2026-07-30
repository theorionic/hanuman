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
    """Normalize a window value to `None` (full causal) or a positive int.

    0 and None both mean "full causal attention", so a 0 becomes None and the
    downstream `local_window_size=None` path is taken.

    The window must be *static*. It shapes the attention mask, and Splash
    materializes that mask's block-sparse metadata at trace time, so a traced
    window silently disables Splash and falls back to `dot_product_attention`
    -- which is 5.5x slower on TPU (measured: 75.2 ms vs 13.6 ms for 8 layers
    fwd+bwd at S=4096). That is a big enough cliff to be worth an exception
    rather than a silent deoptimization: per-layer windows are selected with
    `lax.switch` over a static tuple of choices (see `causal_attention`), never
    by threading a traced window through.
    """
    if window is None:
        return None
    if _is_traced(window):
        raise TypeError(
            "causal_attention got a traced sliding-window size. The window must "
            "be static (it shapes the Splash block-sparse mask at trace time). "
            "Use window_sel + window_choices to pick a per-layer window inside a "
            "scan.")
    return int(window) or None


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
    # Tile sizes decide how dense the per-tile matmuls handed to the MXU are.
    # Measured fwd+bwd for one layer at 1 sequence/chip, bf16, causal:
    #
    #   block_q/block_kv |  S=4096  | S=16384
    #     256 / 512      |  3.44 ms | 46.25 ms
    #     512 / 1024     |  2.24    | 26.62
    #    1024 / 2048     |  2.38    | 24.55   <- 7.2% faster at 16384
    #
    # 512/1024 is optimal at 4096 and 1024/2048 at 16384, so scale with the
    # sequence and cap: past 1024 the tiles stop fitting VMEM well and the win
    # reverses (2048/2048 measured 25.07 ms).
    block = min(1024, max(512, seq_len // 16))
    block = min(block, seq_len)
    kv_block = min(2 * block, seq_len)
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


def causal_attention(q, k, v, head_dim: int, mesh=None, window: int | None = None,
                     window_sel=None, window_choices=None):
    """Causal GQA with an optional sliding window. Splash on TPU, XLA elsewhere.

    q: [B, S, Nq, H], k/v: [B, S, Nkv, H] -> [B, S, Nq, H]

    Two ways to specify the window, both of which keep it *static* so the Splash
    mask can be built at trace time:

      - `window`: a plain int (or None for full causal). Used when the caller
        knows the window at trace time, i.e. outside a scan.
      - `window_sel` + `window_choices`: `window_choices` is a static tuple of
        the distinct windows this layer might use, and `window_sel` is a traced
        index into it. This is what the scanned BlockStack uses for the SWA
        hybrid: the scan compiles one body, but the body dispatches through
        `lax.switch` to a branch per window, each with its own statically-masked
        Splash kernel. XLA's Conditional executes only the selected branch, so
        the cost is one attention call, not len(window_choices) of them.

    Threading a *traced* window straight into the mask instead is what the
    naive version did, and it silently drops every layer onto the XLA fallback
    (see `_normalize_window`).
    """
    if window_sel is not None and window_choices and len(window_choices) > 1:
        branches = [functools.partial(_attend, head_dim=head_dim, mesh=mesh,
                                      window=_normalize_window(w))
                    for w in window_choices]
        return jax.lax.switch(window_sel, branches, q, k, v)
    if window_sel is not None and window_choices:
        window = window_choices[0]
    return _attend(q, k, v, head_dim=head_dim, mesh=mesh,
                   window=_normalize_window(window))


def _attend(q, k, v, *, head_dim: int, mesh=None, window: int | None = None):
    """One causal (optionally sliding-window) attention with a static window.

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
    splash_window = _splash_window(window)
    if not _can_splash(S, head_dim, splash_window):
        # XLA fallback (CPU/GPU, or a shape Splash cannot tile).
        # local_window_size=(left, right): `window` tokens in the past, 0 in the
        # future. is_causal=True still applies. window=None -> full causal.
        lws = None if window is None else (int(window), 0)
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
                 dtype, rngs: nnx.Rngs, mesh=None, window: int | None = None,
                 window_choices=None):
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
        # When this layer lives in a scanned BlockStack whose layers do NOT all
        # share a window, one compiled body has to serve several windows. The
        # distinct windows are enumerated here (static) and the scan supplies a
        # traced index into them at call time; see `causal_attention`.
        self.window_choices = tuple(window_choices) if window_choices else None

        qkv_dim = n_q_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        std = (1.0 / d_model) ** 0.5
        self.wq = nnx.Param(jax.random.normal(rngs.params(), (d_model, qkv_dim)) * std)
        self.wk = nnx.Param(jax.random.normal(rngs.params(), (d_model, kv_dim)) * std)
        self.wv = nnx.Param(jax.random.normal(rngs.params(), (d_model, kv_dim)) * std)
        self.wo = nnx.Param(jax.random.normal(rngs.params(), (qkv_dim, d_model)) * std)

    def __call__(self, x, cos, sin, positions=None, window: int | None = None,
                 window_sel=None):
        # `window` (static int) can be overridden at call time; `window_sel` is
        # the scanned per-layer index into self.window_choices used by the SWA
        # hybrid. Neither set -> full causal attention.
        w = window if window is not None else self.window
        B, S, D = x.shape
        x = x.astype(self.dtype)
        q = x @ self.wq.astype(self.dtype)  # [B, S, Nq*H]
        k = x @ self.wk.astype(self.dtype)  # [B, S, Nkv*H]
        v = x @ self.wv.astype(self.dtype)
        q = q.reshape(B, S, self.n_q_heads, self.head_dim)
        k = k.reshape(B, S, self.n_kv_heads, self.head_dim)
        v = v.reshape(B, S, self.n_kv_heads, self.head_dim)

        q, k = apply_rope(q, k, cos.astype(self.dtype), sin.astype(self.dtype), positions)

        out = causal_attention(q, k, v, self.head_dim, self.mesh, window=w,
                               window_sel=window_sel,
                               window_choices=self.window_choices)
        out = out.reshape(B, S, self.n_q_heads * self.head_dim).astype(self.dtype)
        out = out @ self.wo.astype(self.dtype)
        return out.astype(jnp.float32)
