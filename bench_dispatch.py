"""Microbenchmark for the MoE dispatch, at real per-device shapes.

The step profile (BENCHMARKS.md) puts the dispatch permutation at 21% of the
step at seq 16384 -- pure HBM traffic materializing the [T*K, D] buffer -- and
names a fused gather+grouped-matmul Pallas kernel as the fix. Before writing that
kernel (and especially its custom backward), this isolates the dispatch and times
the candidates head to head, so the decision rests on measured ground truth:

  (a) reference: XLA `x[order//K]` gather + `jax.lax.ragged_dot`  (what moe.py does)
  (b) megablox `gmm` on the pre-gathered rows                     (if available)

Run on the TPU after the training run frees the devices:
    python bench_dispatch.py --seq 4096
    python bench_dispatch.py --seq 16384

Per-device shapes match the `full` config with batch 1/chip (the batch axis is
the sharded one), so T = seq_len, D=1536, F=768, N=80, K=8.
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np


def _inputs(T, D, F, N, K, dtype, seed=0):
    """Build (x_flat, order, group_sizes, w_gate, w_up, w_down) like ragged_dispatch."""
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((T, D)), dtype=dtype)
    # Random top-K expert ids per token, then the same sort ragged_dispatch does.
    top_idx = jnp.asarray(rng.integers(0, N, size=(T, K)), dtype=jnp.int32)
    slot_expert = top_idx.reshape(-1)
    order = jnp.argsort(slot_expert)
    group_sizes = jnp.bincount(slot_expert, length=N).astype(jnp.int32)
    wg = jnp.asarray(rng.standard_normal((N, D, F)) * (D ** -0.5), dtype=dtype)
    wu = jnp.asarray(rng.standard_normal((N, D, F)) * (D ** -0.5), dtype=dtype)
    wd = jnp.asarray(rng.standard_normal((N, F, D)) * (D ** -0.5), dtype=dtype)
    return x, order, group_sizes, wg, wu, wd, K


def ref_forward(x, order, group_sizes, wg, wu, wd, K, dtype):
    """The moe.py forward dispatch: gather -> 3 ragged_dots -> SwiGLU."""
    align = 128
    xs = x[order // K]
    rows = xs.shape[0]
    padded = -(-rows // align) * align
    if padded != rows:
        xs = jnp.pad(xs, ((0, padded - rows), (0, 0)))
    g = jax.lax.ragged_dot(xs, wg, group_sizes)
    u = jax.lax.ragged_dot(xs, wu, group_sizes)
    h = (jax.nn.silu(g) * u).astype(dtype)
    ys = jax.lax.ragged_dot(h, wd, group_sizes)
    return ys[:rows]


def megablox_forward(x, order, group_sizes, wg, wu, wd, K, dtype):
    """megablox gmm on the pre-gathered rows (gather still in XLA)."""
    from jax.experimental.pallas.ops.tpu.megablox import gmm
    xs = x[order // K]
    g = gmm(xs, wg, group_sizes)
    u = gmm(xs, wu, group_sizes)
    h = (jax.nn.silu(g) * u).astype(dtype)
    return gmm(h, wd, group_sizes)


def _time(fn, *args, iters=50):
    f = jax.jit(fn)
    out = f(*args)
    jax.block_until_ready(out)                       # compile
    t = time.time()
    for _ in range(iters):
        out = f(*args)
    jax.block_until_ready(out)
    return (time.time() - t) / iters * 1e3           # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--D", type=int, default=1536)
    ap.add_argument("--F", type=int, default=768)
    ap.add_argument("--N", type=int, default=80)
    ap.add_argument("--K", type=int, default=8)
    args = ap.parse_args()

    dtype = jnp.bfloat16
    T = args.seq
    print(f"backend={jax.default_backend()} T={T} D={args.D} F={args.F} "
          f"N={args.N} K={args.K}  (rows=T*K={T*args.K})")
    x, order, gs, wg, wu, wd, K = _inputs(T, args.D, args.F, args.N, args.K, dtype)

    def ref():
        return ref_forward(x, order, gs, wg, wu, wd, K, dtype)
    ms_ref = _time(ref)
    # useful FLOPs = 3 grouped matmuls over T*K rows: 3 * 2 * (T*K) * D * F
    flop = 3 * 2 * (T * K) * args.D * args.F
    print(f"(a) XLA gather + ragged_dot : {ms_ref:7.3f} ms  "
          f"({flop/ms_ref/1e9:6.1f} GFLOP/ms useful)")

    try:
        def mb():
            return megablox_forward(x, order, gs, wg, wu, wd, K, dtype)
        ms_mb = _time(mb)
        print(f"(b) megablox gmm + gather   : {ms_mb:7.3f} ms  ({ms_ref/ms_mb:.2f}x vs ref)")
    except Exception as e:
        print(f"(b) megablox gmm            : unavailable ({str(e)[:80]})")


if __name__ == "__main__":
    main()
