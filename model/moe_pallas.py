"""Forward gather-fused MoE dispatch: prototype + NEGATIVE feasibility result.

Goal (BENCHMARKS.md target #4): gather the routed rows straight into the MXU
tile inside the grouped matmul, so the [T*K, D] buffer is never materialized in
HBM. The profile shows this buffer's gather is 22% of the forward at seq 4096
and 33% at 16384 (measured), all pure HBM traffic.

CONCLUSION: not feasible on TPU v5e with current Mosaic. A per-row scatter-gather
into a matmul operand is blocked at EVERY level by the (8, 128) VMEM tiling,
which the two probes below hit empirically:

  probe 1 (`pallas_gather`, BlockSpec gather): a (1, D) per-row block is
    rejected -- "the last two dimensions of your block shape [must be] divisible
    by 8 and 128, or equal the array dimensions". Only contiguous >=8-row blocks
    are addressable, and routed rows are scattered.

  probe 2 (`fused_gather_matmul`, in-kernel DMA gather): DMA'ing a single row
    into scratch row r is rejected -- "Offsets along tiled dimensions must be
    aligned to tiles ... index at dimension 0 [must be] divisible by 8". The
    destination row of a tiled VMEM buffer must be 8-aligned; a scattered
    per-row gather cannot satisfy that.

  and there is no `pltpu` gather/scatter primitive (only `roll`).

So XLA's separate `x[order//K]` gather that materializes [T*K, D] is not a missed
optimization -- it is what the hardware tiling model forces, which is why MaxText
and megablox also eat the gather. Both functions are kept as the evidence; NEITHER
COMPILES on TPU, by design of this finding. They are not wired into the model.

The practical way to cut the SAME [T*K, D] traffic is to move fewer rows:
expert granularity top-4-of-40 instead of top-8-of-80 halves T*K at identical
params/FLOPs -- measured +19% throughput / +4 MFU (BENCHMARKS.md, and reproduced
this session). That is the realizable version of what this kernel aimed at.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _gather_kernel(idx_ref, x_ref, o_ref):
    o_ref[...] = x_ref[...]


def pallas_gather(x, idx):
    """Probe 1: out[i] = x[idx[i]] via a BlockSpec index_map gather.

    DOES NOT COMPILE on TPU -- Mosaic rejects the (1, D) per-row block ("last two
    dimensions ... divisible by 8 and 128, or equal the array dimensions").
    Kept as evidence of the tiling constraint; see module docstring.
    """
    T, D = x.shape
    M = idx.shape[0]
    return pl.pallas_call(
        _gather_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(M,),
            in_specs=[pl.BlockSpec((1, D), lambda i, idx_ref: (idx_ref[i], 0))],
            out_specs=pl.BlockSpec((1, D), lambda i, idx_ref: (i, 0)),
        ),
        out_shape=jax.ShapeDtypeStruct((M, D), x.dtype),
    )(idx, x)


def _fused_kernel(row_map_ref, x_hbm, w_ref, o_ref, xs, sem):
    """One m-block: gather block_m rows of x via DMA, then matmul with w.

    row_map_ref: [M] int32 scalar-prefetched (sorted row -> source token row)
    x_hbm:       [T, D] in ANY (HBM); rows are DMA-gathered on demand
    w_ref:       [D, F] in VMEM (single expert for this prototype)
    o_ref:       [block_m, F] VMEM output for this block
    xs:          [block_m, D] VMEM scratch; sem: DMA semaphore
    """
    b = pl.program_id(0)
    bm = o_ref.shape[0]

    def gather_row(r, _):
        src = row_map_ref[b * bm + r]
        cp = pltpu.make_async_copy(x_hbm.at[pl.ds(src, 1)], xs.at[pl.ds(r, 1)], sem)
        cp.start()
        cp.wait()
        return ()

    jax.lax.fori_loop(0, bm, gather_row, ())
    o_ref[...] = jnp.dot(xs[...], w_ref[...], preferred_element_type=jnp.float32)


def fused_gather_matmul(x, w, row_map, block_m: int = 512):
    """out[i] = x[row_map[i]] @ w, without materializing x[row_map] in HBM.

    Prototype: a single shared weight `w` [D, F] (one expert), to isolate the
    cost of the fused per-row DMA gather from the grouped-routing machinery.
    x: [T, D], row_map: [M] int32 (M % block_m == 0) -> out [M, F].
    """
    T, D = x.shape
    M = row_map.shape[0]
    F = w.shape[1]
    return pl.pallas_call(
        _fused_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(M // block_m,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # x resident in HBM
                pl.BlockSpec((D, F), lambda b, r: (0, 0)),   # w whole in VMEM
            ],
            out_specs=pl.BlockSpec((block_m, F), lambda b, r: (b, 0)),
            scratch_shapes=[
                pltpu.VMEM((block_m, D), x.dtype),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((M, F), jnp.float32),
    )(row_map, x, w)
