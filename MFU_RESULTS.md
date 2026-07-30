# MFU Optimization Test Results

Measured on TPU v5e-8 (8 chips, 197 TFLOP/s bf16 each = 1576 TFLOP/s peak),
JAX 0.11.0, `full_g4` config base (24 layers, d_model 1536, 16K context, batch 8,
bf16 compute / fp32 master, remat_policy=full). All runs use random data,
11-16 steps, first loss ~10.89 (parity confirmed across configs).

## Summary

The default config was changed from `full` to `full_g4` (top-4-of-40, d_ff 1536)
in `main.py`. A new optimization lever not in OPTIMIZATION.md was found and
validated: **widening the shared expert (d_ff_dense) to shift compute from the
dispatch-bound MoE path (~46% of peak) to the dispatch-free dense path (~92% of
peak)**, combined with **top-1 routing** to minimize the [T*K, D] dispatch buffer
and free HBM for the wider shared expert.

Best MFU reached: **34.9%** (top-1 + d_ff_dense=12288) — +61% over the old
default (21.6%). Best throughput: **54,123 tok/s** (top-2-of-40).

## Results table

| config | ms/step | tok/s | MFU | hw MFU | HBM peak | vs old default |
|---|---|---|---|---|---|---|
| `full` (old default, top-8-of-80, d_ff=768) | 3437 | 38,132 | 21.6% | 28.5% | 8.94 GB | baseline |
| `full_g4` (new default, top-4-of-40, d_ff=1536) | 2883 | 45,459 | 25.7% | 34.0% | 8.94 GB | +4.1 MFU, +19% tok/s |
| `full_g4` + d_ff_dense=3072 | 3160 | 41,481 | 27.5% | 36.4% | 9.12 GB | +5.9 MFU |
| `full_g4` + d_ff_dense=4096 | 3299 | 39,729 | 28.0% | 37.1% | 9.20 GB | +6.4 MFU |
| `full_g4` + d_ff_dense=6144 | 3537 | 37,062 | 29.3% | 38.9% | 9.37 GB | +7.7 MFU |
| `full_g4` + d_ff_dense=8192 | 3683 | 35,593 | 31.2% | 41.4% | 9.54 GB | +9.6 MFU |
| top-2-of-40, d_ff=1536 (no dense widen) | 2422 | 54,123 | 24.5% | 32.3% | 8.94 GB | +2.9 MFU, +42% tok/s |
| top-2-of-40, d_ff=1536 + d_ff_dense=8192 | 3250 | 40,332 | 30.8% | 40.9% | 9.53 GB | +9.2 MFU |
| top-1-of-40, d_ff=1536 + d_ff_dense=8192 | 2815 | 46,566 | 33.0% | 43.7% | 9.54 GB | +11.4 MFU |
| top-1-of-40, d_ff=1536 + d_ff_dense=10240 | 3107 | 42,187 | 33.5% | 44.4% | 9.71 GB | +11.9 MFU |
| **top-1-of-40, d_ff=1536 + d_ff_dense=12288** | **3311** | **39,592** | **34.9%** | **46.2%** | **9.88 GB** | **+13.3 MFU** |

### SWA hybrid variants (throughput levers on full_g4, window 1024)

SWA removes real attention FLOPs, so tok/s rises while MFU falls (the chip does
less work, not more efficient work). These are modelling tradeoffs (fewer
full-attention layers), not pure perf wins.

| config | ms/step | tok/s | MFU | hw MFU | notes |
|---|---|---|---|---|---|
| `full_g4` + SWA p2 w1024 | 2648 | 49,503 | 23.0% | 30.3% | +9% tok/s vs full_g4 |
| `full_g4` + SWA p4 w1024 | 2527 | 51,881 | 21.5% | 28.3% | +14% tok/s |
| `full_g4` + SWA p8 w1024 | 2466 | 53,161 | 20.7% | 27.2% | +17% tok/s (best SWA throughput) |
| `full_g4` + d_ff_dense=8192 + SWA p4 w1024 | 3328 | 39,388 | 28.6% | 37.9% | MFU drops with SWA |

## Tested and ruled out (OOM or no improvement)

| idea | result | reason |
|---|---|---|
| 2D mesh (data=4, expert=2) | OOM (1.85 GB alloc fail) | expert_axis>1 still gathers weights; needs unimplemented all-to-all token dispatch |
| 32K context (full_g4 + SWA p4) | OOM (19.58 GB) | SWA cuts attention memory but the [T*K,D] dispatch buffer is the wall, not attention |
| 16K + dots_no_batch remat on full_g4 | OOM (21.41 GB) | g4 halves dispatch but dots_no_batch keeps too many intermediates |
| batch 16 (full_g4, any remat) | OOM (17.64-34 GB) | dispatch buffer scales with batch |
| batch 12 | error | batch must be multiple of 8 (data axis sharding) |
| d_ff_dense=13312 (top-1) | OOM (15.87 GB, +126 MB) | 12288 is the exact HBM ceiling |
| d_ff_dense=14336 (top-1) | OOM (15.98 GB) | exceeds HBM |
| d_ff_dense=16384 (top-1) | OOM (16.48 GB) | exceeds HBM |
| d_ff_dense=10240 (top-4) | OOM (15.88 GB) | top-4 dispatch buffer larger than top-1 |
| d_ff_dense=12288 (top-4) | OOM (16.24 GB) | top-4 dispatch buffer larger than top-1 |
| top-2-of-40, d_ff=3072 | OOM (21.63 GB) | wider expert weights (40x1536x3072x3) blow up memory |
| full_g4 at 8K, dots_no_batch remat | 1579 ms, 41,505 tok/s, 18.7% MFU | fits in 6.82 GB but less context amortization -> lower MFU |
| entropy diagnostic removal (HANUMAN_NO_ENTROPY=1) | 2883 ms (identical) | XLA already optimizes it away; no win |
| gate/up ragged_dot merge on full_g4 | 3463 ms (regression) | 2F=3072 tiles worse than two 1536-wide dots; merge is config-dependent |

## The new lever: wider shared expert (d_ff_dense)

### Why it works

The MoE dispatch (scattered gather of [T*K, D]) is the structural wall — 21% of
the step at 16K, pure HBM traffic with the MXU idle, and impossible to fuse on
TPU (the (8,128) VMEM tiling blocks scattered row gather at every level, on all
TPU generations v5e through v7/Ironwood — verified in JAX source).

But the **shared expert runs as a plain dense matmul at ~92% of peak** — no
dispatch, no scattered gather, no routing overhead. It is the one part of the
MoE layer that already runs efficiently.

So instead of fighting the dispatch, move more of the model's compute into the
shared expert by widening d_ff_dense (the shared/dense FFN width, independent of
the routed experts). This:
  - adds compute that runs at ~92% peak (vs ~46% for the dispatched ragged_dot)
  - adds zero dispatch traffic (the shared expert sees all tokens directly)
  - keeps the routed experts for specialization, just smaller in the compute mix

Combining with **top-1 routing** (every token -> 1 routed expert) minimizes the
[T*K, D] buffer (K=1) and frees HBM for the wider shared expert, pushing
d_ff_dense from 768 (baseline) to 12288 (the HBM ceiling).

### The tradeoffs (not a free win)

1. **The shared expert becomes the dominant compute path.** At d_ff_dense=12288
   with top-1 routing, the shared expert does ~90% of FFN compute. The routed
   experts become a small specialization signal. This is closer to "dense model
   + light MoE seasoning" than a true fine-grained MoE. Whether that matches
   your modelling goals is a real question — DeepSeekMoE's claim is that
   fine-grained experts *are* the value, and this dilutes them.

2. **top-1 routing is aggressive.** Every token goes to exactly 1 routed expert
   (plus the shared). Most MoE models use top-2 or higher. top-1 minimizes
   dispatch but reduces expert diversity per token. Loss parity at step 10 looks
   fine (10.89 vs 10.89), but a real loss-curve comparison over thousands of
   steps is needed before adopting for training.

3. **Parameter count rises.** The wider shared expert adds params: d_ff_dense=12288
   adds ~3 x 1536 x 12288 x 21 ~ 1.2B params (shared expert is replicated, not
   sharded across experts). Total params go from 6.23B to ~7.4B. Active params
   rise too. This is more model capacity, not the same model faster.

4. **Throughput drops as MFU rises.** The 34.9% MFU config is slower in tok/s
   (39,592) than the 25.7% config (45,459) — because it does more real FLOPs per
   token (the wider shared expert). MFU measures chip efficiency; tok/s measures
   training speed. They diverge here. If wall-clock training speed is the goal,
   full_g4 (25.7% MFU, 45.5k tok/s) or top-2 (24.5% MFU, 54k tok/s) may be better.

## Pallas dispatch kernel: infeasible on all TPU generations

Verified in JAX source (jax/_src/pallas/mosaic/tpu_info.py, lowering.py): the
VMEM tile shape is hardcoded (8, 128) with a TODO to make it dynamic, across
every TPU generation from v5e through v7/Ironwood. The BlockSpec alignment check
uses literal 8 and 128, never consulting the generation. Trillium (v6e) and
Ironwood (v7) double the MXU width (128->256 cols) but the VMEM/DMA alignment is
unchanged. No gather/scatter primitive exists in Pallas for TPU (only `roll`).
Megablox does not gather scattered rows — it loads contiguous tile-aligned blocks
and masks. The scattered-row gather that blocks the MoE dispatch is blocked on
every TPU generation that exists. On NVIDIA GPU (Triton) it is feasible because
shared memory has no alignment constraint.

## Recommendation

| goal | config | MFU | tok/s | how |
|---|---|---|---|---|
| Max MFU (chip efficiency) | top-1 + d_ff_dense=12288 | 34.9% | 39,592 | `--n_active 1 --d_ff 1536 --d_ff_dense 12288` |
| Max throughput (training speed) | top-2-of-40 | 24.5% | 54,123 | `--n_active 2 --d_ff 1536` |
| Balanced (safe default) | full_g4 | 25.7% | 45,459 | already the default |

The 34.9% MFU config is real and measured but is a different model (more params,
top-1 routing, dense-heavy). Before adopting for real training, run a loss-curve
comparison over a few thousand steps to confirm the modelling tradeoff is
acceptable. The default remains full_g4 (25.7%) as the safe, validated choice.

## Code changes applied

- `main.py`: default config changed from `smoke` to `full_g4` for `train`,
  `count`, and `report` subcommands.
- No other code changes. The gate/up ragged_dot merge was tested and reverted
  (it regressed full_g4). The entropy-diagnostic skip was tested and reverted
  (it is already free). moe.py is unchanged from its original state.