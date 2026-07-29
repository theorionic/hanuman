# Optimization avenues

A survey of where MFU/throughput can still be won on this 7B MoE on TPU v5e-8,
and what has already been ruled out. Numbers are measured (see `BENCHMARKS.md`);
speculative items are marked.

The governing fact: **the bottleneck changes with sequence length.**

- At **seq 4096** the step is **communication-bound** (>50% inter-chip bytes;
  the constant ~230 ms weight-gradient reduce-scatter is the single biggest op).
- At **16K** (the current `full` default) it flips to **compute-bound** (~82%
  compute).

A lever only helps in the regime it targets.

## Unexplored levers that could move the needle

### 1. int8 / AQT quantized matmuls — biggest untapped one
v5e's int8 MXU is ~2x bf16. Attacks the *compute-bound* 16K regime directly and
is the real reason MaxText posts high numbers. Estimated ~1.2-1.5x throughput
(speculative). Cost: quantization-aware training (scale calibration, numerics
validation) -- a real project, not a flag. Highest ceiling of anything left.

### 2. Expert parallelism — the only thing that removes the 230 ms constant
Today experts are replicated and FSDP-gathered; the weight-grad reduce-scatter
(230 ms, constant regardless of tokens) is the largest single op. Expert
parallelism all-to-alls the *tokens* instead of gathering the *weights*,
dropping ~306 ms of comm. The `mesh_expert_axis` knob exists but the all-to-all
path is not implemented, and it needs a capacity factor + token dropping (a
modelling decision this repo currently avoids).

### 3. Break the 16K -> 32K memory wall -> even more context
Sequence length is the proven MFU lever (it doubled MFU from 4096 to 16K). 32K
OOMs today purely because of the `[T*K, D]` dispatch buffer. A chunked/streaming
dispatch (process routed rows in slices) or sequence parallelism would let
context grow further and amortize the fixed comm even more.

### 4. Host offload of optimizer state / activations
Optimizer state is ~2/3 of HBM. Offloading it (or activations) to host RAM frees
HBM headroom, which could enable a bigger batch or longer context (the levers
that actually help). MaxText does this; untried here.

### 5. The ~10-13% "router / norms / loss / scan-plumbing" bucket
A real slice nobody has dissected. Some of it is *logging-only* diagnostics
computed every step (routing entropy, effective-experts, histograms). Computing
those every N steps instead of every step is a free, safe win if that bucket is
meaningfully diagnostic overhead. Worth a targeted profile.

### 6. MLA (multi-head latent attention) instead of GQA — modelling
DeepSeek-V3's KV-compressed attention cuts attention compute and KV memory. Big
modelling change, but this model is already DeepSeek-style, so it is a natural
direction if architecture changes are on the table.

## Cheap/safe but small
- **gate/up `ragged_dot` merge** (~+1.8%, mathematically identical): the only
  pure, no-risk code win left. Merges two `[80,1536,768]` matmuls into one
  `[80,1536,1536]`. Small.

## Already ruled out -- do not spend time here (measured)
- Fused Pallas dispatch kernel: **infeasible** on TPU -- the (8,128) VMEM tiling
  blocks per-row scatter-gather at every level (see `model/moe_pallas.py`).
- Bigger batch: OOM, or MFU *drops* (batch 16 -> 11.3%).
- Every remat policy and collective-scheduler flag: flat.
- Non-expert weight sharding via shard_map: 1.4% and it NaN'd.
- Beating `ragged_dot` / attention kernels: already 51% / 50% of peak.

## How to hunt for more (methodology)
The repo already has the right tool: trace-based profiling that folds every XLA
op back to its source line (`BENCHMARKS.md`, "Where the step actually goes"). To
find the next lever: capture a `jax.profiler` trace at the *current* config
(16K), classify each op as compute- / memory- / comm-bound (roofline), and
attack the largest bucket that is not already near peak. That will show whether
comm (#2) or the router/loss bucket (#5) is the better next target at 16K.

## Recommendation
- **Maximum impact:** int8/AQT (#1) -- the lever the MaxText comparison points at.
- **Lowest risk per gain:** a fresh 16K profile to confirm the 230 ms
  reduce-scatter still dominates, then expert parallelism (#2).

Realistic ceiling on 8x v5e with a fine-grained MoE: stacking 16K + int8 +
granularity reaches roughly the **30-40%** range. MaxText's 60% is dense models
on large pods -- not reachable here without changing the model class, which is
not a failing of this code.

## Confirmed stack so far
| config | MFU | tok/s | status |
|---|---|---|---|
| 4096 baseline (80x768 top-8) | 11.5% | 29,300 | superseded |
| 16K context | 21.6% | 38,215 | **default** |
| 16K + granularity (40x1536 top-4) | 25.7% | 45,458 | `full_g4` preset (needs loss check) |
