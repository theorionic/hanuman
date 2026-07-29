# Measured performance notes

All numbers from TPU v5e-8 (8 chips, 197 TFLOP/s bf16 each = 1576 TFLOP/s),
`full` config unless stated: 24 layers (3 dense + 21 MoE), d_model 1536,
80 experts top-8, d_ff 768, batch 8, seq_len 4096, bf16 compute / fp32 master.
203.1 TFLOP/step model, 267.6 TFLOP/step with remat.

## Baseline

| | ms/step | tok/s | MFU (model) | MFU (hw) | HBM peak |
|---|---|---|---|---|---|
| default (all full causal) | 1119 | 29,300 | 11.5% | 15.2% | 4.81 GB |
| `use_swa=True`, window 1024, period 2 | 1111 | 29,490 | 11.6% | 15.3% | 4.81 GB |

SWA is roughly neutral at seq_len=4096 -- the attention core is only ~15% of
model FLOPs there, so halving it is worth ~1%. Its payoff is at long context,
where the S^2 term dominates.

## Where the step actually goes

From a `jax.profiler` trace, per device per step (the count=168 ops, i.e.
21 MoE layers x 8 devices; the enclosing `while` ops are scan containers and
double-count their children):

| op | ms/device/step | what |
|---|---|---|
| `reduce_scatter` x3 | 230 | expert-weight **gradient** reduce-scatter (backward) |
| `ragged-dot` x9 | 131 | the expert matmuls (fwd + 2 backward) |
| `all-gather` x2 | 76 | expert-weight gather (forward + remat recompute) |
| fusions / gathers / copies | 135 | dispatch permute, un-permute, combine |
| `splash_mha_dq/dkv` | 29 | attention |

One MoE layer's dispatch, forward only, measured standalone at the per-device
shapes (4096 tokens, 32768 rows, D=1536, F=768, N=80):

| ms | stage |
|---|---|
| 0.066 | `argsort(T*K)` |
| 0.301 | `bincount` |
| 0.067 | inverse permutation |
| 0.311 | gather rows `x[order // K]` |
| **2.537** | **3x `ragged_dot`** (232 GFLOP -> 46.4% of peak) |
| 1.111 | un-permute + weighted combine |
| 4.394 | total |

For reference a single plain `dot` of the same shape runs at ~92% of peak, so
`ragged_dot` costs about 2x what the equivalent dense matmuls would.

## Hypotheses that were tested and are FALSE

These were all in the original optimization plan. Each was measured; none help.

1. **"The forward expert all-gather is ~45% of the step; expert parallelism
   would win 10 points of MFU."** The all-gather is 76 ms, not 476 ms, and it is
   already overlapped. Proof: step time is linear in batch. Fitting
   `t = fixed + var*B` to 1119 ms @ B=8 and 2274 ms @ B=16 gives `fixed ~= -40 ms`
   -- there is essentially no fixed per-step cost to amortize. The dominant
   collective is the *backward* gradient reduce-scatter (230 ms), not the
   forward gather.

2. **"HBM is only 4.81 / 16 GB, so batch can go to 24-32."** `peak_bytes_in_use`
   reports the persistent allocation. XLA's step temporaries are what bind:
   batch 24 needs 16.99 GB and batch 32 needs 19.33 GB against 15.75 GB
   available. Both OOM. Batch 16 fits but MFU *drops* to 11.3%.

3. **"Drop remat entirely; at batch 8 it may not be needed."** `remat=False`
   needs 36.68 GB of temporaries. OOM.

4. **`remat_policy` and collective-scheduling flags are all flat.**

| variant | ms/step |
|---|---|
| `dots_no_batch` (default) | 1119 |
| `dots` | 1121 |
| `experts` | 3157 |
| `remat=False` | OOM (36.7 GB) |
| `HANUMAN_MOE_BARRIER=0` | 1118 |
| `--xla_tpu_enable_all_experimental_scheduler_features=true` | 1119 |

The `optimization_barrier` in `model/moe.py` costs nothing measurable -- it can
stay for its OOM-safety value.

## The one lever that is left

1117 ms is a hard floor for the current structure; every config knob is flat.
The remaining ~306 ms of expert-weight communication (230 reduce-scatter +
76 all-gather) is only removable by **expert parallelism**: keep each device's
10 local experts in place and all-to-all the *tokens* instead of gathering the
*weights*. Communication drops from ~1.7 GB/layer to ~0.5 GB/layer.

This was NOT implemented, because it is not a drop-in change: `all_to_all`
needs a fixed number of rows per destination, and per-device token counts vary
with routing. Every standard formulation handles that with a capacity factor
plus token dropping, which the current implementation is deliberately free of.
Adopting it is a modelling decision (what capacity factor, and is dropping
acceptable), not just a performance one.

Second-order, and independent: `ragged_dot` at 46% of peak against ~92% for the
equivalent dense matmuls, and the 1.1 ms un-permute/combine. Both point at a
fused Pallas dispatch kernel, which is the largest and last piece of work.

## KDA

`model/kda.py` recurrent form, 7B config cut to 4 layers, seq_len 4096:

| | ms/step |
|---|---|
| all full attention | 187 |
| 3 KDA + 1 full (`kda_period=4`) | 1421 |

~460 ms per KDA layer against ~47 ms for the whole transformer layer it
replaces. Correct and trainable, but ~10x too slow to use; see the module
docstring for what the chunked parallel form would need.
