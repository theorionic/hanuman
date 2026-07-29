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
where the S^2 term dominates; see the sequence-length section below, where at
16384 the attention core is ~41% of model FLOPs.

`step_flops()` models the per-layer SWA/KDA plan, so MFU stays honest when
those hybrids are on. It did not originally, and counting windowed layers as
full causal overstated SWA MFU by up to 8 points at 16384.

## Sequence length is the biggest MFU lever found

Every knob at seq_len 4096 is flat (see below), but training longer is not:

| seq_len | remat | attention | ms/step | tok/s | MFU | hw MFU |
|---|---|---|---|---|---|---|
| 4,096 | dots_no_batch | full | 1119 | 29,300 | 11.5% | 15.2% |
| 8,192 | dots_no_batch | full | 2388 | 27,449 | 12.4% | 16.3% |
| 8,192 | dots_no_batch | SWA p2 w1024 | 2317 | 28,287 | -- | -- |
| 16,384 | dots_no_batch | either | OOM (22.1 GB) | | | |
| 4,096 | full | full | 2483 | 13,196 | 5.2% | 6.8% |
| 8,192 | full | full | 5098 | 12,856 | 5.8% | 7.6% |
| **16,384** | **full** | **full** | **3496** | **37,487** | **21.2%** | **28.0%** |
| 16,384 | full | SWA p2 w1024 | 3197 | 40,970 | 19.1% | 25.2% |
| 16,384 | full | SWA p4 w1024 | 3043 | 43,078 | 17.8% | 23.5% |
| 16,384 | full | SWA p8 w1024 | 2965 | **44,201** | 17.2% | 22.6% |
| 32,768 | full | either | OOM (21.0 GB) | | | |

Best MFU is 21.2% at seq_len 16384 with full attention -- nearly double the
4096 baseline. Best *throughput* is 44,201 tok/s with SWA period 8, +51% over
the 4096 baseline.

MFU and throughput diverge under SWA because SWA removes real FLOPs: the model
gets cheaper rather than the chip busier, so tok/s rises while MFU falls. Both
numbers are correct; tok/s is the one that decides how long a run takes.

Two counter-intuitive results here, both reproduced across three timing windows
and two separate processes:

- **seq 16384 is faster in absolute terms than seq 8192 under remat=full**
  (3496 ms for 2x the tokens of 8192's 5098 ms). 4096 and 8192 scale linearly
  with each other, so 16384 is the outlier in the *good* direction. The likely
  cause is `ragged_dot` selecting a better tiling once the row count reaches
  131072 (cf. the `scoped_vmem_limit_kib` note in main.py, where the same op is
  already known to pick tilings sensitive to size and VMEM budget). Not chased
  further, but it means **do not assume the 4096 and 8192 numbers extrapolate**.
- **remat_policy=full is 2.2x slower at 4096 and 8192, but is the only policy
  that fits at 16384** -- and at 16384 it beats every measurement at shorter
  context anyway. The right policy depends on sequence length; `dots_no_batch`
  is right at 4096, `full` at 16384.

Attention mechanism in isolation (8 layers, fwd+bwd, batch 8, window 1024),
which is what SWA is actually for:

| seq_len | full causal | SWA | alternating hybrid | SWA speedup |
|---|---|---|---|---|
| 4,096 | 14.5 ms | 10.4 | 13.6 | 1.40x |
| 8,192 | 49.9 | 22.4 | 38.6 | 2.23x |
| 16,384 | 178.4 | 47.5 | 119.3 | 3.76x |
| 32,768 | 688.7 | 96.8 | 399.3 | 7.12x |
| 65,536 | 2669.8 | 194.0 | 1444.6 | 13.76x |

Full causal scales ~3.9x per doubling (O(S^2)); SWA ~2.05x (O(S)). The hybrid
is asymptotically bounded by its full-attention layers -- at period 2 it can
never beat 2x, which is why period 4 and 8 are worth the modelling tradeoff at
long context.

Note that `swa_period=8` on a 24-layer model leaves only 3 full-attention
layers (7, 15, 23). That is an aggressive ratio -- Qwen2/Gemma2 use 1:1 -- and
whether those 3 layers carry enough long-range capability is a modelling
question, not a throughput one.

The context ceiling is the MoE dispatch, not attention: Splash is already O(S)
in memory, and the 16384 OOM under dots_no_batch and the 32768 OOM under full
are both driven by the `[T*K, D]` dispatch intermediates. Going past 16K needs
either a chunked dispatch or sequence parallelism.

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
