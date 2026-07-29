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
| **16,384** | **full** | **full** | **3437** | **38,133** | **21.6%** | **28.5%** |
| 16,384 | full | SWA p2 w1024 | 3197 | 40,970 | 19.1% | 25.2% |
| 16,384 | full | SWA p4 w1024 | 3043 | 43,078 | 17.8% | 23.5% |
| 16,384 | full | SWA p8 w1024 | 2965 | **44,201** | 17.2% | 22.6% |
| 32,768 | full | either | OOM (21.0 GB) | | | |

Best MFU is 21.6% at seq_len 16384 with full attention -- nearly double the
4096 baseline (21.2% before the Splash tiles were tuned; see below). Best *throughput* is 44,201 tok/s with SWA period 8, +51% over
the 4096 baseline.

MFU and throughput diverge under SWA because SWA removes real FLOPs: the model
gets cheaper rather than the chip busier, so tok/s rises while MFU falls. Both
numbers are correct; tok/s is the one that decides how long a run takes.

Two counter-intuitive results here, both reproduced across three timing windows
and two separate processes:

- **seq 16384 is faster in absolute terms than seq 8192 under remat=full**
  (3496 ms for 2x the tokens of 8192's 5098 ms). 4096 and 8192 scale linearly
  with each other, so 16384 is the outlier in the *good* direction. Root cause
  found by profiling: it is the **expert weight-gradient**, not `ragged_dot`.
  See "the expert weight-gradient is the fragile op" below -- it costs 1522 ms
  at seq 4096 under remat=full and only 418 ms at 16384 for 4x the work.
  **Do not assume the 4096 and 8192 numbers extrapolate.**
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

Every XLA op carries the Python source line that emitted it, so a trace folds
straight back onto the model source. The tables below are one device (all 8 run
the same program), with `while`/`conditional` excluded -- those are the scan and
switch *containers*, whose duration already includes their children -- and async
collectives charged to their `-done` (the wait), not their `-start` (the issue).
Device busy time reconciles with wall step time to within 0.1%, so nothing is
double-counted or missing.

Two configurations, because the answer is completely different in each:

| component | 4096 / dots_no_batch | 16384 / full | ratio for 4x the tokens |
|---|---|---|---|
| Attention | 72.8 ms (6.5%) | 876.6 ms (25.1%) | 12.0x |
| MoE expert matmul | 191.1 (17.1%) | 808.2 (23.1%) | 4.2x |
| MoE dispatch (permute/gather/combine) | 120.4 (10.8%) | 802.6 (23.0%) | 6.7x |
| **MoE expert-weight comm** | **374.6 (33.5%)** | 476.5 (13.6%) | 1.3x |
| Other weight comm (FSDP) | 215.1 (19.2%) | 158.9 (4.5%) | 0.7x |
| scan plumbing / router / norms / loss | 144.4 (12.9%) | 370.8 (10.6%) | -- |
| **TOTAL** | **1118.4** | **3493.6** | |
| *of which compute / communication* | *47.3% / 52.7%* | *81.8% / 18.2%* | |

**At seq 4096 this is a communication-bound model, not a compute-bound one.**
Over half the step is moving bytes. At 16384 the same absolute communication is
amortized over 4x the tokens and compute takes over, which is the whole reason
MFU nearly doubles -- see the sequence-length section.

The single largest op is the same in both, and it does not move with sequence
length at all:

| op | 4096 | 16384 | ratio |
|---|---|---|---|
| expert-weight **reduce-scatter** (weight grad) | 230.2 ms | 230.7 ms | **1.00x** |
| splash attention core | 61.2 | 812.1 | 13.3x (O(S^2)) |
| **dispatch row gather + bwd scatter** (`moe.py:55`) | 23.6 | **395.3** | **16.7x** |
| expert weight-gradient (dW) | 103.0 | 410.0 | 4.0x |
| `ragged_dot` (fwd + dgrad) | 73.8 | 302.3 | 4.1x |
| un-permute gather + combine + bwd (`moe.py:80`) | 70.0 | 283.7 | 4.1x |
| expert-weight all-gather | 75.5 | 187.9 | 2.5x |
| `bincount` (`moe.py:56`) | 12.1 | 48.5 | 4.0x |
| **the actual `argsort`** (`moe.py:54`) | **0.9** | **5.4** | 5.9x |

The reduce-scatter is **exactly constant** at 230 ms: it depends on parameter
count, not on tokens. That is the cleanest statement of why long context pays.

### The dispatch permutation, not the sort, is the MoE overhead

The routing *sort* is negligible -- 5.4 ms/step at 16384. Isolated, an
`argsort` of 131072 int32 keys takes 0.12 ms, and neither `stable=False` nor an
O(n) counting sort beats it (0.099 ms and 2.06 ms respectively). There is
nothing to win there.

What costs is moving the **tokens**. Top-8 routing replicates every token 8x
into a `[T*K, D]` buffer -- 131072 x 1536 bf16 = 402 MB per layer per call at
seq 16384 -- and that buffer is gathered on the way in, scatter-added on the way
back, and gathered again for the un-permute:

| moe.py | stage | 4096 | 16384 | ratio |
|---|---|---|---|---|
| :54 | `argsort` | 0.9 | 5.4 | 5.9x |
| :55 | row gather `x[order//K]` + bwd scatter | 23.6 | **395.3** | **16.7x** |
| :56 | `bincount` | 12.1 | 48.5 | 4.0x |
| :79 | inverse permutation | 0.9 | 5.4 | 5.8x |
| :80 | un-permute gather + combine + bwd | 70.0 | 283.7 | 4.1x |
| | **total dispatch data movement** | **107.7** | **738.4** | **6.9x** |

738 ms is 21% of the step at 16384, all of it pure HBM traffic with the MXU
idle, and the `:55` gather is badly superlinear: 4x the bytes for 16.7x the
time, so it is not bandwidth-bound but access-pattern-bound (a random row
gather at 402 MB defeats whatever locality it had at 100 MB). The same
`[T*K, D]` buffer is also the memory wall -- it is why batch 16 OOMs at seq
16384 (18.86 GB of temporaries) and why 32768 does not fit at all.

This is the case for a fused Pallas dispatch kernel: one that gathers rows
directly into the MXU tile inside the grouped matmul and never materializes
`[T*K, D]` at all. It would attack the largest remaining time cost, the memory
wall, and the context ceiling with one change.

### Expert granularity is the cheapest large win

Dispatch cost is proportional to `T*K`, and `K` can be traded against expert
width at **constant parameter count and constant FLOPs**: 80 experts of d_ff 768
with top-8 activates exactly as much as 40 experts of d_ff 1536 with top-4.
Halving `K` halves every byte the dispatch moves. Measured at seq 16384,
remat=full, with `d_ff_dense=768` pinned so the shared expert is untouched:

| | params | active | TFLOP/step | ms/step | tok/s | MFU |
|---|---|---|---|---|---|---|
| 80 x 768, top-8 | 6.233B | 0.880B | 1168.8 | 3437 | 38,132 | 21.6% |
| **40 x 1536, top-4** | 6.232B | 0.880B | 1167.8 | **2883** | **45,457** | **25.7%** |

**+19% throughput and +4.1 points of MFU for a config change**, reproduced
across two processes (2883/2884 ms). Nothing about the arithmetic changed --
only how many times each token is copied.

This is a *modelling* tradeoff, not a free win: fine-grained experts are the
core claim of the DeepSeekMoE line, and top-4-of-40 is a coarser routing space
than top-8-of-80. It needs a loss-curve comparison before adoption. It is also
not enough on its own to move the memory wall -- batch 16 at 16384 still needs
17.64 GB (was 18.86) against 15.75 GB available, and 32768 needs 19.57 GB.

### The expert weight-gradient is the fragile op

XLA lowers the grouped weight-gradient (`dW = xs^T @ dy` per expert) not as a
grouped matmul but as a **masked dense** one: a `convolution_select_fusion`
taking a `pred[80, rows]` mask and a full `[rows, 1536] x [rows, 768]` product
per expert. Its declared FLOP count is 82x the useful work in every config
measured. The kernel clearly skips most masked blocks -- 397 declared TFLOP in
103 ms would be 3855 TFLOP/s on a 197 TFLOP/s chip -- but the strategy is
unstable:

| config | dW ms/step | useful TFLOP | effective | % of chip peak |
|---|---|---|---|---|
| 4096, `dots_no_batch` | 103.0 | 4.87 | 47.3 TFLOP/s | 24% |
| 4096, `full` | **1521.7** | 4.87 | **3.2 TFLOP/s** | **1.6%** |
| 16384, `full` | 417.8 | 19.48 | 46.6 TFLOP/s | 24% |

At seq 4096 with `remat_policy=full` this one op collapses to 1.6% of peak and
costs 1.4 extra seconds per step -- 61% of the whole step. That is the entire
explanation for both "remat=full is 2.2x slower at 4096" and "16384 is faster
than 8192 under remat=full". It is not `ragged_dot`, which behaves linearly
everywhere.

### What runs in parallel, and what only looks like it

Measured from the trace timeline, not assumed: over 34,513 leaf ops per step,
op-busy time is **100.0% of the wall span** and only 4 op-pairs overlap at all,
for 0.00 ms total. **A TPU core executes exactly one op at a time.** No two
matmuls ever run concurrently on a chip. The only true concurrency is:

- **across the 8 chips** -- every matmul runs on all 8 simultaneously, each on
  its own sequence (the batch is the sharded axis), and
- **DMA vs. MXU** -- async collectives (`*-start` ... `*-done`) transfer on the
  ICI engines while the core keeps issuing compute. This is real but partial:
  the exposed `-done` waits still cost 119 ms/step at 4096.

So "sequential vs parallel" is really about the *dependency graph*, which
decides what XLA is allowed to reorder, and about which independent matmuls
could be merged into one wider op. Per layer, per device (seq 4096, batch 1/chip):

| stage | matmul | shape | GFLOP | depends on |
|---|---|---|---|---|
| 1 | `wq` | [4096,1536] x [1536,1536] | 19.3 | norm1 |
| 1 | `wk` | [4096,1536] x [1536,512] | 6.4 | norm1 |
| 1 | `wv` | [4096,1536] x [1536,512] | 6.4 | norm1 |
| 2 | splash QK^T + PV | causal, 12 heads | 51.5 | q,k,v |
| 3 | `wo` | [4096,1536] x [1536,1536] | 19.3 | attn out |
| 4 | router | [4096,1536] x [1536,80] | 1.0 | norm2 |
| 5 | ragged `gate` | [32768,1536] x [80,1536,768] | 77.3 | dispatch gather |
| 5 | ragged `up` | [32768,1536] x [80,1536,768] | 77.3 | dispatch gather |
| 5 | shared `gate` | [4096,1536] x [1536,768] | 9.7 | norm2 |
| 5 | shared `up` | [4096,1536] x [1536,768] | 9.7 | norm2 |
| 6 | ragged `down` | [32768,768] x [80,768,1536] | 77.3 | SwiGLU |
| 6 | shared `down` | [4096,768] x [768,1536] | 9.7 | SwiGLU |

Six dependent stages; everything sharing a stage number is mutually independent
and could be one wider matmul instead of several. Measured value of merging:

| merge | before | after | win |
|---|---|---|---|
| ragged `gate`+`up` -> one `[80,1536,1536]` | 9.81 ms | 9.05 ms | **+7.8%** of that stage |
| `wq`+`wk`+`wv` -> one `[1536,2560]` | 3.10 ms | 2.64 ms | +15% of the projections alone |

The q/k/v merge evaporates in context: a whole attention block measures 3.29 ms
either way, because the projections are not the bottleneck within it. The
gate/up merge is worth ~1.3% of the step at 4096 and ~1.8% at 16384.

### Inside attention: the projections are not the problem

At seq 16384 attention is 876.6 ms, and it decomposes as:

| | ms/step | share of attention |
|---|---|---|
| Splash kernel (the S^2 core) | 812.1 | 92.6% |
| RoPE | 45.6 | 5.2% |
| reshape / cast | 14.3 | 1.6% |
| **`wq` + `wk` + `wv` + `wo` matmuls** | **4.5** | **0.5%** |

Merging q/k/v into one wide matmul cannot pay: there is only 4.5 ms there to
win. (The projections look bigger at seq 4096 -- 26 ms -- but even then most of
their cost is the ring collective-matmul, not MXU time.)

The MXU is *not* idle during the Splash kernel, but it is only about half fed.
Measured for one layer, 1 sequence/chip, bf16 causal, against the 197 TFLOP/s
chip peak (forward = 2*Nq*H*S^2 FLOPs; fwd+bwd = 3x that):

| block_q / block_kv | S=4096 fwd+bwd | % peak | S=16384 fwd+bwd | % peak |
|---|---|---|---|---|
| 256 / 512 | 3.44 ms | 22.8% | 46.25 ms | 27.2% |
| 512 / 512 | 2.41 | 32.5% | 30.74 | 40.8% |
| **512 / 1024** (was hardcoded) | **2.24** | **35.0%** | 26.62 | 47.2% |
| 1024 / 1024 | 2.29 | 34.2% | 25.08 | 49.9% |
| **1024 / 2048** | 2.38 | 33.0% | **24.55** | **50.8%** |
| 2048 / 2048 | 2.39 | 32.8% | 25.07 | 50.1% |

Forward alone reaches 61.6% of peak at 16384 (66.1% tuned). The gap to peak is
inherent to flash attention: the online softmax (exp, running max, rescale) runs
on the VPU and is serialized with the MXU matmuls inside each tile, and causal
masking half-wastes every diagonal tile. ~50% of peak for fwd+bwd is normal.

`_splash_kernel` now scales its tiles with the sequence
(`block_q = min(1024, max(512, S//16))`, `block_kv = 2*block_q`) instead of
hardcoding 512/1024. This is a **no-op at or below seq 8192** -- the rule
returns 512/1024 there, and seq 4096 reproduces the baseline first loss of
10.8940 and 1117 ms/step exactly. At 16384 it is worth 1.7% end to end:

| seq 16384, remat=full | ms/step | tok/s | MFU |
|---|---|---|---|
| block 512/1024 | 3496 | 37,487 | 21.2% |
| block 1024/2048 | **3437** | **38,133** | **21.6%** |

### Weight sharding: a big isolated win that does not survive contact

`param_spec` shards each weight's largest divisible axis, which for every
non-expert weight is its *contracting* axis. XLA therefore turns each of those
matmuls into a ring collective-matmul: 5112 `collective-permute` pairs per step
moving 294 KB each, 136 ms/step, latency-bound. Attributed by source line:
shared/dense FFN 82 ms, attention projections 41 ms, router 10 ms.

Replacing that with an explicit `all_gather` inside a `shard_map` (what the MoE
path already does for expert weights) measures **4.7x faster on an isolated
shared-expert block** -- 3.21 ms -> 0.68 ms fwd+bwd.

End to end it is worth **1.4%**: 1118 -> 1102 ms/step. The ring permutes are
already ~85% hidden behind other compute in the full model, so almost all of the
isolated win is latency that was never on the critical path. Prototyped,
measured, and reverted; the same change applied to `DenseFFN` also produced NaNs
(first loss 10.70 vs 10.89), which was not chased down. **A block-level
microbenchmark is not a step-level result** -- this is the third time in this
file that an isolated speedup failed to reproduce end to end.

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

## Ranked targets, from the profile

Every config *knob* is flat; these are structural changes, ordered by measured
size. The first two are the whole game.

1. **Train at 16K context.** Free, already validated: 11.5% -> 21.2% MFU. It
   works precisely because the 230 ms reduce-scatter and the rest of the
   parameter-proportional communication are fixed per step.
2. **Expert granularity**: top-4-of-40 x d_ff 1536 instead of top-8-of-80 x
   768 is +19% throughput at identical params and FLOPs (see above). Config
   only. Needs a loss-curve check first -- it is a modelling change.
3. **The expert weight-gradient's masked-dense lowering** (82x declared FLOP
   inflation, 410 ms/step at 16384). NOT fixable by swapping in
   `pallas.ops.tpu.megablox.gmm`: measured 0.79x even after tuning its tiling
   to (512, 1536, 768), where its *forward* does match `ragged_dot`
   (2.20 vs 2.12 ms, 71% vs 74% of peak) but its grouped backward loses. A
   hand-written Pallas dW would have to beat XLA's masked-dense form, which
   already reaches 58% of peak for the full fwd+bwd in isolation.
4. **The dispatch permutation** -- 738 ms/step (21%) at 16384 of pure HBM
   traffic, and the `[T*K, D]` buffer behind it is a large part of what caps
   batch at 8 and context at 16K. A fused Pallas dispatch kernel that gathers
   rows inside the grouped matmul is the fix, and it is the largest remaining
   piece of work. (Not the sort: that is 5.4 ms and nothing beats it.)
5. **Expert parallelism**, to remove the 230 ms reduce-scatter. Details below.
6. Merging the `gate`/`up` ragged_dots: +7.8% of that stage, ~1.8% of the step.

Not worth it, measured: non-expert weight sharding (1.4%, and the prototype
NaN'd), batch size, every remat policy except the seq-length-dependent choice
already documented, and every collective-scheduling flag.

The ~306 ms of expert-weight communication (230 reduce-scatter +
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
