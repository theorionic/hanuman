#!/usr/bin/env bash
# ============================================================================
# hanuman — best training configs on TPU v5e-8
#
# All configs use full_g4 base (40 experts, d_model 1536, 16K context).
# Measured on TPU v5e-8 (8 chips, 1576 TFLOP/s bf16 peak), JAX 0.11.0,
# random data, loss parity confirmed (~10.88).
#
# Old default (full):      21.6% MFU,  38,132 tok/s
# full_g4 default:         25.7% MFU,  45,459 tok/s
#
# SOTA shared-expert reference (shared_d_ff / routed_d_ff ratio):
#   DeepSeek-V3:   1.0x  (11% shared compute)
#   DeepSeek-V2:   2.0x  (25% shared compute)
#   Llama 4:       1.0x  (50% shared, top-1 routing)
#   Granite-tiny:  2.0x  (25% shared compute)
# Nobody in SOTA uses ratio > 2x. The quality profile below uses 2.0x.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------------------------------------------------------
# 1. QUALITY — SOTA-grounded, DeepSeek-V2 style (RECOMMENDED for real training)
#    batch 8, 3 dense + 21 MoE (default), top-4, d_ff_dense=3072 (2.0x ratio)
#    25% of FFN compute in the shared expert — matches DeepSeek-V2/MoE-16B.
#    Top-4 routing preserves expert diversity. Default 3 dense / 21 MoE keeps
#    maximum expert specialization capacity. This is the config to use when
#    model quality matters more than raw throughput.
# ----------------------------------------------------------------------------
quality() {
  python main.py train \
    --config full_g4 \
    --n_active 4 \
    --d_ff 1536 \
    --d_ff_dense 3072 \
    --batch 8 \
    "$@"
}

# ----------------------------------------------------------------------------
# 2. QUALITY + THROUGHPUT — SOTA ratio, batch 16, top-2
#    batch 16, 3 dense + 21 MoE, top-2, d_ff_dense=3072 (2.0x ratio)
#    Same 25% shared compute as the quality profile, but top-2 routing (less
#    dispatch) and batch 16 (2x throughput). d_ff_dense=3072 fits batch 16
#    with top-2. Use this when you want SOTA-grounded quality AND speed.
# ----------------------------------------------------------------------------
quality_speed() {
  python main.py train \
    --config full_g4 \
    --n_active 2 \
    --d_ff 1536 \
    --d_ff_dense 3072 \
    --batch 16 \
    "$@"
}

# ----------------------------------------------------------------------------
# 3. BALANCED — efficiency-focused, still defensible (37.7% MFU, 75.7k tok/s)
#    batch 16, 10 dense + 14 MoE, top-1, d_ff=768, d_ff_dense=4096 (2.7x ratio)
#    Slightly above SOTA ratio (2.7x vs 2.0x max). Smaller routed experts
#    (d_ff=768) shrink dispatch buffer, wider shared expert shifts compute
#    to the efficient dense path. 10 dense layers fill dispatch gaps.
#    Use for speed when you can tolerate a less conservative shared ratio.
# ----------------------------------------------------------------------------
balanced() {
  python main.py train \
    --config full_g4 \
    --n_active 1 \
    --d_ff 768 \
    --d_ff_dense 4096 \
    --dense_layers 10 \
    --batch 16 \
    "$@"
}

# ----------------------------------------------------------------------------
# 4. BEST THROUGHPUT — max tokens/sec (83,034 tok/s, 34.8% MFU)
#    batch 16, 14 dense + 10 MoE, top-1, d_ff_dense=2048 (1.3x ratio)
#    14 dense layers minimize MoE dispatch buffers to fit batch 16.
#    d_ff_dense=2048 (1.3x) is within SOTA range. Top-1 routing minimizes
#    dispatch. Highest raw throughput — use for iteration speed / sweeps.
# ----------------------------------------------------------------------------
throughput() {
  python main.py train \
    --config full_g4 \
    --n_active 1 \
    --d_ff 1536 \
    --d_ff_dense 2048 \
    --dense_layers 14 \
    --batch 16 \
    "$@"
}

# ----------------------------------------------------------------------------
# 5. BEST MFU — max chip efficiency (41.4% MFU, 40k tok/s) [BENCHMARK ONLY]
#    batch 8, 10 dense + 14 MoE, top-1, d_ff_dense=16384 (10.7x ratio)
#    WARNING: 10.7x shared ratio is far outside SOTA practice (max 2.0x).
#    This is a dense model with MoE seasoning, not a true fine-grained MoE.
#    Use only for MFU benchmarking / hardware utilization experiments.
#    Do NOT use for production training — quality will likely degrade.
# ----------------------------------------------------------------------------
mfu() {
  python main.py train \
    --config full_g4 \
    --n_active 1 \
    --d_ff 1536 \
    --d_ff_dense 16384 \
    --dense_layers 10 \
    --batch 8 \
    "$@"
}

# ----------------------------------------------------------------------------
# Usage
# ----------------------------------------------------------------------------
usage() {
  cat <<EOF
Usage: $0 <profile> [extra args passed to main.py]

Profiles (ordered by quality -> speed):
  quality      SOTA 2.0x shared, top-4, batch 8  [RECOMMENDED for real training]
  quality_speed SOTA 2.0x shared, top-2, batch 16 [quality + 2x throughput]
  balanced     2.7x shared, top-1, batch 16       [efficiency-focused]
  throughput   1.3x shared, top-1, batch 16       [max tok/s: 83k]
  mfu          10.7x shared, top-1, batch 8       [BENCHMARK ONLY - not for training]

SOTA shared-expert ratio reference:
  DeepSeek-V3: 1.0x | DeepSeek-V2: 2.0x | Llama 4: 1.0x | Granite: 2.0x
  Nobody in SOTA uses ratio > 2.0x for production.

Examples:
  $0 quality --steps 10000 --log_every 100 --use_grain
  $0 quality_speed --steps 5000 --log_every 50
  $0 throughput --random_data --steps 100 --no_save
EOF
}

case "${1:-}" in
  quality)       shift; quality "$@" ;;
  quality_speed) shift; quality_speed "$@" ;;
  balanced)      shift; balanced "$@" ;;
  throughput)    shift; throughput "$@" ;;
  mfu)           shift; mfu "$@" ;;
  -h|--help)     usage ;;
  *)             usage; exit 1 ;;
esac