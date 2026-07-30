#!/usr/bin/env bash
# ============================================================================
# hanuman — best training configs on TPU v5e-8
#
# All configs use: full_g4 base, top-1 routing (--n_active 1 --d_ff 1536),
# 10-14 dense layers (shifts compute from dispatch-bound MoE path to the
# ~92%-peak dense path), wider shared expert (--d_ff_dense), batch 8 or 16.
#
# Measured on TPU v5e-8 (8 chips, 1576 TFLOP/s bf16 peak), JAX 0.11.0,
# 16K context, random data, loss parity confirmed (~10.88).
#
# Old default (full):      21.6% MFU,  38,132 tok/s
# full_g4 default:         25.7% MFU,  45,459 tok/s
# Best MFU below:          41.4% MFU,  40,014 tok/s  (+91% MFU vs old default)
# Best throughput below:   34.8% MFU,  83,034 tok/s  (+118% tok/s vs old default)
# Balanced below:          37.7% MFU,  75,687 tok/s
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------------------------------------------------------
# 1. BEST MFU — max chip efficiency (41.4% MFU, 40k tok/s)
#    batch 8, 10 dense + 14 MoE, top-1, d_ff_dense=16384
#    The wider shared expert shifts ~90% of FFN compute to the dispatch-free
#    dense path that runs at ~92% of MXU peak. 10 dense layers is the sweet
#    spot: fewer MoE layers = less dispatch overhead, but enough MoE remains
#    for expert specialization.
# ----------------------------------------------------------------------------
best_mfu() {
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
# 2. BEST THROUGHPUT — max tokens/sec (83,034 tok/s, 34.8% MFU)
#    batch 16, 14 dense + 10 MoE, top-1, d_ff_dense=2048
#    Doubling the batch amortizes dispatch overhead across 2x tokens.
#    14 dense layers minimizes the number of MoE dispatch buffers (the
#    memory wall at batch 16). Narrower shared expert fits the HBM budget.
# ----------------------------------------------------------------------------
best_throughput() {
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
# 3. BALANCED — quality + efficiency (37.7% MFU, 75,687 tok/s)
#    batch 16, 10 dense + 14 MoE, top-1, d_ff=768, d_ff_dense=4096
#    Smaller routed experts (d_ff=768) shrink the dispatch buffer, freeing
#    HBM for a wider shared expert (4096) at batch 16. Keeps 14 MoE layers
#    for expert specialization capacity. Best quality/efficiency tradeoff.
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
# 4. QUALITY-PRESERVING — top-2 routing, batch 16 (67,248 tok/s, 31.5% MFU)
#    batch 16, 10 dense + 14 MoE, top-2, d_ff_dense=2048
#    Top-2 routing preserves expert diversity per token (most MoE models use
#    top-2+). Lower MFU than top-1 but better modelling quality. Use this
#    if top-1 loss curves diverge over long training.
# ----------------------------------------------------------------------------
quality() {
  python main.py train \
    --config full_g4 \
    --n_active 2 \
    --d_ff 1536 \
    --d_ff_dense 2048 \
    --dense_layers 10 \
    --batch 16 \
    "$@"
}

# ----------------------------------------------------------------------------
# Usage
# ----------------------------------------------------------------------------
usage() {
  cat <<EOF
Usage: $0 <profile> [extra args passed to main.py]

Profiles:
  mfu         Best MFU:        41.4% MFU,  40,014 tok/s  (batch 8)
  throughput  Best throughput: 34.8% MFU,  83,034 tok/s  (batch 16)
  balanced    Quality+eff:     37.7% MFU,  75,687 tok/s  (batch 16)
  quality     Top-2 routing:   31.5% MFU,  67,248 tok/s  (batch 16)

Examples:
  $0 mfu --steps 1000 --log_every 50
  $0 throughput --random_data --steps 100 --no_save
  $0 balanced --use_grain --steps 5000
EOF
}

case "${1:-}" in
  mfu)        shift; best_mfu "$@" ;;
  throughput) shift; best_throughput "$@" ;;
  balanced)   shift; balanced "$@" ;;
  quality)    shift; quality "$@" ;;
  -h|--help)  usage ;;
  *)          usage; exit 1 ;;
esac