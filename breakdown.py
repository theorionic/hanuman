"""Per-component FLOP breakdown for one training step.

Usage:
  python breakdown.py                  # use default 3897 ms/step from your log
  python breakdown.py --ms 3897         # explicit ms/step
  python breakdown.py --ms 4500         # any measured ms/step

The breakdown uses the same step_flops() accounting as train.py, but splits
the total into: attention, dense FFN, MoE routed FFN, MoE shared expert,
router, and lm head. Times are estimated by distributing the measured ms/step
proportionally to FLOP share.

IMPORTANT CAVEAT: the FLOP-share split assumes every component runs at the same
efficiency. In reality MoE dispatch (gather/scatter of expert inputs/outputs)
is memory-bound, not compute-bound, so the MoE routed FFN takes MORE wall-time
than its FLOP share suggests, and attention (compute-bound) takes LESS. See
the note at the bottom of the output. For true per-component TPU timing,
capture a trace:

    python -c "
    import jax, jax.numpy as jnp
    from jax.experimental import profiler
    # ... your train step call site ...
    " with jax.profiler.trace('trace_dir'):
        # run ~5 steps

    tensorboard --logdir trace_dir
"""
from __future__ import annotations
import argparse


def make_config():
    from config import full_g4
    c = full_g4()
    # quality_speed profile (the one currently running)
    c.n_active = 2
    c.d_ff = 1536
    c.d_ff_dense = 2048
    c.dense_layers = 10
    c.batch_size = 16
    return c


def analytical_breakdown(config):
    """FLOP breakdown per component, matching train.py's step_flops accounting."""
    D, dff, S = config.d_model, config.d_ff, config.seq_len
    Nq, Nkv, H = config.n_q_heads, config.n_kv_heads, config.head_dim
    L, n_dense = config.layers, config.dense_layers
    n_moe = L - n_dense
    dffd = config.d_ff_dense or dff

    # Per-token, per-layer forward FLOPs
    attn_proj = 2 * D * (Nq * H + 2 * Nkv * H + Nq * H)
    attn_core = 4 * Nq * H * (S + 1) / 2          # full causal
    attn_per_layer = attn_proj + attn_core

    dense_ffn = 2 * 3 * D * dffd                   # SwiGLU: gate, up, down

    moe_routed = 2 * 3 * D * (dff * config.n_active)
    moe_shared = 2 * 3 * D * (dffd * config.n_shared_experts)
    router = 2 * D * config.n_experts

    head = 2 * D * config.vocab_size

    # Totals (forward only, per token)
    attn_total = L * attn_per_layer
    dense_ffn_total = n_dense * dense_ffn
    moe_routed_total = n_moe * moe_routed
    moe_shared_total = n_moe * moe_shared
    router_total = n_moe * router
    head_total = head

    fwd_per_token = (attn_total + dense_ffn_total + moe_routed_total +
                     moe_shared_total + router_total + head_total)
    tokens = config.batch_size * config.seq_len

    # model = 3x forward (fwd + bwd), hardware = +1x forward for remat
    model = 3 * fwd_per_token * tokens
    hardware = model + (fwd_per_token * tokens if config.remat else 0)

    components = {
        "Attention (24 layers)":      3 * attn_total * tokens,
        "Dense FFN (10 layers)":      3 * dense_ffn_total * tokens,
        "MoE routed FFN (14 layers)": 3 * moe_routed_total * tokens,
        "MoE shared expert (14)":     3 * moe_shared_total * tokens,
        "Router (14 layers)":         3 * router_total * tokens,
        "LM head":                    3 * head_total * tokens,
    }

    return components, model, hardware, tokens


def print_breakdown(components, model, hardware, tokens, ms_per_step):
    peak_tflops = 8 * 197  # v5e-8
    print(f"\n{'='*72}")
    print(f"  FLOP BREAKDOWN (fwd + bwd = 3x fwd, {tokens:,} tokens/step)")
    print(f"  Config: batch 16, 10 dense + 14 MoE, top-2/40, d_ff=1536, d_ff_dense=2048")
    print(f"{'='*72}")
    print(f"{'Component':<32} {'TFLOP':>10} {'% of step':>10} {'est. ms':>10}")
    print(f"{'-'*72}")
    for name, flops in components.items():
        tflop = flops / 1e12
        pct = 100 * flops / model
        est_ms = flops / model * ms_per_step
        print(f"{name:<32} {tflop:>10.1f} {pct:>9.1f}% {est_ms:>10.0f}")
    print(f"{'-'*72}")
    tflop_total = model / 1e12
    hw_total = hardware / 1e12
    print(f"{'TOTAL (model)':<32} {tflop_total:>10.1f} {'100.0%':>10} {ms_per_step:>10.0f}")
    print(f"{'TOTAL (w/ remat)':<32} {hw_total:>10.1f}")
    achieved = tflop_total / (ms_per_step / 1000)
    mfu = 100 * achieved / peak_tflops
    print(f"\n  {ms_per_step:.0f} ms/step  ->  {achieved:.0f} TFLOP/s achieved"
          f"  ->  MFU {mfu:.1f}%  (peak {peak_tflops:.0f} TFLOP/s)")
    print(f"\n  NOTE: FLOP-share assumes uniform efficiency. In reality:")
    print(f"    - MoE routed FFN is MEMORY-BOUND (gather/scatter dispatch),")
    print(f"      so its real wall-time is HIGHER than the est. ms above.")
    print(f"    - Attention is compute-bound, so its real wall-time is LOWER.")
    print(f"    - For true per-component timing, capture a jax.profiler trace.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=float, default=3897,
                     help="measured ms/step (default: 3897 from your log)")
    args = ap.parse_args()

    config = make_config()
    components, model, hardware, tokens = analytical_breakdown(config)
    print_breakdown(components, model, hardware, tokens, args.ms)


if __name__ == "__main__":
    main()