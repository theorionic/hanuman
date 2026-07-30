"""CLI entry point for Hanuman MoE transformer.

Usage:
  python main.py train --config smoke
  python main.py train --config full
  python main.py generate --config smoke --prompt "hello" --max_tokens 20
  python main.py count --config full
"""
from __future__ import annotations

import argparse
import os
import sys

# Ensure local imports work when run from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# TPU compiler flags. Must be set before the libtpu backend initializes, i.e.
# before anything imports jax -- hence at module scope, above the jax imports
# that live inside the command functions below.
#
#   scoped_vmem_limit_kib   ragged_dot picks a [4096, 512, 768] tiling whose
#                           working set is 33.5 MB; the stock 16 MB scoped VMEM
#                           limit rejects it outright at batch >= 16.
#   async_collective_fusion  let the expert-weight all-gathers start early and
#   overlap_compute_collective_tc  overlap with the matmuls that precede them.
_TPU_FLAGS = [
    "--xla_tpu_scoped_vmem_limit_kib=98304",
    "--xla_tpu_enable_async_collective_fusion=true",
    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true",
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
    "--xla_tpu_overlap_compute_collective_tc=true",
    "--xla_enable_async_all_gather=true",
]
if os.environ.get("HANUMAN_NO_TPU_FLAGS") != "1":
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " " + " ".join(_TPU_FLAGS)
    ).strip()

# Persistent XLA compilation cache. The 7B step takes ~50 s to compile, which
# is paid again on every process start -- and dominates short runs and config
# sweeps, where the compiled program is usually identical. The cache key covers
# the HLO, so a genuine shape/config change still recompiles.
_CACHE_DIR = os.environ.get("HANUMAN_CACHE_DIR",
                            os.path.expanduser("~/.cache/hanuman-jax"))
if _CACHE_DIR and _CACHE_DIR != "off":
    os.makedirs(_CACHE_DIR, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE_DIR)
    # Defaults skip small/fast entries; we want the big step function cached.
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1.0")


def cmd_train(args):
    from config import get_config
    from train import train
    config = get_config(args.config)
    if args.steps is not None:
        config.train_steps = args.steps
    if args.batch is not None:
        config.batch_size = args.batch
    if args.seq_len is not None:
        config.seq_len = args.seq_len
    if args.remat is not None:
        config.remat = args.remat
    if args.remat_policy is not None:
        config.remat_policy = args.remat_policy
    if args.experts is not None:
        config.n_experts = args.experts
    if args.d_ff is not None:
        config.d_ff = args.d_ff
    if args.d_ff_dense is not None:
        config.d_ff_dense = args.d_ff_dense
    if args.n_active is not None:
        config.n_active = args.n_active
    if args.dense_layers is not None:
        config.dense_layers = args.dense_layers
    if args.z_weight is not None:
        config.z_loss_weight = args.z_weight
    if args.bias_rate is not None:
        config.bias_update_rate = args.bias_rate
    if args.balance_weight is not None:
        config.balance_loss_weight = args.balance_weight
    if args.warmup is not None:
        config.warmup_steps = args.warmup
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.log_every is not None:
        config.log_every = args.log_every
    if args.data_axis is not None:
        config.mesh_data_axis = args.data_axis
    if args.expert_axis is not None:
        config.mesh_expert_axis = args.expert_axis
    if args.use_swa is not None:
        config.use_swa = args.use_swa
    if args.swa_window is not None:
        config.swa_window = args.swa_window
    if args.swa_period is not None:
        config.swa_period = args.swa_period
    if args.use_grain is not None:
        config.use_grain = args.use_grain
    if args.gen_every is not None:
        config.gen_every = args.gen_every
    if args.gen_prompt is not None:
        config.gen_prompt = args.gen_prompt
    if args.gen_max_tokens is not None:
        config.gen_max_tokens = args.gen_max_tokens
    if args.use_kda is not None:
        config.use_kda = args.use_kda
    if args.kda_period is not None:
        config.kda_period = args.kda_period
    use_random = args.config.startswith("smoke") or args.random_data
    train(config, use_random_data=use_random, save=not args.no_save)


def cmd_generate(args):
    from config import get_config
    from generate import load_for_generate
    config = get_config(args.config)
    if args.yarn_factor is not None:
        config.yarn_factor = args.yarn_factor
    text, ids = load_for_generate(config, args.prompt, args.max_tokens,
                                  temperature=args.temperature, top_k=args.top_k, seed=args.seed)
    print("=== Generated ===")
    print(text)
    print("=== Token ids ===")
    print(ids)


def cmd_report(args):
    from config import get_config
    from train import memory_report
    print(memory_report(get_config(args.config)))


def cmd_count(args):
    from config import get_config
    from train import count_total_params, count_active_params
    config = get_config(args.config)
    total = count_total_params(config)
    active = count_active_params(config)
    print(f"Config: {config.name}")
    print(f"  Layers: {config.layers} (dense={config.dense_layers}, moe={config.layers - config.dense_layers})")
    print(f"  d_model={config.d_model}, d_ff={config.d_ff}, heads q/kv={config.n_q_heads}/{config.n_kv_heads} head_dim={config.head_dim}")
    print(f"  Experts: {config.n_experts} routed + {config.n_shared_experts} shared, top-{config.n_active}")
    print(f"  Total params:   {total:,} ({total/1e9:.3f}B)")
    print(f"  Active params:  {active:,} ({active/1e9:.3f}B)")


def main():
    parser = argparse.ArgumentParser(description="Hanuman MoE transformer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Run training")
    # full_g4 is the default: top-4-of-40 x d_ff 1536 (identical params/FLOPs to
    # full's top-8-of-80 x 768) halves the dispatch row count, measured 25.7% MFU
    # / 45.5k tok/s vs full's 21.6% / 38.2k (+19% throughput, +4 MFU). See
    # BENCHMARKS.md "Expert granularity is the cheapest large win".
    p_train.add_argument("--config", default="full_g4", choices=list(__import__("config").PRESETS))
    p_train.add_argument("--steps", type=int, default=None)
    p_train.add_argument("--batch", type=int, default=None)
    p_train.add_argument("--seq_len", type=int, default=None)
    p_train.add_argument("--random_data", action="store_true", help="Force random data")
    p_train.add_argument("--no_save", action="store_true", help="Skip the final checkpoint")
    p_train.add_argument("--experts", type=int, default=None, help="Routed experts per MoE layer")
    p_train.add_argument("--d_ff", type=int, default=None, help="Per-routed-expert SwiGLU hidden size")
    p_train.add_argument("--d_ff_dense", type=int, default=None,
                         help="Shared-expert / dense-layer FFN width (default: same as --d_ff)")
    p_train.add_argument("--n_active", type=int, default=None, help="Top-k routed experts")
    p_train.add_argument("--dense_layers", type=int, default=None,
                         help="Number of first layers using dense FFN (rest are MoE)")
    p_train.add_argument("--z_weight", type=float, default=None,
                         help="Router z-loss weight")
    p_train.add_argument("--bias_rate", type=float, default=None,
                         help="Router-bias load-balancing step (DeepSeek gamma)")
    p_train.add_argument("--balance_weight", type=float, default=None,
                         help="Weight on the sequence-wise balance loss")
    p_train.add_argument("--warmup", type=int, default=None, help="Warmup steps")
    p_train.add_argument("--lr", type=float, default=None, help="Peak learning rate")
    p_train.add_argument("--remat_policy", default=None,
                         choices=["full", "dots", "dots_no_batch", "experts", "none"],
                         help="Which intermediates to keep for the backward pass")
    p_train.add_argument("--log_every", type=int, default=None)
    p_train.add_argument("--data_axis", type=int, default=None, help="Mesh 'data' axis size")
    p_train.add_argument("--expert_axis", type=int, default=None, help="Mesh 'expert' axis size")
    p_train.add_argument("--remat", dest="remat", action="store_true", default=None,
                         help="Force block rematerialization on")
    p_train.add_argument("--no_remat", dest="remat", action="store_false",
                         help="Disable block rematerialization")
    p_train.add_argument("--use_swa", dest="use_swa", action="store_true", default=None,
                         help="Sliding-window attention hybrid")
    p_train.add_argument("--no_swa", dest="use_swa", action="store_false",
                         help="Disable the SWA hybrid")
    p_train.add_argument("--swa_window", type=int, default=None,
                         help="Tokens each SWA layer attends to (rounded up to 128 on TPU)")
    p_train.add_argument("--swa_period", type=int, default=None,
                         help="Every Nth layer is full attention, the rest are SWA")
    p_train.add_argument("--gen_every", type=int, default=None,
                         help="Generate a sample from the live weights every N steps (0=off)")
    p_train.add_argument("--gen_prompt", type=str, default=None,
                         help="Prompt for in-training generation")
    p_train.add_argument("--gen_max_tokens", type=int, default=None,
                         help="Tokens to generate per in-training sample")
    p_train.add_argument("--use_grain", dest="use_grain", action="store_true", default=None,
                         help="Stream real data through the Grain dataloader (grain_data.py)")
    p_train.add_argument("--no_grain", dest="use_grain", action="store_false",
                         help="Use the data.py thread-prefetch pipeline instead of Grain")
    p_train.add_argument("--use_kda", dest="use_kda", action="store_true", default=None,
                         help="KDA linear-attention hybrid (SLOW: see model/kda.py)")
    p_train.add_argument("--no_kda", dest="use_kda", action="store_false",
                         help="Disable the KDA hybrid")
    p_train.add_argument("--kda_period", type=int, default=None,
                         help="Every Nth layer is full attention, the rest are KDA")
    p_train.set_defaults(func=cmd_train)

    p_gen = sub.add_parser("generate", help="Generate text")
    p_gen.add_argument("--config", default="smoke", choices=list(__import__("config").PRESETS))
    p_gen.add_argument("--prompt", default="Once upon a time")
    p_gen.add_argument("--max_tokens", type=int, default=100)
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--top_k", type=int, default=0)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--yarn_factor", type=float, default=None)
    p_gen.set_defaults(func=cmd_generate)

    p_count = sub.add_parser("count", help="Print param counts")
    p_count.add_argument("--config", default="full_g4", choices=list(__import__("config").PRESETS))

    p_report = sub.add_parser("report", help="Per-tensor sharding and HBM table")
    p_report.add_argument("--config", default="full_g4", choices=list(__import__("config").PRESETS))
    p_report.set_defaults(func=cmd_report)
    p_count.set_defaults(func=cmd_count)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()