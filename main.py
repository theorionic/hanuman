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
    use_random = args.config == "smoke" or args.random_data
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
    p_train.add_argument("--config", default="smoke", choices=["smoke", "full"])
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
    p_train.set_defaults(func=cmd_train)

    p_gen = sub.add_parser("generate", help="Generate text")
    p_gen.add_argument("--config", default="smoke", choices=["smoke", "full"])
    p_gen.add_argument("--prompt", default="Once upon a time")
    p_gen.add_argument("--max_tokens", type=int, default=100)
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--top_k", type=int, default=0)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--yarn_factor", type=float, default=None)
    p_gen.set_defaults(func=cmd_generate)

    p_count = sub.add_parser("count", help="Print param counts")
    p_count.add_argument("--config", default="full", choices=["smoke", "full"])

    p_report = sub.add_parser("report", help="Per-tensor sharding and HBM table")
    p_report.add_argument("--config", default="full", choices=["smoke", "full"])
    p_report.set_defaults(func=cmd_report)
    p_count.set_defaults(func=cmd_count)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()