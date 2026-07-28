"""Configuration dataclasses for the Hanuman MoE transformer.

Two presets:
  - 'smoke': tiny, CPU-runnable in <2 min (random data)
  - 'full' : ~7B total / ~1B active MoE transformer
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Config:
    # ---- Model architecture ----
    name: str = "smoke"
    vocab_size: int = 256
    layers: int = 2
    d_model: int = 64
    n_q_heads: int = 2
    n_kv_heads: int = 1
    head_dim: int = 32
    n_experts: int = 4          # routed experts
    n_active: int = 2           # top-k routed experts
    n_shared_experts: int = 1   # always-on shared expert(s)
    d_ff: int = 128             # per-expert FFN hidden (SwiGLU intermediate)
    dense_layers: int = 1       # first N layers use dense FFN instead of MoE

    # ---- Norm ----
    norm_eps: float = 1e-6
    residual_scale_init: float = 1.0

    # ---- RoPE / position ----
    rope_base: float = 10000.0
    yarn_factor: float = 1.0       # 1.0 for train, 8.0 for 32K infer
    yarn_attn_factor: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0

    # ---- MoE routing ----
    router_init_std: float = 0.01
    routed_scaling_factor: float = 2.5
    bias_update_rate: float = 0.001   # gamma
    z_loss_weight: float = 1e-3
    balance_loss_weight: float = 1e-4
    n_group: int = 1                 # group-limited routing groups (1 = disabled)

    # ---- Training ----
    batch_size: int = 2
    seq_len: int = 128
    train_steps: int = 10
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 2000
    decay_fraction: float = 0.2      # last 20% of training is cosine decay
    weight_decay: float = 1.0
    grad_clip: float = 1.0
    dtype: str = "bf16"              # compute dtype
    master_dtype: str = "fp32"       # master weights
    remat: bool = True               # rematerialize each block in the backward pass

    # ---- Data ----
    dataset: str = "openbmb/Ultra-FineWeb-L3"
    dataset_config: Optional[str] = "Ultra-FineWeb-L3-en-Multi-Style-Synthetic"
    tokenizer: str = "byte"          # 'byte' | 'llama3' | path
    pack_eos_id: Optional[int] = None

    # ---- Hardware / sharding ----
    mesh_data_axis: int = 1
    mesh_expert_axis: int = 1
    checkpoint_every: int = 0        # 0 = only at end
    checkpoint_dir: str = "checkpoints"

    # ---- Inference ----
    infer_seq_len: int = 32768

    # ---- Logging ----
    log_every: int = 1

    def compute_dtype(self):
        import jax.numpy as jnp
        return {"bf16": jnp.bfloat16, "fp32": jnp.float32, "fp16": jnp.float16}[self.dtype]

    def master_dtype_jnp(self):
        import jax.numpy as jnp
        return {"fp32": jnp.float32, "bf16": jnp.bfloat16, "fp16": jnp.float16}[self.master_dtype]

    def as_dict(self):
        return asdict(self)


def smoke() -> Config:
    return Config(
        name="smoke",
        vocab_size=256,
        layers=2,
        d_model=64,
        n_q_heads=2,
        n_kv_heads=1,
        head_dim=32,
        n_experts=4,
        n_active=2,
        n_shared_experts=1,
        d_ff=128,
        dense_layers=1,
        batch_size=2,
        seq_len=128,
        train_steps=10,
        warmup_steps=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        tokenizer="byte",
        log_every=1,
        mesh_data_axis=1,
        mesh_expert_axis=1,
    )


def full() -> Config:
    return Config(
        name="full",
        vocab_size=32000,
        layers=24,
        d_model=1536,
        n_q_heads=12,
        n_kv_heads=4,
        head_dim=128,
        n_experts=80,
        n_active=8,
        n_shared_experts=1,
        d_ff=768,
        dense_layers=3,
        # One sequence per v5e chip: the batch axis is sharded over the 8-way
        # 'data' mesh axis, so batch_size must stay a multiple of the device count.
        batch_size=8,
        seq_len=4096,
        train_steps=100000,
        warmup_steps=2000,
        learning_rate=1e-4,
        min_lr=1e-5,
        decay_fraction=0.2,
        weight_decay=1.0,
        dtype="bf16",
        master_dtype="fp32",
        tokenizer="llama3",
        mesh_data_axis=8,
        mesh_expert_axis=1,
        checkpoint_every=1000,
        log_every=10,
        infer_seq_len=32768,
    )


PRESETS = {"smoke": smoke, "full": full}


def get_config(name: str) -> Config:
    if name not in PRESETS:
        raise ValueError(f"Unknown config preset: {name!r}. Choose from {list(PRESETS)}")
    return PRESETS[name]()