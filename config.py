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
    d_ff: int = 128             # per-routed-expert FFN hidden (SwiGLU intermediate)
    # Width of the always-on paths: the shared expert and the dense-layer FFN.
    # None means "same as d_ff". Separating them lets routed-expert granularity
    # be changed without also resizing every dense matmul in the model.
    d_ff_dense: Optional[int] = None
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
    # Which intermediates survive into the backward pass. See model.transformer.
    # remat_policy: 'full' | 'dots' | 'dots_no_batch' | 'experts' | 'none'
    # Measured on the 7B config, fwd+bwd: full 2472 ms / dots 1036 ms /
    # dots_no_batch 1021 ms. 'experts' saves too much and runs out of HBM.
    remat_policy: str = "dots_no_batch"
    fused_step: bool = True          # backward + optimizer in one jit
    opt_state_dtype: str = "bf16"    # Lion momentum dtype ('bf16' halves opt state)

    # ---- Data ----
    dataset: str = "openbmb/Ultra-FineWeb-L3"
    dataset_config: Optional[str] = "Ultra-FineWeb-L3-en-Multi-Style-Synthetic"
    # Direct parquet glob. Set this and dataset_config is bypassed -- resolving
    # the named config against this repo's 1771 files takes minutes, the glob
    # takes ~2s. Set to None to go back through the config machinery.
    dataset_files: Optional[str] = (
        "hf://datasets/openbmb/Ultra-FineWeb-L3/"
        "data/ultrafineweb_en_l3/multi_style/*.parquet")
    tokenizer: str = "byte"          # 'byte' | 'llama3' | path
    pack_eos_id: Optional[int] = None

    # ---- Grain dataloader (grain_data.py) ----
    # When True, real-data runs use the Grain pipeline: parquet read + tokenize +
    # concat-split packing + batch all on a CPU prefetch thread, with batches
    # double-buffered onto the TPU so the step never waits on the host. False
    # keeps the hand-rolled thread-prefetch pipeline in data.py.
    use_grain: bool = False
    grain_interleave: int = 4        # parquet files read concurrently
    grain_cpu_buffer: int = 8        # batches buffered on the host prefetch thread
    grain_device_buffer: int = 2     # batches prefetched per device (in HBM)
    grain_shuffle_files: bool = True # shuffle shard order before streaming
    grain_seed: int = 0

    # ---- Hardware / sharding ----
    mesh_data_axis: int = 1
    mesh_expert_axis: int = 1
    checkpoint_every: int = 0        # 0 = only at end
    checkpoint_dir: str = "checkpoints"

    # ---- Sliding window attention (SWA) hybrid ----
    # When use_swa=True, layers alternate between local (sliding window)
    # attention and full causal attention. This halves attention FLOPs at long
    # context while keeping global context through the full-attention layers.
    #   use_swa    : enable SWA hybrid mode
    #   swa_window : number of tokens each SWA layer attends to (to the left,
    #                plus itself). Must be >= 1.
    #   swa_period : every Nth layer is full attention, the rest are SWA.
    #                period=2 -> alternating (layer 0 SWA, layer 1 full, ...).
    #                period=4 -> 3 SWA then 1 full (Mistral-style).
    #                Layer i is full when (i % swa_period == swa_period - 1).
    use_swa: bool = False
    swa_window: int = 4096
    swa_period: int = 2

    # ---- Kimi Delta Attention (KDA) hybrid ----
    # When use_kda=True, most layers use KDA (linear attention with the delta
    # update rule, Kimi-Linear arXiv:2507.05927) and every kda_period-th layer
    # is full causal attention. This is the Kimi-Linear pattern: 3 KDA : 1 full
    # (kda_period=4). Layer i is "full" when (i % kda_period == kda_period-1),
    # else "kda" -- the same convention as swa_period.
    #   kda_heads     : number of KDA heads (can differ from n_q_heads)
    #   kda_head_dim  : per-head dim for KDA layers (V=K)
    #   kda_chunk_size: chunk size for the (future) chunked parallel form
    #   use_scan      : when False, run layers in a plain Python loop instead of
    #                   lax.scan. KDA and full-attention layers have different
    #                   module structures, so a mixed stack cannot be scanned;
    #                   we disable scan for the KDA hybrid. The scan was a
    #                   compile-time optimization, irrelevant at smoke scale.
    use_kda: bool = False
    kda_period: int = 4
    kda_heads: int = 64
    kda_head_dim: int = 128
    kda_chunk_size: int = 64
    use_scan: bool = True

    # ---- Inference ----
    infer_seq_len: int = 32768

    # ---- In-training sampling ----
    # Every `gen_every` steps, generate a short sample from the live weights so
    # generation quality can be tracked as training proceeds. 0 disables it.
    gen_every: int = 0
    gen_prompt: str = "The meaning of life is"
    gen_max_tokens: int = 40
    gen_temperature: float = 0.8
    gen_top_k: int = 40

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
        # Exercise the SWA code path. smoke seq_len=128, so a window of 64
        # means each SWA layer attends to 64 tokens (half the sequence) --
        # small enough to actually constrain the window, large enough to train.
        #
        # KDA is deliberately NOT enabled here as well. `use_kda` outranks
        # `use_swa` in Transformer.layer_attention_type, so a preset that sets
        # both silently tests only KDA and leaves SWA -- the path that is
        # actually on by default in long-context runs -- completely uncovered.
        # See smoke_kda() for the KDA hybrid.
        use_swa=True,
        swa_window=64,
        swa_period=2,
    )


def smoke_kda() -> Config:
    """Smoke preset for the KDA hybrid (see smoke() for why it is separate).

    With 2 layers and kda_period=2: layer 0 = KDA, layer 1 = full attention.
    KDA heads/dim are tiny so this stays fast on CPU. `use_scan` is off because
    KDA and full-attention layers have different module structures and a mixed
    stack cannot be lax.scan'd.
    """
    c = smoke()
    c.name = "smoke_kda"
    c.use_swa = False
    c.use_kda = True
    c.kda_period = 2
    c.kda_heads = 2
    c.kda_head_dim = 32
    c.kda_chunk_size = 16
    c.use_scan = False
    return c


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
        # 16K context is the biggest MFU lever measured: 21.6% vs 11.5% at 4096
        # (1.88x), because the fixed ~230ms expert weight-gradient reduce-scatter
        # amortizes over 4x the tokens. remat_policy MUST be 'full' here -- at
        # 16384 'dots_no_batch' OOMs (22.1 GB of temporaries); 'full' fits at
        # batch 8. See BENCHMARKS.md "Sequence length is the biggest MFU lever".
        seq_len=16384,
        train_steps=100000,
        warmup_steps=2000,
        learning_rate=1e-4,
        min_lr=1e-5,
        decay_fraction=0.2,
        weight_decay=1.0,
        dtype="bf16",
        master_dtype="fp32",
        remat_policy="full",   # required at seq_len 16384 (dots_no_batch OOMs)
        tokenizer="llama3",
        use_grain=True,
        mesh_data_axis=8,
        mesh_expert_axis=1,
        checkpoint_every=1000,
        log_every=10,
        infer_seq_len=32768,
        gen_every=1000,
    )


def full_g4() -> Config:
    """Coarser-granularity variant of `full`: 40 experts x d_ff 1536, top-4.

    Identical params (0.880B active) and FLOPs as full's 80 x 768 top-8, but it
    halves the dispatch row count (T*K), which is the largest reclaimable piece
    of the MoE step. Measured at 16K: 25.7% MFU / 45.5k tok/s vs full's 21.6% /
    38.2k -- +19% throughput, +4 MFU, reproduced across runs.

    This is a *modelling* change: top-4-of-40 is a coarser routing space than
    top-8-of-80, so validate the loss curve before preferring it to `full`.
    d_ff_dense is pinned to 768 so the shared/dense FFN width is unchanged.
    """
    c = full()
    c.name = "full_g4"
    c.n_experts = 40
    c.n_active = 4
    c.d_ff = 1536
    c.d_ff_dense = 768
    return c


PRESETS = {"smoke": smoke, "smoke_kda": smoke_kda, "full": full, "full_g4": full_g4}


def get_config(name: str) -> Config:
    if name not in PRESETS:
        raise ValueError(f"Unknown config preset: {name!r}. Choose from {list(PRESETS)}")
    return PRESETS[name]()