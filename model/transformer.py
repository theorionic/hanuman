"""Transformer block (pre-norm, residual scale) and full model."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .attention import Attention
from .kda import KDA
from .moe import MoE, DenseFFN
from .rope import RopeCache, precompute_rope


class RMSNorm(nnx.Module):
    def __init__(self, d_model: int, eps: float, dtype):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((d_model,), dtype=jnp.float32))
        self.dtype = dtype

    def __call__(self, x):
        x = x.astype(jnp.float32)
        var = jnp.mean(x ** 2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.eps)
        return (x * self.weight).astype(self.dtype)


class Block(nnx.Module):
    """Pre-norm transformer block with residual scale.

    Uses dense FFN for first `dense_layers` layers, MoE for the rest.

    `attention_type` selects the attention submodule:
      - "full": standard GQA with full causal attention (window=None).
      - "swa" : GQA with a sliding window (set via `window`).
      - "kda" : Kimi Delta Attention (linear attention, delta update rule).
    KDA layers have a completely different parameter set from GQA layers, so
    they cannot share a BlockStack with full/swa layers (see Transformer).

    `window` is the sliding-window size for this block's attention, or None for
    full causal attention. It is stored on the Attention module as a static
    (Python int) default. At call time a per-layer window can be passed in
    (see BlockStack) to override it -- this is how the SWA hybrid applies a
    different window to each layer from a single scanned stack.
    """

    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int, head_dim: int,
                 d_ff: int, n_experts: int, n_active: int, n_shared_experts: int,
                 d_ff_dense: int,
                 router_init_std: float, routed_scaling_factor: float,
                 norm_eps: float, residual_scale_init: float,
                 use_moe: bool, dtype, rngs: nnx.Rngs, mesh=None,
                 window: int | None = None,
                 attention_type: str = "full",
                 kda_heads: int = 64, kda_head_dim: int = 128,
                 kda_chunk_size: int = 64,
                 window_choices=None):
        self.attention_type = attention_type
        self.norm1 = RMSNorm(d_model, norm_eps, dtype)
        if attention_type == "kda":
            # KDA ignores RoPE and the window; it has its own ShortConv
            # front-end and recurrent state. heads/dim are independent of the
            # GQA heads so a KDA layer can use a different head count.
            self.attn = KDA(d_model, kda_heads, kda_head_dim, dtype, rngs,
                            chunk_size=kda_chunk_size)
        else:
            self.attn = Attention(d_model, n_q_heads, n_kv_heads, head_dim, dtype,
                                  rngs, mesh=mesh, window=window,
                                  window_choices=window_choices)
        self.norm2 = RMSNorm(d_model, norm_eps, dtype)
        if use_moe:
            self.ffn = MoE(d_model, d_ff, n_experts, n_active, n_shared_experts,
                           router_init_std, routed_scaling_factor, dtype, rngs,
                           mesh=mesh, d_ff_shared=d_ff_dense)
        else:
            self.ffn = DenseFFN(d_model, d_ff_dense, dtype, rngs)
        self.use_moe = use_moe
        # residual scale (DeepSeek-style learnable scalar, init 1.0)
        self.residual_scale = nnx.Param(jnp.array(residual_scale_init, dtype=jnp.float32))

    def __call__(self, x, cos, sin, positions=None, window=None, window_sel=None):
        # pre-norm attention
        h = self.norm1(x)
        if self.attention_type == "kda":
            # KDA ignores cos/sin/positions/window (no RoPE, no sliding window).
            a = self.attn(h)
        else:
            a = self.attn(h, cos, sin, positions=positions, window=window,
                          window_sel=window_sel)
        x = x + self.residual_scale.astype(jnp.float32) * a
        # pre-norm FFN
        h = self.norm2(x)
        if self.use_moe:
            f, aux = self.ffn(h)
            x = x + self.residual_scale.astype(jnp.float32) * f
            return x, aux
        else:
            f = self.ffn(h)
            x = x + self.residual_scale.astype(jnp.float32) * f
            return x, {}


def remat_policy(name: str):
    """Map a config string to a `jax.checkpoint` policy.

    Which intermediates are kept for the backward pass is the single biggest
    throughput knob in this model. `full` (save nothing, recompute everything)
    is the textbook choice, but it measures at ~76 ms per MoE layer against an
    11 ms forward: the recompute drags the expert-weight all-gather and the
    routing sort onto the backward's critical path, where there is no other
    work to overlap them with. Saving just the grouped-matmul outputs removes
    most of that while keeping the big [T*K, D] tensors out of HBM.
    """
    p = jax.checkpoint_policies
    return {
        "none": None,                                   # no remat at all
        "full": p.nothing_saveable,                     # recompute everything
        "dots": p.checkpoint_dots,                      # keep every matmul output
        "dots_no_batch": p.checkpoint_dots_with_no_batch_dims,
        "experts": p.save_only_these_names("moe_gate", "moe_up", "moe_out"),
    }[name]


class BlockStack(nnx.Module):
    """`n` structurally identical blocks with their params stacked on axis 0.

    The forward pass is one `lax.scan`, so XLA compiles a single block body
    instead of `n` unrolled copies. That keeps compile time and compiler memory
    flat in depth -- unrolling 21 MoE layers (each with its own shard_map and
    remat region) produced an HLO graph that took minutes and hundreds of GB of
    host RAM to optimize.

    SWA hybrid: when the layers in this stack do not all share one window,
    `window_sel` is a static per-layer tuple of indices into the block's
    `window_choices`. The index is scanned alongside the layer state, and the
    body dispatches on it with `lax.switch` -- one branch per distinct window,
    each carrying its own statically-masked Splash kernel.

    The window must never be threaded through the scan as a *value*. Splash
    builds its block-sparse mask at trace time, so a traced window silently
    drops every layer onto `jax.nn.dot_product_attention`, which materializes
    the full [Nq, S, S] scores: 75.2 ms vs 13.6 ms for 8 layers fwd+bwd at
    S=4096. When every layer shares a window (the common case, including the
    default all-full-causal model) `window_sel` is None and the scan carries no
    window input at all.
    """

    def __init__(self, n: int, make_block, remat: bool, policy: str = "full",
                 window_sel=None):
        blocks = [make_block() for _ in range(n)]
        graphdefs, states = zip(*[nnx.split(b) for b in blocks])
        # Same structure for every block, so stacking leaf-by-leaf gives one
        # Block whose parameters carry a leading layer axis.
        stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
        self.blocks = nnx.merge(graphdefs[0], stacked)
        self.n = n
        self.remat = remat
        self.policy = policy
        # Plain tuple of ints (or None) -- static, so it is not a state leaf.
        self.window_sel = tuple(window_sel) if window_sel is not None else None

    def __call__(self, x, cos, sin, positions=None):
        graphdef, state = nnx.split(self.blocks)

        def run(carry, layer_state, layer_sel):
            block = nnx.merge(graphdef, layer_state)
            return block(carry, cos, sin, positions=positions,
                         window_sel=layer_sel)

        if self.window_sel is None:
            # Homogeneous stack: no extra scan input, the block's static
            # init-time window is authoritative.
            body = lambda carry, layer_state: run(carry, layer_state, None)
            xs = state
        else:
            body = lambda carry, packed: run(carry, packed[0], packed[1])
            xs = (state, jnp.asarray(self.window_sel, dtype=jnp.int32))

        pol = remat_policy(self.policy) if self.remat else None
        f = body if (not self.remat or self.policy == "none") else jax.checkpoint(body, policy=pol)
        x, aux = jax.lax.scan(f, x, xs)
        return x, aux


class Transformer(nnx.Module):
    """Full MoE transformer with embedding tying."""

    def __init__(self, config, dtype, rngs: nnx.Rngs, mesh=None):
        self.config = config
        self.dtype = dtype
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

        # Token embedding (tied with output)
        std = (1.0 / config.d_model) ** 0.5
        self.wte = nnx.Param(jax.random.normal(rngs.params(), (config.vocab_size, config.d_model)) * std)

        # One RoPE table shared by every layer: it is a constant, identical
        # across blocks, so storing it per-block would replicate it `layers`
        # times in HBM for no reason. KDA layers ignore it; only full/swa
        # attention layers use it.
        max_seq_len = max(config.seq_len, config.infer_seq_len)
        cos, sin = precompute_rope(config.head_dim, max_seq_len,
                                    config.rope_base, config.yarn_factor,
                                    config.yarn_beta_fast, config.yarn_beta_slow)
        self.rope_cos = RopeCache(cos.astype(dtype))
        self.rope_sin = RopeCache(sin.astype(dtype))

        # ---- Per-layer attention type ----
        # Three modes, in priority order:
        #   1. use_kda  : Kimi-Linear hybrid. Layer i is "full" when
        #                 (i % kda_period == kda_period-1), else "kda".
        #   2. use_swa  : SWA hybrid. Layer i is "full" when
        #                 (i % swa_period == swa_period-1), else "swa".
        #   3. else     : all "full".
        # KDA and full/swa layers have DIFFERENT module structures (KDA has
        # ShortConv + recurrent state params, no RoPE), so they cannot live in
        # the same BlockStack. When use_kda=True we set use_scan=False and run
        # layers in a plain Python loop (no lax.scan) -- the scan was a
        # compile-time optimization, irrelevant at smoke scale and incompatible
        # with mixed module types.
        use_kda = getattr(config, "use_kda", False)
        use_swa = getattr(config, "use_swa", False)
        kda_period = getattr(config, "kda_period", 4)
        swa_window = getattr(config, "swa_window", 4096)
        swa_period = getattr(config, "swa_period", 2)
        use_scan = getattr(config, "use_scan", True)
        # If KDA is on, scan must be off (mixed module types in one stack).
        if use_kda:
            use_scan = False

        self.use_kda = use_kda
        self.use_swa = use_swa
        self.use_scan = use_scan
        self.swa_window = swa_window
        self.swa_period = swa_period
        self.kda_period = kda_period

        def layer_attention_type(i):
            if use_kda:
                return "full" if (i % kda_period == kda_period - 1) else "kda"
            if use_swa:
                return "full" if (i % swa_period == swa_period - 1) else "swa"
            return "full"

        def layer_window(i):
            """Sliding-window size for layer i (0 = full causal, >0 = SWA)."""
            at = layer_attention_type(i)
            if at == "swa":
                return swa_window
            return 0  # full causal (KDA ignores the window entirely)

        # Print the per-layer plan (helps verify the hybrid layout).
        self._layer_types = nnx.data([layer_attention_type(i) for i in range(config.layers)])

        # ---- Block factory ----
        def make_block(use_moe, attention_type, window=None, window_choices=None):
            return lambda: Block(
                d_model=config.d_model,
                n_q_heads=config.n_q_heads,
                n_kv_heads=config.n_kv_heads,
                head_dim=config.head_dim,
                d_ff=config.d_ff,
                d_ff_dense=(config.d_ff if getattr(config, "d_ff_dense", None) is None
                            else config.d_ff_dense),
                n_experts=config.n_experts,
                n_active=config.n_active,
                n_shared_experts=config.n_shared_experts,
                router_init_std=config.router_init_std,
                routed_scaling_factor=config.routed_scaling_factor,
                norm_eps=config.norm_eps,
                residual_scale_init=config.residual_scale_init,
                use_moe=use_moe,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
                window=window,
                window_choices=window_choices,
                attention_type=attention_type,
                kda_heads=getattr(config, "kda_heads", 64),
                kda_head_dim=getattr(config, "kda_head_dim", 128),
                kda_chunk_size=getattr(config, "kda_chunk_size", 64),
            )

        remat = getattr(config, "remat", True)
        policy = getattr(config, "remat_policy", "full")
        self.remat = remat
        self.remat_policy_name = policy
        self.n_dense = config.dense_layers
        self.n_moe = config.layers - config.dense_layers

        if not use_scan:
            # ---- No-scan path: keep every block as its own module and run them
            # in a plain Python loop. Required for the KDA hybrid (mixed module
            # types) and also a fine fallback for tiny smoke configs. We still
            # build the blocks in dense-then-moe order to match the scan path.
            self.blocks = []
            for i in range(config.layers):
                use_moe = i >= self.n_dense
                at = layer_attention_type(i)
                w = layer_window(i)
                # Static window hint (only used by the TPU Splash path; on CPU
                # the call-time window is authoritative). None for full/kda.
                static_w = None if w == 0 else w
                self.blocks.append(
                    make_block(use_moe, at, static_w)())
            # nnx.data marks this list as a static (non-pytree-leaf) container
            # of modules -- the blocks themselves are nnx modules tracked via
            # their own params, so the list wrapper must be static.
            self.blocks = nnx.data(self.blocks)
            # Per-layer static windows for the call path (0 = full causal).
            self._windows = tuple(layer_window(i) for i in range(config.layers))
            self.dense_stack = None
            self.moe_stack = None
        else:
            # ---- Scan path: two homogeneous stacks (dense, moe). Each stack
            # is lax.scan'd so XLA compiles one block body. The SWA hybrid
            # varies the window per layer through a static `window_choices`
            # tuple plus a scanned index (see BlockStack); KDA is incompatible
            # with this path (see above) so use_kda forces the no-scan path.
            def _stack_plan(ws):
                """(window_choices, window_sel) for one stack's per-layer windows.

                `ws` is a list of ints, 0 meaning full causal. The distinct
                values become the static choice tuple; `window_sel` indexes into
                it per layer, and is None when the stack is homogeneous so the
                scan carries no window input at all.
                """
                if not ws:
                    return None, None
                uniq = sorted(set(ws))
                choices = tuple(None if w == 0 else int(w) for w in uniq)
                if len(choices) == 1:
                    return choices, None
                pos = {w: i for i, w in enumerate(uniq)}
                return choices, tuple(pos[w] for w in ws)

            dense_windows = [layer_window(i) for i in range(self.n_dense)]
            moe_windows = [layer_window(self.n_dense + i) for i in range(self.n_moe)]
            dense_choices, dense_sel = _stack_plan(dense_windows)
            moe_choices, moe_sel = _stack_plan(moe_windows)
            # When a stack is homogeneous the single choice is the block's
            # static init-time window; when it is mixed the block gets the whole
            # choice tuple and the scanned index selects among them.
            dense_static = dense_choices[0] if (dense_choices and dense_sel is None) else None
            moe_static = moe_choices[0] if (moe_choices and moe_sel is None) else None

            self.dense_stack = (BlockStack(
                self.n_dense,
                make_block(False, "full", dense_static, dense_choices),
                remat, policy, window_sel=dense_sel)
                if self.n_dense else None)
            self.moe_stack = (BlockStack(
                self.n_moe,
                make_block(True, "full", moe_static, moe_choices),
                remat, policy, window_sel=moe_sel)
                if self.n_moe else None)
            self.blocks = None
            self._windows = None

        self.norm_f = RMSNorm(config.d_model, config.norm_eps, dtype)

    def __call__(self, tokens, positions=None):
        """tokens: [B, S] int. Returns logits [B, S, vocab]."""
        x = self.wte[tokens].astype(jnp.float32)  # [B, S, D]
        cos, sin = self.rope_cos.value, self.rope_sin.value

        if self.use_scan:
            # ---- Scan path (dense + moe stacks) ----
            if self.dense_stack is not None:
                x, _ = self.dense_stack(x, cos, sin, positions)
            aux_stacked = {}
            if self.moe_stack is not None:
                x, aux_stacked = self.moe_stack(x, cos, sin, positions)
        else:
            # ---- No-scan path: plain Python loop over blocks. Required for
            # the KDA hybrid (mixed module types). MoE aux is summed across
            # MoE layers; scalars are averaged, expert_counts is stacked.
            #
            # Remat is applied per block here just as BlockStack applies it per
            # scan step. Without it this path holds every layer's activations
            # live at once -- the scan path's `remat=True` would silently stop
            # meaning anything the moment use_scan went False.
            pol = (remat_policy(self.remat_policy_name)
                   if self.remat and self.remat_policy_name != "none" else False)
            aux_list = []
            for i, block in enumerate(self.blocks):
                w = self._windows[i]
                # 0 -> None (full causal); KDA ignores the window anyway.
                w_arg = None if w == 0 else w
                gd, st = nnx.split(block)

                def call(carry, layer_state, _gd=gd, _w=w_arg):
                    return nnx.merge(_gd, layer_state)(
                        carry, cos, sin, positions=positions, window=_w)

                f = call if pol is False else jax.checkpoint(call, policy=pol)
                x, aux = f(x, st)
                if aux:
                    aux_list.append(aux)
            # Merge aux the same way the scan path does (see below).
            aux_stacked = {}
            if aux_list:
                keys = aux_list[0].keys()
                for k in keys:
                    stacked = jnp.stack([a[k] for a in aux_list], axis=0)
                    aux_stacked[k] = stacked

        x = self.norm_f(x)
        # tied output projection: logits = x @ wte.T
        logits = x @ self.wte.T.astype(jnp.float32)  # [B, S, vocab]
        # scan stacks each MoE layer's aux on a leading axis. Scalars (the
        # losses, the entropy) average over layers; expert_counts stays
        # [n_moe, n_experts] because each layer's router is balanced separately.
        n_moe = max(1, self.n_moe)
        aux_out = {k: (v if k == "expert_counts" else jnp.sum(v, axis=0) / n_moe)
                   for k, v in aux_stacked.items()}
        return logits, aux_out

    def generate(self, tokens, positions=None):
        """Forward for generation: returns logits only (no aux)."""
        logits, _ = self(tokens, positions=positions)
        return logits