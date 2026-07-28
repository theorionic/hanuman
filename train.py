"""Training loop: init model + optax state on mesh, jit step, checkpoint, log."""
from __future__ import annotations

import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp

from config import Config
from model import Transformer
from model.sharding import get_mesh, state_shardings, data_sharding, describe
from optimizer import build_optimizer
from data import make_batches, random_batches, get_tokenizer


def count_params(state) -> int:
    """Count total params in an nnx State (or pytree)."""
    leaves = jax.tree_util.tree_leaves(state)
    return int(sum(x.size for x in leaves))


def count_active_params(config: Config) -> int:
    """Estimate active params per forward pass (top-k experts + shared + dense)."""
    D = config.d_model
    dff = config.d_ff
    # embedding (tied, counted once)
    emb = config.vocab_size * D
    # per attention layer: wq, wk, wv, wo
    qkv = D * (config.n_q_heads * config.head_dim) + 2 * D * (config.n_kv_heads * config.head_dim)
    attn = qkv + (config.n_q_heads * config.head_dim) * D
    # norms: 2 per block + final
    norms = 2 * D + D
    # dense FFN (SwiGLU: 3 matrices)
    dffd = getattr(config, "d_ff_dense", None) or dff
    dense_ffn = 3 * D * dffd
    # MoE active: n_active routed experts (3 matrices each) + shared (3 matrices)
    moe_active = config.n_active * 3 * D * dff + config.n_shared_experts * 3 * D * dffd
    # router (active in compute)
    router = D * config.n_experts
    # residual scales: one per block
    res = config.layers

    n_dense = config.dense_layers
    n_moe = config.layers - n_dense
    total = emb + n_dense * (attn + dense_ffn + 2 * D) + n_moe * (attn + moe_active + 2 * D) + norms + res + router
    return total


def count_total_params(config: Config) -> int:
    """Total params (all experts stored)."""
    D = config.d_model
    dff = config.d_ff
    emb = config.vocab_size * D
    qkv = D * (config.n_q_heads * config.head_dim) + 2 * D * (config.n_kv_heads * config.head_dim)
    attn = qkv + (config.n_q_heads * config.head_dim) * D
    dffd = getattr(config, "d_ff_dense", None) or dff
    dense_ffn = 3 * D * dffd
    moe_total = config.n_experts * 3 * D * dff + config.n_shared_experts * 3 * D * dffd
    router = D * config.n_experts
    bias = config.n_experts
    norms = 2 * D + D
    res = config.layers
    n_dense = config.dense_layers
    n_moe = config.layers - n_dense
    total = (emb + n_dense * (attn + dense_ffn + 2 * D)
             + n_moe * (attn + moe_total + 2 * D) + norms + res + router + bias)
    return total


def step_flops(config: Config) -> tuple[int, int]:
    """(model_flops, hardware_flops) for one training step.

    model_flops counts the arithmetic the model mathematically requires:
    forward + backward = 3x forward. hardware_flops adds the extra forward that
    rematerialization pays inside each block. MFU against the first is the
    standard number quoted for a model; against the second it is what the chip
    actually executed.
    """
    D, dff, S = config.d_model, config.d_ff, config.seq_len
    Nq, Nkv, H = config.n_q_heads, config.n_kv_heads, config.head_dim
    L, n_dense = config.layers, config.dense_layers
    n_moe = L - n_dense

    # attention projections (wq, wk, wv, wo)
    proj = 2 * D * (Nq * H + 2 * Nkv * H + Nq * H)
    # causal attention core: QK^T and PV, each averaging S/2 keys per query
    core = 2 * Nq * H * S
    attn = proj + core

    dffd = getattr(config, "d_ff_dense", None) or dff
    dense_ffn = 2 * 3 * D * dffd
    # Routed experts use d_ff; the always-on shared expert uses d_ff_dense.
    moe_ffn = 2 * 3 * D * (dff * config.n_active + dffd * config.n_shared_experts)
    router = 2 * D * config.n_experts
    head = 2 * D * config.vocab_size

    fwd_per_token = n_dense * (attn + dense_ffn) + n_moe * (attn + moe_ffn + router) + head
    tokens = config.batch_size * config.seq_len

    model = 3 * fwd_per_token * tokens
    # remat recomputes every block's forward; the head and embedding are outside.
    block_fwd = n_dense * (attn + dense_ffn) + n_moe * (attn + moe_ffn + router)
    hardware = model + (block_fwd * tokens if getattr(config, "remat", True) else 0)
    return model, hardware


def peak_flops_per_device() -> float:
    """Advertised bf16 peak for the attached accelerator, FLOP/s per chip."""
    kind = jax.devices()[0].device_kind.lower()
    table = {"tpu v5 lite": 197e12, "tpu v5e": 197e12, "tpu v5p": 459e12,
             "tpu v4": 275e12, "tpu v6 lite": 918e12, "tpu v6e": 918e12}
    for k, v in table.items():
        if k in kind:
            return v
    return 0.0


def loss_fn(graphdef, params, rest, tokens, config: Config):
    """Cross-entropy + z-loss + balance loss.

    tokens: [B, S]. Shift by 1 for next-token prediction.
    """
    model = nnx.merge(graphdef, params, rest)
    # Run the model on the *full* [B, S] sequence and drop the last position
    # afterwards, rather than feeding it tokens[:, :-1]. Trimming the input
    # first makes S odd (4095), which is not tile-aligned: it silently disables
    # the splash attention kernel (which needs S % 128 == 0) and leaves every
    # matmul and the MoE row count on a ragged boundary.
    logits, aux = model(tokens)              # [B, S, V]
    logits = logits[:, :-1]                  # [B, S-1, V]
    targets = tokens[:, 1:]                  # [B, S-1]
    # cross entropy (integer-label form: never materializes a [B, S, V] one-hot)
    nll = optax.softmax_cross_entropy_with_integer_labels(logits, targets)  # [B, S-1]
    ce_loss = jnp.mean(nll)

    z_loss = config.z_loss_weight * aux.get("router_z_loss", 0.0)
    bal_loss = config.balance_loss_weight * aux.get("balance_loss", 0.0)
    total = ce_loss + z_loss + bal_loss
    return total, {"ce": ce_loss, "z_loss": z_loss, "bal_loss": bal_loss,
                   "mean_entropy": aux.get("mean_entropy", 0.0),
                   "expert_counts": aux.get("expert_counts", jnp.zeros(config.n_experts))}


def make_train_step(graphdef, config: Config, opt: optax.GradientTransformation,
                    shardings=None):
    """Create a jitted train step.

    Returns: step_fn(params, rest, opt_state, tokens) ->
        (new_params, new_opt_state, loss, metrics)
    The router bias is updated OUTSIDE the optimizer (manual, aux-loss-free style).
    `shardings` is (param_sharding, opt_sharding) so the step is guaranteed to
    return state laid out exactly like it consumed it (no silent resharding
    between steps, which would re-copy the whole model every iteration).

    Backward and update live in one jit. They used to be split, because XLA:TPU
    could not compile a computation that both produced the backward's
    reduce-scatter and then combined it with the same parameter ("Pattern match
    for backwards collectives + grad_y - NYI"). That was a stale-libtpu bug;
    with a current runtime the fused form compiles, and fusing matters: split
    across two jits every gradient has to be fully materialized at the boundary
    (3.15 GB/device for this model), whereas fused XLA can consume and free them
    layer by layer.
    """
    param_sh, opt_sh = shardings if shardings is not None else (None, None)

    def _apply(params, opt_state, grads, counts):
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # ---- Bias update (outside gradient) ----
        # DeepSeek-V3 aux-loss-free balancing: nudge each expert's routing bias
        # by a fixed step in the direction that evens out load. The update is
        # sign-based on purpose -- raw (count - mean) is in units of tokens
        # (thousands), which would instantly swamp the sigmoid scores in (0,1).
        delta = config.bias_update_rate * jnp.sign(counts - jnp.mean(counts))
        return _update_biases(new_params, delta, config), new_opt_state

    # Donate params + opt_state: they are ~2/3 of HBM and are dead after the
    # step, so letting XLA write the update in place avoids a second copy.
    @partial(jax.jit, donate_argnums=(0, 2), out_shardings=(param_sh, opt_sh, None, None))
    def fused_step(params, rest, opt_state, tokens):
        (loss, metrics), grads = jax.value_and_grad(
            lambda p: loss_fn(graphdef, p, rest, tokens, config), has_aux=True)(params)
        counts = jax.lax.stop_gradient(metrics["expert_counts"])
        params, opt_state = _apply(params, opt_state, grads, counts)
        return params, opt_state, loss, metrics

    @jax.jit
    def grad_step(params, rest, tokens):
        return jax.value_and_grad(
            lambda p: loss_fn(graphdef, p, rest, tokens, config), has_aux=True
        )(params)

    @partial(jax.jit, donate_argnums=(0, 1, 2), out_shardings=(param_sh, opt_sh))
    def update_step(params, opt_state, grads, counts):
        return _apply(params, opt_state, grads, counts)

    def split_step(params, rest, opt_state, tokens):
        (loss, metrics), grads = grad_step(params, rest, tokens)
        counts = jax.lax.stop_gradient(metrics["expert_counts"])
        params, opt_state = update_step(params, opt_state, grads, counts)
        return params, opt_state, loss, metrics

    return fused_step if getattr(config, "fused_step", True) else split_step


def _update_biases(state, delta, config: Config):
    """Subtract `delta` from every 'bias' Param in the nnx State.

    `delta` is [n_experts]; broadcast to each MoE layer's bias.
    Uses tree_map_with_path to preserve the exact State/Param pytree structure
    (manually rebuilding State containers breaks jax pytree structure equality).
    """
    def keystr(path):
        return "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)

    def fn(path, x):
        ks = keystr(path)
        # bias Param leaves have path ending in 'bias/.value'
        if ks.endswith("/bias/.value") or ks == "bias/.value":
            return x - delta.astype(x.dtype)
        return x

    return jax.tree_util.tree_map_with_path(fn, state)


def save_checkpoint(state, opt_state, config: Config, step: int):
    """Save params + optimizer state via orbax."""
    path = os.path.abspath(os.path.join(config.checkpoint_dir, config.name, f"step_{step}"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckptr = ocp.PyTreeCheckpointer()
    # Convert nnx State to a plain dict of arrays for orbax
    flat = jax.tree_util.tree_map(lambda x: x if isinstance(x, jnp.ndarray) else np.asarray(x), state)
    ckptr.save(path, {"state": flat, "step": step}, force=True)
    print(f"[train] Saved checkpoint to {path}")


def hbm_report() -> str:
    """Peak HBM in use across devices."""
    peaks, limits = [], []
    for d in jax.local_devices():
        s = d.memory_stats() or {}
        peaks.append(s.get("peak_bytes_in_use", 0) / 1e9)
        limits.append(s.get("bytes_limit", 0) / 1e9)
    if not peaks:
        return "n/a"
    return f"peak {max(peaks):.2f} GB / {max(limits):.2f} GB per device"


def memory_report(config: Config) -> str:
    """Per-tensor sharding and HBM table, without allocating the model.

    Built from `nnx.eval_shape`, so it costs nothing and can be run on a laptop
    to sanity-check a config before booking accelerators.
    """
    from model.sharding import param_spec, _keystr

    mesh = get_mesh(config.mesh_data_axis * config.mesh_expert_axis,
                    config.mesh_data_axis, config.mesh_expert_axis)
    abs_model = nnx.eval_shape(
        lambda: Transformer(config, config.compute_dtype(), nnx.Rngs(0), mesh=mesh))
    _, params, buffers = nnx.split(abs_model, nnx.Param, ...)

    rows, total, per_dev = [], 0, 0.0
    for state in (params, buffers):
        for path, x in jax.tree_util.tree_leaves_with_path(state):
            name, shape = _keystr(path), jnp.shape(x)
            spec = param_spec(name, shape, mesh)
            n = int(np.prod(shape)) if shape else 1
            div = int(np.prod([mesh.shape[a] for a in spec if a is not None])) or 1
            rows.append((n * 4 / 1e6, n * 4 / 1e6 / div, name, shape, spec))
            total += n * 4 / 1e6
            per_dev += n * 4 / 1e6 / div

    out = [f"{'tensor':46s} {'shape':24s} {'spec':24s} {'fp32 MB':>9s} {'per-dev':>9s}",
           "-" * 116]
    for mb, pmb, name, shape, spec in sorted(rows, reverse=True):
        mark = "" if any(a is not None for a in spec) else "  replicated"
        out.append(f"{name[:46]:46s} {str(shape)[:24]:24s} {str(spec)[:24]:24s} "
                   f"{mb:9.1f} {pmb:9.1f}{mark}")
    out.append("-" * 116)
    out.append(f"{'TOTAL':46s} {'':24s} {'':24s} {total:9.1f} {per_dev:9.1f}")

    mu = 2 if getattr(config, "opt_state_dtype", "bf16") == "bf16" else 4
    opt = per_dev * mu / 4
    out += ["",
            f"per device: master weights {per_dev/1e3:.2f} GB"
            f" + Lion momentum {opt/1e3:.2f} GB ({config.opt_state_dtype})"
            f" = {(per_dev+opt)/1e3:.2f} GB resident"]
    if not getattr(config, "fused_step", True):
        out.append(f"  + {per_dev/1e3:.2f} GB of fp32 gradients held at the "
                   f"grad/update jit boundary (fused_step=False)")
    model_fl, hw_fl = step_flops(config)
    out.append(f"step FLOPs: {model_fl/1e12:.1f} T model, {hw_fl/1e12:.1f} T with remat "
               f"(policy={getattr(config, 'remat_policy', 'full')})")
    return "\n".join(out)


def init_sharded(config: Config, dtype, mesh):
    """Build the model directly in sharded form.

    A 7B model in fp32 is ~25 GB, which does not fit in one v5e chip's 16 GB of
    HBM -- so the parameters can never exist unsharded, not even briefly. We
    take the graphdef from an abstract (shape-only) model, derive the layout
    from that, and run the real initializer under jit with those out_shardings
    so every device only ever allocates its own slice.
    """
    def build():
        return Transformer(config, dtype, nnx.Rngs(0), mesh=mesh)

    abs_model = nnx.eval_shape(build)
    graphdef, abs_params, abs_rest = nnx.split(abs_model, nnx.Param, ...)
    param_sh = state_shardings(abs_params, mesh)
    rest_sh = state_shardings(abs_rest, mesh)

    def _init():
        _, params, rest = nnx.split(build(), nnx.Param, ...)
        return params, rest

    params, rest = jax.jit(_init, out_shardings=(param_sh, rest_sh))()
    return graphdef, params, rest, param_sh, rest_sh


def train(config: Config, use_random_data: bool = True, save: bool = True):
    """Main training entry point."""
    print(f"[train] config={config.name}")
    devices = jax.devices()
    print(f"[train] {len(devices)} devices: {devices[0].device_kind} x{len(devices)}")

    mesh = get_mesh(config.mesh_data_axis * config.mesh_expert_axis,
                    config.mesh_data_axis, config.mesh_expert_axis)
    print(f"[train] mesh data={mesh.shape['data']} expert={mesh.shape['expert']}")

    dtype = config.compute_dtype()
    print("[train] Initializing model (sharded)...")
    t_init = time.time()
    graphdef, params, rest, param_sh, rest_sh = init_sharded(config, dtype, mesh)
    jax.block_until_ready(params)
    print(f"[train] init took {time.time()-t_init:.1f}s")

    n_params = count_params(params)
    n_buffers = count_params(rest)
    print(f"[train] Trainable params: {n_params:,} ({n_params/1e9:.3f}B)")
    print(f"[train] Non-trainable buffers (RoPE tables): {n_buffers:,}")
    print(f"[train] sharding: {describe(params, param_sh)}")
    if config.name == "full":
        print(f"[train] Active params (est): {count_active_params(config):,} "
              f"({count_active_params(config)/1e9:.3f}B)")

    # Optimizer. Its state is laid out with the same rule as the params (Lion's
    # momentum mirrors the param tree); deriving it explicitly rather than
    # reading back `.sharding` keeps scalars like the step counter on the mesh
    # instead of committing them to a single device.
    opt, schedule = build_optimizer(config, params)
    opt_sh = state_shardings(jax.eval_shape(opt.init, params), mesh)
    opt_state = jax.jit(opt.init, out_shardings=opt_sh)(params)
    jax.block_until_ready(opt_state)
    print(f"[train] optimizer state ready ({count_params(opt_state)/1e9:.3f}B leaves)")
    print(f"[train] HBM after init: {hbm_report()}")

    # Data
    batch_sh = data_sharding(mesh)
    if use_random_data or config.tokenizer == "byte":
        data_iter = random_batches(config, config.batch_size, config.seq_len, seed=42)
        print("[train] Using random token data (smoke / no-data mode)")
    else:
        tokenizer = get_tokenizer(config)
        data_iter = make_batches(config, config.batch_size, config.seq_len, tokenizer)
        print(f"[train] Streaming {config.dataset}")

    step_fn = make_train_step(graphdef, config, opt, shardings=(param_sh, opt_sh))

    print(f"[train] Starting training for {config.train_steps} steps...")
    model_fl, hw_fl = step_flops(config)
    peak = peak_flops_per_device() * len(jax.local_devices())
    print(f"[train] {model_fl/1e12:.2f} TFLOP/step model, {hw_fl/1e12:.2f} TFLOP/step with remat"
          f"{f'; chip peak {peak/1e12:.0f} TFLOP/s' if peak else ''}")

    losses = []
    t_win = None       # wall clock at the start of the current logging window
    steps_win = 0      # steps completed inside it
    for step in range(config.train_steps):
        tokens = jax.device_put(next(data_iter), batch_sh)
        params, opt_state, loss, metrics = step_fn(params, rest, opt_state, tokens)
        if step == 0:
            # First step pays for compilation; start the throughput clock after it.
            jax.block_until_ready(loss)
            print(f"[train] compile + first step: {time.time()-t_init:.1f}s")
            losses.append(float(loss))
            t_win, steps_win = time.time(), 0
            continue
        steps_win += 1

        if step % config.log_every == 0 or step == config.train_steps - 1:
            # Timing only means anything once the queued steps have retired, so
            # sync here -- and nowhere else, to leave the dispatch pipelined.
            jax.block_until_ready((loss, params))
            dt = max(time.time() - t_win, 1e-9)
            sec = dt / steps_win
            tps = config.batch_size * config.seq_len * steps_win / dt
            mfu = f" | MFU {100*model_fl/sec/peak:.1f}% (hw {100*hw_fl/sec/peak:.1f}%)" if peak else ""
            losses.append(float(loss))
            print(f"  step {step:5d} | loss {float(loss):.4f} | ce {float(metrics['ce']):.4f} "
                  f"| z {float(metrics['z_loss']):.6f} | bal {float(metrics['bal_loss']):.6f} "
                  f"| ent {float(metrics['mean_entropy']):.3f} "
                  f"| lr {float(schedule(step)):.2e} | {sec*1e3:.0f} ms/step "
                  f"| tok/s {tps:,.0f}{mfu}")
            t_win, steps_win = time.time(), 0

    print(f"[train] HBM peak: {hbm_report()}")
    if save:
        save_checkpoint(params, opt_state, config, config.train_steps)

    print(f"[train] Done. First loss={losses[0]:.4f}, last loss={losses[-1]:.4f}")
    return params, graphdef, losses
