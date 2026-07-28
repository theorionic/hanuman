"""Training loop: init model + optax state on mesh, jit step, checkpoint, log."""
from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp

from config import Config
from model import Transformer
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
    dense_ffn = 3 * D * dff
    # MoE active: n_active routed experts (3 matrices each) + shared (3 matrices)
    moe_active = config.n_active * 3 * D * dff + config.n_shared_experts * 3 * D * dff
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
    dense_ffn = 3 * D * dff
    moe_total = config.n_experts * 3 * D * dff + config.n_shared_experts * 3 * D * dff
    router = D * config.n_experts
    bias = config.n_experts
    norms = 2 * D + D
    res = config.layers
    n_dense = config.dense_layers
    n_moe = config.layers - n_dense
    total = (emb + n_dense * (attn + dense_ffn + 2 * D)
             + n_moe * (attn + moe_total + 2 * D) + norms + res + router + bias)
    return total


def loss_fn(graphdef, state, tokens, config: Config):
    """Cross-entropy + z-loss + balance loss.

    tokens: [B, S]. Shift by 1 for next-token prediction.
    """
    model = nnx.merge(graphdef, state)
    logits, aux = model(tokens[:, :-1])  # [B, S-1, V]
    targets = tokens[:, 1:]  # [B, S-1]
    # cross entropy
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # gather target log-probs
    one_hot = jax.nn.one_hot(targets, config.vocab_size, dtype=log_probs.dtype)
    nll = -jnp.sum(one_hot * log_probs, axis=-1)  # [B, S-1]
    ce_loss = jnp.mean(nll)

    z_loss = config.z_loss_weight * aux.get("router_z_loss", 0.0)
    bal_loss = config.balance_loss_weight * aux.get("balance_loss", 0.0)
    total = ce_loss + z_loss + bal_loss
    return total, {"ce": ce_loss, "z_loss": z_loss, "bal_loss": bal_loss,
                   "mean_entropy": aux.get("mean_entropy", 0.0),
                   "expert_counts": aux.get("expert_counts", jnp.zeros(config.n_experts))}


def make_train_step(graphdef, config: Config, opt: optax.GradientTransformation):
    """Create a jitted train step.

    Returns: step_fn(state, opt_state, tokens, bias) ->
        (new_state, new_opt_state, new_bias, loss, metrics)
    The bias is updated OUTSIDE the optimizer (manual, aux-loss-free style).
    """
    @jax.jit
    def step(state, opt_state, tokens):
        (loss, metrics), grads = jax.value_and_grad(
            lambda s: loss_fn(graphdef, s, tokens, config), has_aux=True
        )(state)
        updates, new_opt_state = opt.update(grads, opt_state, state)
        new_state = optax.apply_updates(state, updates)

        # ---- Bias update (outside gradient) ----
        # expert_counts from metrics; update bias to balance load.
        # bias -= gamma * (count - mean_count)  (stop_gradient on counts)
        counts = jax.lax.stop_gradient(metrics["expert_counts"])  # [N]
        mean_count = jnp.mean(counts)
        delta = config.bias_update_rate * (counts - mean_count)
        # bias is a Param in state; update it in-place in the state dict
        # Find the bias entries: they live under blocks[i].ffn.bias for MoE layers.
        # We update all of them by walking the state.
        new_state = _update_biases(new_state, delta, config)
        return new_state, new_opt_state, loss, metrics

    return step


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


def train(config: Config, use_random_data: bool = True):
    """Main training entry point."""
    print(f"[train] config={config.name}")
    print(f"[train] devices={jax.devices()}")

    dtype = config.compute_dtype()
    rngs = nnx.Rngs(0)
    print("[train] Initializing model...")
    model = Transformer(config, dtype, rngs)
    graphdef, state = nnx.split(model)
    del model

    n_params = count_params(state)
    print(f"[train] Total params: {n_params:,} ({n_params/1e9:.3f}B)")
    if config.name == "full":
        print(f"[train] Active params (est): {count_active_params(config):,} "
              f"({count_active_params(config)/1e9:.3f}B)")

    # Optimizer
    opt, schedule = build_optimizer(config, state)
    opt_state = opt.init(state)

    # Data
    if use_random_data or config.tokenizer == "byte":
        data_iter = random_batches(config, config.batch_size, config.seq_len, seed=42)
        print("[train] Using random token data (smoke / no-data mode)")
    else:
        tokenizer = get_tokenizer(config)
        data_iter = make_batches(config, config.batch_size, config.seq_len, tokenizer)
        print(f"[train] Streaming {config.dataset}")

    step_fn = make_train_step(graphdef, config, opt)

    print(f"[train] Starting training for {config.train_steps} steps...")
    t0 = time.time()
    losses = []
    for step in range(config.train_steps):
        tokens = next(data_iter)
        state, opt_state, loss, metrics = step_fn(state, opt_state, tokens)
        losses.append(float(loss))
        if step % config.log_every == 0 or step == config.train_steps - 1:
            lr = float(schedule(step))
            dt = time.time() - t0
            tps = config.batch_size * config.seq_len * (step + 1) / max(dt, 1e-6)
            print(f"  step {step:5d} | loss {float(loss):.4f} | ce {float(metrics['ce']):.4f} "
                  f"| z {float(metrics['z_loss']):.6f} | bal {float(metrics['bal_loss']):.6f} "
                  f"| lr {lr:.2e} | tps {tps:.0f}")

    # Save checkpoint
    save_checkpoint(state, opt_state, config, config.train_steps)

    print(f"[train] Done. First loss={losses[0]:.4f}, last loss={losses[-1]:.4f}")
    return state, graphdef, losses