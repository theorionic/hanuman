"""Optimizer: Lion + WSD schedule + grad clip + no-decay mask."""
from __future__ import annotations

import optax
import jax
import jax.numpy as jnp
import flax.nnx as nnx


def wsd_schedule(learning_rate: float, min_lr: float, warmup_steps: int,
                 total_steps: int, decay_fraction: float) -> optax.Schedule:
    """Warmup-Stable-Decay schedule.

    - Linear warmup from 0 -> learning_rate over warmup_steps
    - Stable at learning_rate
    - Cosine decay from learning_rate -> min_lr over last decay_fraction of total_steps
    """
    decay_steps = int(total_steps * decay_fraction)
    stable_steps = max(1, total_steps - warmup_steps - decay_steps)
    decay_start = warmup_steps + stable_steps

    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=max(1, warmup_steps),
        transition_begin=0,
    )
    stable = optax.constant_schedule(learning_rate)
    decay = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=max(1, decay_steps),
        alpha=min_lr / learning_rate if learning_rate > 0 else 0.0,
    )

    schedule = optax.join_schedules(
        schedules=[warmup, stable, decay],
        boundaries=[warmup_steps, decay_start],
    )
    return schedule


def is_no_decay(name: str) -> bool:
    """Return True for params that should NOT receive weight decay.

    No decay for: norms, biases, embeddings, router bias, router weights.
    """
    n = name.lower()
    if "norm" in n or "weight" in n and "norm" in n:
        return True
    if "residual_scale" in n:
        return True
    if "bias" in n:
        return True
    if "wte" in n or "embed" in n:
        return True
    if "router" in n:
        return True
    # RoPE tables (cos/sin) - not params anyway
    if "cos" in n or "sin" in n:
        return True
    return False


def build_no_decay_mask(state):
    """Build a pytree mask (same structure as state) where True = apply WD.

    We apply WD to all params EXCEPT those matched by is_no_decay
    (norms, biases, embeddings, router bias, router weights, residual scales).
    Non-Param arrays (RoPE cos/sin buffers) are masked out (False).

    Uses tree_map_with_path WITHOUT is_leaf so that Param custom nodes are
    traversed and preserved, keeping the mask structure identical to `state`.
    """
    def keystr(path):
        return "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)

    def mask_fn(path, x):
        ks = keystr(path)
        # Param leaves have a '.value' suffix in the path; plain arrays don't.
        is_param_leaf = ks.endswith(".value")
        if not is_param_leaf:
            return False  # buffer (cos/sin), no WD
        name = ks[: -len(".value")]
        return not is_no_decay(name)

    mask = jax.tree_util.tree_map_with_path(mask_fn, state)
    return mask


def _custom_weight_decay(wd: float, mask_tree, schedule: optax.Schedule):
    """Decoupled (AdamW-style) weight decay for masked (True) leaves.

    Works with nnx State/Param custom pytree nodes, unlike optax's built-in
    `mask` which breaks on Param custom nodes.

    The decay must be scaled by the current learning rate. This transform runs
    *after* `optax.lion`, which has already multiplied its update by lr, so an
    unscaled `u - wd * p` would be a full-magnitude step: at the configured
    wd=1.0 the update becomes `lion_update - p`, and `p + update` collapses
    every decayed parameter to ~0 on the very first step. Tracking the step
    count here lets us apply `lr(t) * wd * p` to match optax's convention.
    """
    def init_fn(params):
        return {"count": jnp.zeros([], jnp.int32)}

    def update_fn(updates, state, params=None):
        lr = schedule(state["count"])

        def f(u, m, p):
            if m and p is not None:
                return u - lr * wd * p
            return u

        new_updates = jax.tree_util.tree_map(f, updates, mask_tree, params)
        return new_updates, {"count": optax.safe_increment(state["count"])}

    return optax.GradientTransformation(init_fn, update_fn)


def build_optimizer(config, state):
    """Build Lion optimizer with WSD schedule, grad clip, and WD mask.

    Weight decay is applied via a custom transform (not optax's `mask` kwarg)
    because optax's masking is incompatible with nnx Param custom pytree nodes.
    """
    schedule = wsd_schedule(
        learning_rate=config.learning_rate,
        min_lr=config.min_lr,
        warmup_steps=config.warmup_steps,
        total_steps=config.train_steps,
        decay_fraction=config.decay_fraction,
    )
    mask = build_no_decay_mask(state)
    opt = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.lion(
            learning_rate=schedule,
            b1=0.9,
            b2=0.99,
        ),
        _custom_weight_decay(config.weight_decay, mask, schedule),
    )
    return opt, schedule