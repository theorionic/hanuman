"""Inference: YaRN context extension, simple KV cache, sampling."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from config import Config
from model import Transformer
from data import get_tokenizer


def load_model_from_state(config: Config, state, graphdef):
    """Reconstruct a Transformer from saved state."""
    model = nnx.merge(graphdef, state)
    return model


def load_for_generate(config: Config, prompt: str, max_tokens: int = 100,
                      temperature: float = 1.0, top_k: int = 0, seed: int = 0):
    """Load latest checkpoint and generate text. Returns (text, ids)."""
    import os
    import orbax.checkpoint as ocp
    import jax.tree_util as tree_util

    # Build a fresh model to get graphdef + state structure. Split trainable
    # params from buffers exactly like training does: checkpoints hold params
    # only (the RoPE tables are constants, rebuilt here from the config).
    dtype = config.compute_dtype()
    model = Transformer(config, dtype, nnx.Rngs(0))
    graphdef, state, rest = nnx.split(model, nnx.Param, ...)
    del model

    # Find latest checkpoint
    ckpt_dir = os.path.abspath(os.path.join(config.checkpoint_dir, config.name))
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoint dir at {ckpt_dir}. Train first.")
    steps = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("step_")],
                   key=lambda d: int(d.split("_")[1]))
    if not steps:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    latest = steps[-1]
    path = os.path.join(ckpt_dir, latest)
    print(f"[generate] Loading checkpoint {path}")
    ckptr = ocp.PyTreeCheckpointer()
    restored = ckptr.restore(path)
    saved_state = restored["state"]

    # Merge saved arrays back into the nnx State structure.
    # The saved state (from orbax) is a plain dict with string keys; the fresh
    # state is an nnx State with typed keys. We walk both in lockstep by
    # flattening and comparing leaf counts, then rebuild via tree_map.
    # Simplest robust approach: flatten both, take saved leaves, and
    # tree_unflatten into the fresh state's treedef.
    fresh_leaves, fresh_treedef = jax.tree_util.tree_flatten(state)
    saved_leaves, _ = jax.tree_util.tree_flatten(saved_state)
    assert len(fresh_leaves) == len(saved_leaves), (
        f"Leaf count mismatch: fresh={len(fresh_leaves)} saved={len(saved_leaves)}"
    )
    # Cast saved leaves to match fresh leaf dtypes
    merged_leaves = []
    for fl, sl in zip(fresh_leaves, saved_leaves):
        merged_leaves.append(jnp.asarray(sl).astype(fl.dtype) if hasattr(fl, 'dtype') else sl)
    state = jax.tree_util.tree_unflatten(fresh_treedef, merged_leaves)

    return generate(config, state, graphdef, prompt, max_tokens, temperature, top_k, seed,
                    rest=rest)


def sample_next(logits, temperature: float = 1.0, top_k: int = 0, rng=None):
    """Sample next token id from logits [vocab]."""
    if temperature <= 0:
        return int(jnp.argmax(logits))
    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        top_vals, _ = jax.lax.top_k(logits, k)
        thresh = top_vals[-1]
        logits = jnp.where(logits < thresh, -1e30, logits)
    probs = jax.nn.softmax(logits)
    if rng is None:
        rng = np.random.default_rng()
    idx = int(rng.choice(len(probs), p=np.asarray(probs)))
    return idx


def build_sampler(config: Config, graphdef, tokenizer, mesh=None, seq_len: int | None = None):
    """Build a reusable sampler for in-training generation.

    The forward is jitted ONCE over ``(params, rest, tokens)`` -- params and rest
    carried as arguments (not closed over), so it compiles a single time and is
    reused at every generation checkpoint instead of recompiling the 7B forward
    each call. Sampling itself runs on the host.

    No KV cache: each new token reprocesses the fixed ``seq_len`` context (padded
    right), which keeps the compiled shape constant and the code trivial. That is
    fine for a short periodic sample of a few dozen tokens; it would be too slow
    for long-form serving.
    """
    from model.sharding import data_sharding

    seq_len = seq_len or config.seq_len
    pad_id = 0
    # The model's attention runs under a shard_map that shards the batch axis
    # over the 'data' mesh axis, so the generation batch must be a multiple of it
    # (batch-1 fails: "8 does not divide 1"). Replicate the prompt across the data
    # axis and read row 0 -- each device then processes one row exactly like a
    # training forward.
    gdev = mesh.shape["data"] if (mesh is not None and mesh.devices.size > 1) else 1
    tok_sh = data_sharding(mesh) if gdev > 1 else None

    @jax.jit
    def forward(params, rest, tokens):
        model = nnx.merge(graphdef, params, rest)
        return model.generate(tokens)                      # [gdev, seq_len, V]

    eos = getattr(tokenizer, "eos_id", None)
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None)

    def encode(prompt: str):
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(prompt)
        else:
            ids = tokenizer(prompt)["input_ids"]
        return ids or [0]

    def sample(params, rest, prompt: str, max_tokens: int = 40,
               temperature: float = 0.8, top_k: int = 40, seed: int = 0) -> str:
        rng = np.random.default_rng(seed)
        generated = list(encode(prompt))
        for _ in range(max_tokens):
            ctx = generated[-seq_len:]
            n_real = len(ctx)
            row = np.full((seq_len,), pad_id, dtype=np.int32)
            row[:n_real] = ctx
            x = np.broadcast_to(row, (gdev, seq_len)).copy()   # replicate over data axis
            xj = jax.device_put(x, tok_sh) if tok_sh is not None else jnp.asarray(x)
            logits = forward(params, rest, xj)
            next_id = sample_next(np.asarray(logits[0, n_real - 1]),
                                  temperature=temperature, top_k=top_k, rng=rng)
            generated.append(next_id)
            if eos is not None and next_id == eos:
                break
        return tokenizer.decode(generated)

    return sample


def generate(config: Config, state, graphdef, prompt: str, max_tokens: int = 100,
             temperature: float = 1.0, top_k: int = 0, seed: int = 0, rest=None):
    """Generate text from a prompt using the model.

    For YaRN: set config.yarn_factor=8.0 before building the model for 32K context.
    Uses a simple incremental forward (no KV cache for simplicity; reprocesses
    the full context each step). The forward is jitted with a fixed seq_len
    (padded) to avoid recompilation every step.
    """
    tokenizer = get_tokenizer(config)
    rng = np.random.default_rng(seed)

    # Encode prompt
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(prompt)
    else:
        ids = tokenizer(prompt)["input_ids"]
    if not ids:
        ids = [0]

    model = nnx.merge(graphdef, state) if rest is None else nnx.merge(graphdef, state, rest)
    seq_len = config.seq_len
    pad_id = 0

    # Jit the forward pass with fixed shape (seq_len) to avoid recompilation.
    @jax.jit
    def forward(tokens):
        return model.generate(tokens)  # [1, seq_len, V]

    generated = list(ids)
    for _ in range(max_tokens):
        # Right-pad context to seq_len; we take logits at the last real token.
        ctx = generated[-seq_len:]
        n_real = len(ctx)
        pad_len = seq_len - n_real
        x = jnp.array([ctx + [pad_id] * pad_len], dtype=jnp.int32)
        logits = forward(x)  # [1, seq_len, V]
        # logits at position n_real-1 (the last real token)
        next_logits = np.asarray(logits[0, n_real - 1])  # [V]
        next_id = sample_next(next_logits, temperature=temperature, top_k=top_k, rng=rng)
        generated.append(next_id)
        if next_id == getattr(tokenizer, "eos_id", None):
            break

    # Decode
    if hasattr(tokenizer, "decode"):
        text = tokenizer.decode(generated)
    else:
        text = tokenizer.decode(generated)
    return text, generated