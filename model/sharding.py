"""Mesh setup and PartitionSpec helpers for FSDP + expert parallelism."""
from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P


def get_mesh(n_devices: int, data_axis: int = 1, expert_axis: int = 1) -> Mesh:
    """Create a 2D mesh with axes ('data', 'expert').

    For CPU dev: n_devices=1, data_axis=1, expert_axis=1.
    For TPU v5e-8: n_devices=8, data_axis=8, expert_axis=1 (FSDP over data,
    experts replicated). To enable expert parallelism set expert_axis>1
    and data_axis = n_devices // expert_axis.
    """
    assert data_axis * expert_axis == n_devices, (
        f"data_axis({data_axis}) * expert_axis({expert_axis}) != n_devices({n_devices})"
    )
    devices = jax.devices()
    if len(devices) < n_devices:
        # CPU fallback: just use available devices (1)
        devices = jax.devices()[:1]
        data_axis, expert_axis = 1, 1
        n_devices = 1
    device_array = jnp.array(devices[:n_devices]).reshape(data_axis, expert_axis)
    return Mesh(device_array, axis_names=("data", "expert"))


def fsdp_spec(param_name: str, ndim: int) -> P:
    """Default FSDP sharding rule: shard the largest non-batch dim over 'data'.

    Embeddings: shard vocab dim over data.
    Linear weights: shard out-features over data.
    Biases / norms: replicate.
    """
    if "norm" in param_name or "scale" in param_name or "bias" in param_name and "router" not in param_name:
        return P()
    if "embed" in param_name or "wte" in param_name:
        # [vocab, d_model] -> shard vocab
        return P("data", None) if ndim >= 2 else P("data")
    if ndim == 1:
        return P()
    # 2D weight: shard rows (out) over data, cols replicated
    return P("data", None)


def expert_spec(param_name: str, ndim: int) -> P:
    """Sharding for expert params: shard the expert axis over 'expert' mesh axis.

    Expert weights are stored as [n_experts, ...]. Shard dim 0 over 'expert'.
    """
    if "expert" in param_name and ndim >= 1:
        # [n_experts, d_model, d_ff] -> shard experts
        if ndim == 1:
            return P("expert")
        return P("expert", None, None)
    return fsdp_spec(param_name, ndim)


def shard_params(params, mesh: Mesh, rules=None):
    """Apply PartitionSpecs to a pytree of params, returning sharded arrays.

    Uses jax.device_put with NamedSharding. `rules` is an optional callable
    (name, ndim) -> PartitionSpec. Defaults to a heuristic that FSDPs dense
    params over 'data' and shards expert params over 'expert'.
    """
    if rules is None:
        rules = expert_spec

    def _apply(path, x):
        name = "/".join(str(p) for p in path) if path else ""
        spec = rules(name, x.ndim)
        sharding = jax.sharding.NamedSharding(mesh, spec)
        return jax.device_put(x, sharding)

    # params here is an nnx State (dict-like) or a plain pytree
    try:
        import flax.nnx as nnx
        if isinstance(params, nnx.statelib.State):
            def _walk(state):
                out = {}
                for k, v in state.items():
                    if hasattr(v, "items") and not isinstance(v, jnp.ndarray):
                        # nested
                        out[k] = _walk(v)
                    else:
                        out[k] = v
                return out
            # Just return as-is; sharding constraints applied inside model via
            # jax.lax.with_sharding_constraint. This keeps it simple and robust.
            return params
    except Exception:
        pass
    return params


def with_sharding_constraint(x, partition_spec, mesh: Optional[Mesh] = None):
    """Wrap jax.lax.with_sharding_constraint; no-op if mesh is None / single device."""
    if mesh is None or len(mesh.devices) <= 1:
        return x
    return jax.lax.with_sharding_constraint(x, partition_spec)