"""Mesh setup and PartitionSpec helpers for FSDP + expert parallelism."""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


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
    device_array = np.array(devices[:n_devices]).reshape(data_axis, expert_axis)
    return Mesh(device_array, axis_names=("data", "expert"))


def _keystr(path) -> str:
    return "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)


_REPLICATE_BELOW = 1 << 16  # elements; norms, router biases, residual scales


def expert_shard_axis(n_experts: int, mesh: Mesh):
    """Which mesh axis the expert dimension is split over, or None.

    Prefers a real 'expert' axis when the mesh has one; otherwise the experts
    ride the 'data' axis (that is still FSDP -- the weights get gathered per
    layer -- but sharding along the expert dimension is what lets the gather be
    expressed as one all_gather with a reduce-scatter transpose).
    """
    e, d = mesh.shape["expert"], mesh.shape["data"]
    if e > 1 and n_experts % e == 0:
        return "expert"
    if d > 1 and n_experts % d == 0:
        return "data"
    return None


def _leaf_name(name: str) -> str:
    """Last meaningful path component, ignoring nnx's '.value' leaf marker.

    nnx paths end in a '.value' element (e.g. 'rope_cos/.value'), so taking the
    final component naively yields '.value' for every leaf and any name-based
    rule silently never matches.
    """
    parts = [p for p in name.split("/") if p not in (".value", "value", "raw_value")]
    return parts[-1] if parts else name


def param_spec(name: str, shape, mesh: Mesh) -> P:
    """FSDP PartitionSpec for one parameter.

    Blocks are stored stacked (a leading layer axis added by BlockStack), so
    this rule is written to be shape-driven rather than rank-specific:

      - anything small (norms, router bias, residual scales) and the RoPE
        tables are replicated -- sharding them would buy nothing but collectives.
      - stacked expert weights additionally shard their expert axis (third from
        last, ahead of the two matrix axes) over the 'expert' mesh axis.
      - otherwise the largest axis divisible by the 'data' mesh size is sharded,
        which is the model dimension for weights and the vocab for embeddings.

    A parameter whose axes are all indivisible by the mesh stays replicated.
    """
    d = mesh.shape["data"]
    e = mesh.shape["expert"]
    base = _leaf_name(name)

    # The RoPE tables are indexed by position, so their long axis is *sequence*.
    # The generic "shard the largest divisible axis" rule below would split them
    # across devices, and GSPMD then propagates that sequence sharding into q
    # and k -- costing an all-gather of cos and sin inside every layer of the
    # scan, in the forward, the remat recompute, and the backward alike.
    if not shape or base in ("cos", "sin", "rope_cos", "rope_sin"):
        return P()
    if int(np.prod(shape)) < _REPLICATE_BELOW:
        return P()

    specs = [None] * len(shape)
    expert_axis = len(shape) - 3
    if "expert_w" in name and expert_axis >= 0:
        # Shard stacked expert weights along the expert axis specifically, and
        # nothing else. MoE gathers them inside a shard_map, and it can only
        # emit the matching all_gather (whose transpose is the reduce-scatter
        # that keeps the gradient sharded) if it knows the axis statically.
        axis_name = expert_shard_axis(shape[expert_axis], mesh)
        if axis_name is not None:
            specs[expert_axis] = axis_name
            return P(*specs)

    if d > 1:
        candidates = [i for i, s in enumerate(shape)
                      if specs[i] is None and s % d == 0 and s // d > 0]
        if candidates:
            specs[max(candidates, key=lambda i: shape[i])] = "data"
    return P(*specs)


def state_shardings(state, mesh: Mesh):
    """Map a (possibly abstract) state pytree to a matching pytree of NamedShardings."""
    def fn(path, x):
        return NamedSharding(mesh, param_spec(_keystr(path), jnp.shape(x), mesh))
    return jax.tree_util.tree_map_with_path(fn, state)


def data_sharding(mesh: Mesh) -> NamedSharding:
    """Batches are [B, S]; shard the batch over 'data'."""
    return NamedSharding(mesh, P("data", None))


def describe(state, shardings) -> str:
    """Human-readable summary of how many params are sharded vs replicated."""
    leaves = jax.tree_util.tree_leaves_with_path(state)
    shard_leaves = jax.tree_util.tree_leaves(shardings)
    sharded = sum(int(np.prod(jnp.shape(x)))
                  for (_, x), s in zip(leaves, shard_leaves)
                  if any(a is not None for a in s.spec))
    total = sum(int(np.prod(jnp.shape(x))) for _, x in leaves)
    return f"{sharded/1e9:.3f}B / {total/1e9:.3f}B params sharded ({100*sharded/max(total,1):.1f}%)"


def with_sharding_constraint(x, partition_spec, mesh: Optional[Mesh] = None):
    """Wrap jax.lax.with_sharding_constraint; no-op if mesh is None / single device."""
    if mesh is None or mesh.devices.size <= 1:
        return x
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, partition_spec))
