"""Build-time mount joint slots for runtime tool attach/detach.

XPBD/VBD solvers honor ``model.joint_enabled`` and receive a disabled revolute
(or fixed) joint between host hull and tool root.

MuJoCo does not honor ``joint_enabled``; a disabled revolute would still couple
bodies. MuJoCo levels therefore reserve a disabled WELD equality constraint
instead, toggled through ``model.mujoco.equality_constraint_enabled``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import newton
import warp as wp

from script.role.objects.tool_anchor import compute_weld_relpose_from_anchors


@dataclass
class ToolMountMetadata:
    tool_free_joint_idx: Optional[int]
    tool_root_body_idx: int
    tool_internal_joint_idxs: List[int]
    host_body_idx: int
    tool_body_idx: int


@dataclass
class ToolMountBuildResult:
    mount_joint_idx: Optional[int]
    mount_eq_idx: Optional[int]
    mount_joint_dof_idx: Optional[int]
    mount_joint_coord_idx: Optional[int]
    mount_joint_type: str
    uses_weld_fallback: bool


def compute_max_mount_joints_per_env(tool_configs: Sequence[dict]) -> int:
    """Total mount joints required per environment (one per tool config)."""
    return len(tool_configs)


def resolve_joint_index_by_leaf_name(
    path_joint_map: dict,
    joint_name: str,
    joint_start: int,
    joint_end: int,
) -> int:
    """Resolve a joint index by exact USD prim leaf name (no substring matching)."""
    leaf = str(joint_name).strip()
    if not leaf:
        raise ValueError("joint_name must be a non-empty string")

    matches = [
        (path, int(idx))
        for path, idx in path_joint_map.items()
        if path.rstrip("/").split("/")[-1] == leaf
    ]
    if not matches:
        raise KeyError(
            f"Joint leaf '{leaf}' not found in path_joint_map "
            f"(keys sample: {list(path_joint_map)[:8]})"
        )
    if len(matches) > 1:
        paths = [path for path, _ in matches]
        raise KeyError(
            f"Joint leaf '{leaf}' is ambiguous ({len(matches)} matches): {paths}"
        )

    joint_idx = matches[0][1]
    if not (joint_start <= joint_idx < joint_end):
        raise IndexError(
            f"Joint '{leaf}' index {joint_idx} outside tool joint range "
            f"[{joint_start}, {joint_end})"
        )
    return joint_idx


def _resolve_internal_joint_indices(
    path_joint_map: dict,
    internal_joint_names: Sequence[str],
    joint_start: int,
    joint_end: int,
) -> List[int]:
    if not internal_joint_names:
        return []

    resolved: List[int] = []
    for name in internal_joint_names:
        joint_idx = resolve_joint_index_by_leaf_name(
            path_joint_map, str(name), joint_start, joint_end
        )
        resolved.append(joint_idx)
    return resolved


def _find_free_joint(builder: newton.ModelBuilder, joint_start: int, joint_end: int) -> Optional[int]:
    for j in range(joint_start, joint_end):
        if int(builder.joint_type[j]) == int(newton.JointType.FREE):
            return j
    return None


def collect_tool_mount_metadata(
    builder: newton.ModelBuilder,
    host_body_idx: int,
    tool_body_idx: int,
    tool_joint_start: int,
    tool_joint_end: int,
    path_joint_map: dict,
    internal_joint_names: Sequence[str],
) -> ToolMountMetadata:
    tool_free_joint_idx = _find_free_joint(builder, tool_joint_start, tool_joint_end)
    if tool_free_joint_idx is not None:
        tool_root_body_idx = int(builder.joint_child[tool_free_joint_idx])
    else:
        tool_root_body_idx = int(tool_body_idx)

    internal_joint_idxs = _resolve_internal_joint_indices(
        path_joint_map,
        internal_joint_names,
        tool_joint_start,
        tool_joint_end,
    )

    return ToolMountMetadata(
        tool_free_joint_idx=tool_free_joint_idx,
        tool_root_body_idx=tool_root_body_idx,
        tool_internal_joint_idxs=internal_joint_idxs,
        host_body_idx=int(host_body_idx),
        tool_body_idx=int(tool_body_idx),
    )


def _joint_dof_coord_indices(builder: newton.ModelBuilder, joint_idx: int) -> tuple[int, int]:
    dof_idx = int(builder.joint_qd_start[joint_idx])
    coord_idx = int(builder.joint_q_start[joint_idx])
    return dof_idx, coord_idx


def add_mujoco_weld_equality_constraint(
    builder: newton.ModelBuilder,
    *,
    body1: int,
    body2: int,
    anchor: wp.vec3,
    relpose: wp.transform,
    enabled: bool,
    label: str,
) -> int:
    """Reserve one MuJoCo WELD equality row via the ``mujoco:equality_constraint_*``
    custom-attribute frequency (newton 1.4 API)."""
    from newton.solvers import SolverMuJoCo

    SolverMuJoCo.register_custom_attributes(builder)
    eq_type = int(SolverMuJoCo.EqType.WELD)
    indices = builder.add_custom_values(
        **{
            "mujoco:equality_constraint_type": eq_type,
            "mujoco:equality_constraint_body1": int(body1),
            "mujoco:equality_constraint_body2": int(body2),
            "mujoco:equality_constraint_anchor": anchor,
            "mujoco:equality_constraint_relpose": relpose,
            "mujoco:equality_constraint_enabled": bool(enabled),
            "mujoco:equality_constraint_label": label,
        }
    )
    return int(indices["mujoco:equality_constraint_type"])


def build_tool_mount_joint(
    builder: newton.ModelBuilder,
    *,
    host_body_idx: int,
    tool_root_body_idx: int,
    host_anchor_local: wp.transform,
    tool_anchor_local: wp.transform,
    mount_joint_type: str,
    mount_axis: Sequence[float],
    mount_limits: Sequence[float],
    label: str,
    solver_type: str,
) -> ToolMountBuildResult:
    """Reserve one runtime mount slot on *builder* (template world, ``enabled=False``)."""
    joint_type = str(mount_joint_type or "revolute").lower()
    solver = str(solver_type or "").lower()
    use_weld = solver == "mujoco" and joint_type in {"revolute", "fixed"}

    if use_weld:
        # MuJoCo WELD: anchor = body2 weld point; relpose.p = body1 weld point;
        # relpose.q = body2 orientation relative to body1 (see tool_anchor.py).
        relpose = compute_weld_relpose_from_anchors(host_anchor_local, tool_anchor_local)
        anchor = tool_anchor_local.p if hasattr(tool_anchor_local, "p") else wp.vec3(0.0, 0.0, 0.0)
        eq_idx = add_mujoco_weld_equality_constraint(
            builder,
            body1=int(host_body_idx),
            body2=int(tool_root_body_idx),
            anchor=anchor,
            relpose=relpose,
            enabled=False,
            label=f"{label}_weld",
        )
        return ToolMountBuildResult(
            mount_joint_idx=None,
            mount_eq_idx=int(eq_idx),
            mount_joint_dof_idx=None,
            mount_joint_coord_idx=None,
            mount_joint_type=joint_type,
            uses_weld_fallback=True,
        )

    axis = wp.vec3(float(mount_axis[0]), float(mount_axis[1]), float(mount_axis[2]))
    lo = float(mount_limits[0])
    hi = float(mount_limits[1])

    if joint_type == "fixed":
        joint_idx = builder.add_joint_fixed(
            parent=int(host_body_idx),
            child=int(tool_root_body_idx),
            parent_xform=host_anchor_local,
            child_xform=tool_anchor_local,
            enabled=False,
            label=label,
            collision_filter_parent=True,
        )
        dof_idx, coord_idx = None, None
    elif joint_type == "prismatic":
        joint_idx = builder.add_joint_prismatic(
            parent=int(host_body_idx),
            child=int(tool_root_body_idx),
            parent_xform=host_anchor_local,
            child_xform=tool_anchor_local,
            axis=axis,
            limit_lower=lo,
            limit_upper=hi,
            enabled=False,
            label=label,
            collision_filter_parent=True,
        )
        dof_idx, coord_idx = _joint_dof_coord_indices(builder, joint_idx)
    elif joint_type == "ball":
        joint_idx = builder.add_joint_ball(
            parent=int(host_body_idx),
            child=int(tool_root_body_idx),
            parent_xform=host_anchor_local,
            child_xform=tool_anchor_local,
            enabled=False,
            label=label,
            collision_filter_parent=True,
        )
        dof_idx, coord_idx = _joint_dof_coord_indices(builder, joint_idx)
    else:
        joint_idx = builder.add_joint_revolute(
            parent=int(host_body_idx),
            child=int(tool_root_body_idx),
            parent_xform=host_anchor_local,
            child_xform=tool_anchor_local,
            axis=axis,
            limit_lower=lo,
            limit_upper=hi,
            enabled=False,
            label=label,
            collision_filter_parent=True,
        )
        dof_idx, coord_idx = _joint_dof_coord_indices(builder, joint_idx)

    return ToolMountBuildResult(
        mount_joint_idx=int(joint_idx),
        mount_eq_idx=None,
        mount_joint_dof_idx=dof_idx,
        mount_joint_coord_idx=coord_idx,
        mount_joint_type=joint_type,
        uses_weld_fallback=False,
    )
