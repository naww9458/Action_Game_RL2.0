"""Shared runtime helpers for articulation-body control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import warp as wp


@dataclass(frozen=True)
class RuntimeNominalsGpuSpec:
    """GPU-side runtime nominal adjustment inputs built at configure time."""

    upright_dot_threshold: float
    passive_dof_mask: List[int]

    def has_passive_dofs(self) -> bool:
        return any(int(v) != 0 for v in self.passive_dof_mask)


@wp.func
def body_upright_dot_from_quat(q: wp.quat) -> float:
    """World-frame Z component of the body up axis."""
    qx = q[0]
    qy = q[1]
    return 1.0 - 2.0 * (qx * qx + qy * qy)


@wp.kernel
def adjust_runtime_joint_nominals_kernel(
    body_q: wp.array(dtype=wp.transform),
    joint_q: wp.array(dtype=float, ndim=3),
    index_players_gpu: wp.array(dtype=wp.int32),
    view_object_indices: wp.array(dtype=int),
    view_body_local_indices: wp.array(dtype=int),
    num_objects_env: int,
    num_rigid_bodies_env: int,
    count_per_world: int,
    joint_dof_count: int,
    joint_nominal_qs: wp.array(dtype=float),
    default_nominal_qs: wp.array(dtype=float),
    passive_dof_mask: wp.array(dtype=wp.int32),
    upright_dot_threshold: float,
):
    tid = wp.tid()
    player_idx = index_players_gpu[tid]
    world = player_idx // num_objects_env
    local_idx = player_idx % num_objects_env

    local_body = view_body_local_indices[0]
    global_body = world * num_rigid_bodies_env + local_body
    upright_dot = body_upright_dot_from_quat(body_q[global_body].q)

    obj_idx = wp.int32(-1)
    for i in range(count_per_world):
        if view_object_indices[i] == local_idx:
            obj_idx = i
            break

    if obj_idx == -1:
        return

    for dof in range(joint_dof_count):
        if passive_dof_mask[dof] == 0:
            continue
        if upright_dot < upright_dot_threshold:
            joint_nominal_qs[dof] = joint_q[world, obj_idx, dof]
        else:
            joint_nominal_qs[dof] = default_nominal_qs[dof]


def body_upright_dot(body_q_transform) -> float:
    """Host helper: world-frame Z component of the body up axis."""
    quat = getattr(body_q_transform, "q", body_q_transform)
    qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return 1.0 - 2.0 * (qx * qx + qy * qy)


def build_runtime_nominals_gpu_spec_from_masks(
    joint_labels: Sequence[str],
    passive_label_predicate,
    upright_dot_threshold: float,
) -> RuntimeNominalsGpuSpec | None:
    mask = [1 if passive_label_predicate(label) else 0 for label in joint_labels]
    if not any(mask):
        return None
    return RuntimeNominalsGpuSpec(
        upright_dot_threshold=float(upright_dot_threshold),
        passive_dof_mask=mask,
    )
