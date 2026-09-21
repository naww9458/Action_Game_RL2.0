"""Pose / joint queries for boxing_v1. Returns Torch tensors.

Newton / Warp details stay in this boxing_v1 plugin. Task-layer rewards and
environments read tensors only and must TryGet when a query cannot be built.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
import warp as wp


@wp.kernel
def _gather_link_pose_kernel(
    link_q: wp.array(dtype=wp.transform, ndim=3),
    link_qd: wp.array(dtype=wp.spatial_vector, ndim=3),
    world_idx: wp.array(dtype=wp.int32),
    view_idx: wp.array(dtype=wp.int32),
    link_idx: int,
    out_pos: wp.array2d(dtype=float),
    out_vel: wp.array2d(dtype=float),
    out_grav_xy: wp.array(dtype=float),
):
    tid = wp.tid()
    world = world_idx[tid]
    view = view_idx[tid]
    tf = link_q[world, view, link_idx]
    pos = wp.transform_get_translation(tf)
    out_pos[tid, 0] = pos[0]
    out_pos[tid, 1] = pos[1]
    out_pos[tid, 2] = pos[2]
    qd = link_qd[world, view, link_idx]
    out_vel[tid, 0] = qd[0]
    out_vel[tid, 1] = qd[1]
    out_vel[tid, 2] = qd[2]
    rot = wp.transform_get_rotation(tf)
    local_g = wp.quat_rotate(wp.quat_inverse(rot), wp.vec3(0.0, 0.0, -1.0))
    out_grav_xy[tid] = local_g[0] * local_g[0] + local_g[1] * local_g[1]


@wp.kernel
def _gather_link_yaw_kernel(
    link_q: wp.array(dtype=wp.transform, ndim=3),
    world_idx: wp.array(dtype=wp.int32),
    view_idx: wp.array(dtype=wp.int32),
    link_idx: int,
    out_yaw: wp.array(dtype=float),
):
    tid = wp.tid()
    world = world_idx[tid]
    view = view_idx[tid]
    tf = link_q[world, view, link_idx]
    rot = wp.transform_get_rotation(tf)
    siny_cosp = 2.0 * (rot[3] * rot[2] + rot[0] * rot[1])
    cosy_cosp = 1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2])
    out_yaw[tid] = wp.atan2(siny_cosp, cosy_cosp)


@wp.kernel
def _gather_joint_row_kernel(
    joint_q: wp.array(dtype=float, ndim=3),
    world_idx: wp.array(dtype=wp.int32),
    view_idx: wp.array(dtype=wp.int32),
    dof_count: int,
    out_q: wp.array2d(dtype=float),
):
    tid = wp.tid()
    world = world_idx[tid]
    view = view_idx[tid]
    for d in range(dof_count):
        out_q[tid, d] = joint_q[world, view, d]


def _torch_device(device: Any) -> torch.device:
    try:
        return wp.device_to_torch(device)
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_newton_body_ids(
    physics_manager: Any,
    suffixes: Sequence[str],
    device: str,
) -> Optional[torch.Tensor]:
    """Per-world Newton body ids for label suffixes. Shape ``(nworld, n_suffixes)``.

    Returns None when the solver mapping is missing (TryGet).
    """
    pm = physics_manager
    solver = getattr(getattr(pm, "solver_handler", None), "solver", None)
    if solver is None or not hasattr(solver, "mjc_body_to_newton"):
        return None
    model = getattr(pm, "model", None)
    if model is None:
        return None
    names = tuple(str(s) for s in suffixes if str(s).strip())
    if not names:
        return None

    mt = solver.mjc_body_to_newton.numpy()
    nworld = int(mt.shape[0])
    bodies_per_world = int(model.body_count) // int(getattr(model, "world_count", 1) or 1)
    body_labels = model.body_label
    if hasattr(body_labels, "numpy"):
        body_labels = body_labels.numpy()
    labels = [str(x) for x in body_labels]

    local_ids: list[int] = []
    for suffix in names:
        found = None
        for i, name in enumerate(labels):
            if name.endswith(suffix):
                found = i % bodies_per_world
                break
        if found is None:
            return None
        local_ids.append(int(found))

    row0 = mt[0]
    nbody_mj = int(mt.shape[1])
    mj_ids: list[int] = []
    for local in local_ids:
        matches = [j for j in range(nbody_mj) if int(row0[j]) == local]
        if len(matches) != 1:
            return None
        mj_ids.append(int(matches[0]))

    out = np.zeros((nworld, len(names)), dtype=np.int32)
    for w in range(nworld):
        for k, mj_b in enumerate(mj_ids):
            out[w, k] = int(mt[w, mj_b])
    torch_device = _torch_device(device)
    return torch.as_tensor(out, device=torch_device, dtype=torch.int32)


def _resolve_link_index(view: Any, body_name: str) -> Optional[int]:
    names = list(getattr(view, "link_names", ()) or ())
    for i, name in enumerate(names):
        if str(name).endswith(str(body_name)):
            return i
    return None


class ArticulationState:
    """Link / joint tensors for one articulation pattern (one object per world)."""

    def __init__(
        self,
        *,
        view: Any,
        physics_manager: Any,
        pattern: str,
        articulation_body: Any,
        world_indices: Sequence[int],
        view_indices: Sequence[int],
        device: Any,
    ) -> None:
        self.view = view
        self.physics_manager = physics_manager
        self.pattern = pattern
        self.articulation_body = articulation_body
        self.device = device
        self.num_instances = len(world_indices)
        self._world_wp = wp.array(list(world_indices), dtype=wp.int32, device=device)
        self._view_wp = wp.array(list(view_indices), dtype=wp.int32, device=device)
        self._pos = wp.zeros((self.num_instances, 3), dtype=float, device=device)
        self._vel = wp.zeros((self.num_instances, 3), dtype=float, device=device)
        self._grav_xy = wp.zeros(self.num_instances, dtype=float, device=device)

    @classmethod
    def try_from_environment(cls, environment: Any, pattern: Optional[str] = None):
        ab = getattr(environment, "articulation_body", None)
        pm = getattr(environment, "physics_manager", None)
        if ab is None or pm is None:
            return None
        resolved = str(pattern or "")
        if not resolved:
            obs_actor = getattr(environment, "g1_obs_actor", None)
            resolved = str(getattr(obs_actor, "pattern", "") or "")
        if not resolved:
            return None
        view_idx = next(
            (i for i, p in enumerate(getattr(ab, "patterns", ()) or ()) if p == resolved),
            -1,
        )
        if view_idx < 0:
            return None
        view = ab.views[view_idx]
        obs_actor = getattr(environment, "g1_obs_actor", None)
        worlds = getattr(obs_actor, "instance_world_indices", None)
        views = getattr(obs_actor, "instance_view_indices", None)
        num_env = int(getattr(environment, "num_env", 0) or 0)
        if not worlds:
            worlds = list(range(num_env))
            views = [0] * num_env
        device = getattr(environment, "device", None) or getattr(pm, "device", None)
        return cls(
            view=view,
            physics_manager=pm,
            pattern=resolved,
            articulation_body=ab,
            world_indices=list(worlds),
            view_indices=list(views),
            device=device,
        )

    def copy_link_position_velocity(self, body_name: str, out_pos: wp.array, out_vel: wp.array) -> None:
        link_idx = _resolve_link_index(self.view, body_name)
        if link_idx is None or self.num_instances <= 0:
            out_pos.zero_()
            out_vel.zero_()
            return
        pm = self.physics_manager
        link_q = self.view.get_link_transforms(pm.state_0)
        link_qd = self.view.get_link_velocities(pm.state_0)
        wp.launch(
            _gather_link_pose_kernel,
            dim=self.num_instances,
            inputs=[
                link_q,
                link_qd,
                self._world_wp,
                self._view_wp,
                int(link_idx),
                out_pos,
                out_vel,
                self._grav_xy,
            ],
            device=self.device,
        )

    def copy_projected_gravity_xy(self, body_name: str, out: wp.array) -> None:
        link_idx = _resolve_link_index(self.view, body_name)
        if link_idx is None or self.num_instances <= 0:
            out.zero_()
            return
        pm = self.physics_manager
        link_q = self.view.get_link_transforms(pm.state_0)
        link_qd = self.view.get_link_velocities(pm.state_0)
        wp.launch(
            _gather_link_pose_kernel,
            dim=self.num_instances,
            inputs=[
                link_q,
                link_qd,
                self._world_wp,
                self._view_wp,
                int(link_idx),
                self._pos,
                self._vel,
                out,
            ],
            device=self.device,
        )

    def copy_link_yaw(self, body_name: str, out_yaw: wp.array) -> None:
        """Z-XY Euler yaw of one link per instance (pelvis yaw frame for rewards)."""
        link_idx = _resolve_link_index(self.view, body_name)
        if link_idx is None or self.num_instances <= 0:
            out_yaw.zero_()
            return
        pm = self.physics_manager
        link_q = self.view.get_link_transforms(pm.state_0)
        wp.launch(
            _gather_link_yaw_kernel,
            dim=self.num_instances,
            inputs=[link_q, self._world_wp, self._view_wp, int(link_idx), out_yaw],
            device=self.device,
        )

    def copy_joint_positions(self, out: wp.array) -> None:
        dof = int(getattr(self.view, "joint_dof_count", 0) or 0)
        if dof <= 0 or self.num_instances <= 0:
            return
        joint_q = self.view.get_dof_positions(self.physics_manager.state_0)
        wp.launch(
            _gather_joint_row_kernel,
            dim=self.num_instances,
            inputs=[joint_q, self._world_wp, self._view_wp, dof, out],
            device=self.device,
        )

    def joint_nominal_positions(self) -> Optional[torch.Tensor]:
        buf = getattr(self.articulation_body, "control_joint_nominal_qs_gpus", {}).get(self.pattern)
        if buf is None:
            return None
        return wp.to_torch(buf).reshape(-1)

    def joint_rl_mask(self) -> Optional[torch.Tensor]:
        buf = getattr(self.articulation_body, "control_joint_rl_mask_gpus", {}).get(self.pattern)
        if buf is None:
            return None
        return wp.to_torch(buf).reshape(-1)

    def copy_joint_torques(self, out: wp.array) -> bool:
        buf = getattr(self.articulation_body, "control_joint_torque_gpus", {}).get(self.pattern)
        if buf is None or self.num_instances <= 0:
            return False
        dof = int(getattr(self.view, "joint_dof_count", 0) or 0)
        wp.launch(
            _gather_joint_row_kernel,
            dim=self.num_instances,
            inputs=[buf, self._world_wp, self._view_wp, dof, out],
            device=self.device,
        )
        return True


@wp.kernel
def _gather_override_body_pose_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    worlds: wp.array(dtype=wp.int32),
    local_body: wp.array(dtype=wp.int32),
    bodies_per_env: int,
    out_pos: wp.array2d(dtype=float),
    out_vel: wp.array2d(dtype=float),
    out_flag: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    body = local_body[tid]
    if body < 0 or bodies_per_env <= 0 or body >= bodies_per_env:
        out_pos[tid, 0] = 0.0
        out_pos[tid, 1] = 0.0
        out_pos[tid, 2] = 0.0
        out_vel[tid, 0] = 0.0
        out_vel[tid, 1] = 0.0
        out_vel[tid, 2] = 0.0
        out_flag[tid] = 0
        return
    world = worlds[tid]
    if world < 0:
        out_pos[tid, 0] = 0.0
        out_pos[tid, 1] = 0.0
        out_pos[tid, 2] = 0.0
        out_vel[tid, 0] = 0.0
        out_vel[tid, 1] = 0.0
        out_vel[tid, 2] = 0.0
        out_flag[tid] = 0
        return
    gid = world * bodies_per_env + body
    tf = body_q[gid]
    pos = wp.transform_get_translation(tf)
    out_pos[tid, 0] = pos[0]
    out_pos[tid, 1] = pos[1]
    out_pos[tid, 2] = pos[2]
    qd = body_qd[gid]
    out_vel[tid, 0] = qd[0]
    out_vel[tid, 1] = qd[1]
    out_vel[tid, 2] = qd[2]
    out_flag[tid] = 1


@wp.kernel
def _clear_selected_newton_ids_kernel(out_ids: wp.array2d(dtype=wp.int32)):
    tid = wp.tid()
    out_ids[tid, 0] = -1


@wp.kernel
def _fill_selected_newton_ids_kernel(
    worlds: wp.array(dtype=wp.int32),
    local_body: wp.array(dtype=wp.int32),
    bodies_per_env: int,
    num_env: int,
    out_ids: wp.array2d(dtype=wp.int32),
):
    tid = wp.tid()
    world = worlds[tid]
    if world < 0 or world >= num_env:
        return
    body = local_body[tid]
    if body < 0 or bodies_per_env <= 0:
        out_ids[world, 0] = -1
        return
    out_ids[world, 0] = world * bodies_per_env + body


def fill_selected_newton_body_ids(
    physics_manager: Any,
    worlds: wp.array,
    local_body: wp.array,
    out_ids: wp.array,
    *,
    count: int,
    num_env: int,
    device: Any,
) -> None:
    """Write per-env Newton ids for the selected env-local body (graph-safe)."""
    if count <= 0 or num_env <= 0 or out_ids is None or physics_manager is None:
        return
    model = getattr(physics_manager, "model", None)
    if model is None:
        return
    nworld = max(int(getattr(model, "world_count", 1) or 1), 1)
    bodies_per_env = int(model.body_count) // nworld
    wp.launch(
        _clear_selected_newton_ids_kernel,
        dim=int(num_env),
        inputs=[out_ids],
        device=device,
    )
    wp.launch(
        _fill_selected_newton_ids_kernel,
        dim=int(count),
        inputs=[
            worlds,
            local_body,
            int(bodies_per_env),
            int(num_env),
            out_ids,
        ],
        device=device,
    )


def gather_override_body_poses(
    physics_manager: Any,
    worlds: wp.array,
    local_body: wp.array,
    out_pos: wp.array,
    out_vel: wp.array,
    out_flag: wp.array,
    *,
    count: int,
    device: Any,
) -> None:
    """Copy selected env-local body poses into instance target buffers (TryGet)."""
    if count <= 0 or physics_manager is None:
        return
    state = getattr(physics_manager, "state_0", None)
    body_q = getattr(state, "body_q", None) if state is not None else None
    body_qd = getattr(state, "body_qd", None) if state is not None else None
    model = getattr(physics_manager, "model", None)
    if body_q is None or body_qd is None or model is None:
        return
    nworld = int(getattr(model, "world_count", 1) or 1)
    bodies_per_env = int(model.body_count) // max(nworld, 1)
    if bodies_per_env <= 0:
        return
    wp.launch(
        _gather_override_body_pose_kernel,
        dim=count,
        inputs=[
            body_q,
            body_qd,
            worlds,
            local_body,
            bodies_per_env,
            out_pos,
            out_vel,
            out_flag,
        ],
        device=device,
    )


@wp.kernel
def _fill_scaled_term_kernel(
    raw: wp.array(dtype=float),
    term_buf: wp.array(dtype=float),
    scale: float,
):
    world = wp.tid()
    term_buf[world] = raw[world] * scale


@wp.kernel
def _scatter_player_reward_kernel(
    raw: wp.array(dtype=float),
    player_shape_ids: wp.array(dtype=wp.int32),
    env_map: wp.array(dtype=wp.int32),
    scale: float,
    total: wp.array(dtype=float),
):
    tid = wp.tid()
    world = env_map[tid]
    wp.atomic_add(total, player_shape_ids[tid], raw[world] * scale)


def scatter_env_reward(
    raw: wp.array,
    *,
    scale: float,
    step_total_rewards: wp.array,
    term_buf: Any,
    player_shape_ids: wp.array,
    env_map: wp.array,
    num_players: int,
    num_env: int,
    device: Any,
) -> None:
    """Scale per-env raw terms into the per-object total and optional log buffer."""
    if term_buf is not None and num_env > 0:
        wp.launch(
            _fill_scaled_term_kernel,
            dim=num_env,
            inputs=[raw, term_buf, float(scale)],
            device=device,
        )
    if num_players > 0:
        wp.launch(
            _scatter_player_reward_kernel,
            dim=num_players,
            inputs=[raw, player_shape_ids, env_map, float(scale), step_total_rewards],
            device=device,
        )
