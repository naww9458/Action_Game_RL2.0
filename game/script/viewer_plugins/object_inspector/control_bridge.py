from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import warp as wp

from .metadata import BodyParamSpec, JointParamSpec, ObjectInspectorSpec, PlayerActionSpec

if TYPE_CHECKING:
    from script.game import Game


def _assign_host(arr: wp.array, host):
    arr.assign(host)


@dataclass
class BodyState:
    pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    quat: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    lin_vel: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ang_vel: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    force: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    torque: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class JointState:
    q: float = 0.0
    qd: float = 0.0
    torque: float = 0.0


@dataclass
class PinnedBodyFields:
    pos: bool = False
    quat: bool = False
    lin_vel: bool = False
    ang_vel: bool = False


@dataclass
class PinnedJointFields:
    q: bool = False
    qd: bool = False


class ControlBridge:
    def __init__(self, game: "Game"):
        self.game = game
        self.pm = game.physics_manager
        self.ab = game.articulation_body
        self.db = game.deformable_body

    def read_body(self, spec: ObjectInspectorSpec, world_idx: int, body: BodyParamSpec) -> BodyState:
        state = BodyState()
        if spec.body_kind == "deformable":
            return self._read_deformable(spec, world_idx, state)
        return self._read_rigid_body(body.global_body_index, world_idx, state)

    def read_joint(self, spec: ObjectInspectorSpec, world_idx: int, joint: JointParamSpec) -> JointState:
        js = JointState()
        if spec.body_kind != "articulation":
            return js
        view_idx = next((i for i, p in enumerate(self.ab.patterns) if p == spec.pattern), -1)
        if view_idx < 0:
            return js
        view = self.ab.views[view_idx]
        dof_pos = view.get_dof_positions(self.pm.state_0).numpy()
        dof_vel = view.get_dof_velocities(self.pm.state_0).numpy()
        w = min(world_idx, dof_pos.shape[0] - 1)
        o = min(spec.view_obj_idx, dof_pos.shape[1] - 1)
        d = min(joint.dof_in_obj_idx, dof_pos.shape[2] - 1)
        js.q = float(dof_pos[w, o, d])
        js.qd = float(dof_vel[w, o, d])
        return js

    def apply_body_pinned(
        self,
        spec: ObjectInspectorSpec,
        world_idx: int,
        body: BodyParamSpec,
        values: BodyState,
        pinned: PinnedBodyFields,
    ):
        if spec.body_kind == "deformable":
            self._apply_deformable_pinned(spec, world_idx, values, pinned)
            return

        global_idx = self._global_body_index(body.global_body_index, world_idx)
        if global_idx < 0:
            return

        if spec.body_kind == "articulation":
            if body.is_base_body and (pinned.pos or pinned.quat or pinned.lin_vel or pinned.ang_vel):
                self._apply_articulation_body_control(spec, world_idx, body, values, pinned)
            elif not body.is_base_body and (pinned.lin_vel or pinned.ang_vel):
                self._write_body_qd_pinned(global_idx, values, pinned)
            return

        self._write_rigid_pinned(global_idx, body, values, pinned)

    def apply_body_impulse(
        self,
        spec: ObjectInspectorSpec,
        world_idx: int,
        body: BodyParamSpec,
        values: BodyState,
        apply_force: bool,
        apply_torque: bool,
    ):
        if spec.body_kind == "deformable":
            if apply_force:
                self._write_deformable_force(spec, world_idx, values.force)
            return

        global_idx = self._global_body_index(body.global_body_index, world_idx)
        if global_idx < 0:
            return

        if spec.body_kind == "articulation" and body.is_base_body:
            pattern = spec.pattern
            if pattern in self.ab.control_force_gpus:
                w, o, b = world_idx, spec.view_obj_idx, body.body_in_obj_idx
                if apply_force:
                    host_f = self.ab.control_force_gpus[pattern].numpy()
                    host_f[w, o, b] = values.force
                    self.ab.control_force_gpus[pattern].assign(host_f)
                if apply_torque and pattern in self.ab.control_torque_gpus:
                    host_t = self.ab.control_torque_gpus[pattern].numpy()
                    host_t[w, o, b] = values.torque
                    self.ab.control_torque_gpus[pattern].assign(host_t)
            return

        if not apply_force and not apply_torque:
            return
        if self.pm.inspector_body_f is None:
            return
        inspector_f = self.pm.inspector_body_f.numpy()
        if global_idx >= len(inspector_f):
            return
        if apply_force:
            inspector_f[global_idx, 0:3] += values.force
        if apply_torque:
            inspector_f[global_idx, 3:6] += values.torque
        self.pm.inspector_body_f.assign(inspector_f)

    def apply_joint_pinned(
        self,
        spec: ObjectInspectorSpec,
        world_idx: int,
        joint: JointParamSpec,
        values: JointState,
        pinned: PinnedJointFields,
    ):
        if spec.body_kind != "articulation" or not (pinned.q or pinned.qd):
            return
        pattern = spec.pattern
        if pattern not in self.ab.control_joint_mask_gpus:
            return
        w, o, d = world_idx, spec.view_obj_idx, joint.dof_in_obj_idx
        mask_arr = self.ab.control_joint_mask_gpus[pattern]
        mask_np = mask_arr.numpy()
        if pinned.q:
            host = self.ab.control_joint_pos_gpus[pattern].numpy()
            host[w, o, d] = values.q
            self.ab.control_joint_pos_gpus[pattern].assign(host)
            mask_np[w, o, d] |= 4
        if pinned.qd:
            host = self.ab.control_joint_vel_gpus[pattern].numpy()
            host[w, o, d] = values.qd
            self.ab.control_joint_vel_gpus[pattern].assign(host)
            mask_np[w, o, d] |= 2
        mask_arr.assign(mask_np)

    def apply_joint_impulse(
        self,
        spec: ObjectInspectorSpec,
        world_idx: int,
        joint: JointParamSpec,
        torque: float,
    ):
        if spec.body_kind != "articulation":
            return
        pattern = spec.pattern
        if pattern not in self.ab.control_joint_mask_gpus:
            return
        w, o, d = world_idx, spec.view_obj_idx, joint.dof_in_obj_idx
        host = self.ab.control_joint_torque_gpus[pattern].numpy()
        host[w, o, d] = torque
        self.ab.control_joint_torque_gpus[pattern].assign(host)
        mask_np = self.ab.control_joint_mask_gpus[pattern].numpy()
        mask_np[w, o, d] |= 1
        self.ab.control_joint_mask_gpus[pattern].assign(mask_np)

    def _global_body_index(self, body_index_in_env0: int, world_idx: int) -> int:
        num_env = max(self.game.num_env, 1)
        bodies_per_env = self.pm.model.body_count // num_env
        local = body_index_in_env0 % bodies_per_env
        return world_idx * bodies_per_env + local

    def _read_rigid_body(self, local_body_index: int, world_idx: int, state: BodyState) -> BodyState:
        global_idx = self._global_body_index(local_body_index, world_idx)
        body_q = self.pm.state_0.body_q.numpy()
        body_qd = self.pm.state_0.body_qd.numpy()
        if global_idx < 0 or global_idx >= len(body_q):
            return state
        state.pos = body_q[global_idx, 0:3].tolist()
        state.quat = body_q[global_idx, 3:7].tolist()
        state.lin_vel = body_qd[global_idx, 0:3].tolist()
        state.ang_vel = body_qd[global_idx, 3:6].tolist()
        return state

    def _write_body_qd_pinned(self, global_idx: int, values: BodyState, pinned: PinnedBodyFields):
        body_qd = self.pm.state_0.body_qd.numpy()
        if global_idx >= len(body_qd):
            return
        changed = False
        if pinned.lin_vel:
            body_qd[global_idx, 0:3] = values.lin_vel
            changed = True
        if pinned.ang_vel:
            body_qd[global_idx, 3:6] = values.ang_vel
            changed = True
        if changed:
            _assign_host(self.pm.state_0.body_qd, body_qd)

    def _write_rigid_pinned(
        self,
        global_idx: int,
        body: BodyParamSpec,
        values: BodyState,
        pinned: PinnedBodyFields,
    ):
        body_q = self.pm.state_0.body_q.numpy()
        body_qd = self.pm.state_0.body_qd.numpy()
        if global_idx >= len(body_q):
            return

        changed_q = False
        changed_qd = False
        if pinned.pos and body.can_edit_position:
            body_q[global_idx, 0:3] = values.pos
            changed_q = True
        if pinned.quat and body.can_edit_orientation:
            body_q[global_idx, 3:7] = values.quat
            changed_q = True
        if pinned.lin_vel:
            body_qd[global_idx, 0:3] = values.lin_vel
            changed_qd = True
        if pinned.ang_vel:
            body_qd[global_idx, 3:6] = values.ang_vel
            changed_qd = True
        if changed_q:
            _assign_host(self.pm.state_0.body_q, body_q)
            self._sync_body_transform_prev(global_idx, body_q[global_idx])
        if changed_qd:
            _assign_host(self.pm.state_0.body_qd, body_qd)

    def _sync_body_transform_prev(self, global_idx: int, transform_row):
        if self.pm.state_0.body_q_prev is not None:
            prev = self.pm.state_0.body_q_prev.numpy()
            prev[global_idx] = transform_row
            _assign_host(self.pm.state_0.body_q_prev, prev)
        if self.pm.solver_body_q_prev is not None:
            solver_prev = self.pm.solver_body_q_prev.numpy()
            solver_prev[global_idx] = transform_row
            _assign_host(self.pm.solver_body_q_prev, solver_prev)

    def _read_deformable(self, spec: ObjectInspectorSpec, world_idx: int, state: BodyState) -> BodyState:
        pattern = spec.pattern
        view_idx = next((i for i, p in enumerate(self.db.patterns) if p == spec.pattern), -1)
        if view_idx < 0:
            return state
        view = self.db.views[view_idx]
        offsets = self.db.offset[pattern]
        o_idx = spec.view_obj_idx
        if o_idx >= len(offsets):
            return state
        start = offsets[o_idx] + world_idx * view.stride_between_worlds
        pq = self.pm.state_0.particle_q.numpy()[start]
        pqd = self.pm.state_0.particle_qd.numpy()[start]
        state.pos = [float(pq[0]), float(pq[1]), float(pq[2])]
        state.lin_vel = [float(pqd[0]), float(pqd[1]), float(pqd[2])]
        return state

    def _apply_articulation_body_control(
        self,
        spec: ObjectInspectorSpec,
        world_idx: int,
        body: BodyParamSpec,
        values: BodyState,
        pinned: PinnedBodyFields,
    ):
        pattern = spec.pattern
        if pattern not in self.ab.control_mask_gpus:
            return
        w, o = world_idx, spec.view_obj_idx
        b = body.body_in_obj_idx
        mask_arr = self.ab.control_mask_gpus[pattern]
        mask_np = mask_arr.numpy()
        if pinned.pos and body.can_edit_position:
            host = self.ab.control_pos_gpus[pattern].numpy()
            host[w, o, b] = values.pos
            self.ab.control_pos_gpus[pattern].assign(host)
            mask_np[w, o, b] |= 1
        if pinned.quat and body.can_edit_orientation:
            q = values.quat
            host = self.ab.control_rot_gpus[pattern].numpy()
            host[w, o, b] = [q[0], q[1], q[2], q[3]]
            self.ab.control_rot_gpus[pattern].assign(host)
            mask_np[w, o, b] |= 2
        if pinned.lin_vel:
            host = self.ab.control_vel_gpus[pattern].numpy()
            host[w, o, b] = values.lin_vel
            self.ab.control_vel_gpus[pattern].assign(host)
            mask_np[w, o, b] |= 4
        if pinned.ang_vel:
            host = self.ab.control_omega_gpus[pattern].numpy()
            host[w, o, b] = values.ang_vel
            self.ab.control_omega_gpus[pattern].assign(host)
            mask_np[w, o, b] |= 8
        mask_arr.assign(mask_np)

    def _apply_deformable_pinned(
        self,
        spec: ObjectInspectorSpec,
        world_idx: int,
        values: BodyState,
        pinned: PinnedBodyFields,
    ):
        pattern = spec.pattern
        if pattern not in self.db.control_mask_gpus:
            return
        w, o, p = world_idx, spec.view_obj_idx, 0
        mask_arr = self.db.control_mask_gpus[pattern]
        mask_np = mask_arr.numpy()
        if pinned.pos:
            host = self.db.control_pos_gpus[pattern].numpy()
            host[w, o, p] = values.pos
            self.db.control_pos_gpus[pattern].assign(host)
            mask_np[w, o, p] |= 1
        if pinned.lin_vel:
            host = self.db.control_vel_gpus[pattern].numpy()
            host[w, o, p] = values.lin_vel
            self.db.control_vel_gpus[pattern].assign(host)
            mask_np[w, o, p] |= 4
        mask_arr.assign(mask_np)

    def _write_deformable_force(self, spec: ObjectInspectorSpec, world_idx: int, force: List[float]):
        pattern = spec.pattern
        if pattern not in self.db.control_force_gpus:
            return
        w, o, p = world_idx, spec.view_obj_idx, 0
        host = self.db.control_force_gpus[pattern].numpy()
        host[w, o, p] = force
        self.db.control_force_gpus[pattern].assign(host)
        mask_np = self.db.control_mask_gpus[pattern].numpy()
        mask_np[w, o, p] |= 1
        self.db.control_mask_gpus[pattern].assign(mask_np)

    def read_gravity(self) -> List[float]:
        return self.pm.read_runtime_gravity()

    def set_gravity(self, gravity: List[float]):
        self.pm.set_runtime_gravity(gravity)

    def has_commands(self) -> bool:
        level = self.game.level
        return level.commands is not None and len(level.command_labels) > 0

    def read_commands(self, world_idx: int) -> Dict[int, float]:
        level = self.game.level
        if level.commands is None:
            return {}
        host = level.commands.numpy()
        if world_idx < 0 or world_idx >= len(host):
            return {}
        return {idx: float(host[world_idx, idx]) for idx in range(min(len(level.command_labels), host.shape[1]))}

    def apply_command_pins(self, world_idx: int, values: Dict[int, float], pinned_indices: List[int]):
        level = self.game.level
        if level.commands is None or not pinned_indices:
            return
        host = level.commands.numpy()
        if world_idx < 0 or world_idx >= len(host):
            return
        changed = False
        for dim_index in pinned_indices:
            if dim_index not in values:
                continue
            if dim_index >= host.shape[1]:
                continue
            host[world_idx, dim_index] = values[dim_index]
            changed = True
        if changed:
            level.commands.assign(host)

    def resolve_rl_action_row(self, local_role_idx: int, world_idx: int) -> int:
        num_objects_env = self.game.num_objects_env
        global_role_idx = world_idx * num_objects_env + local_role_idx
        try:
            role_list_idx = self.game.players.index_obj_role.index(global_role_idx)
        except ValueError:
            return -1
        mask = getattr(self.game.level, "is_rl_player_mask", None)
        if mask is None or role_list_idx >= len(mask):
            return -1
        return int(mask[role_list_idx])

    def read_rl_actions(
        self,
        player_action: PlayerActionSpec,
        world_idx: int,
        local_role_idx: int,
    ) -> Dict[int, float]:
        values: Dict[int, float] = {}
        rl_row = self.resolve_rl_action_row(local_role_idx, world_idx)
        if rl_row < 0:
            return values
        actions = self.game.default_action.numpy()
        if rl_row >= len(actions):
            return values
        row = actions[rl_row]
        for ability in player_action.abilities:
            for dim in ability.dims:
                if dim.dim_index < len(row):
                    values[dim.dim_index] = float(row[dim.dim_index])
        return values

    def apply_rl_action_pinned(
        self,
        player_action: PlayerActionSpec,
        world_idx: int,
        local_role_idx: int,
        values: Dict[int, float],
        pinned_indices: List[int],
    ):
        row_idx = self.resolve_rl_action_row(local_role_idx, world_idx)
        if row_idx < 0 or not pinned_indices:
            return
        actions = self.game.default_action.numpy()
        if row_idx >= len(actions):
            return
        changed = False
        for dim_index in pinned_indices:
            if dim_index not in values:
                continue
            if dim_index >= len(actions[row_idx]):
                continue
            actions[row_idx, dim_index] = values[dim_index]
            changed = True
        if changed:
            self.game.default_action.assign(actions)

    def apply_rl_action_pins_to_buffer(
        self,
        actions_wp,
        player_action: PlayerActionSpec,
        world_idx: int,
        local_role_idx: int,
        values: Dict[int, float],
        pinned_indices: List[int],
    ):
        row_idx = self.resolve_rl_action_row(local_role_idx, world_idx)
        if row_idx < 0 or not pinned_indices:
            return
        host = actions_wp.numpy()
        if row_idx >= len(host):
            return
        changed = False
        for dim_index in pinned_indices:
            if dim_index not in values:
                continue
            if dim_index >= len(host[row_idx]):
                continue
            host[row_idx, dim_index] = values[dim_index]
            changed = True
        if changed:
            actions_wp.assign(host)


def euler_from_quat(q) -> List[float]:
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]
