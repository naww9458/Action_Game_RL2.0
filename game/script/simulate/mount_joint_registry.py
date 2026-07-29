"""Runtime registry for tool mount joints (toggle on U press)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import newton
import numpy as np
import warp as wp

from script.role.objects.tool_anchor import (
    anchor_within_mount_proximity,
    compose_body_snap_transform,
    compose_mounted_weld_relpose,
)
from script.simulate.tool_camera_aim import (
    ToolAimControlConfig,
    camera_forward_z_up,
    clamp_angle_to_limits,
    compute_host_local_aim_errors,
    measure_mount_yaw_in_host_frame,
    pd_torque,
    soft_limit_torque,
    _wrap_pi,
)


@dataclass
class AttachedToolDofSpec:
    global_dof_idx: int
    limit_lower: float
    limit_upper: float
    mouse_axis: str  # "pitch"
    current_target: float = 0.0
    sensitivity: float = 1.0


@dataclass
class ToolMountRecord:
    tool_key: str
    host_player_index: int
    host_role_object_id: int
    tool_role_object_id: int
    host_body_idx: int
    tool_body_idx: int
    tool_root_body_idx: int
    tool_free_joint_idx: Optional[int]
    tool_internal_joint_idxs: List[int]
    host_anchor_local: wp.transform
    tool_anchor_local: wp.transform
    mount_axis: Tuple[float, float, float]
    mount_yaw_limits: Tuple[float, float]
    proximity_threshold: float
    proximity_height_threshold: float = 3.5
    tool_body_indices: List[int] = field(default_factory=list)
    pitch_joint_name: str = ""
    pitch_joint_idx: Optional[int] = None
    mount_joint_idx: Optional[int] = None
    mount_eq_idx: Optional[int] = None
    mount_joint_dof_idx: Optional[int] = None
    mount_joint_coord_idx: Optional[int] = None
    mount_joint_type: str = "revolute"
    uses_weld_fallback: bool = False
    slot_index: int = 0
    mount_yaw: float = 0.0
    attached: bool = False
    prompt_visible: bool = False
    pitch_dof_spec: Optional[AttachedToolDofSpec] = None
    aim_body_idx: int = -1
    aim_config: ToolAimControlConfig = field(default_factory=ToolAimControlConfig)


class MountJointRegistry:
    def __init__(self, max_mount_joints_per_env: int = 0, num_env: int = 1):
        self.records: Dict[str, ToolMountRecord] = {}
        self.max_mount_joints_per_env = int(max_mount_joints_per_env)
        self.num_env = int(num_env)
        self._model: Optional[newton.Model] = None
        self._device = None
        self._num_joint_dof_env = 0
        self._num_joint_coord_env = 0
        self._num_joint_count_env = 0
        self._num_eq_count_env = 0
        self._num_rigid_bodies_env = 0
        self._solver_supports_joint_toggle = False
        self._uses_mujoco_weld = False
        self._solver = None

    def register(self, record: ToolMountRecord) -> None:
        if (
            self.max_mount_joints_per_env > 0
            and len(self.records) >= self.max_mount_joints_per_env
            and record.tool_key not in self.records
        ):
            raise IndexError(
                f"Mount record '{record.tool_key}' exceeds capacity "
                f"max_mount_joints_per_env={self.max_mount_joints_per_env}"
            )
        self.records[record.tool_key] = record

    def bind_model(
        self,
        model: newton.Model,
        device,
        num_joint_dof_env: int,
        num_rigid_bodies_env: int,
        solver_type: str = "",
        num_env: int = 1,
        solver=None,
    ) -> None:
        self._model = model
        self._device = device
        self._solver = solver
        self.num_env = max(1, int(num_env))
        self._num_joint_dof_env = int(num_joint_dof_env)
        self._num_rigid_bodies_env = int(num_rigid_bodies_env)
        self._num_joint_count_env = max(1, int(model.joint_count // self.num_env))
        self._num_eq_count_env = max(0, int(model.equality_constraint_count // self.num_env))
        if model.joint_coord_count > 0 and self.num_env > 0:
            self._num_joint_coord_env = max(1, int(model.joint_coord_count // self.num_env))
        else:
            self._num_joint_coord_env = 0

        solver = solver_type.lower()
        self._solver_supports_joint_toggle = solver in {"xpbd", "vbd"}
        self._uses_mujoco_weld = solver == "mujoco" or any(
            r.uses_weld_fallback for r in self.records.values()
        )

        joint_qd_start = model.joint_qd_start.numpy()
        joint_limit_lower = model.joint_limit_lower.numpy() if model.joint_limit_lower is not None else None
        joint_limit_upper = model.joint_limit_upper.numpy() if model.joint_limit_upper is not None else None

        for record in self.records.values():
            record.pitch_dof_spec = self._build_pitch_dof_spec(
                record,
                joint_qd_start,
                joint_limit_lower,
                joint_limit_upper,
            )

    def _build_pitch_dof_spec(
        self,
        record: ToolMountRecord,
        joint_qd_start: np.ndarray,
        joint_limit_lower: Optional[np.ndarray],
        joint_limit_upper: Optional[np.ndarray],
    ) -> Optional[AttachedToolDofSpec]:
        if record.pitch_joint_idx is None:
            return None

        joint_idx = int(record.pitch_joint_idx)
        dof_idx = int(joint_qd_start[joint_idx])
        lower = float(joint_limit_lower[dof_idx]) if joint_limit_lower is not None else -0.3
        upper = float(joint_limit_upper[dof_idx]) if joint_limit_upper is not None else 1.2
        return AttachedToolDofSpec(
            global_dof_idx=dof_idx,
            limit_lower=lower,
            limit_upper=upper,
            mouse_axis="pitch",
            sensitivity=1.0,
        )

    def _global_body_idx(self, world: int, local_body_idx: int) -> int:
        return world * self._num_rigid_bodies_env + local_body_idx

    def _global_joint_idx(self, world: int, local_joint_idx: int) -> int:
        return world * self._num_joint_count_env + local_joint_idx

    def _global_eq_idx(self, world: int, local_eq_idx: int) -> int:
        return world * self._num_eq_count_env + local_eq_idx

    def _global_dof_idx(self, world: int, local_dof_idx: int) -> int:
        return world * self._num_joint_dof_env + local_dof_idx

    def _global_coord_idx(self, world: int, local_coord_idx: int) -> int:
        return world * self._num_joint_coord_env + local_coord_idx

    def update_proximity(
        self,
        body_q: wp.array,
        world: int = 0,
        body_q_np: np.ndarray | None = None,
    ) -> None:
        if self._model is None:
            return

        if body_q_np is None:
            body_q_np = body_q.numpy()
        for record in self.records.values():
            if record.attached:
                record.prompt_visible = False
                continue

            host_body = body_q_np[self._global_body_idx(world, record.host_body_idx)]
            tool_body = body_q_np[self._global_body_idx(world, record.tool_body_idx)]
            record.prompt_visible = anchor_within_mount_proximity(
                host_body,
                record.host_anchor_local,
                tool_body,
                record.tool_anchor_local,
                horizontal_threshold=record.proximity_threshold,
                vertical_threshold=record.proximity_height_threshold,
            )

    def any_prompt_visible(self, host_role_object_id: int | None = None) -> bool:
        for record in self.records.values():
            if host_role_object_id is not None and record.host_role_object_id != host_role_object_id:
                continue
            if record.prompt_visible and not record.attached:
                return True
        return False

    def prompt_tool_key(self, host_role_object_id: int | None = None) -> Optional[str]:
        for key, record in self.records.items():
            if host_role_object_id is not None and record.host_role_object_id != host_role_object_id:
                continue
            if record.prompt_visible and not record.attached:
                return key
        return None

    def get_attached_tool_key(self, host_role_object_id: int) -> Optional[str]:
        for key, record in self.records.items():
            if record.attached and record.host_role_object_id == host_role_object_id:
                return key
        return None

    def toggle_attachment(
        self,
        tool_key: str,
        body_q: wp.array,
        body_qd: wp.array,
        joint_q: wp.array,
        world: int = 0,
        body_f: wp.array | None = None,
        joint_qd: wp.array | None = None,
        body_q_prev: wp.array | None = None,
    ) -> bool:
        record = self.records.get(tool_key)
        if record is None or self._model is None:
            return False
        if record.attached:
            return self.disable_attachment(tool_key, body_q, body_qd, world=world)
        return self.enable_attachment(
            tool_key,
            body_q,
            body_qd,
            joint_q,
            world=world,
            body_f=body_f,
            joint_qd=joint_qd,
            body_q_prev=body_q_prev,
        )

    def _clear_tool_dynamics(
        self,
        record: ToolMountRecord,
        world: int,
        body_qd: wp.array | None,
        body_f: wp.array | None,
        joint_qd: wp.array | None,
    ) -> None:
        body_indices = record.tool_body_indices or [record.tool_root_body_idx, record.tool_body_idx]
        unique_bodies = sorted({int(idx) for idx in body_indices if idx >= 0})

        if body_qd is not None:
            body_qd_np = body_qd.numpy()
            for local_body in unique_bodies:
                global_body = self._global_body_idx(world, local_body)
                body_qd_np[global_body, 0:6] = 0.0
            body_qd.assign(body_qd_np)

        if body_f is not None:
            body_f_np = body_f.numpy()
            for local_body in unique_bodies:
                global_body = self._global_body_idx(world, local_body)
                body_f_np[global_body, 0:6] = 0.0
            body_f.assign(body_f_np)

        if joint_qd is None or self._model is None:
            return

        joint_qd_start = self._model.joint_qd_start.numpy()
        joint_indices = list(record.tool_internal_joint_idxs)
        if record.tool_free_joint_idx is not None:
            joint_indices.append(int(record.tool_free_joint_idx))

        joint_qd_np = joint_qd.numpy()
        for joint_idx in sorted({int(j) for j in joint_indices if j >= 0}):
            if joint_idx + 1 >= len(joint_qd_start):
                continue
            local_start = int(joint_qd_start[joint_idx])
            local_end = int(joint_qd_start[joint_idx + 1])
            for local_dof in range(local_start, local_end):
                global_dof = self._global_dof_idx(world, local_dof)
                if 0 <= global_dof < joint_qd_np.shape[0]:
                    joint_qd_np[global_dof] = 0.0
        joint_qd.assign(joint_qd_np)

    def enable_attachment(
        self,
        tool_key: str,
        body_q: wp.array,
        body_qd: wp.array,
        joint_q: wp.array,
        world: int = 0,
        body_f: wp.array | None = None,
        joint_qd: wp.array | None = None,
        body_q_prev: wp.array | None = None,
    ) -> bool:
        record = self.records.get(tool_key)
        if record is None or record.attached or self._model is None:
            return False

        host_global = self._global_body_idx(world, record.host_body_idx)
        tool_root_global = self._global_body_idx(world, record.tool_root_body_idx)

        host_xform = body_q.numpy()[host_global]
        record.mount_yaw = 0.0
        desired_tool = compose_body_snap_transform(
            host_xform,
            record.host_anchor_local,
            record.tool_anchor_local,
        )

        body_q_np = body_q.numpy()
        body_q_np[tool_root_global] = desired_tool
        body_q.assign(body_q_np)

        if body_q_prev is not None:
            body_q_prev_np = body_q_prev.numpy()
            body_q_prev_np[tool_root_global] = desired_tool
            body_q_prev.assign(body_q_prev_np)

        self._clear_tool_dynamics(
            record,
            world=world,
            body_qd=body_qd,
            body_f=body_f,
            joint_qd=joint_qd,
        )

        # Keep FREE floating-base coordinates consistent with body_q (required for MuJoCo).
        if joint_q is not None and record.tool_free_joint_idx is not None:
            jq_np = joint_q.numpy()
            global_free = self._global_joint_idx(world, int(record.tool_free_joint_idx))
            q_start = int(self._model.joint_q_start.numpy()[global_free])
            jq_np[q_start : q_start + 7] = desired_tool
            joint_q.assign(jq_np)

        if joint_q is not None and record.mount_joint_coord_idx is not None:
            jq_np = joint_q.numpy()
            global_coord = self._global_coord_idx(world, record.mount_joint_coord_idx)
            jq_np[global_coord] = 0.0
            joint_q.assign(jq_np)

        self._set_mount_active(record, world=world, active=True)
        record.attached = True
        record.prompt_visible = False
        return True

    def disable_attachment(
        self,
        tool_key: str,
        body_q: wp.array,
        body_qd: wp.array,
        world: int = 0,
    ) -> bool:
        record = self.records.get(tool_key)
        if record is None or not record.attached or self._model is None:
            return False

        self._set_mount_active(record, world=world, active=False)
        record.attached = False
        record.mount_yaw = 0.0
        if record.pitch_dof_spec is not None:
            record.pitch_dof_spec.current_target = 0.0
        return True

    def _notify_solver(self) -> None:
        if self._solver is None:
            return
        try:
            from newton.solvers import SolverNotifyFlags
        except ImportError:
            return
        try:
            self._solver.notify_model_changed(SolverNotifyFlags.CONSTRAINT_PROPERTIES)
        except Exception:
            pass

    def _set_mount_active(self, record: ToolMountRecord, world: int, active: bool) -> None:
        model = self._model
        if model is None:
            return

        if record.uses_weld_fallback and record.mount_eq_idx is not None:
            eq_enabled = model.equality_constraint_enabled
            if eq_enabled is not None:
                enabled_np = eq_enabled.numpy()
                global_eq = self._global_eq_idx(world, record.mount_eq_idx)
                enabled_np[global_eq] = bool(active)
                eq_enabled.assign(enabled_np)

                if active:
                    relpose_arr = model.equality_constraint_relpose
                    if relpose_arr is not None:
                        rel_np = relpose_arr.numpy()
                        rel_np[global_eq] = compose_mounted_weld_relpose(
                            record.host_anchor_local,
                            record.tool_anchor_local,
                            yaw_rad=record.mount_yaw,
                            mount_axis=record.mount_axis,
                        )
                        relpose_arr.assign(rel_np)
            self._notify_solver()
            return

        if record.mount_joint_idx is None:
            return

        joint_enabled = model.joint_enabled
        if joint_enabled is None:
            return

        enabled_np = joint_enabled.numpy()
        global_mount = self._global_joint_idx(world, record.mount_joint_idx)
        enabled_np[global_mount] = bool(active)

        if record.tool_free_joint_idx is not None and self._solver_supports_joint_toggle:
            global_free = self._global_joint_idx(world, record.tool_free_joint_idx)
            enabled_np[global_free] = not bool(active)

        joint_enabled.assign(enabled_np)
        self._notify_solver()

    def apply_attached_aim(
        self,
        body_q: wp.array,
        body_qd: wp.array,
        control,
        camera_yaw: float,
        camera_pitch: float,
        world: int = 0,
        dt: float = 0.02,
        host_role_object_id: int | None = None,
        body_q_np: np.ndarray | None = None,
    ) -> None:
        if self._model is None:
            return

        attached_records = [
            record
            for record in self.records.values()
            if record.attached
            and (
                host_role_object_id is None
                or record.host_role_object_id == host_role_object_id
            )
        ]
        if not attached_records:
            return

        if body_q_np is None:
            body_q_np = body_q.numpy()

        needs_joint_torque = any(
            (not record.uses_weld_fallback and record.mount_joint_dof_idx is not None)
            or record.pitch_dof_spec is not None
            for record in attached_records
        )
        needs_joint_q = any(record.mount_joint_coord_idx is not None for record in attached_records)
        needs_relpose = self._uses_mujoco_weld and any(
            record.uses_weld_fallback for record in attached_records
        )

        # body_qd is unused on the aim hot path; skip the full-table sync.
        joint_qd_np = (
            self._model.joint_qd.numpy()
            if needs_joint_torque and self._model.joint_qd is not None
            else None
        )
        joint_f_np = (
            control.joint_f.numpy()
            if needs_joint_torque and control is not None and control.joint_f is not None
            else None
        )
        joint_q_np = (
            self._model.joint_q.numpy()
            if needs_joint_q and self._model.joint_q is not None
            else None
        )
        relpose_np = (
            self._model.equality_constraint_relpose.numpy()
            if needs_relpose and self._model.equality_constraint_relpose is not None
            else None
        )

        desired_world = camera_forward_z_up(float(camera_yaw), float(camera_pitch))
        relpose_dirty = False
        joint_f_dirty = False

        for record in attached_records:
            cfg = record.aim_config
            dead_zone = math.radians(cfg.angle_dead_zone_deg)

            global_host = self._global_body_idx(world, record.host_body_idx)
            aim_local_idx = record.aim_body_idx if record.aim_body_idx >= 0 else record.tool_body_idx
            global_aim = self._global_body_idx(world, aim_local_idx)

            host_body_q = body_q_np[global_host]
            aim_body_q = body_q_np[global_aim]

            yaw_error, pitch_error = compute_host_local_aim_errors(
                host_body_q,
                aim_body_q,
                desired_world,
                cfg.aim_forward_local,
            )

            if abs(yaw_error) < dead_zone:
                yaw_error = 0.0
            if abs(pitch_error) < dead_zone:
                pitch_error = 0.0

            lo, hi = record.mount_yaw_limits
            previous_yaw = float(record.mount_yaw)
            if record.mount_joint_coord_idx is not None and joint_q_np is not None:
                global_coord = self._global_coord_idx(world, record.mount_joint_coord_idx)
                if 0 <= global_coord < joint_q_np.shape[0]:
                    previous_yaw = float(joint_q_np[global_coord])
            else:
                previous_yaw = measure_mount_yaw_in_host_frame(
                    host_body_q,
                    aim_body_q,
                    cfg.aim_forward_local,
                )
                record.mount_yaw = previous_yaw

            if record.uses_weld_fallback:
                if abs(yaw_error) >= dead_zone:
                    proposed_yaw = _wrap_pi(
                        previous_yaw + cfg.weld_yaw_drive_gain * yaw_error * float(dt)
                    )
                    record.mount_yaw = clamp_angle_to_limits(proposed_yaw, lo, hi, previous_yaw)
                if relpose_np is not None and record.mount_eq_idx is not None:
                    global_eq = self._global_eq_idx(world, record.mount_eq_idx)
                    relpose_np[global_eq] = compose_mounted_weld_relpose(
                        record.host_anchor_local,
                        record.tool_anchor_local,
                        yaw_rad=record.mount_yaw,
                        mount_axis=record.mount_axis,
                    )
                    relpose_dirty = True
            elif record.mount_joint_dof_idx is not None and joint_f_np is not None:
                global_mount_dof = self._global_dof_idx(world, record.mount_joint_dof_idx)
                mount_rate = 0.0
                if joint_qd_np is not None and 0 <= global_mount_dof < joint_qd_np.shape[0]:
                    mount_rate = float(joint_qd_np[global_mount_dof])
                yaw_torque = soft_limit_torque(
                    pd_torque(
                        yaw_error,
                        mount_rate,
                        cfg.yaw_torque_gain,
                        cfg.yaw_damping,
                        cfg.max_yaw_torque,
                    ),
                    cfg.max_yaw_torque,
                )
                joint_f_np[global_mount_dof] = float(joint_f_np[global_mount_dof]) + yaw_torque
                joint_f_dirty = True

            pitch_spec = record.pitch_dof_spec
            if pitch_spec is not None and joint_f_np is not None:
                global_pitch_dof = self._global_dof_idx(world, pitch_spec.global_dof_idx)
                pitch_rate = 0.0
                if joint_qd_np is not None and 0 <= global_pitch_dof < joint_qd_np.shape[0]:
                    pitch_rate = float(joint_qd_np[global_pitch_dof])
                pitch_torque = soft_limit_torque(
                    pd_torque(
                        pitch_error,
                        pitch_rate,
                        cfg.pitch_torque_gain,
                        cfg.pitch_damping,
                        cfg.max_pitch_torque,
                    ),
                    cfg.max_pitch_torque,
                )
                joint_f_np[global_pitch_dof] = float(joint_f_np[global_pitch_dof]) + pitch_torque
                joint_f_dirty = True

        if relpose_dirty and self._model.equality_constraint_relpose is not None and relpose_np is not None:
            # Reuse numpy buffer; avoid allocating a fresh Warp array each frame.
            self._model.equality_constraint_relpose.assign(relpose_np)
            self._notify_solver()
        if joint_f_dirty and control is not None and joint_f_np is not None:
            control.joint_f.assign(joint_f_np)

    def get_attached_tool_pattern(self, host_player_index: int) -> Optional[str]:
        for record in self.records.values():
            if record.attached and record.host_player_index == host_player_index:
                return record.tool_key
        return None
