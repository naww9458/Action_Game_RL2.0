"""Runtime registry for tool mount joints (toggle on U press).

The registry only knows about the generic ``ToolAction`` interface — concrete
attached-tool behavior (e.g. turret aiming) is provided by the per-pattern
action stored on each record and is loaded lazily by the tool function registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import newton
import numpy as np
import warp as wp

from script.role.objects.tool_anchor import (
    anchor_within_mount_proximity,
    compose_body_snap_transform,
    compose_mounted_weld_relpose,
)
from script.simulate.tool_action import ToolAction


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
    tool_pattern: str = ""
    # Mount this tool automatically when the level starts (and after resets).
    start_attached: bool = False
    mount_joint_idx: Optional[int] = None
    mount_eq_idx: Optional[int] = None
    mount_joint_dof_idx: Optional[int] = None
    mount_joint_coord_idx: Optional[int] = None
    mount_joint_type: str = "revolute"
    uses_weld_fallback: bool = False
    slot_index: int = 0
    mount_yaw: float = 0.0
    # Virtual-torque yaw rate for the MuJoCo weld fallback (aim.py weld path).
    mount_yaw_rate: float = 0.0
    attached: bool = False
    prompt_visible: bool = False
    # Generic attached-tool behavior (e.g. turret "aim" action).
    action: Optional[ToolAction] = None


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
        self._physics_manager = None
        self._num_objects_env = 1

    @property
    def model(self) -> Optional[newton.Model]:
        return self._model

    @property
    def uses_mujoco_weld(self) -> bool:
        return self._uses_mujoco_weld

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
        num_objects_env: int = 1,
    ) -> None:
        self._model = model
        self._device = device
        self._solver = solver
        self.num_env = max(1, int(num_env))
        self._num_objects_env = max(1, int(num_objects_env))
        self._num_joint_dof_env = int(num_joint_dof_env)
        self._num_rigid_bodies_env = int(num_rigid_bodies_env)
        self._num_joint_count_env = max(1, int(model.joint_count // self.num_env))
        self._num_eq_count_env = max(
            0,
            int(
                getattr(model.mujoco, "equality_constraint_count", 0) // self.num_env
            ),
        )
        if model.joint_coord_count > 0 and self.num_env > 0:
            self._num_joint_coord_env = max(1, int(model.joint_coord_count // self.num_env))
        else:
            self._num_joint_coord_env = 0

        solver = solver_type.lower()
        self._solver_supports_joint_toggle = solver in {"xpbd", "vbd"}
        self._uses_mujoco_weld = solver == "mujoco" or any(
            r.uses_weld_fallback for r in self.records.values()
        )

        for record in self.records.values():
            if record.action is not None:
                record.action.bind_model(self, record)

    def attach_physics_pre_substep(self, physics_manager) -> None:
        """Store physics manager reference for per-frame attached-tool driving."""
        self._physics_manager = physics_manager

    def drive_attached_tools_frame(
        self,
        *,
        camera_yaw: float = 0.0,
        camera_pitch: float = 0.0,
        mouse_buttons=None,
        host_role_object_id: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> None:
        """Run attached-tool actions once per simulation frame (CUDA-graph safe).

        Must be called from Python before ``physics_manager.simulate()`` / graph
        launch — Warp CUDA graphs do not replay Python pre-substep callbacks.
        """
        pm = self._physics_manager
        if pm is None or self._model is None:
            return
        if not any(record.attached for record in self.records.values()):
            return

        state = pm.state_0
        frame_dt = float(dt if dt is not None else getattr(pm, "frame_dt", 1.0 / 50.0))
        body_q_np = state.body_q.numpy()

        if host_role_object_id is not None:
            host_id = int(host_role_object_id)
            world = host_id // self._num_objects_env
            self.apply_attached_actions(
                state.body_q,
                state.body_qd,
                pm.control,
                camera_yaw=float(camera_yaw),
                camera_pitch=float(camera_pitch),
                world=world,
                dt=frame_dt,
                host_role_object_id=host_id,
                body_q_np=body_q_np,
                joint_q=state.joint_q,
                joint_qd=state.joint_qd,
                mouse_buttons=mouse_buttons,
                body_f=state.body_f,
            )
            return

        for record in self.records.values():
            if not record.attached or record.action is None:
                continue
            host_id = int(record.host_role_object_id)
            world = host_id // self._num_objects_env
            self.apply_attached_actions(
                state.body_q,
                state.body_qd,
                pm.control,
                camera_yaw=0.0,
                camera_pitch=0.0,
                world=world,
                dt=frame_dt,
                host_role_object_id=host_id,
                body_q_np=body_q_np,
                joint_q=state.joint_q,
                joint_qd=state.joint_qd,
                mouse_buttons=None,
                body_f=state.body_f,
            )

    def global_body_idx(self, world: int, local_body_idx: int) -> int:
        return world * self._num_rigid_bodies_env + local_body_idx

    def global_joint_idx(self, world: int, local_joint_idx: int) -> int:
        return world * self._num_joint_count_env + local_joint_idx

    def global_eq_idx(self, world: int, local_eq_idx: int) -> int:
        return world * self._num_eq_count_env + local_eq_idx

    def global_dof_idx(self, world: int, local_dof_idx: int) -> int:
        return world * self._num_joint_dof_env + local_dof_idx

    def global_coord_idx(self, world: int, local_coord_idx: int) -> int:
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

            host_body = body_q_np[self.global_body_idx(world, record.host_body_idx)]
            tool_body = body_q_np[self.global_body_idx(world, record.tool_body_idx)]
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

    def get_attached_record(self, host_role_object_id: int) -> Optional[ToolMountRecord]:
        host_id = int(host_role_object_id)
        for record in self.records.values():
            if record.attached and record.host_role_object_id == host_id:
                return record
        return None

    def clear_rl_control_for_host(self, host_role_object_id: int) -> None:
        record = self.get_attached_record(host_role_object_id)
        if record is None or record.action is None:
            return
        record.action.clear_rl_control()

    def apply_rl_control_for_host(
        self,
        host_role_object_id: int,
        values: Sequence[float],
    ) -> bool:
        record = self.get_attached_record(host_role_object_id)
        if record is None or record.action is None:
            return False
        record.action.set_rl_control(values)
        return True

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
                global_body = self.global_body_idx(world, local_body)
                body_qd_np[global_body, 0:6] = 0.0
            body_qd.assign(body_qd_np)

        if body_f is not None:
            body_f_np = body_f.numpy()
            for local_body in unique_bodies:
                global_body = self.global_body_idx(world, local_body)
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
                global_dof = self.global_dof_idx(world, local_dof)
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
        force: bool = False,
    ) -> bool:
        record = self.records.get(tool_key)
        if record is None or self._model is None:
            return False
        if record.attached and not force:
            return False

        host_global = self.global_body_idx(world, record.host_body_idx)
        tool_root_global = self.global_body_idx(world, record.tool_root_body_idx)

        host_xform = body_q.numpy()[host_global]
        record.mount_yaw = 0.0
        record.mount_yaw_rate = 0.0
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
            global_free = self.global_joint_idx(world, int(record.tool_free_joint_idx))
            q_start = int(self._model.joint_q_start.numpy()[global_free])
            jq_np[q_start : q_start + 7] = desired_tool
            joint_q.assign(jq_np)

        if joint_q is not None and record.mount_joint_coord_idx is not None:
            jq_np = joint_q.numpy()
            global_coord = self.global_coord_idx(world, record.mount_joint_coord_idx)
            jq_np[global_coord] = 0.0
            joint_q.assign(jq_np)

        self._set_mount_active(record, world=world, active=True)
        if record.action is not None:
            record.action.on_attach(self, record, world=world)
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
        if record.action is not None:
            record.action.on_detach(self, record, world=world)
        record.attached = False
        record.mount_yaw = 0.0
        record.mount_yaw_rate = 0.0
        return True

    def attach_start_attached(
        self,
        body_q: wp.array,
        body_qd: wp.array,
        joint_q: wp.array,
        *,
        worlds: Optional[Sequence[int]] = None,
        body_f: wp.array | None = None,
        joint_qd: wp.array | None = None,
        body_q_prev: wp.array | None = None,
    ) -> int:
        """Enable mounts for every tool configured with ``start_attached: true``.

        Snaps each tool to its host in the given worlds (default: all worlds)
        and enables the mount joint/equality constraint, so the environment
        begins with the tool already attached. Called right after the initial
        env reset and after each episode reset.
        """
        if self._model is None:
            return 0
        if worlds is None:
            worlds = range(self.num_env)

        count = 0
        for record in self.records.values():
            if not record.start_attached:
                continue
            for world in worlds:
                if self.enable_attachment(
                    record.tool_key,
                    body_q,
                    body_qd,
                    joint_q,
                    world=int(world),
                    body_f=body_f,
                    joint_qd=joint_qd,
                    body_q_prev=body_q_prev,
                    force=True,
                ):
                    count += 1
        return count

    def _mount_active_in_world(self, record: ToolMountRecord, world: int) -> bool:
        """Whether the record's mount joint / weld constraint is currently enabled
        in ``world`` (reads the live model array)."""
        model = self._model
        if model is None:
            return False
        if record.uses_weld_fallback and record.mount_eq_idx is not None:
            eq_enabled = model.mujoco.equality_constraint_enabled
            if eq_enabled is None:
                return False
            return bool(eq_enabled.numpy()[self.global_eq_idx(world, record.mount_eq_idx)])
        if record.mount_joint_idx is None:
            return False
        joint_enabled = model.joint_enabled
        if joint_enabled is None:
            return False
        return bool(joint_enabled.numpy()[self.global_joint_idx(world, record.mount_joint_idx)])

    def reset_attachments(
        self,
        *,
        worlds: Optional[Sequence[int]] = None,
    ) -> int:
        """Detach every tool in the given worlds (default: all worlds).

        Disables each mount joint / equality constraint and clears the
        per-record attached state, so an env reset returns the world to its
        pristine, unattached configuration. Tools configured with
        ``start_attached: true`` are re-attached right after by the caller via
        :meth:`attach_start_attached`.

        Returns the number of records that were attached and got detached.
        """
        if self._model is None:
            return 0
        if worlds is None:
            worlds = range(self.num_env)
        worlds = [int(w) for w in worlds]

        detached = 0
        for record in self.records.values():
            was_attached = record.attached
            detached_here = False
            for world in worlds:
                if not self._mount_active_in_world(record, world):
                    continue
                self._set_mount_active(record, world=world, active=False)
                if record.action is not None:
                    try:
                        record.action.on_detach(self, record, world=world)
                    except Exception:
                        pass
                detached += 1
                detached_here = True
            if was_attached and detached_here:
                # The attached flag is per-record, not per-world: only clear it
                # once the mount is inactive in every world, otherwise a tool
                # still mounted in a non-reset world would be orphaned.
                still_active = any(
                    self._mount_active_in_world(record, w) for w in range(self.num_env)
                )
                if not still_active:
                    record.attached = False
            record.mount_yaw = 0.0
            record.mount_yaw_rate = 0.0
            record.prompt_visible = False
        return detached

    def notify_joint_dof_properties(self) -> None:
        if self._solver is None:
            return
        try:
            from newton import ModelFlags
        except ImportError:
            return
        try:
            self._solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        except Exception:
            pass

    def notify_solver(self) -> None:
        if self._solver is None:
            return
        try:
            from newton import ModelFlags
        except ImportError:
            return
        try:
            self._solver.notify_model_changed(ModelFlags.CONSTRAINT_PROPERTIES)
        except Exception:
            pass

    def _set_mount_active(self, record: ToolMountRecord, world: int, active: bool) -> None:
        model = self._model
        if model is None:
            return

        if record.uses_weld_fallback and record.mount_eq_idx is not None:
            eq_enabled = model.mujoco.equality_constraint_enabled
            if eq_enabled is not None:
                enabled_np = eq_enabled.numpy()
                global_eq = self.global_eq_idx(world, record.mount_eq_idx)
                enabled_np[global_eq] = bool(active)
                eq_enabled.assign(enabled_np)

                if active:
                    relpose_arr = model.mujoco.equality_constraint_relpose
                    if relpose_arr is not None:
                        rel_np = relpose_arr.numpy()
                        rel_np[global_eq] = compose_mounted_weld_relpose(
                            record.host_anchor_local,
                            record.tool_anchor_local,
                            yaw_rad=record.mount_yaw,
                            mount_axis=record.mount_axis,
                        )
                        relpose_arr.assign(rel_np)
            self.notify_solver()
            return

        if record.mount_joint_idx is None:
            return

        joint_enabled = model.joint_enabled
        if joint_enabled is None:
            return

        enabled_np = joint_enabled.numpy()
        global_mount = self.global_joint_idx(world, record.mount_joint_idx)
        enabled_np[global_mount] = bool(active)

        if record.tool_free_joint_idx is not None and self._solver_supports_joint_toggle:
            global_free = self.global_joint_idx(world, record.tool_free_joint_idx)
            enabled_np[global_free] = not bool(active)

        joint_enabled.assign(enabled_np)
        self.notify_solver()

    def apply_attached_actions(
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
        joint_q: wp.array | None = None,
        joint_qd: wp.array | None = None,
        mouse_buttons=None,
        body_f=None,
    ) -> None:
        """Forward per-frame camera input to every attached tool action."""
        for record in self.records.values():
            action = record.action
            if action is None or not record.attached:
                continue
            if (
                host_role_object_id is not None
                and record.host_role_object_id != host_role_object_id
            ):
                continue
            record_world = int(record.host_role_object_id) // self._num_objects_env
            action.step(
                self,
                record,
                world=record_world,
                dt=dt,
                camera_yaw=camera_yaw,
                camera_pitch=camera_pitch,
                host_role_object_id=host_role_object_id,
                body_q=body_q,
                body_qd=body_qd,
                control=control,
                body_q_np=body_q_np,
                joint_q=joint_q,
                joint_qd=joint_qd,
                mouse_buttons=mouse_buttons,
                body_f=body_f,
            )

    def get_attached_tool_pattern(self, host_player_index: int) -> Optional[str]:
        for record in self.records.values():
            if record.attached and record.host_player_index == host_player_index:
                return record.tool_key
        return None

    def get_tool_forward_local(self, tool_role_object_id: int) -> Optional[Tuple[float, float, float]]:
        """Tool-local forward direction (e.g. debug geometry) from its action."""
        for record in self.records.values():
            if record.tool_role_object_id == int(tool_role_object_id) and record.action is not None:
                return tuple(float(v) for v in record.action.forward_local())
        return None
