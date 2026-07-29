"""Load wheeled armored vehicle joint control config from template folder."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import newton
import yaml
from newton import JointTargetMode

from script.role.abilities.articulation_control_config.config_models import (
    JointParam,
    TaskParam,
)
from script.role.abilities.articulation_control_config.runtime_helpers import (
    RuntimeNominalsGpuSpec,
    build_runtime_nominals_gpu_spec_from_masks,
)

from .vehicle_body_model import (
    BodyMassSpec,
    RolloverSpec,
    SuspensionGainSpec,
    VehicleContactSpec,
    WheelMaterialSpec,
    WheelSpinGainSpec,
    apply_body_masses,
    apply_vehicle_collision_proxies,
    apply_vehicle_contact_properties,
    apply_wheel_shape_materials,
    compute_suspension_spec,
    is_suspension_joint_label,
    parse_body_mass_spec,
    parse_possess_offset,
    parse_rollover_spec,
    parse_suspension_gain_spec,
    parse_vehicle_contact_spec,
    parse_wheel_material_spec,
    parse_wheel_spin_gain_spec,
)
from .vehicle_drive import (
    DriveSpec,
    VehicleCommandInterface,
    build_vehicle_command_interface,
    parse_drive_spec,
)
from .vehicle_joint_model import (
    JointRole,
    classify_joint_dof,
    count_joint_label_dofs,
    normalize_joint_label,
    resolve_dof_param_for_view,
    resolve_dof_physics,
    set_active_suspension_spec,
    set_active_wheel_spin_spec,
    static_suspension_angle_rad,
)

VEHICLE_CONTROL_CONFIG_PATH = Path(__file__).resolve().parent / "control_configs.yaml"
VEHICLE_ROBOT_NAME = "wheeled_armored_vehicle_basic"
VEHICLE_DEFAULT_TASK = "differential_drive"

_TASK_CONFIG_CACHE: Dict[Tuple[str, str], "VehicleTaskConfig"] = {}


@dataclass
class VehicleTaskConfig:
    task_name: str
    soft_limit_factor: float
    root_pos: Tuple[float, float, float]
    root_rot: Tuple[float, float, float, float]
    joint_pos_overrides: Dict[str, float]
    non_rl_patterns: Tuple[str, ...]
    body_mass_spec: BodyMassSpec
    suspension_gain_spec: SuspensionGainSpec
    wheel_spin_gain_spec: WheelSpinGainSpec
    wheel_material_spec: WheelMaterialSpec
    contact_spec: VehicleContactSpec
    rollover_spec: RolloverSpec
    possess_offset: Tuple[float, float, float]
    drive_spec: DriveSpec
    human_control: Dict[str, Any]
    gravity: float = 9.81
    possess_anchor_name: Optional[str] = None
    possess_body_prim_suffix: Optional[str] = None
    possess_height_above_anchor: Optional[float] = None
    mount_host: Optional[Dict[str, Any]] = None

    @classmethod
    def from_yaml(
        cls,
        task_name: str = VEHICLE_DEFAULT_TASK,
        config_path: Optional[Path] = None,
    ) -> "VehicleTaskConfig":
        path = config_path or VEHICLE_CONTROL_CONFIG_PATH
        cache_key = (str(path), task_name, path.stat().st_mtime)
        cached = _TASK_CONFIG_CACHE.get(cache_key)
        if cached is not None:
            return cached

        with path.open("r", encoding="utf-8") as fh:
            raw_data = yaml.safe_load(fh) or {}

        robot_cfg = raw_data.get(VEHICLE_ROBOT_NAME, {})
        task_cfg = robot_cfg.get(task_name, {})
        if not isinstance(task_cfg, dict):
            raise KeyError(f"Task '{task_name}' not found in {path}")

        init_state = task_cfg.get("init_state", {})
        root_pos_raw = init_state.get("root_pos", [0.0, 0.0, 0.0])
        root_rot_raw = init_state.get("root_rot", [0.0, 0.0, 0.0, 1.0])

        body_mass_spec = parse_body_mass_spec(task_cfg.get("body_masses"))
        suspension_gain_spec = parse_suspension_gain_spec(task_cfg.get("suspension"))
        wheel_spin_gain_spec = parse_wheel_spin_gain_spec(task_cfg.get("wheel_spin"))
        wheel_material_spec = parse_wheel_material_spec(task_cfg.get("wheel_materials"))
        contact_spec = parse_vehicle_contact_spec(task_cfg.get("contact"))
        rollover_spec = parse_rollover_spec(task_cfg.get("rollover"))
        possess_offset = parse_possess_offset(task_cfg.get("possess_offset"))
        possess_anchor_name = task_cfg.get("possess_anchor_name")
        possess_height_above_anchor = task_cfg.get("possess_height_above_anchor")
        if possess_height_above_anchor is not None:
            possess_height_above_anchor = float(possess_height_above_anchor)
        possess_body_raw = task_cfg.get("possess_body_prim_suffix")
        possess_body_prim_suffix = str(possess_body_raw) if possess_body_raw else None
        gravity = float(task_cfg.get("gravity", 9.81))
        drive_spec = parse_drive_spec(task_cfg.get("drive"))
        human_control = dict(task_cfg.get("human_control") or {})
        mount_host = dict(task_cfg.get("mount_host") or {}) or None

        instance = cls(
            task_name=task_name,
            soft_limit_factor=float(task_cfg.get("soft_limit_factor", 0.95)),
            root_pos=tuple(float(v) for v in root_pos_raw),
            root_rot=tuple(float(v) for v in root_rot_raw),
            joint_pos_overrides=dict(task_cfg.get("joint_pos", {})),
            non_rl_patterns=tuple(task_cfg.get("non_rl_patterns", [])),
            body_mass_spec=body_mass_spec,
            suspension_gain_spec=suspension_gain_spec,
            wheel_spin_gain_spec=wheel_spin_gain_spec,
            wheel_material_spec=wheel_material_spec,
            contact_spec=contact_spec,
            rollover_spec=rollover_spec,
            possess_offset=possess_offset,
            drive_spec=drive_spec,
            human_control=human_control,
            gravity=gravity,
            possess_anchor_name=possess_anchor_name,
            possess_body_prim_suffix=possess_body_prim_suffix,
            possess_height_above_anchor=possess_height_above_anchor,
            mount_host=mount_host,
        )
        suspension_spec = compute_suspension_spec(
            body_mass_spec, suspension_gain_spec, gravity=gravity
        )
        set_active_suspension_spec(suspension_spec)
        from .vehicle_joint_model import DofPhysicsSpec

        spin_spec = wheel_spin_gain_spec
        set_active_wheel_spin_spec(
            DofPhysicsSpec(
                stiffness=0.0,
                damping=0.0,
                armature=float(spin_spec.armature),
                target_mode=JointTargetMode.VELOCITY,
                nominal=0.0,
                rl_controllable=True,
                action_scale=drive_spec.max_wheel_torque_nm,
            )
        )
        _TASK_CONFIG_CACHE[cache_key] = instance
        return instance

    def get_command_interface(self) -> VehicleCommandInterface:
        return build_vehicle_command_interface(
            drive=self.drive_spec,
            human_control=self.human_control,
        )

    def get_task_meta(self) -> TaskParam:
        return TaskParam(soft_limit_factor=self.soft_limit_factor)

    def _is_non_rl(self, label: str) -> bool:
        basename = normalize_joint_label(label)
        for pattern in self.non_rl_patterns:
            if re.search(pattern, basename) or re.search(pattern, label):
                return True
        return False

    def get_joint_param(self, label: str) -> JointParam:
        _, spec = resolve_dof_param_for_view(label, 0, self.joint_pos_overrides)
        non_rl = self._is_non_rl(label)
        if spec is None:
            return JointParam(scale=0.0, nominal=0.0, rl_controllable=False)
        return JointParam(
            scale=spec.action_scale,
            nominal=spec.nominal,
            rl_controllable=spec.rl_controllable and not non_rl,
            stiffness=spec.stiffness,
            kd=spec.damping,
        )

    def resolve_joint_arrays(
        self,
        joint_labels: List[str],
        default_qs: Optional[List[float]] = None,
    ) -> Tuple[List[float], List[float], List[float], List[float], List[int], List[int], float]:
        scales: List[float] = []
        nominals: List[float] = []
        limits_max: List[float] = []
        limits_min: List[float] = []
        rl_mask: List[int] = []
        rl_indices: List[int] = []

        label_occurrence: Dict[str, int] = {}
        label_dof_counts = count_joint_label_dofs(joint_labels)
        action_cursor = 0
        use_command_interface = hasattr(self, "get_command_interface")

        for i, label in enumerate(joint_labels):
            occ = label_occurrence.get(label, 0)
            label_occurrence[label] = occ + 1

            _, spec = resolve_dof_param_for_view(
                label,
                occ,
                self.joint_pos_overrides,
                label_dof_count=label_dof_counts.get(label, 1),
            )
            non_rl = self._is_non_rl(label)

            if spec is not None:
                scale = spec.action_scale
                nominal = spec.nominal
            elif default_qs is not None:
                scale = 0.0
                nominal = float(default_qs[i])
            else:
                scale = 0.0
                nominal = 0.0

            controllable = (
                spec is not None
                and spec.rl_controllable
                and not non_rl
                and scale > 0.0
            )
            scales.append(scale)
            nominals.append(nominal)
            rl_mask.append(1 if controllable else 0)
            if controllable and not use_command_interface:
                rl_indices.append(action_cursor)
                action_cursor += 1
            else:
                rl_indices.append(-1)
            limits_max.append(1e6)
            limits_min.append(-1e6)

        return scales, nominals, limits_max, limits_min, rl_mask, rl_indices, self.soft_limit_factor

    def build_runtime_nominals_gpu_spec(
        self,
        joint_labels: List[str],
    ) -> RuntimeNominalsGpuSpec | None:
        return build_runtime_nominals_gpu_spec_from_masks(
            joint_labels=joint_labels,
            passive_label_predicate=is_suspension_joint_label,
            upright_dot_threshold=self.rollover_spec.passive_suspension_upright_dot,
        )

    def apply_builder_physics_init(
        self,
        builder_env,
        start_q_idx: int,
        joint_start: int,
        joint_end: int,
    ) -> None:
        builder_env.joint_q[start_q_idx : start_q_idx + 3] = list(self.root_pos)
        builder_env.joint_q[start_q_idx + 3 : start_q_idx + 7] = list(self.root_rot)

        bodies_updated = apply_body_masses(builder_env, self.body_mass_spec)
        visual_copies, chassis_proxies, wheel_proxies, suspension_proxies = (
            apply_vehicle_collision_proxies(builder_env, self.contact_spec)
        )
        contacts_updated = apply_vehicle_contact_properties(
            builder_env, self.contact_spec
        )
        wheels_updated = apply_wheel_shape_materials(
            builder_env, self.wheel_material_spec
        )
        suspension_spec = compute_suspension_spec(
            self.body_mass_spec,
            self.suspension_gain_spec,
            gravity=self.gravity,
        )
        set_active_suspension_spec(suspension_spec)

        applied = 0
        spring_actuators = 0
        for joint_idx in range(joint_start, joint_end):
            if int(builder_env.joint_type[joint_idx]) == int(newton.JointType.FREE):
                continue

            label = str(builder_env.joint_label[joint_idx])
            q_start = int(builder_env.joint_q_start[joint_idx])
            qd_start = int(builder_env.joint_qd_start[joint_idx])
            qd_end = (
                int(builder_env.joint_qd_start[joint_idx + 1])
                if joint_idx + 1 < joint_end
                else builder_env.joint_dof_count
            )
            dof_count = qd_end - qd_start

            for local_dof, qd in enumerate(range(qd_start, qd_end)):
                spec = resolve_dof_physics(
                    label, local_dof, dof_count, self.joint_pos_overrides
                )
                if spec is None:
                    continue

                role = classify_joint_dof(label, local_dof, dof_count)
                if role == JointRole.SUSPENSION:
                    spring_spec = suspension_spec
                    target_rad = static_suspension_angle_rad(
                        label, self.suspension_gain_spec.target_angle_rad
                    )
                    max_torque = self.suspension_gain_spec.servo_max_torque_nm
                    stiffness_override = self.suspension_gain_spec.servo_stiffness_nm_rad
                    damping_override = self.suspension_gain_spec.servo_damping_nm_s_rad
                else:
                    spring_spec = None

                if spring_spec is not None:
                    ke = float(
                        spring_spec.stiffness
                        if stiffness_override is None
                        else stiffness_override
                    )
                    kd = float(
                        spring_spec.damping
                        if damping_override is None
                        else damping_override
                    )
                    builder_env.joint_q[q_start + local_dof] = target_rad
                    builder_env.joint_target_pos[qd] = target_rad
                    builder_env.joint_target_ke[qd] = ke
                    builder_env.joint_target_kd[qd] = kd
                    builder_env.joint_target_mode[qd] = int(JointTargetMode.POSITION)
                    builder_env.joint_effort_limit[qd] = float(max_torque)
                    spring_actuators += 1
                elif role == JointRole.WHEEL_SPIN:
                    builder_env.joint_target_ke[qd] = 0.0
                    builder_env.joint_target_kd[qd] = 0.0
                    builder_env.joint_target_mode[qd] = int(JointTargetMode.VELOCITY)
                    builder_env.joint_effort_limit[qd] = float(
                        self.drive_spec.max_wheel_torque_nm
                    )
                    builder_env.joint_armature[qd] = float(
                        self.wheel_spin_gain_spec.armature
                    )
                else:
                    builder_env.joint_target_ke[qd] = spec.stiffness
                    builder_env.joint_target_kd[qd] = spec.damping
                    builder_env.joint_target_mode[qd] = int(spec.target_mode)

                if role != JointRole.WHEEL_SPIN:
                    builder_env.joint_armature[qd] = spec.armature
                applied += 1

        corner_moment = (
            self.body_mass_spec.chassis_mass_kg
            * self.gravity
            / float(self.suspension_gain_spec.num_corners)
            * self.suspension_gain_spec.lever_arm_m
        )
        expected_residual_deg = (
            corner_moment / max(suspension_spec.stiffness, 1e-6) * 57.2957795
        )
        target_deg = self.suspension_gain_spec.target_angle_rad * 57.2957795
        print(
            f"[VehicleTaskConfig] Applied physics init: task={self.task_name}, "
            f"bodies_mass={bodies_updated}, wheel_shapes={wheels_updated}, "
            f"visual_copies={visual_copies}, chassis_proxies={chassis_proxies}, "
            f"wheel_proxies={wheel_proxies}, "
            f"susp_proxies={suspension_proxies}, contact_shapes={contacts_updated}, "
            f"contact_ke={self.contact_spec.ke:.1f}, "
            f"contact_kd={self.contact_spec.kd:.1f}, "
            f"contact_restitution={self.contact_spec.restitution:.2f}, "
            f"contact_margin={self.contact_spec.margin:.3f}, "
            f"use_collision_proxies={self.contact_spec.use_collision_proxies}, "
            f"suspension_target_deg={target_deg:.2f}, "
            f"suspension_ke={suspension_spec.stiffness:.1f}, "
            f"suspension_kd={suspension_spec.damping:.1f}, "
            f"suspension_max_torque={self.suspension_gain_spec.servo_max_torque_nm:.1f}, "
            f"corner_gravity_moment~={corner_moment:.1f} Nm, "
            f"expected_residual_past_target_deg~={expected_residual_deg:.2f}, "
            f"wheel_spin_direct_torque_max={self.drive_spec.max_wheel_torque_nm:.1f}, "
            f"wheel_spin_velocity_actuator=disabled, "
            f"spring_actuators={spring_actuators}, "
            f"joints={joint_end - joint_start}, actuated_dofs={applied}"
        )


def get_vehicle_task_config(
    task_name: str = VEHICLE_DEFAULT_TASK,
) -> VehicleTaskConfig:
    return VehicleTaskConfig.from_yaml(task_name=task_name)
