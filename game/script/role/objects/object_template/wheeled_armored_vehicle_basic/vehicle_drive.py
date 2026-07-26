"""Differential-drive command expansion for wheeled_armored_vehicle_basic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .vehicle_joint_model import (
    JointRole,
    count_joint_label_dofs,
    is_left_side,
    resolve_dof_param_for_view,
)


@dataclass(frozen=True)
class DriveSpec:
    max_wheel_speed_rad_s: float = 40.0
    max_wheel_torque_nm: float = 90000.0
    velocity_control_gain_nm_per_rad_s: float = 800.0
    left_spin_sign: float = 1.0
    right_spin_sign: float = 1.0


def _mix_tank_drive(throttle: float, steer: float) -> Tuple[float, float]:
    """Map throttle/steer ∈ [-1, 1] to per-side torque commands.

    steer follows keyboard binding A - D:
      W/S only → both sides ±1
      A/D only → opposite sides (pivot)
      W/S + A → inside track 0, outside ±1
      W/S + D → outside track ±1, inside 0
    """
    if throttle == 0.0:
        return -steer, steer
    if steer == 0.0:
        return throttle, throttle
    if steer > 0.0:
        return 0.0, throttle
    return throttle, 0.0


@dataclass(frozen=True)
class VehicleCommandInterface:
    command_dim: int
    binding_names: Tuple[str, ...]
    human_control: Dict[str, Any]
    drive: DriveSpec

    @property
    def uses_direct_joint_torque(self) -> bool:
        """Wheel speed targets are tracked by direct torques in control.joint_f."""
        return True

    @property
    def direct_torque_limit(self) -> float:
        return self.drive.max_wheel_torque_nm

    @property
    def direct_velocity_gain(self) -> float:
        return self.drive.velocity_control_gain_nm_per_rad_s

    def expand_commands(
        self,
        throttle: float,
        steer: float,
        joint_labels: Sequence[str],
        rl_mask: Sequence[int],
        joint_pos_overrides: Dict[str, float] | None = None,
    ) -> List[float]:
        overrides = joint_pos_overrides or {}
        throttle = max(-1.0, min(1.0, float(throttle)))
        # Keyboard bindings: steer = A - D (positive = left).
        steer = max(-1.0, min(1.0, float(steer)))

        left_cmd, right_cmd = _mix_tank_drive(throttle, steer)
        target_omega_left = (
            left_cmd
            * self.drive.max_wheel_speed_rad_s
            * self.drive.left_spin_sign
        )
        target_omega_right = (
            right_cmd
            * self.drive.max_wheel_speed_rad_s
            * self.drive.right_spin_sign
        )

        targets = [0.0] * len(joint_labels)
        label_occurrence: Dict[str, int] = {}
        label_dof_counts = count_joint_label_dofs(list(joint_labels))
        for dof_idx, label in enumerate(joint_labels):
            occ = label_occurrence.get(label, 0)
            label_occurrence[label] = occ + 1

            if dof_idx >= len(rl_mask) or rl_mask[dof_idx] == 0:
                continue

            role, _ = resolve_dof_param_for_view(
                label,
                occ,
                overrides,
                label_dof_count=label_dof_counts.get(label, 1),
            )
            if role != JointRole.WHEEL_SPIN:
                continue

            if is_left_side(label):
                targets[dof_idx] = target_omega_left
            else:
                targets[dof_idx] = target_omega_right

        return targets


def parse_drive_spec(raw: Dict[str, Any] | None) -> DriveSpec:
    raw = raw or {}
    return DriveSpec(
        max_wheel_speed_rad_s=float(raw.get("max_wheel_speed_rad_s", 40.0)),
        max_wheel_torque_nm=float(raw.get("max_wheel_torque_nm", 90000.0)),
        velocity_control_gain_nm_per_rad_s=float(
            raw.get("velocity_control_gain_nm_per_rad_s", 800.0)
        ),
        left_spin_sign=float(raw.get("left_spin_sign", 1.0)),
        right_spin_sign=float(raw.get("right_spin_sign", 1.0)),
    )


def build_vehicle_command_interface(
    drive: DriveSpec,
    human_control: Dict[str, Any] | None,
) -> VehicleCommandInterface:
    return VehicleCommandInterface(
        command_dim=2,
        binding_names=("throttle", "steer"),
        human_control=dict(human_control or {}),
        drive=drive,
    )
