"""Differential-drive command expansion for wheeled_armored_vehicle_basic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .vehicle_joint_model import (
    JointRole,
    is_left_side,
    resolve_dof_param_for_view,
)


@dataclass(frozen=True)
class DriveSpec:
    max_linear_speed_m_s: float = 8.0
    max_yaw_rate_rad_s: float = 1.2
    wheel_radius_m: float = 0.35
    track_width_m: float = 2.2
    left_spin_sign: float = 1.0
    right_spin_sign: float = 1.0


@dataclass(frozen=True)
class VehicleCommandInterface:
    command_dim: int
    binding_names: Tuple[str, ...]
    human_control: Dict[str, Any]
    drive: DriveSpec

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
        steer = max(-1.0, min(1.0, float(steer)))

        linear_v = throttle * self.drive.max_linear_speed_m_s
        # Positive steer (D) → clockwise/right turn (negative yaw in Z-up).
        yaw_rate = -steer * self.drive.max_yaw_rate_rad_s
        half_track = self.drive.track_width_m * 0.5
        v_left = linear_v - yaw_rate * half_track
        v_right = linear_v + yaw_rate * half_track
        omega_left = (
            v_left / max(self.drive.wheel_radius_m, 1e-6) * self.drive.left_spin_sign
        )
        omega_right = (
            v_right / max(self.drive.wheel_radius_m, 1e-6) * self.drive.right_spin_sign
        )

        targets = [0.0] * len(joint_labels)
        label_occurrence: Dict[str, int] = {}
        for dof_idx, label in enumerate(joint_labels):
            # Always advance occurrence for repeated joint labels (hinge then spin),
            # even when the DOF is masked out — otherwise right-side spin is
            # misclassified as hinge and never receives drive targets.
            occ = label_occurrence.get(label, 0)
            label_occurrence[label] = occ + 1

            if dof_idx >= len(rl_mask) or rl_mask[dof_idx] == 0:
                continue

            role, _ = resolve_dof_param_for_view(label, occ, overrides)
            if role != JointRole.WHEEL_SPIN:
                continue

            if is_left_side(label):
                targets[dof_idx] = omega_left
            else:
                targets[dof_idx] = omega_right

        return targets


def parse_drive_spec(raw: Dict[str, Any] | None) -> DriveSpec:
    raw = raw or {}
    return DriveSpec(
        max_linear_speed_m_s=float(raw.get("max_linear_speed_m_s", 8.0)),
        max_yaw_rate_rad_s=float(raw.get("max_yaw_rate_rad_s", 1.2)),
        wheel_radius_m=float(raw.get("wheel_radius_m", 0.35)),
        track_width_m=float(raw.get("track_width_m", 2.2)),
        left_spin_sign=float(raw.get("left_spin_sign", 1.0)),
        right_spin_sign=float(raw.get("right_spin_sign", 1.0)),
    )


def compute_max_spin_omega(drive: DriveSpec) -> float:
    """Upper bound on |wheel spin omega| for RL mask scaling."""
    half_track = drive.track_width_m * 0.5
    yaw_term = drive.max_yaw_rate_rad_s * half_track
    v_max = drive.max_linear_speed_m_s
    side_speeds = (
        abs(v_max + yaw_term),
        abs(v_max - yaw_term),
        abs(-v_max + yaw_term),
        abs(-v_max - yaw_term),
    )
    return max(side_speeds) / max(drive.wheel_radius_m, 1e-6)


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
