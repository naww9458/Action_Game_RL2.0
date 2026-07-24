"""Joint physics specs for wheeled_armored_vehicle_basic (USD articulation)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Tuple

from newton import JointTargetMode


class JointRole(Enum):
    FREE = auto()
    SUSPENSION = auto()
    WHEEL_SPIN = auto()
    WHEEL_HINGE = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class DofPhysicsSpec:
    stiffness: float
    damping: float
    armature: float
    target_mode: JointTargetMode
    nominal: float = 0.0
    rl_controllable: bool = False
    action_scale: float = 0.0


SUSPENSION_SPEC = DofPhysicsSpec(
    stiffness=800.0,
    damping=40.0,
    armature=0.05,
    target_mode=JointTargetMode.POSITION,
    nominal=0.0,
)

WHEEL_HINGE_SPEC = DofPhysicsSpec(
    stiffness=5000.0,
    damping=200.0,
    armature=0.02,
    target_mode=JointTargetMode.POSITION,
    nominal=0.0,
)

# Populated from control_configs.yaml via VehicleTaskConfig.
ACTIVE_SUSPENSION_SPEC: DofPhysicsSpec = SUSPENSION_SPEC
ACTIVE_WHEEL_HINGE_SPEC: DofPhysicsSpec = WHEEL_HINGE_SPEC


def set_active_suspension_spec(spec: DofPhysicsSpec) -> None:
    global ACTIVE_SUSPENSION_SPEC
    ACTIVE_SUSPENSION_SPEC = spec


def set_active_wheel_hinge_spec(spec: DofPhysicsSpec) -> None:
    global ACTIVE_WHEEL_HINGE_SPEC
    ACTIVE_WHEEL_HINGE_SPEC = spec


WHEEL_SPIN_SPEC = DofPhysicsSpec(
    stiffness=0.0,
    damping=8.0,
    armature=0.02,
    target_mode=JointTargetMode.VELOCITY,
    nominal=0.0,
    rl_controllable=True,
    action_scale=0.0,
)

ACTIVE_WHEEL_SPIN_SPEC: DofPhysicsSpec = WHEEL_SPIN_SPEC


def set_active_wheel_spin_spec(spec: DofPhysicsSpec) -> None:
    global ACTIVE_WHEEL_SPIN_SPEC
    ACTIVE_WHEEL_SPIN_SPEC = spec


def normalize_joint_label(label: str) -> str:
    text = str(label).strip().lower()
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text


def is_left_side(label: str) -> bool:
    lower = str(label).lower()
    return bool(re.search(r"(?:susp|wheels)_l[0-9]", lower))


def static_suspension_angle_rad(label: str, sag_rad: float) -> float:
    """Equilibrium droop: left arm negative X, right arm positive X."""
    magnitude = abs(float(sag_rad))
    return -magnitude if is_left_side(label) else magnitude


def static_wheel_hinge_angle_rad(label: str, lift_rad: float) -> float:
    """Equilibrium wheel tilt up: left negative rotX, right positive rotX (USD axis)."""
    magnitude = abs(float(lift_rad))
    return -magnitude if is_left_side(label) else magnitude


def equilibrium_wheel_hinge_angle_rad(
    label: str,
    body_spec: BodyMassSpec,
    gain_spec: WheelHingeGainSpec,
    gravity: float = 9.81,
) -> float:
    """Gravity-balanced hinge angle for the current wheel-hinge stiffness."""
    load_n = body_spec.wheel_mass_kg * gravity
    ke = (
        load_n
        * gain_spec.lever_arm_m
        / max(gain_spec.nominal_lift_rad, 1e-4)
        * gain_spec.stiffness_multiplier
    )
    if ke <= 1e-3:
        return static_wheel_hinge_angle_rad(label, gain_spec.nominal_lift_rad)
    theta = load_n * gain_spec.lever_arm_m / ke
    sign = -1.0 if is_left_side(label) else 1.0
    return sign * theta


def classify_joint_dof(label: str, local_dof: int, dof_count: int) -> JointRole:
    basename = normalize_joint_label(label)
    lower = label.lower()

    if "vb_susp" in basename and "revolute" in basename:
        return JointRole.SUSPENSION

    if "d6joint" in basename or "wheels_" in lower:
        if dof_count > 1 and local_dof == 0:
            return JointRole.WHEEL_HINGE
        return JointRole.WHEEL_SPIN

    return JointRole.UNKNOWN


def resolve_dof_physics(
    label: str,
    local_dof: int,
    dof_count: int,
    joint_pos_overrides: Dict[str, float],
) -> Optional[DofPhysicsSpec]:
    role = classify_joint_dof(label, local_dof, dof_count)
    basename = normalize_joint_label(label)

    if role == JointRole.SUSPENSION:
        nominal = _resolve_nominal(basename, label, joint_pos_overrides, 0.0)
        return DofPhysicsSpec(
            stiffness=ACTIVE_SUSPENSION_SPEC.stiffness,
            damping=ACTIVE_SUSPENSION_SPEC.damping,
            armature=ACTIVE_SUSPENSION_SPEC.armature,
            target_mode=ACTIVE_SUSPENSION_SPEC.target_mode,
            nominal=nominal,
        )

    if role == JointRole.WHEEL_HINGE:
        nominal = _resolve_nominal(basename, label, joint_pos_overrides, 0.0)
        return DofPhysicsSpec(
            stiffness=ACTIVE_WHEEL_HINGE_SPEC.stiffness,
            damping=ACTIVE_WHEEL_HINGE_SPEC.damping,
            armature=ACTIVE_WHEEL_HINGE_SPEC.armature,
            target_mode=ACTIVE_WHEEL_HINGE_SPEC.target_mode,
            nominal=nominal,
        )

    if role == JointRole.WHEEL_SPIN:
        return ACTIVE_WHEEL_SPIN_SPEC

    return None


def _resolve_nominal(
    basename: str,
    full_label: str,
    overrides: Dict[str, float],
    default: float,
) -> float:
    for pattern, value in overrides.items():
        if re.search(pattern, basename) or re.search(pattern, full_label):
            return float(value)
    return default


def resolve_dof_param_for_view(
    label: str,
    occurrence_index: int,
    joint_pos_overrides: Dict[str, float],
) -> Tuple[JointRole, Optional[DofPhysicsSpec]]:
    """Map articulation-view DOF rows (same label may repeat) to physics specs."""
    lower = label.lower()
    basename = normalize_joint_label(label)

    if "vb_susp" in basename and "revolute" in basename:
        spec = resolve_dof_physics(label, 0, 1, joint_pos_overrides)
        return JointRole.SUSPENSION, spec

    if "d6joint" in basename or "wheels_" in lower:
        if is_left_side(label):
            spec = resolve_dof_physics(label, 0, 1, joint_pos_overrides)
            return JointRole.WHEEL_SPIN, spec
        local_dof = occurrence_index % 2
        spec = resolve_dof_physics(label, local_dof, 2, joint_pos_overrides)
        role = classify_joint_dof(label, local_dof, 2)
        return role, spec

    return JointRole.UNKNOWN, None
