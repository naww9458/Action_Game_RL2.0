"""Joint physics specs for wheeled_armored_vehicle_basic (USD articulation)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from newton import JointTargetMode


class JointRole(Enum):
    FREE = auto()
    SUSPENSION = auto()
    WHEEL_SPIN = auto()
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

# Populated from control_configs.yaml via VehicleTaskConfig.
ACTIVE_SUSPENSION_SPEC: DofPhysicsSpec = SUSPENSION_SPEC


def set_active_suspension_spec(spec: DofPhysicsSpec) -> None:
    global ACTIVE_SUSPENSION_SPEC
    ACTIVE_SUSPENSION_SPEC = spec


WHEEL_SPIN_SPEC = DofPhysicsSpec(
    stiffness=0.0,
    damping=0.0,
    armature=0.01,
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


def classify_joint_dof(label: str, local_dof: int, dof_count: int) -> JointRole:
    basename = normalize_joint_label(label)
    lower = label.lower()

    if "vb_susp" in basename and "revolute" in basename:
        return JointRole.SUSPENSION

    if "d6joint" in basename or "wheels_" in lower:
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


def count_joint_label_dofs(joint_labels: List[str]) -> Dict[str, int]:
    """How many articulation-view rows each joint label contributes."""
    counts: Dict[str, int] = {}
    for label in joint_labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def resolve_dof_param_for_view(
    label: str,
    occurrence_index: int,
    joint_pos_overrides: Dict[str, float],
    label_dof_count: Optional[int] = None,
) -> Tuple[JointRole, Optional[DofPhysicsSpec]]:
    """Map articulation-view DOF rows (same label may repeat) to physics specs."""
    lower = label.lower()
    basename = normalize_joint_label(label)

    if "vb_susp" in basename and "revolute" in basename:
        spec = resolve_dof_physics(label, 0, 1, joint_pos_overrides)
        return JointRole.SUSPENSION, spec

    if "d6joint" in basename or "wheels_" in lower:
        spec = resolve_dof_physics(label, 0, 1, joint_pos_overrides)
        return JointRole.WHEEL_SPIN, spec

    return JointRole.UNKNOWN, None
