# This file contains code adapted from:
# https://github.com/mujocolab/mjlab
#
# Modified for Action_Game_RL.
#
# The original project is licensed under the Apache License 2.0.

"""Unitree G1 actuator motor model, mjlab action scales, and velocity task constants."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .mjlab_utils_actuator import (
    ElectricActuator,
    reflected_inertia_from_two_stage_planetary,
)

##
# Motor specs (from Unitree).
##

ROTOR_INERTIAS_5020 = (
    0.139e-4,
    0.017e-4,
    0.169e-4,
)
GEARS_5020 = (
    1,
    1 + (46 / 18),
    1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
    0.489e-4,
    0.098e-4,
    0.533e-4,
)
GEARS_7520_14 = (
    1,
    4.5,
    1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
    0.489e-4,
    0.109e-4,
    0.738e-4,
)
GEARS_7520_22 = (
    1,
    4.5,
    5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
    0.068e-4,
    0.0,
    0.0,
)
GEARS_4010 = (
    1,
    5,
    5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
    reflected_inertia=ARMATURE_5020,
    velocity_limit=37.0,
    effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
    reflected_inertia=ARMATURE_7520_14,
    velocity_limit=32.0,
    effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
    reflected_inertia=ARMATURE_7520_22,
    velocity_limit=20.0,
    effort_limit=139.0,
)
ACTUATOR_4010 = ElectricActuator(
    reflected_inertia=ARMATURE_4010,
    velocity_limit=22.0,
    effort_limit=5.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

_NON_RL_PATTERNS = frozenset({"finger", "thumb", "hand"})
_NON_RL_SCALE_KEYS = frozenset({"finger", "thumb", "hand"})

# Hand joints present in g1_29dof_with_hand USD but absent from mjlab.
# Defaults match Action_Game_RL_Assets/assets/external_sources/newton-assets-main/unitree_g1/rl_policies/g1_29dof.yaml.
HAND_JOINT_NAMES: frozenset[str] = frozenset({
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
})

KNEES_BENT_JOINT_POS: Dict[str, float] = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

HOME_JOINT_POS: Dict[str, float] = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
}

G1_INIT_STATES: Dict[str, Dict[str, object]] = {
    "knees_bent": {
        "root_pos": (0.0, 0.0, 0.76),
        "root_rot": (0.0, 0.0, 0.7071, 0.7071),
        "joint_pos": KNEES_BENT_JOINT_POS,
    },
    "home": {
        "root_pos": (0.0, 0.0, 0.783675),
        "root_rot": (0.0, 0.0, 0.7071, 0.7071),
        "joint_pos": HOME_JOINT_POS,
    },
}


@dataclass(frozen=True)
class JointPhysics:
    stiffness: float
    damping: float
    armature: float
    effort_limit: float
    scale: float


HAND_JOINT_PHYSICS = JointPhysics(
    stiffness=10.0,
    damping=2.0,
    armature=0.1,
    effort_limit=0.0,
    scale=0.0,
)


def configure_hand_joints(
    names: Optional[Sequence[str]] = None,
    stiffness: float = 10.0,
    damping: float = 2.0,
    armature: float = 0.1,
    nominal: float = 0.0,
) -> None:
    """Override hand joint physics (defaults from g1_29dof.yaml)."""
    global HAND_JOINT_NAMES, HAND_JOINT_PHYSICS, HAND_JOINT_NOMINAL
    if names is not None:
        HAND_JOINT_NAMES = frozenset(names)
    HAND_JOINT_PHYSICS = JointPhysics(
        stiffness=stiffness,
        damping=damping,
        armature=armature,
        effort_limit=0.0,
        scale=0.0,
    )
    HAND_JOINT_NOMINAL = nominal


HAND_JOINT_NOMINAL = 0.0


def mjlab_action_scale(effort_limit: float, stiffness: float) -> float:
    return 0.25 * effort_limit / stiffness


def _build_actuator_physics_specs() -> List[Tuple[Sequence[str], float, float, float, float]]:
    return [
        (
            (
                ".*_elbow_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_wrist_roll_joint",
            ),
            STIFFNESS_5020,
            DAMPING_5020,
            ARMATURE_5020,
            ACTUATOR_5020.effort_limit,
        ),
        (
            (".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
            STIFFNESS_7520_14,
            DAMPING_7520_14,
            ARMATURE_7520_14,
            ACTUATOR_7520_14.effort_limit,
        ),
        (
            (".*_hip_roll_joint", ".*_knee_joint"),
            STIFFNESS_7520_22,
            DAMPING_7520_22,
            ARMATURE_7520_22,
            ACTUATOR_7520_22.effort_limit,
        ),
        (
            (".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
            STIFFNESS_4010,
            DAMPING_4010,
            ARMATURE_4010,
            ACTUATOR_4010.effort_limit,
        ),
        (
            ("waist_pitch_joint", "waist_roll_joint"),
            STIFFNESS_5020 * 2,
            DAMPING_5020 * 2,
            ARMATURE_5020 * 2,
            ACTUATOR_5020.effort_limit * 2,
        ),
        (
            (".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
            STIFFNESS_5020 * 2,
            DAMPING_5020 * 2,
            ARMATURE_5020 * 2,
            ACTUATOR_5020.effort_limit * 2,
        ),
    ]


G1_ACTUATOR_PHYSICS: List[Tuple[str, JointPhysics]] = []
for names, stiffness, damping, armature, effort_limit in _build_actuator_physics_specs():
    scale = mjlab_action_scale(effort_limit, stiffness)
    physics = JointPhysics(
        stiffness=stiffness,
        damping=damping,
        armature=armature,
        effort_limit=effort_limit,
        scale=scale,
    )
    for name in names:
        G1_ACTUATOR_PHYSICS.append((name, physics))

_COMPILED_PHYSICS_RULES = [
    (re.compile(pattern), physics) for pattern, physics in G1_ACTUATOR_PHYSICS
]

G1_JOINT_SCALE_RULES: List[Tuple[str, float]] = [
    (pattern, physics.scale) for pattern, physics in G1_ACTUATOR_PHYSICS
]

_COMPILED_SCALE_RULES = [
    (re.compile(pattern), scale) for pattern, scale in G1_JOINT_SCALE_RULES
]

_COMPILED_KEYFRAME_RULES: Dict[str, List[Tuple[re.Pattern[str], float]]] = {}
for keyframe_name, patterns in (
    ("knees_bent", KNEES_BENT_JOINT_POS),
    ("home", HOME_JOINT_POS),
):
    _COMPILED_KEYFRAME_RULES[keyframe_name] = [
        (re.compile(pattern), value) for pattern, value in patterns.items()
    ]

# print(f"G1_ACTUATOR_PHYSICS: {G1_ACTUATOR_PHYSICS}")
# print(f"G1_JOINT_SCALE_RULES: {G1_JOINT_SCALE_RULES}")
# print(f"_COMPILED_SCALE_RULES: {_COMPILED_SCALE_RULES}")


def normalize_joint_label(joint_label: str) -> str:
    """Extract joint basename from a USD path label."""
    label = joint_label.lower().strip()
    if "/" in label:
        label = label.rsplit("/", 1)[-1]
    return label


def is_hand_joint(joint_label: str) -> bool:
    return normalize_joint_label(joint_label) in HAND_JOINT_NAMES


def is_non_rl_joint(joint_label: str, extra_patterns: Optional[Sequence[str]] = None) -> bool:
    label = normalize_joint_label(joint_label)
    if is_hand_joint(label):
        return True
    patterns = list(_NON_RL_PATTERNS)
    if extra_patterns:
        patterns.extend(p.lower() for p in extra_patterns)
    return any(p in label for p in patterns)


def resolve_joint_physics(joint_label: str) -> Optional[JointPhysics]:
    label = normalize_joint_label(joint_label)
    if label in HAND_JOINT_NAMES:
        return HAND_JOINT_PHYSICS
    for pattern, physics in _COMPILED_PHYSICS_RULES:
        if pattern.fullmatch(label):
            return physics
    return None


def resolve_joint_scale(joint_label: str) -> float:
    physics = resolve_joint_physics(joint_label)
    return physics.scale if physics is not None else 0.0


def resolve_joint_nominal(
    joint_label: str,
    keyframe: str = "knees_bent",
    joint_pos_overrides: Optional[Dict[str, float]] = None,
) -> float:
    label = normalize_joint_label(joint_label)
    if label in HAND_JOINT_NAMES:
        return HAND_JOINT_NOMINAL
    if joint_pos_overrides:
        for pattern, value in joint_pos_overrides.items():
            if re.compile(pattern).fullmatch(label):
                return float(value)
    rules = _COMPILED_KEYFRAME_RULES.get(keyframe, _COMPILED_KEYFRAME_RULES["knees_bent"])
    for pattern, value in rules:
        if pattern.fullmatch(label):
            return float(value)
    return 0.0


def scale_for_config_key(key: str) -> float:
    if key in _NON_RL_SCALE_KEYS:
        return 0.0

    if key.endswith("_joint"):
        candidates = [key]
    elif key.startswith("left_") or key.startswith("right_"):
        candidates = [f"{key}_joint", key]
    else:
        candidates = [f"left_{key}_joint", f"right_{key}_joint", f"{key}_joint", key]

    for candidate in candidates:
        scale = resolve_joint_scale(candidate)
        if scale > 0.0:
            return scale
    return 0.0


##
# Mjlab velocity locomotion task (29-DOF action vector).
##

# Timing (policy runs once every ``DECIMATION`` physics substeps).
DECIMATION = 4
SIM_DT = 0.005
STEP_DT = SIM_DT * DECIMATION  # 0.02 s policy / env step

# Actuated joints in mjlab action-vector order (natural G1 joint order).
JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

ACTION_DIM = len(JOINT_NAMES)

# Reference ctrl indices for the stock mjlab ``g1.xml`` (one position actuator per joint).
CTRL_IDS_MJLAB: tuple[int, ...] = (
    10,
    15,
    11,
    16,
    25,
    26,
    12,
    17,
    13,
    18,
    27,
    28,
    14,
    23,
    24,
    0,
    1,
    2,
    3,
    4,
    19,
    20,
    5,
    6,
    7,
    8,
    9,
    21,
    22,
)

# mjlab G1 velocity task does not clip actions by default (clip_actions=None).
CLIP_ACTIONS: float | None = None


def _tuple_for_velocity_joints(fn) -> tuple[float, ...]:
    return tuple(float(fn(name)) for name in JOINT_NAMES)


ACTION_SCALE = _tuple_for_velocity_joints(resolve_joint_scale)
DEFAULT_JOINT_POS = _tuple_for_velocity_joints(
    lambda name: resolve_joint_nominal(name, keyframe="knees_bent")
)
STIFFNESS = _tuple_for_velocity_joints(
    lambda name: resolve_joint_physics(name).stiffness  # type: ignore[union-attr]
)
DAMPING = _tuple_for_velocity_joints(
    lambda name: resolve_joint_physics(name).damping  # type: ignore[union-attr]
)

JOINT_NAME_TO_ACTION_INDEX: dict[str, int] = {
    name: idx for idx, name in enumerate(JOINT_NAMES)
}
