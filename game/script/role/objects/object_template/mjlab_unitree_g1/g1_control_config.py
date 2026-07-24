# This file contains code adapted from:
# https://github.com/mujocolab/mjlab
#
# Modified for Action_Game_RL.
#
# The original project is licensed under the Apache License 2.0.

"""Load Unitree G1 joint control config from the g1 template folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from newton import JointTargetMode

from script.role.abilities.articulation_control_config.config_models import (
    JointParam,
    TaskParam,
)

from .g1_actuator_model import (
    configure_hand_joints,
    is_non_rl_joint,
    normalize_joint_label,
    resolve_joint_nominal,
    resolve_joint_physics,
    resolve_joint_scale,
)

G1_CONTROL_CONFIG_PATH = Path(__file__).resolve().parent / "control_configs.yaml"
G1_ROBOT_NAME = "unitree_g1"
G1_DEFAULT_TASK = "velocity_locomotion"

_TASK_CONFIG_CACHE: Dict[Tuple[str, str], "G1TaskConfig"] = {}


@dataclass
class G1TaskConfig:
    task_name: str
    soft_limit_factor: float
    keyframe: str
    root_pos: Tuple[float, float, float]
    root_rot: Tuple[float, float, float, float]
    joint_pos_overrides: Dict[str, float]
    non_rl_patterns: Tuple[str, ...]

    @classmethod
    def from_yaml(
        cls,
        task_name: str = G1_DEFAULT_TASK,
        config_path: Optional[Path] = None,
    ) -> "G1TaskConfig":
        cache_key = (str(config_path or G1_CONTROL_CONFIG_PATH), task_name)
        cached = _TASK_CONFIG_CACHE.get(cache_key)
        if cached is not None:
            return cached

        path = config_path or G1_CONTROL_CONFIG_PATH
        with path.open("r", encoding="utf-8") as fh:
            raw_data = yaml.safe_load(fh) or {}

        robot_cfg = raw_data.get(G1_ROBOT_NAME, {})
        task_cfg = robot_cfg.get(task_name, {})
        if not isinstance(task_cfg, dict):
            raise KeyError(f"Task '{task_name}' not found in {path}")

        init_state = task_cfg.get("init_state", {})
        keyframe = str(init_state.get("keyframe", "knees_bent")).lower()
        if keyframe == "home":
            root_pos_raw = task_cfg.get("home_init_state", {}).get(
                "root_pos", init_state.get("root_pos", [0.0, 0.0, 0.783675])
            )
            root_rot_raw = task_cfg.get("home_init_state", {}).get(
                "root_rot", init_state.get("root_rot", [0.0, 0.0, 0.7071, 0.7071])
            )
            joint_pos_overrides = dict(task_cfg.get("home_joint_pos", {}))
        else:
            keyframe = "knees_bent"
            root_pos_raw = init_state.get("root_pos", [0.0, 0.0, 0.76])
            root_rot_raw = init_state.get("root_rot", [0.0, 0.0, 0.7071, 0.7071])
            joint_pos_overrides = dict(task_cfg.get("joint_pos", {}))

        hand_cfg = task_cfg.get("hand_joints", {})
        if isinstance(hand_cfg, dict) and hand_cfg:
            configure_hand_joints(
                names=hand_cfg.get("names"),
                stiffness=float(hand_cfg.get("stiffness", 10.0)),
                damping=float(hand_cfg.get("damping", 2.0)),
                armature=float(hand_cfg.get("armature", 0.1)),
                nominal=float(hand_cfg.get("nominal", 0.0)),
            )

        instance = cls(
            task_name=task_name,
            soft_limit_factor=float(task_cfg.get("soft_limit_factor", 0.9)),
            keyframe=keyframe,
            root_pos=tuple(float(v) for v in root_pos_raw),
            root_rot=tuple(float(v) for v in root_rot_raw),
            joint_pos_overrides=joint_pos_overrides,
            non_rl_patterns=tuple(task_cfg.get("non_rl_patterns", ["finger", "thumb", "hand"])),
        )
        _TASK_CONFIG_CACHE[cache_key] = instance
        return instance

    def get_task_meta(self) -> TaskParam:
        return TaskParam(soft_limit_factor=self.soft_limit_factor)

    def get_joint_param(self, joint_label: str) -> JointParam:
        label = normalize_joint_label(joint_label)
        non_rl = is_non_rl_joint(label, self.non_rl_patterns)
        physics = resolve_joint_physics(label)
        scale = physics.scale if physics is not None else 0.0
        nominal = resolve_joint_nominal(
            label,
            keyframe=self.keyframe,
            joint_pos_overrides=self.joint_pos_overrides,
        )
        return JointParam(
            scale=scale,
            nominal=nominal,
            rl_controllable=not non_rl,
            effort_limit=physics.effort_limit if physics else None,
            stiffness=physics.stiffness if physics else None,
            kd=physics.damping if physics else None,
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

        action_cursor = 0
        for i, label in enumerate(joint_labels):
            param = self.get_joint_param(label)
            scale = param.resolved_scale()
            if param.nominal is not None:
                nominal = float(param.nominal)
            elif default_qs is not None:
                nominal = float(default_qs[i])
            else:
                nominal = 0.0

            controllable = param.rl_controllable and scale > 0.0
            scales.append(scale)
            nominals.append(nominal)
            rl_mask.append(1 if controllable else 0)
            if controllable:
                rl_indices.append(action_cursor)
                action_cursor += 1
            else:
                rl_indices.append(-1)
            limits_max.append(1e6)
            limits_min.append(-1e6)

        return scales, nominals, limits_max, limits_min, rl_mask, rl_indices, self.soft_limit_factor

    def apply_builder_physics_init(
        self,
        builder_env,
        start_q_idx: int,
        joint_start: int,
        joint_end: int,
    ) -> None:
        builder_env.joint_q[start_q_idx : start_q_idx + 3] = list(self.root_pos)
        builder_env.joint_q[start_q_idx + 3 : start_q_idx + 7] = list(self.root_rot)

        applied = 0
        for joint_idx in range(joint_start, joint_end):
            label = str(builder_env.joint_label[joint_idx])
            basename = normalize_joint_label(label)
            q_start = builder_env.joint_q_start[joint_idx]
            qd_start = builder_env.joint_qd_start[joint_idx]

            if q_start >= 7:
                nominal = resolve_joint_nominal(
                    basename,
                    keyframe=self.keyframe,
                    joint_pos_overrides=self.joint_pos_overrides,
                )
                builder_env.joint_q[q_start] = nominal

            physics = resolve_joint_physics(basename)
            if physics is None:
                continue

            builder_env.joint_target_ke[qd_start] = physics.stiffness
            builder_env.joint_target_kd[qd_start] = physics.damping
            builder_env.joint_armature[qd_start] = physics.armature
            builder_env.joint_target_mode[qd_start] = int(JointTargetMode.POSITION)
            applied += 1

        print(
            f"[G1TaskConfig] Applied mjlab physics init: keyframe={self.keyframe}, "
            f"joints={joint_end - joint_start}, actuated={applied}"
        )


def get_g1_task_config(task_name: str = G1_DEFAULT_TASK) -> G1TaskConfig:
    return G1TaskConfig.from_yaml(task_name=task_name)


def load_unitree_g1_config() -> Dict[str, Any]:
    """Legacy helper for callers expecting raw yaml dict."""
    with G1_CONTROL_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
