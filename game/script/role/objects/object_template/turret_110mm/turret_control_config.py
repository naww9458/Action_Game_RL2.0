"""Load turret joint control config from template folder."""

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

TURRET_CONTROL_CONFIG_PATH = Path(__file__).resolve().parent / "control_configs.yaml"
TURRET_ROBOT_NAME = "turret_110mm"
TURRET_DEFAULT_TASK = "turret_aim"

_TASK_CONFIG_CACHE: Dict[Tuple[str, str], "TurretTaskConfig"] = {}


@dataclass
class TurretTaskConfig:
    task_name: str
    soft_limit_factor: float
    root_pos: Tuple[float, float, float]
    root_rot: Tuple[float, float, float, float]
    joint_pos_overrides: Dict[str, float]
    non_rl_patterns: Tuple[str, ...]
    human_control: Dict[str, Any]
    joint_stiffness: float
    joint_damping: float
    joint_armature: float
    # 俯仰關節物理限位剛度：0.0 = MuJoCo 硬限位（有限角度物理上不可超越）。
    # use_mujoco_policy_init 預設的 soft-limit（limit_ke=1000）彈簧過弱，
    # 會被 1200 kg 炮管的重力矩推過限位，故砲管限位需用硬限位守護。
    joint_limit_ke: float
    joint_limit_kd: float

    @classmethod
    def from_yaml(
        cls,
        task_name: str = TURRET_DEFAULT_TASK,
        config_path: Optional[Path] = None,
    ) -> "TurretTaskConfig":
        path = config_path or TURRET_CONTROL_CONFIG_PATH
        cache_key = (str(path), task_name, path.stat().st_mtime)
        cached = _TASK_CONFIG_CACHE.get(cache_key)
        if cached is not None:
            return cached

        with path.open("r", encoding="utf-8") as fh:
            raw_data = yaml.safe_load(fh) or {}

        robot_cfg = raw_data.get(TURRET_ROBOT_NAME, {})
        task_cfg = robot_cfg.get(task_name, {})
        if not isinstance(task_cfg, dict):
            raise KeyError(f"Task '{task_name}' not found in {path}")

        init_state = task_cfg.get("init_state", {})
        root_pos_raw = init_state.get("root_pos", [0.0, 0.0, 0.0])
        root_rot_raw = init_state.get("root_rot", [0.0, 0.0, 0.0, 1.0])
        actuation = task_cfg.get("joint_actuation", {})

        instance = cls(
            task_name=task_name,
            soft_limit_factor=float(task_cfg.get("soft_limit_factor", 0.95)),
            root_pos=tuple(float(v) for v in root_pos_raw),
            root_rot=tuple(float(v) for v in root_rot_raw),
            joint_pos_overrides=dict(task_cfg.get("joint_pos", {})),
            non_rl_patterns=tuple(task_cfg.get("non_rl_patterns", [])),
            human_control=dict(task_cfg.get("human_control") or {}),
            joint_stiffness=float(actuation.get("stiffness", 500.0)),
            joint_damping=float(actuation.get("damping", 50.0)),
            joint_armature=float(actuation.get("armature", 0.01)),
            joint_limit_ke=float(actuation.get("limit_ke", 0.0)),
            joint_limit_kd=float(actuation.get("limit_kd", 0.0)),
        )
        _TASK_CONFIG_CACHE[cache_key] = instance
        return instance

    def get_task_meta(self) -> TaskParam:
        return TaskParam(soft_limit_factor=self.soft_limit_factor)

    def _is_non_rl(self, label: str) -> bool:
        for pattern in self.non_rl_patterns:
            if re.search(pattern, label):
                return True
        return False

    def _nominal_for_label(self, label: str) -> float:
        for pattern, value in self.joint_pos_overrides.items():
            if re.search(pattern, label) or pattern.lower() in label.lower():
                return float(value)
        return 0.0

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
            nominal = self._nominal_for_label(label)
            if default_qs is not None and i < len(default_qs):
                nominal = float(default_qs[i])
            non_rl = self._is_non_rl(label)
            scales.append(0.35)
            nominals.append(nominal)
            limits_max.append(1.2)
            limits_min.append(-0.3)
            rl_mask.append(0 if non_rl else 1)
            if non_rl:
                rl_indices.append(-1)
            else:
                rl_indices.append(action_cursor)
                action_cursor += 1

        return (
            scales,
            nominals,
            limits_max,
            limits_min,
            rl_mask,
            rl_indices,
            self.soft_limit_factor,
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

        applied = 0
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

            nominal = self._nominal_for_label(label)
            for local_dof, qd in enumerate(range(qd_start, qd_end)):
                coord_idx = q_start + local_dof
                builder_env.joint_q[coord_idx] = nominal
                builder_env.joint_target_pos[qd] = nominal
                builder_env.joint_target_ke[qd] = self.joint_stiffness
                builder_env.joint_target_kd[qd] = self.joint_damping
                builder_env.joint_armature[qd] = self.joint_armature
                builder_env.joint_target_mode[qd] = int(JointTargetMode.POSITION)
                builder_env.joint_limit_ke[qd] = self.joint_limit_ke
                builder_env.joint_limit_kd[qd] = self.joint_limit_kd
                applied += 1

        print(
            f"[TurretTaskConfig] Applied physics init: task={self.task_name}, "
            f"joints={joint_end - joint_start}, actuated={applied}"
        )


def get_turret_task_config(task_name: str | None = None) -> TurretTaskConfig:
    return TurretTaskConfig.from_yaml(task_name or TURRET_DEFAULT_TASK)
