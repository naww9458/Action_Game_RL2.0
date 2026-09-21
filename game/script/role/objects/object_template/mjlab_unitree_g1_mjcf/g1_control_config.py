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
from typing import Any, Dict, List, Optional, Tuple

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
)

TEMPLATE_DIR = Path(__file__).resolve().parent
# Distinct from USD ``mjlab_unitree_g1`` (pattern ``unitree_g1``) so both
# templates can register loaders / policy bundles without overwriting.
G1_ROBOT_NAME = "unitree_g1_mjcf"
# Nested key inside models/<version>/control_configs.yaml (shared G1 layout).
G1_CONTROL_CONFIG_KEY = "unitree_g1"
G1_DEFAULT_TASK = "velocity_locomotion"

_TASK_CONFIG_CACHE: Dict[Tuple[str, str, str], "G1TaskConfig"] = {}


def resolve_g1_version_dir(
    *,
    control_policy_version: Optional[str] = None,
    task_name: Optional[str] = None,
) -> Path:
    """Return ``models/<version>/`` for this robot.

    Prefers ``control_policy_version``, then a version whose ``control_task``
    matches ``task_name``. Version id must come from object / preset config.
    """
    models_root = TEMPLATE_DIR / "models"
    if control_policy_version:
        candidate = models_root / str(control_policy_version)
        if (candidate / "control_configs.yaml").is_file() or (candidate / "control_policy.yaml").is_file():
            return candidate
        raise FileNotFoundError(
            f"G1 control-policy version '{control_policy_version}' has no folder "
            f"under {models_root}"
        )
    if task_name:
        for folder in sorted(models_root.iterdir() if models_root.is_dir() else []):
            if not folder.is_dir() or folder.name.startswith("_"):
                continue
            policy_path = folder / "control_policy.yaml"
            if not policy_path.is_file():
                continue
            data = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
            if str(data.get("control_task") or "") == str(task_name):
                return folder
    raise FileNotFoundError(
        f"No G1 model version folder under {models_root}. "
        "Set object.control_policy_version or object.control_task."
    )


_COLLISION_CACHE: Dict[str, Dict[str, Any]] = {}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _cfg_set(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[key] = value
        return
    try:
        cfg[key] = value
        return
    except Exception:
        setattr(cfg, key, value)


def load_g1_collision(
    *,
    control_policy_version: Optional[str] = None,
    task_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load the complete collision table for this G1 model.

    Prefer ``models/<version>/collision.yaml``; if that file is absent, use
    ``collision.yaml`` next to this template. The chosen file is the whole
    table — callers replace object collision fields with it, they do not merge.
    """
    paths: List[Path] = []
    try:
        version_dir = resolve_g1_version_dir(
            control_policy_version=control_policy_version,
            task_name=task_name,
        )
        paths.append(version_dir / "collision.yaml")
    except FileNotFoundError:
        pass
    paths.append(TEMPLATE_DIR / "collision.yaml")

    seen: set[str] = set()
    for path in paths:
        resolved = str(path.resolve()) if path.exists() else str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.is_file():
            continue
        cached = _COLLISION_CACHE.get(resolved)
        if cached is not None:
            return cached
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            continue
        _COLLISION_CACHE[resolved] = raw
        return raw
    return None


def apply_g1_collision(_object_cfg: Any) -> None:
    """MJCF g1.xml already contains mjlab collision geoms; do not overlay a table."""
    return


def _control_configs_path(version_dir: Path) -> Path:
    path = version_dir / "control_configs.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing control_configs.yaml for G1 model at {version_dir}")
    return path


def _parse_task_mapping(raw: Dict[str, Any], *, task_name: str, path: Path) -> Dict[str, Any]:
    """Read ``unitree_g1.<task>`` (and robot-level ``foot_sensor``) from control_configs.yaml."""
    robot_cfg = raw.get(G1_CONTROL_CONFIG_KEY, {})
    if isinstance(robot_cfg, dict):
        nested = robot_cfg.get(task_name)
        if isinstance(nested, dict):
            mapping = dict(nested)
            if "foot_sensor" in robot_cfg and "foot_sensor" not in mapping:
                mapping["foot_sensor"] = robot_cfg.get("foot_sensor")
            return mapping
    raise KeyError(f"Task '{task_name}' not found in {path}")


def _require_key(mapping: Dict[str, Any], key: str, *, path: Path, ctx: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise KeyError(f"Missing '{ctx}.{key}' in {path}")
    return mapping[key]


def _require_vec(mapping: Dict[str, Any], key: str, n: int, *, path: Path, ctx: str) -> Tuple[float, ...]:
    raw = _require_key(mapping, key, path=path, ctx=ctx)
    if not isinstance(raw, (list, tuple)) or len(raw) != n:
        raise ValueError(f"{path}: '{ctx}.{key}' must have {n} values")
    return tuple(float(v) for v in raw)


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
        control_policy_version: Optional[str] = None,
    ) -> "G1TaskConfig":
        if config_path is None:
            version_dir = resolve_g1_version_dir(
                control_policy_version=control_policy_version,
                task_name=task_name,
            )
            path = _control_configs_path(version_dir)
        else:
            path = Path(config_path)
        cache_key = (str(path), str(task_name or ""), str(control_policy_version or ""))
        cached = _TASK_CONFIG_CACHE.get(cache_key)
        if cached is not None:
            return cached

        with path.open("r", encoding="utf-8") as fh:
            raw_data = yaml.safe_load(fh) or {}
        if not isinstance(raw_data, dict):
            raise ValueError(f"Invalid YAML mapping: {path}")
        task_cfg = _parse_task_mapping(raw_data, task_name=task_name, path=path)

        init_state = task_cfg.get("init_state")
        if not isinstance(init_state, dict):
            raise KeyError(f"Missing '{task_name}.init_state' in {path}")
        keyframe = str(_require_key(init_state, "keyframe", path=path, ctx="init_state")).lower()
        if keyframe == "home":
            home_state = task_cfg.get("home_init_state")
            if not isinstance(home_state, dict):
                raise KeyError(f"Missing '{task_name}.home_init_state' in {path}")
            root_pos_raw = _require_vec(home_state, "root_pos", 3, path=path, ctx="home_init_state")
            root_rot_raw = _require_vec(home_state, "root_rot", 4, path=path, ctx="home_init_state")
            joint_pos_overrides = dict(task_cfg.get("home_joint_pos") or {})
            if not joint_pos_overrides:
                raise KeyError(f"Missing '{task_name}.home_joint_pos' in {path}")
        else:
            root_pos_raw = _require_vec(init_state, "root_pos", 3, path=path, ctx="init_state")
            root_rot_raw = _require_vec(init_state, "root_rot", 4, path=path, ctx="init_state")
            joint_pos_overrides = dict(task_cfg.get("joint_pos") or {})

        hand_cfg = task_cfg.get("hand_joints", {})
        if isinstance(hand_cfg, dict) and hand_cfg:
            configure_hand_joints(
                names=_require_key(hand_cfg, "names", path=path, ctx="hand_joints"),
                stiffness=float(_require_key(hand_cfg, "stiffness", path=path, ctx="hand_joints")),
                damping=float(_require_key(hand_cfg, "damping", path=path, ctx="hand_joints")),
                armature=float(_require_key(hand_cfg, "armature", path=path, ctx="hand_joints")),
                nominal=float(_require_key(hand_cfg, "nominal", path=path, ctx="hand_joints")),
            )

        non_rl = _require_key(task_cfg, "non_rl_patterns", path=path, ctx=task_name)
        if not isinstance(non_rl, (list, tuple)) or not non_rl:
            raise ValueError(f"{path}: '{task_name}.non_rl_patterns' must be a non-empty list")

        instance = cls(
            task_name=task_name,
            soft_limit_factor=float(
                _require_key(task_cfg, "soft_limit_factor", path=path, ctx=task_name)
            ),
            keyframe=keyframe,
            root_pos=tuple(float(v) for v in root_pos_raw),
            root_rot=tuple(float(v) for v in root_rot_raw),
            joint_pos_overrides=joint_pos_overrides,
            non_rl_patterns=tuple(str(p) for p in non_rl),
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


def get_g1_task_config(
    task_name: str = G1_DEFAULT_TASK,
    control_policy_version: Optional[str] = None,
) -> G1TaskConfig:
    return G1TaskConfig.from_yaml(
        task_name=task_name,
        control_policy_version=control_policy_version,
    )
