"""Robot joint config registry for articulation RL control."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from script.role.abilities.articulation_control_config.config_models import (
    apply_soft_limits,
)
from script.role.abilities.articulation_control_config.robot_pattern import (
    normalize_robot_pattern,
)

RobotConfigLoader = Callable[[], object]

_ROBOT_LOADERS: Dict[str, RobotConfigLoader] = {}


def register_robot_loader(pattern: str, loader: RobotConfigLoader) -> None:
    _ROBOT_LOADERS[normalize_robot_pattern(pattern)] = loader


def _resolve_robot_pattern(pattern: str) -> str:
    from script.role.objects.object_template.loader import ensure_object_templates_registered

    ensure_object_templates_registered()
    robot_pattern = normalize_robot_pattern(pattern)
    if robot_pattern not in _ROBOT_LOADERS:
        raise KeyError(
            f"No joint config loader registered for pattern '{robot_pattern}'. "
            f"Registered: {sorted(_ROBOT_LOADERS.keys())}"
        )
    return robot_pattern


def _try_load_robot_config(pattern: str, task_name: str | None):
    from script.role.objects.object_template.loader import ensure_object_templates_registered

    ensure_object_templates_registered()
    robot_pattern = normalize_robot_pattern(pattern)
    if robot_pattern not in _ROBOT_LOADERS:
        return None
    return _load_robot_config(robot_pattern, task_name)


def _load_robot_config(robot_pattern: str, task_name: str | None):
    loader = _ROBOT_LOADERS[robot_pattern]
    config = loader()
    if task_name is not None and hasattr(config, "from_yaml"):
        return config.from_yaml(task_name=task_name)
    return config


def apply_physics_init_for_pattern(
    pattern: str,
    builder_env,
    start_q_idx: int,
    joint_start: int,
    joint_end: int,
    task_name: str | None = None,
) -> None:
    robot_pattern = _resolve_robot_pattern(pattern)
    config = _load_robot_config(robot_pattern, task_name)
    config.apply_builder_physics_init(builder_env, start_q_idx, joint_start, joint_end)


def resolve_joint_arrays_for_pattern(
    pattern: str,
    joint_labels: List[str],
    default_qs: Optional[List[float]] = None,
    task_name: str | None = None,
) -> Tuple[List[float], List[float], List[float], List[float], List[int], List[int], float]:
    robot_pattern = _resolve_robot_pattern(pattern)
    config = _load_robot_config(robot_pattern, task_name)
    return config.resolve_joint_arrays(joint_labels=joint_labels, default_qs=default_qs)


def resolve_command_interface_for_pattern(
    pattern: str,
    task_name: str | None = None,
):
    config = _try_load_robot_config(pattern, task_name)
    if config is None or not hasattr(config, "get_command_interface"):
        return None
    return config.get_command_interface()


def resolve_rl_action_dim_for_pattern(
    pattern: str,
    per_dof_rl_dim: int,
    task_name: str | None = None,
) -> int:
    command_iface = resolve_command_interface_for_pattern(pattern, task_name=task_name)
    if command_iface is not None:
        return int(command_iface.command_dim)
    return int(per_dof_rl_dim)


def resolve_possess_offset_for_pattern(
    pattern: str,
    task_name: str | None = None,
) -> Optional[Tuple[float, float, float]]:
    config = _try_load_robot_config(pattern, task_name)
    if config is None:
        return None
    offset = getattr(config, "possess_offset", None)
    if offset is None:
        return None
    values = tuple(float(v) for v in offset)
    if len(values) != 3:
        return None
    return values


def resolve_runtime_nominals_gpu_spec(
    pattern: str,
    joint_labels: List[str],
    task_name: str | None = None,
):
    from script.role.abilities.articulation_control_config.runtime_helpers import (
        RuntimeNominalsGpuSpec,
    )

    config = _try_load_robot_config(pattern, task_name)
    if config is None or not hasattr(config, "build_runtime_nominals_gpu_spec"):
        return None
    spec = config.build_runtime_nominals_gpu_spec(joint_labels)
    if spec is None or not isinstance(spec, RuntimeNominalsGpuSpec):
        return None
    if not spec.has_passive_dofs():
        return None
    return spec


__all__ = [
    "apply_physics_init_for_pattern",
    "apply_soft_limits",
    "register_robot_loader",
    "resolve_command_interface_for_pattern",
    "resolve_joint_arrays_for_pattern",
    "resolve_possess_offset_for_pattern",
    "resolve_rl_action_dim_for_pattern",
    "resolve_runtime_nominals_gpu_spec",
]
