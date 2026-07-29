"""Runtime profiles for articulation-body abilities (control-policy version sourced)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from script.role.abilities.articulation_control_config.robot_pattern import (
    compose_runtime_pattern,
    normalize_robot_pattern,
    patterns_compatible,
    player_pattern,
)
from script.role.controller_utils import normalize_controller
from script.role.policies.policy_bundle import PolicyBundleSpec, get_policy_bundle


@dataclass
class AxisKeyBindings:
    positive_keyboard: List[int] = field(default_factory=list)
    positive_mouse: List[int] = field(default_factory=list)
    negative_keyboard: List[int] = field(default_factory=list)
    negative_mouse: List[int] = field(default_factory=list)


@dataclass
class ArticulationAbilityProfile:
    control_mode: int = 3
    gait_torque: float = 50.0
    force: float = 150.0
    control_bindings: Tuple[str, ...] = ()
    human_control: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandAbilityProfile:
    command_dim: int
    command_labels: Tuple[str, ...]
    command_ranges: Tuple[Tuple[float, float], ...]
    human_control: Dict[str, Any] = field(default_factory=dict)


def _articulation_profile_from_spec(spec: PolicyBundleSpec) -> ArticulationAbilityProfile:
    raw = dict(spec.articulation_ability or {})
    human_control = deepcopy(raw.get("human_control") or {"keyboard": {}, "mouse": {}})
    bindings_raw = human_control.pop("bindings", None) or ()
    control_bindings = tuple(str(name) for name in bindings_raw)
    if not control_bindings:
        raise ValueError(
            f"articulation_ability.human_control.bindings is required in control policy '{spec.bundle_id}'"
        )
    return ArticulationAbilityProfile(
        control_mode=int(raw.get("control_mode", 3)),
        gait_torque=float(raw.get("gait_torque", 50.0)),
        force=float(raw.get("force", 150.0)),
        control_bindings=control_bindings,
        human_control=human_control,
    )


def get_articulation_ability_profile(
    control_policy_version: str,
    *,
    robot_pattern: str,
) -> ArticulationAbilityProfile:
    spec = get_policy_bundle(control_policy_version, robot_pattern=robot_pattern)
    if not patterns_compatible(robot_pattern, spec.robot_pattern):
        raise ValueError(
            f"Control policy '{control_policy_version}' targets robot '{spec.robot_pattern}', "
            f"but player pattern is '{robot_pattern}'."
        )
    return _articulation_profile_from_spec(spec)


def command_profile_from_bundle(spec: PolicyBundleSpec) -> CommandAbilityProfile:
    labels = tuple(spec.command_labels or tuple(f"command_{i}" for i in range(spec.command_dim)))
    if spec.command_ranges:
        ranges = tuple(tuple(r) for r in spec.command_ranges)
    else:
        ranges = tuple((-1.0, 1.0) for _ in range(spec.command_dim))
    if len(labels) != spec.command_dim or len(ranges) != spec.command_dim:
        raise ValueError(
            f"Control policy '{spec.bundle_id}' command metadata mismatch: "
            f"dim={spec.command_dim}, labels={len(labels)}, ranges={len(ranges)}"
        )
    human_control = deepcopy(spec.human_control or {"keyboard": {}, "mouse": {}})
    return CommandAbilityProfile(
        command_dim=spec.command_dim,
        command_labels=labels,
        command_ranges=ranges,
        human_control=human_control,
    )


def resolve_human_control_bindings(
    human_control: Dict[str, Any],
    binding_names: Sequence[str],
) -> Dict[str, AxisKeyBindings]:
    """Resolve {name}_positive/{name}_negative entries from model human_control config."""
    from script.role.abilities.key_mapping import KeyMapping

    resolved = KeyMapping.get(deepcopy(human_control))
    keyboard = resolved.get("keyboard", {})
    mouse = resolved.get("mouse", {})
    bindings: Dict[str, AxisKeyBindings] = {}
    for name in binding_names:
        bindings[name] = AxisKeyBindings(
            positive_keyboard=list(keyboard.get(f"{name}_positive", [])),
            positive_mouse=list(mouse.get(f"{name}_positive", [])),
            negative_keyboard=list(keyboard.get(f"{name}_negative", [])),
            negative_mouse=list(mouse.get(f"{name}_negative", [])),
        )
    return bindings


def command_binding_names(command_dim: int) -> Tuple[str, ...]:
    return tuple(f"command_{i}" for i in range(command_dim))


def find_player_config_for_ability(
    player_configs: Sequence[Dict[str, Any]],
    ability_name: str,
    *,
    robot_pattern: str | None = None,
) -> Dict[str, Any]:
    """Return the first player config that lists *ability_name*.

    When *robot_pattern* is set, only configs whose ``object.pattern`` normalizes
    to that robot kind are considered (for per-articulation ability instances).
    """
    wanted = normalize_robot_pattern(robot_pattern) if robot_pattern else None
    for cfg in player_configs or []:
        abilities = cfg.get("abilities") or []
        if ability_name not in abilities:
            continue
        if wanted is not None:
            object_cfg = dict(cfg.get("object") or {})
            cfg_pattern = object_cfg.get("pattern")
            if not cfg_pattern:
                continue
            if normalize_robot_pattern(str(cfg_pattern)) != wanted:
                continue
        return dict(cfg)
    detail = f" with robot_pattern='{wanted}'" if wanted else ""
    raise RuntimeError(
        f"No player config references ability '{ability_name}'{detail}."
    )


def find_tool_config_for_ability(
    tool_configs: Sequence[Dict[str, Any]],
    ability_name: str,
    *,
    robot_pattern: str | None = None,
) -> Dict[str, Any]:
    """Return the first tool config that lists *ability_name* (or uses defaults).

    Tools often omit ``abilities`` and rely on ``DEFAULT_TOOL_ABILITIES``; those
    still match when the ability is an articulation-control class used by tools.
    When *robot_pattern* is set, filter by ``object.pattern``.
    """
    from script.role.tool import DEFAULT_TOOL_ABILITIES

    wanted = normalize_robot_pattern(robot_pattern) if robot_pattern else None
    for cfg in tool_configs or []:
        abilities = list(cfg.get("abilities") or DEFAULT_TOOL_ABILITIES)
        if ability_name not in abilities:
            continue
        if wanted is not None:
            object_cfg = dict(cfg.get("object") or {})
            cfg_pattern = object_cfg.get("pattern")
            if not cfg_pattern:
                continue
            if normalize_robot_pattern(str(cfg_pattern)) != wanted:
                continue
        return dict(cfg)
    detail = f" with robot_pattern='{wanted}'" if wanted else ""
    raise RuntimeError(
        f"No tool config references ability '{ability_name}'{detail}."
    )


def resolve_object_robot_pattern(object_config: Dict[str, Any]) -> str:
    pattern = object_config.get("pattern")
    if not pattern:
        raise ValueError("Player object config is missing required field 'pattern'.")
    return normalize_robot_pattern(str(pattern))


def resolve_player_runtime_pattern(player_config: Dict[str, Any]) -> str:
    """Compose runtime player pattern from controller + role type + object.pattern."""
    role_type = str(player_config.get("type") or "player")
    object_config = dict(player_config.get("object") or {})
    job_pattern = resolve_object_robot_pattern(object_config)
    controller = normalize_controller(player_config.get("controller"))
    return compose_runtime_pattern(controller, role_type, job_pattern)


def resolve_articulation_player_pattern(
    object_config: Dict[str, Any],
    player_config: Dict[str, Any] | None = None,
) -> str:
    if player_config is not None:
        return resolve_player_runtime_pattern(player_config)
    return player_pattern(resolve_object_robot_pattern(object_config))


def resolve_role_articulation_pattern(role_type: str, object_config: Dict[str, Any]) -> str:
    """Build articulation-body pattern id from role type and object.pattern (e.g. player_default)."""
    pattern = object_config.get("pattern")
    if not pattern:
        raise ValueError(f"{role_type} object config is missing required field 'pattern'.")
    return f"{role_type}_{pattern}"


def resolve_ability_generated_object_pattern(object_config: Dict[str, Any]) -> str:
    return resolve_role_articulation_pattern("ability_generated_object", object_config)


def resolve_policy_checkpoint(object_config: Dict[str, Any]) -> str | None:
    """Optional checkpoint filename override within the version directory."""
    checkpoint = object_config.get("policy_checkpoint") or object_config.get("policy_checkpoint_path")
    if not checkpoint:
        return None
    return Path(str(checkpoint)).name


def resolve_control_policy_version(object_config: Dict[str, Any]) -> str:
    """Read control-policy version id from player object config."""
    version = object_config.get("control_policy_version")
    if version:
        return str(version)
    legacy_bundle = object_config.get("policy_bundle")
    if legacy_bundle:
        return str(legacy_bundle)
    
    return None
