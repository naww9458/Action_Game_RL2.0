"""Lazy per-pattern tool action registry.

Registers only *module paths* (no code imports at registration time). The
concrete action implementation is imported only when a level actually uses the
matching object template, so turret/weapon-specific code never enters the
common mount/simulation hot path.
"""

from __future__ import annotations

import importlib
from typing import Dict, Optional, Sequence, Tuple

from script.role.objects.object_template.loader import get_object_template
from script.simulate.tool_action import ToolAction

# pattern -> (module_path, class_name)
_REGISTERED_ACTIONS: Dict[str, Tuple[str, str]] = {}


def register_tool_action(pattern: str, module_path: str, class_name: str) -> None:
    """Declare which class implements a pattern's attached-tool action.

    No import happens here — the module is loaded lazily by
    :func:`create_tool_action` only when a level config uses ``pattern``.
    """
    if not pattern:
        return
    _REGISTERED_ACTIONS[str(pattern)] = (str(module_path), str(class_name))


def create_tool_action(pattern: str | None) -> Optional[ToolAction]:
    """Instantiate the action for ``pattern`` (lazy import), or None if none."""
    if not pattern:
        return None
    spec = _REGISTERED_ACTIONS.get(str(pattern))
    if spec is None:
        return None
    module_path, class_name = spec
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None or not callable(cls):
        raise AttributeError(
            f"Tool action module '{module_path}' has no callable class '{class_name}'."
        )
    return cls()


def _rl_action_section_from_template(pattern: str) -> dict:
    """Read ``aim.rl_action`` (or top-level ``rl_action``) from object template YAML."""
    template = get_object_template(str(pattern))
    if not template:
        return {}
    aim = template.get("aim") or {}
    rl = aim.get("rl_action") or template.get("rl_action") or {}
    return dict(rl) if isinstance(rl, dict) else {}


def resolve_rl_action_dim_for_pattern(pattern: str) -> int:
    """Return RL action dimension declared for ``pattern`` (0 when absent)."""
    rl = _rl_action_section_from_template(pattern)
    shape = rl.get("shape", 0)
    try:
        return max(0, int(shape))
    except (TypeError, ValueError):
        return 0


def resolve_rl_action_spec_for_pattern(pattern: str) -> dict:
    """Build a Gym-style action spec dict from template ``rl_action`` metadata."""
    rl = _rl_action_section_from_template(pattern)
    if not rl:
        return {"type": "box", "shape": 0, "range": [-1.0, 1.0]}
    try:
        shape = max(0, int(rl.get("shape", 0)))
    except (TypeError, ValueError):
        shape = 0
    rng = rl.get("range", [-1.0, 1.0])
    if not isinstance(rng, list) or len(rng) != 2:
        rng = [-1.0, 1.0]
    spec: dict = {
        "type": "box",
        "shape": shape,
        "range": [float(rng[0]), float(rng[1])],
    }
    dims = rl.get("dims")
    if isinstance(dims, list) and dims:
        spec["dims"] = dims
    description = rl.get("description")
    if description:
        spec["description"] = str(description)
    return spec


def resolve_max_rl_action_dim_for_patterns(patterns: Sequence[str]) -> int:
    """Max ``rl_action.shape`` across tool patterns used in a level."""
    max_dim = 0
    for pattern in patterns:
        if not pattern:
            continue
        max_dim = max(max_dim, resolve_rl_action_dim_for_pattern(str(pattern)))
    return max_dim


def resolve_rl_action_spec_for_max_pattern(patterns: Sequence[str]) -> dict:
    """Action spec for the pattern with the largest RL tool action dimension."""
    best_pattern: Optional[str] = None
    best_dim = 0
    for pattern in patterns:
        if not pattern:
            continue
        pattern_str = str(pattern)
        dim = resolve_rl_action_dim_for_pattern(pattern_str)
        if dim > best_dim:
            best_dim = dim
            best_pattern = pattern_str
    if best_pattern is None or best_dim <= 0:
        return {"type": "box", "shape": 0, "range": [-1.0, 1.0]}
    return resolve_rl_action_spec_for_pattern(best_pattern)
