"""Lazy per-pattern tool action registry.

Registers only *module paths* (no code imports at registration time). The
concrete action implementation is imported only when a level actually uses the
matching object template, so turret/weapon-specific code never enters the
common mount/simulation hot path.
"""

from __future__ import annotations

import importlib
from typing import Dict, Optional, Tuple

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
