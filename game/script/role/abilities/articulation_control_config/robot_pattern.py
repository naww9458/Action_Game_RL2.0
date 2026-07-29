"""Robot pattern helpers shared across articulation control modules."""

from __future__ import annotations

CONTROLLER_PREFIXES: tuple[str, ...] = ("Human_", "RL_", "Bot_")


def strip_controller_prefix(pattern: str) -> str:
    text = str(pattern).strip()
    for prefix in CONTROLLER_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def normalize_robot_pattern(pattern: str) -> str:
    """Strip controller and optional role prefixes from articulation-body pattern ids."""
    text = strip_controller_prefix(str(pattern).strip())
    for role_prefix in ("player_", "tool_", "entity_", "platform_"):
        if text.startswith(role_prefix):
            return text[len(role_prefix) :]
    return text


def compose_runtime_pattern(
    controller: str | None, role_type: str, job_pattern: str
) -> str:
    """Build runtime pattern: ``[controller]_[role_type]_[job_pattern]``."""
    role = str(role_type or "player").strip()
    job = normalize_robot_pattern(str(job_pattern))
    base = f"{role}_{job}"
    if controller in ("Human", "RL", "Bot"):
        return f"{controller}_{base}"
    return base


def player_pattern(robot_pattern: str) -> str:
    """Build legacy articulation-body player pattern id from a robot/job pattern."""
    text = normalize_robot_pattern(robot_pattern)
    return f"player_{text}"


def patterns_compatible(left: str, right: str) -> bool:
    """Return True when two pattern strings refer to the same robot."""
    return normalize_robot_pattern(left) == normalize_robot_pattern(right)
