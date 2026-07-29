"""Register turret articulation joint loader."""

from __future__ import annotations


def register() -> None:
    from script.role.abilities.articulation_control_config.joint_config_registry import (
        register_robot_loader,
    )

    from .turret_control_config import TURRET_ROBOT_NAME, get_turret_task_config

    register_robot_loader(TURRET_ROBOT_NAME, get_turret_task_config)
