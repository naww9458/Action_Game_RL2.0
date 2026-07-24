"""Register wheeled armored vehicle articulation joint loader."""

from __future__ import annotations


def register() -> None:
    from script.role.abilities.articulation_control_config.joint_config_registry import (
        register_robot_loader,
    )

    from .vehicle_control_config import VEHICLE_ROBOT_NAME, get_vehicle_task_config

    register_robot_loader(VEHICLE_ROBOT_NAME, get_vehicle_task_config)
