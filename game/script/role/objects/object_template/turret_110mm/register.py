"""Register turret articulation joint loader + lazy tool action mapping."""

from __future__ import annotations


def register() -> None:
    from script.role.abilities.articulation_control_config.joint_config_registry import (
        register_robot_loader,
    )
    from script.role.objects.object_template.tool_function_registry import (
        register_tool_action,
    )

    from .turret_control_config import TURRET_ROBOT_NAME, get_turret_task_config

    register_robot_loader(TURRET_ROBOT_NAME, get_turret_task_config)

    # Declare the "aim" action for this pattern. Only the module path is stored
    # here; the actual turret code is imported lazily when a level uses the turret.
    register_tool_action(
        TURRET_ROBOT_NAME,
        "script.role.objects.object_template.turret_110mm.functions.aim",
        "Turret110mmAimAction",
    )
