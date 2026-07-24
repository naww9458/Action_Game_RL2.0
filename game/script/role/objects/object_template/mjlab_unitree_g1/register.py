"""Register G1 articulation joint loader and observation providers."""

from __future__ import annotations


def register() -> None:
    from script.role.abilities.articulation_control_config.joint_config_registry import (
        register_robot_loader,
    )
    from script.role.policies.policy_bundle import PolicyBundleRegistry

    from .g1_control_config import G1_ROBOT_NAME, get_g1_task_config
    from .g1_velocity_locomotion_provider import create_g1_velocity_locomotion_provider

    register_robot_loader(G1_ROBOT_NAME, get_g1_task_config)
    PolicyBundleRegistry.register_obs_provider(
        "g1_velocity_locomotion",
        create_g1_velocity_locomotion_provider,
    )
