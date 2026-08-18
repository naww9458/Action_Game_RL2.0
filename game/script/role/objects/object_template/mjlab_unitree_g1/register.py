"""Register G1 articulation joint loader and observation providers."""

from __future__ import annotations


def register() -> None:
    from script.role.abilities.articulation_control_config.joint_config_registry import (
        register_robot_loader,
    )
    from script.role.policies.policy_bundle import PolicyBundleRegistry

    from .g1_control_config import G1_ROBOT_NAME, get_g1_task_config

    register_robot_loader(G1_ROBOT_NAME, get_g1_task_config)
    PolicyBundleRegistry.ensure_loaded()


def prepare_object(object_cfg) -> None:
    from .g1_control_config import apply_g1_collision

    apply_g1_collision(object_cfg)


def setup(environment) -> None:
    from .g1_policy_runtime import attach_if_present

    attach_if_present(environment)
