"""Register striking-target joint defaults and runtime soft-anchor setup."""

from __future__ import annotations

from typing import List, Optional, Tuple

STRIKING_TARGET_PATTERN = "striking_target"


class _StrikingTargetJointConfig:
    """Keep builder D6 coordinates; do not expose the free D6 to RL."""

    def resolve_joint_arrays(
        self,
        joint_labels: List[str],
        default_qs: Optional[List[float]] = None,
    ) -> Tuple[List[float], List[float], List[float], List[float], List[int], List[int], float]:
        n = len(joint_labels)
        qs = [float(v) for v in (default_qs or [])]
        if len(qs) < n:
            qs.extend([0.0] * (n - len(qs)))
        qs = qs[:n]
        return (
            [0.0] * n,
            qs,
            [1.0e6] * n,
            [-1.0e6] * n,
            [0] * n,
            [-1] * n,
            1.0,
        )


def get_striking_target_joint_config() -> _StrikingTargetJointConfig:
    return _StrikingTargetJointConfig()


def register() -> None:
    from script.role.abilities.articulation_control_config.joint_config_registry import (
        register_robot_loader,
    )

    register_robot_loader(STRIKING_TARGET_PATTERN, get_striking_target_joint_config)


def setup(environment) -> None:
    from .soft_anchor_force import attach_if_present

    attach_if_present(environment)
