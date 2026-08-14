"""Generic asymmetric-critic observation provider registry (service locator).

Robot-specific critic-observation extensions (e.g. Unitree G1 foot-state
extras) register a factory here under their normalized robot pattern. Levels
discover the extension lazily via ``CriticObsProviderRegistry.create`` and
gracefully fall back to the symmetric (policy) observation when no provider is
registered for the player pattern.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from script.role.abilities.articulation_control_config.robot_pattern import (
    normalize_robot_pattern,
)


class CriticObsProvider:
    """Duck-typed protocol for an asymmetric-critic observation extension.

    Concrete implementations live next to their robot's ``object_template``
    and expose at least:

    * ``critic_obs_dim`` (``int``): number of extra columns appended after the
      policy observation.
    * ``setup()``: allocate GPU buffers and resolve robot-specific mappings.
    * ``get_critic_observation(policy_obs_wp) -> torch.Tensor``: build and
      return the full critic observation for the current state.
    """

    critic_obs_dim: int = 0

    def setup(self) -> None:
        raise NotImplementedError

    def get_critic_observation(self, policy_obs_wp: Any):  # -> torch.Tensor
        raise NotImplementedError


class CriticObsProviderRegistry:
    """Service locator mapping normalized robot pattern -> critic-obs factory."""

    _factories: Dict[str, Callable[..., CriticObsProvider]] = {}

    @classmethod
    def register(cls, robot_pattern: str, factory: Callable[..., CriticObsProvider]) -> None:
        cls._factories[normalize_robot_pattern(robot_pattern)] = factory

    @classmethod
    def create(cls, robot_pattern: str, **kwargs) -> Optional[CriticObsProvider]:
        """Create the provider for ``robot_pattern`` or return ``None`` when absent."""
        factory = cls._factories.get(normalize_robot_pattern(robot_pattern))
        if factory is None:
            return None
        provider = factory(**kwargs)
        provider.setup()
        return provider
