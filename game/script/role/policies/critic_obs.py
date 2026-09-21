"""Asymmetric-critic observation registry (service locator).

Version-specific critic extras (e.g. G1 V1 foot-state) register a factory under
the id in ``control_policy.yaml`` ``observation.obs_critic``. Tasks call
``CriticObsRegistry.create(obs_critic_id, ...)`` and fall back to the
symmetric (policy) observation when no factory is registered.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class CriticObs:
    """Duck-typed protocol for an asymmetric-critic observation extension.

    Concrete implementations live in ``object_template/.../models/<version>/``
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


class CriticObsRegistry:
    """Service locator mapping ``observation.obs_critic`` id -> factory."""

    _factories: Dict[str, Callable[..., CriticObs]] = {}

    @classmethod
    def register(cls, obs_critic_id: str, factory: Callable[..., CriticObs]) -> None:
        cls._factories[str(obs_critic_id).strip()] = factory

    @classmethod
    def is_registered(cls, obs_critic_id: str) -> bool:
        return str(obs_critic_id).strip() in cls._factories

    @classmethod
    def create(cls, obs_critic_id: str | None, **kwargs) -> Optional[CriticObs]:
        """Create the critic observation for ``obs_critic_id``, or ``None`` when absent."""
        if not obs_critic_id:
            return None
        factory = cls._factories.get(str(obs_critic_id).strip())
        if factory is None:
            return None
        critic_obs = factory(**kwargs)
        critic_obs.setup()
        return critic_obs
