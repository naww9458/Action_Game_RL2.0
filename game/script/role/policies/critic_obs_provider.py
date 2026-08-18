"""Generic asymmetric-critic observation provider registry (service locator).

Version-specific critic extras (e.g. G1 V1 foot-state) register a factory under
the id in ``control_policy.yaml`` ``observation.obs_critic``. Tasks call
``CriticObsProviderRegistry.create(provider_id, ...)`` and fall back to the
symmetric (policy) observation when no factory is registered.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class CriticObsProvider:
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


class CriticObsProviderRegistry:
    """Service locator mapping critic-provider id -> factory."""

    _factories: Dict[str, Callable[..., CriticObsProvider]] = {}

    @classmethod
    def register(cls, provider_id: str, factory: Callable[..., CriticObsProvider]) -> None:
        cls._factories[str(provider_id).strip()] = factory

    @classmethod
    def is_registered(cls, provider_id: str) -> bool:
        return str(provider_id).strip() in cls._factories

    @classmethod
    def create(cls, provider_id: str | None, **kwargs) -> Optional[CriticObsProvider]:
        """Create the provider for ``provider_id`` or return ``None`` when absent."""
        if not provider_id:
            return None
        factory = cls._factories.get(str(provider_id).strip())
        if factory is None:
            return None
        provider = factory(**kwargs)
        provider.setup()
        return provider
