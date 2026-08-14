"""Placeholder policy classes for the RSL-rl trainer.

``training.loader.TrainingPresetLoader`` requires every preset to declare a
``policy_module`` exposing ``Policy`` and ``Value`` attributes. RSL-rl builds its
own actor/critic models internally (``MLPModel`` with ELU activations and
``EmpiricalNormalization``), so these classes are never instantiated — they only
satisfy the preset-loading contract and document the architecture.
"""

from __future__ import annotations


class Policy:
    """Marker class; RSL-rl constructs the actor from ``RslRlPpoRunnerCfg.actor``."""

    def __init__(self, *args, **kwargs):  # pragma: no cover - never instantiated
        raise NotImplementedError("RSL-rl builds its actor internally; do not instantiate this class.")


class Value:
    """Marker class; RSL-rl constructs the critic from ``RslRlPpoRunnerCfg.critic``."""

    def __init__(self, *args, **kwargs):  # pragma: no cover - never instantiated
        raise NotImplementedError("RSL-rl builds its critic internally; do not instantiate this class.")
