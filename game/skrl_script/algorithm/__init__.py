from __future__ import annotations

from typing import Callable, Dict

from training.schema import TrainingPresetConfig

AgentConfigBuilder = Callable[[TrainingPresetConfig], dict]

_ALGORITHM_BUILDERS: Dict[str, AgentConfigBuilder] = {}


def register_algorithm(name: str, builder: AgentConfigBuilder) -> None:
    _ALGORITHM_BUILDERS[name.upper()] = builder


def build_agent_cfg_for_algorithm(algorithm: str, preset: TrainingPresetConfig) -> dict:
    key = algorithm.upper()
    if key not in _ALGORITHM_BUILDERS:
        _register_builtin(key)
    if key not in _ALGORITHM_BUILDERS:
        available = sorted(_ALGORITHM_BUILDERS)
        raise KeyError(f"Unknown algorithm: {algorithm}. Available: {available}")
    return _ALGORITHM_BUILDERS[key](preset)


def _register_builtin(name: str) -> None:
    if name == "PPO":
        from skrl_script.algorithm.ppo.config import build_agent_cfg

        register_algorithm("PPO", build_agent_cfg)
    elif name == "APG":
        from skrl_script.algorithm.apg.config import build_agent_cfg

        register_algorithm("APG", build_agent_cfg)
