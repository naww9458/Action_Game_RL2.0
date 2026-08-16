"""Training-framework registry (SKRL and RSL-RL are peers).

SKRL currently exposes PPO and APG. RSL-RL currently exposes PPO only.
Algorithm builders for SKRL live under ``rl_framework.skrl_script.algorithm``; RSL-RL
builds its runner config in ``rl_framework.rsl_rl_script.trainer`` from ``model.ppo``.
"""

from __future__ import annotations

from training.schema import (
    FRAMEWORK_ALGORITHMS,
    FRAMEWORK_RSL_RL,
    FRAMEWORK_SKRL,
    TrainingPresetConfig,
    validate_framework_algorithm,
)


def build_agent_cfg(preset: TrainingPresetConfig) -> dict:
    """Build the framework-specific agent dict consumed by ``ModelConfigView``.

    SKRL trainers read this dict as the skrl agent config. RSL-RL does not
    use it (returns ``{}``); its PPO hyperparameters stay in ``model.ppo``.
    """
    framework = str(preset.meta.framework).upper()
    algorithm = str(preset.meta.algorithm).upper()
    validate_framework_algorithm(framework, algorithm)
    if framework == FRAMEWORK_SKRL:
        from rl_framework.skrl_script.algorithm import build_agent_cfg_for_algorithm

        return build_agent_cfg_for_algorithm(algorithm, preset)
    if framework == FRAMEWORK_RSL_RL:
        return {}
    raise KeyError(
        f"Unknown framework: {framework}. Available: {sorted(FRAMEWORK_ALGORITHMS)}"
    )
