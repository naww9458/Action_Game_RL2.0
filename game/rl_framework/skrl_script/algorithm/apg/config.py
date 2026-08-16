from __future__ import annotations

from rl_framework.skrl_script.algorithm.apg.apg import APG_DEFAULT_CONFIG
from training.runtime_env import framework_runs_dir
from training.schema import TrainingPresetConfig


def build_agent_cfg(preset: TrainingPresetConfig) -> dict:
    apg = preset.model.apg
    train = preset.train

    cfg = APG_DEFAULT_CONFIG.copy()
    cfg["learning_rate"] = apg.learning_rate
    cfg["mixed_precision"] = apg.mixed_precision
    cfg["experiment"]["checkpoint_interval"] = apg.checkpoint_interval
    cfg["experiment"]["directory"] = str(framework_runs_dir(preset.meta.framework))
    cfg["horizon"] = train.horizon # TODO Hardcode attribute
    print("cfg[] = train.horizon: ", train.horizon)
    return cfg
