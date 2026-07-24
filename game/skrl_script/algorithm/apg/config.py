from __future__ import annotations

from skrl_script.algorithm.apg.apg import APG_DEFAULT_CONFIG
from training.schema import TrainingPresetConfig


def build_agent_cfg(preset: TrainingPresetConfig) -> dict:
    apg = preset.model.apg
    train = preset.train

    cfg = APG_DEFAULT_CONFIG.copy()
    cfg["learning_rate"] = apg.learning_rate
    cfg["mixed_precision"] = apg.mixed_precision
    cfg["experiment"]["checkpoint_interval"] = apg.checkpoint_interval
    cfg["horizon"] = train.horizon # TODO Hardcode attribute
    print("cfg[] = train.horizon: ", train.horizon)
    return cfg
