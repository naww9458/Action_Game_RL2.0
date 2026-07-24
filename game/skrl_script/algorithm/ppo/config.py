from __future__ import annotations

from typing import Optional

from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG

from training.registry import PREPROCESSOR_REGISTRY, SCHEDULER_REGISTRY, resolve_class_ref
from training.schema import ClassRefConfig, TrainingPresetConfig


def build_agent_cfg(preset: TrainingPresetConfig) -> dict:
    ppo = preset.model.ppo

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = ppo.rollouts
    cfg["memory_size"] = ppo.rollouts
    cfg["grad_norm_clip"] = ppo.grad_norm_clip
    cfg["entropy_loss_scale"] = ppo.entropy_loss_scale
    cfg["value_loss_scale"] = ppo.value_loss_scale
    cfg["ratio_clip"] = ppo.ratio_clip
    cfg["value_clip"] = ppo.value_clip
    cfg["discount_factor"] = ppo.discount_factor
    cfg["lambda"] = ppo.lambda_
    cfg["learning_epochs"] = ppo.learning_epochs
    cfg["mini_batches"] = ppo.mini_batches
    cfg["random_timesteps"] = ppo.random_timesteps
    cfg["rewards_shaper"] = None
    cfg["learning_rate"] = ppo.learning_rate
    cfg["kl_threshold"] = ppo.kl_threshold
    cfg["mixed_precision"] = ppo.mixed_precision
    cfg["experiment"]["checkpoint_interval"] = ppo.checkpoint_interval

    _apply_preprocessor(cfg, "state", preset.model.preprocessors.state)
    _apply_preprocessor(cfg, "value", preset.model.preprocessors.value)
    _apply_scheduler(cfg, preset.model.learning_rate_scheduler)
    return cfg


def _apply_preprocessor(cfg: dict, key: str, ref: Optional[ClassRefConfig]) -> None:
    if ref is None:
        cfg[f"{key}_preprocessor"] = None
        cfg[f"{key}_preprocessor_kwargs"] = {}
        return
    cls = resolve_class_ref(PREPROCESSOR_REGISTRY, ref.type)
    cfg[f"{key}_preprocessor"] = cls
    cfg[f"{key}_preprocessor_kwargs"] = dict(ref.kwargs)


def _apply_scheduler(cfg: dict, ref: Optional[ClassRefConfig]) -> None:
    if ref is None:
        return
    cls = resolve_class_ref(SCHEDULER_REGISTRY, ref.type)
    if cls is not None:
        cfg["learning_rate_scheduler"] = cls
        cfg["learning_rate_scheduler_kwargs"] = dict(ref.kwargs)
