from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type

import numpy as np
import yaml
from gymnasium import spaces

from skrl_script.algorithm import build_agent_cfg_for_algorithm
from training.registry import (
    TrainingPresetRegistry,
    import_policy_classes,
    import_trainer_class,
)
from training.reward_imports import ensure_reward_registered
from training.schema import TrainingPresetConfig


class ModelConfigView:
    """Runtime view compatible with existing Trainer code."""

    def __init__(self, preset: TrainingPresetConfig):
        meta = preset.meta
        model = preset.model

        self.model_obs_type = meta.obs_type
        self.obs_width = model.obs_width
        self.obs_height = model.obs_height
        self.stack_size = model.stack_size
        self.state_obs_size = model.state_obs_size
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(model.state_obs_size,),
            dtype=np.float32,
        )
        self.level = meta.level
        self.sub_level = meta.sub_level

        self.cfg = build_agent_cfg_for_algorithm(meta.algorithm, preset)


class TrainConfigView:
    """Runtime view compatible with existing Trainer code."""

    def __init__(self, preset: TrainingPresetConfig):
        train = preset.train
        self.cfg_trainer = {
            "timesteps": train.timesteps,
            "write_interval": train.write_interval,
            "enable_namespaces": train.enable_namespaces,
        }
        self.max_episode_step = train.max_episode_step # TODO Hardcode
        self.horizon = train.horizon
        self.max_episode_epochs = train.max_episode_epochs
        self.total_epochs = train.total_epochs
        self.max_episode_step_evaluate = train.max_episode_step_evaluate
        self.seed = train.seed
        self.num_envs_default = train.num_envs_default
        from script.levels.rewards.reward_calculator import RewardComponent
        self.reward_components = [RewardComponent.resolve(name) for name in train.reward_components]
        self.reward_components_diff = [RewardComponent.resolve(name) for name in train.reward_components_diff]
        self.reward_parameters = dict(train.reward_parameters)
        self.player_controllers = list(train.player_ids)
        rl_count = sum(1 for c in self.player_controllers if c == "RL")
        self.num_agents_each_env = rl_count if rl_count > 0 else train.num_agents_each_env


@dataclass
class LoadedTrainingConfig:
    preset: TrainingPresetConfig
    preset_path: Optional[Path]
    model_cfg: ModelConfigView
    train_cfg: TrainConfigView
    Policy: Type
    Value: Type
    Trainer: Type

    @property
    def meta(self):
        return self.preset.meta


class TrainingPresetLoader:
    @classmethod
    def load(cls, preset_id_or_path: str) -> LoadedTrainingConfig:
        preset_path = TrainingPresetRegistry.resolve_preset_path(preset_id_or_path)
        preset = TrainingPresetRegistry.load_preset_yaml(preset_id_or_path)
        Policy, Value = import_policy_classes(preset.meta.policy_module)
        Trainer = import_trainer_class(preset.meta.trainer_module)
        return LoadedTrainingConfig(
            preset=preset,
            preset_path=preset_path,
            model_cfg=ModelConfigView(preset),
            train_cfg=TrainConfigView(preset),
            Policy=Policy,
            Value=Value,
            Trainer=Trainer,
        )

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any], preset_path: Optional[Path] = None) -> LoadedTrainingConfig:
        preset = TrainingPresetConfig.model_validate(data)
        Policy, Value = import_policy_classes(preset.meta.policy_module)
        Trainer = import_trainer_class(preset.meta.trainer_module)
        return LoadedTrainingConfig(
            preset=preset,
            preset_path=preset_path,
            model_cfg=ModelConfigView(preset),
            train_cfg=TrainConfigView(preset),
            Policy=Policy,
            Value=Value,
            Trainer=Trainer,
        )

    @classmethod
    def load_from_run_config(cls, run_dir: Path) -> LoadedTrainingConfig:
        config_dir = run_dir / "config"
        preset_yaml = config_dir / "preset.yaml"
        if preset_yaml.exists():
            with open(preset_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return cls.load_from_dict(data, preset_path=preset_yaml)

        manifest_json = config_dir / "run_manifest.json"
        if manifest_json.exists():
            with open(manifest_json, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            preset_id = manifest.get("preset_id")
            if preset_id:
                return cls.load(preset_id)

        raise FileNotFoundError(f"No preset.yaml or run_manifest.json in {config_dir}")

    @classmethod
    def from_legacy_pickle(cls, model_cfg, train_cfg, manifest: Optional[Dict[str, Any]] = None) -> LoadedTrainingConfig:
        """Build LoadedTrainingConfig from legacy pickle objects when YAML is unavailable."""
        policy_module = (manifest or {}).get("policy_module")
        trainer_module = (manifest or {}).get("trainer_module")
        algorithm = str((manifest or {}).get("algorithm", "PPO")).upper()
        if not policy_module or not trainer_module:
            from training.level_defaults import resolve_preset_id
            from training.registry import TrainingPresetRegistry

            preset_id = (manifest or {}).get("preset_id") or resolve_preset_id(
                algorithm,
                getattr(model_cfg, "level", 4),
                getattr(model_cfg, "sub_level", 0),
                getattr(model_cfg, "model_obs_type", "state_based"),
            )
            preset_meta = TrainingPresetRegistry.load_preset_yaml(preset_id).meta
            policy_module = policy_module or preset_meta.policy_module
            trainer_module = trainer_module or preset_meta.trainer_module
        Policy, Value = import_policy_classes(policy_module)
        Trainer = import_trainer_class(trainer_module)

        preset_data = {
            "meta": {
                "id": (manifest or {}).get("preset_id", "legacy"),
                "display_name": "Legacy Run",
                "level": getattr(model_cfg, "level", 4),
                "sub_level": getattr(model_cfg, "sub_level", 0),
                "obs_type": getattr(model_cfg, "model_obs_type", "state_based"),
                "algorithm": (manifest or {}).get("algorithm", "PPO"),
                "policy_module": policy_module,
                "trainer_module": trainer_module,
            },
            "model": {
                "state_obs_size": getattr(model_cfg, "state_obs_size", 18),
                "obs_width": getattr(model_cfg, "obs_width", 0),
                "obs_height": getattr(model_cfg, "obs_height", 0),
                "stack_size": getattr(model_cfg, "stack_size", 1),
            },
            "train": {
                "timesteps": train_cfg.cfg_trainer.get("timesteps", 5000),
                "write_interval": train_cfg.cfg_trainer.get("write_interval", 1),
                "enable_namespaces": train_cfg.cfg_trainer.get("enable_namespaces", True),
                "max_episode_step": getattr(train_cfg, "max_episode_step", 1000),
                "seed": getattr(train_cfg, "seed", 31415926),
                "num_agents_each_env": getattr(train_cfg, "num_agents_each_env", 1),
                "reward_components": [c.__name__ for c in getattr(train_cfg, "reward_components", [])],
                "reward_components_diff": [c.__name__ for c in getattr(train_cfg, "reward_components_diff", [])],
                "reward_parameters": getattr(train_cfg, "reward_parameters", {}),
                "player_ids": getattr(train_cfg, "player_ids", []),
            },
        }
        preset = TrainingPresetConfig.model_validate(preset_data)
        return LoadedTrainingConfig(
            preset=preset,
            preset_path=None,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            Policy=Policy,
            Value=Value,
            Trainer=Trainer,
        )
