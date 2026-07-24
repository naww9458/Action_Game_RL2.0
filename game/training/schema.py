from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from script.role.controller_utils import normalize_player_controller_overrides


class PresetMetaConfig(BaseModel):
    id: str
    display_name: str = ""
    level: int
    sub_level: int
    obs_type: str = "state_based"
    algorithm: str = "PPO"
    policy_module: str
    trainer_module: str = "skrl_script.trainer_PPO"


class ClassRefConfig(BaseModel):
    type: Optional[str] = None
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class PreprocessorsConfig(BaseModel):
    state: Optional[ClassRefConfig] = None
    value: Optional[ClassRefConfig] = None


class PPOHyperparamsConfig(BaseModel):
    rollouts: int = 24
    grad_norm_clip: float = 1.0
    entropy_loss_scale: float = 0.01
    value_loss_scale: float = 1.0
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    discount_factor: float = 0.99
    lambda_: float = Field(default=0.95, alias="lambda")
    learning_epochs: int = 5
    mini_batches: int = 4
    random_timesteps: int = 0
    learning_rate: float = 1e-3
    kl_threshold: float = 0.01
    checkpoint_interval: int = 50
    mixed_precision: bool = False

    model_config = {"populate_by_name": True}


class APGHyperparamsConfig(BaseModel):
    learning_rate: float = 1e-3
    checkpoint_interval: int = 50
    mixed_precision: bool = False


class ModelPresetConfig(BaseModel):
    state_obs_size: int
    obs_width: int = 0
    obs_height: int = 0
    stack_size: int = 1
    ppo: PPOHyperparamsConfig = Field(default_factory=PPOHyperparamsConfig)
    apg: APGHyperparamsConfig = Field(default_factory=APGHyperparamsConfig)
    preprocessors: PreprocessorsConfig = Field(default_factory=PreprocessorsConfig)
    learning_rate_scheduler: Optional[ClassRefConfig] = None


class TrainPresetConfig(BaseModel):
    timesteps: int = 5000
    write_interval: int = 1
    enable_namespaces: bool = True
    max_episode_step: int = 1000
    horizon: int = 16
    max_episode_epochs: int = 40 # TODO Hardcode
    total_epochs: int = 4000
    max_episode_step_evaluate: int = 3000
    seed: int = 31415926
    num_agents_each_env: int = 1
    num_envs_default: int = 4096
    reward_components: List[str] = Field(default_factory=list)
    reward_components_diff: List[str] = Field(default_factory=list)
    reward_parameters: Dict[str, Any] = Field(default_factory=dict)
    # Per-player controller overrides applied on top of level YAML (Human / RL / Bot).
    player_ids: List[str] = Field(default_factory=list)

    @field_validator("player_ids", mode="before")
    @classmethod
    def _normalize_player_ids(cls, value):
        if not value:
            return []
        return normalize_player_controller_overrides(list(value))


class TrainingPresetConfig(BaseModel):
    meta: PresetMetaConfig
    model: ModelPresetConfig
    train: TrainPresetConfig


class ManifestEntry(BaseModel):
    id: str
    file: str
    display_name: str = ""
    level: int = 0
    sub_level: int = 0
    algorithm: str = "PPO"


class ManifestConfig(BaseModel):
    presets: List[ManifestEntry] = Field(default_factory=list)
