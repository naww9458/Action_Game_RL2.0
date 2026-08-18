from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from script.role.controller_utils import normalize_player_controller_overrides

FRAMEWORK_SKRL = "SKRL"
FRAMEWORK_RSL_RL = "RSL_RL"
FRAMEWORK_ALGORITHMS: Dict[str, frozenset[str]] = {
    FRAMEWORK_SKRL: frozenset({"PPO", "APG"}),
    FRAMEWORK_RSL_RL: frozenset({"PPO"}),
}
# Folder names under ``runs/``. Internal ids stay SKRL / RSL_RL.
FRAMEWORK_RUN_FOLDER: Dict[str, str] = {
    FRAMEWORK_SKRL: "SKRL",
    FRAMEWORK_RSL_RL: "RSL-rl",
}


def framework_run_folder_name(framework: Optional[str]) -> str:
    fw = str(framework or FRAMEWORK_SKRL).upper().replace("-", "_")
    if fw not in FRAMEWORK_RUN_FOLDER and fw.startswith("RSL"):
        fw = FRAMEWORK_RSL_RL
    if fw not in FRAMEWORK_RUN_FOLDER:
        fw = FRAMEWORK_SKRL
    return FRAMEWORK_RUN_FOLDER[fw]


def normalize_framework_id(framework: Optional[str]) -> str:
    fw = str(framework or FRAMEWORK_SKRL).upper().replace("-", "_")
    if fw not in FRAMEWORK_RUN_FOLDER and fw.startswith("RSL"):
        return FRAMEWORK_RSL_RL
    if fw in FRAMEWORK_RUN_FOLDER:
        return fw
    return FRAMEWORK_SKRL


def coerce_framework_algorithm(data: Any) -> Any:
    """Normalize meta/manifest dicts: framework is peer to SKRL, algorithm is PPO/APG.

    Legacy presets stored ``algorithm: RSL_RL``; that value is a framework, not
    an algorithm, and is rewritten to ``framework=RSL_RL, algorithm=PPO``.
    """
    if not isinstance(data, dict):
        return data
    data = dict(data)
    raw_algo = str(data.get("algorithm", "PPO") or "PPO").upper()
    raw_fw = data.get("framework")
    if raw_algo == FRAMEWORK_RSL_RL:
        data["framework"] = FRAMEWORK_RSL_RL
        data["algorithm"] = "PPO"
        return data
    data["framework"] = str(raw_fw or FRAMEWORK_SKRL).upper()
    data["algorithm"] = raw_algo
    return data


def validate_framework_algorithm(framework: str, algorithm: str) -> None:
    fw = framework.upper()
    algo = algorithm.upper()
    allowed_fw = sorted(FRAMEWORK_ALGORITHMS)
    if fw not in FRAMEWORK_ALGORITHMS:
        raise ValueError(f"Unknown framework '{fw}'. Available: {allowed_fw}")
    allowed_algo = sorted(FRAMEWORK_ALGORITHMS[fw])
    if algo not in FRAMEWORK_ALGORITHMS[fw]:
        raise ValueError(
            f"Framework {fw} does not support algorithm {algo}. Supported: {allowed_algo}"
        )


class PresetMetaConfig(BaseModel):
    id: str
    display_name: str = ""
    env_id: str
    obs_type: str = "state_based"
    framework: str = FRAMEWORK_SKRL
    algorithm: str = "PPO"
    policy_module: str
    trainer_module: str = "rl_framework.skrl_script.trainer_PPO"

    @model_validator(mode="before")
    @classmethod
    def _coerce_framework_algorithm(cls, data):
        return coerce_framework_algorithm(data)

    @model_validator(mode="after")
    def _check_framework_algorithm(self):
        validate_framework_algorithm(self.framework, self.algorithm)
        return self


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
    # Asymmetric critic observation size. ``None`` falls back to ``state_obs_size``
    # (symmetric critic).
    critic_obs_size: Optional[int] = None
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
    # Per-player controller overrides applied on top of environment YAML (Human / RL / Bot).
    player_ids: List[str] = Field(default_factory=list)
    # When set, overrides player ``object.control_policy_version`` (preset wins over env YAML).
    control_policy_version: Optional[str] = None

    @field_validator("player_ids", mode="before")
    @classmethod
    def _normalize_player_ids(cls, value):
        if not value:
            return []
        return normalize_player_controller_overrides(list(value))

    @field_validator("control_policy_version", mode="before")
    @classmethod
    def _normalize_control_policy_version(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class TrainingPresetConfig(BaseModel):
    meta: PresetMetaConfig
    model: ModelPresetConfig
    train: TrainPresetConfig


class ManifestEntry(BaseModel):
    id: str
    file: str
    display_name: str = ""
    env_id: str
    framework: str = FRAMEWORK_SKRL
    algorithm: str = "PPO"

    @model_validator(mode="before")
    @classmethod
    def _coerce_framework_algorithm(cls, data):
        return coerce_framework_algorithm(data)

    @model_validator(mode="after")
    def _check_framework_algorithm(self):
        validate_framework_algorithm(self.framework, self.algorithm)
        return self


class ManifestConfig(BaseModel):
    presets: List[ManifestEntry] = Field(default_factory=list)
