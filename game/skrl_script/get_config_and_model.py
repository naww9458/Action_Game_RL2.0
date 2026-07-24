
from training.level_defaults import resolve_preset_id
from training.loader import TrainingPresetLoader


def get_config_and_model(algorithm: str, level: int, sub_level: int, obs_type: str):
    preset_id = resolve_preset_id(algorithm, level, sub_level, obs_type)
    loaded = TrainingPresetLoader.load(preset_id)
    value_cls = loaded.Value
    return loaded.model_cfg, loaded.train_cfg, loaded.Policy, value_cls
