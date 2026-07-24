from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import yaml
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR

from training.schema import ManifestConfig, TrainingPresetConfig

PREPROCESSOR_REGISTRY: Dict[str, Optional[Type]] = {
    "RunningStandardScaler": RunningStandardScaler,
    "null": None,
    "none": None,
}

SCHEDULER_REGISTRY: Dict[str, Optional[Type]] = {
    "KLAdaptiveLR": KLAdaptiveLR,
    "null": None,
    "none": None,
}


def resolve_class_ref(registry: Dict[str, Optional[Type]], type_name: Optional[str]) -> Optional[Type]:
    if type_name is None or type_name in ("null", "none"):
        return None
    if type_name not in registry:
        raise KeyError(f"Unknown type '{type_name}'. Available: {sorted(registry.keys())}")
    return registry[type_name]


def import_policy_classes(policy_module: str) -> Tuple[Type, Type]:
    mod = importlib.import_module(policy_module)
    return mod.Policy, mod.Value if hasattr(mod, "Value") else None # TODO 


def import_trainer_class(trainer_module: str) -> Type:
    mod = importlib.import_module(trainer_module)
    return mod.Trainer


class TrainingPresetRegistry:
    _presets_dir = Path(__file__).parent / "presets"

    @classmethod
    def presets_dir(cls) -> Path:
        return cls._presets_dir

    @classmethod
    def manifest_path(cls) -> Path:
        return cls._presets_dir / "manifest.yaml"

    @classmethod
    def load_manifest(cls) -> ManifestConfig:
        path = cls.manifest_path()
        if not path.exists():
            return ManifestConfig(presets=[])
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return ManifestConfig.model_validate(data)

    @classmethod
    def save_manifest(cls, manifest: ManifestConfig) -> None:
        path = cls.manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(manifest.model_dump(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @classmethod
    def list_presets(cls) -> List[Dict[str, Any]]:
        manifest = cls.load_manifest()
        result = []
        for entry in manifest.presets:
            result.append({
                "id": entry.id,
                "file": entry.file,
                "display_name": entry.display_name or entry.id,
                "level": entry.level,
                "sub_level": entry.sub_level,
                "algorithm": entry.algorithm,
                "path": str(cls._presets_dir / entry.file),
            })
        return result

    @classmethod
    def resolve_preset_path(cls, preset_id_or_path: str) -> Path:
        candidate = Path(preset_id_or_path)
        if candidate.exists():
            return candidate.resolve()

        manifest = cls.load_manifest()
        for entry in manifest.presets:
            if entry.id == preset_id_or_path:
                path = cls._presets_dir / entry.file
                if not path.exists():
                    raise FileNotFoundError(f"Preset file not found: {path}")
                return path.resolve()

        path = cls._presets_dir / f"{preset_id_or_path}.yaml"
        if path.exists():
            return path.resolve()

        raise FileNotFoundError(f"Training preset not found: {preset_id_or_path}")

    @classmethod
    def load_preset_yaml(cls, preset_id_or_path: str) -> TrainingPresetConfig:
        path = cls.resolve_preset_path(preset_id_or_path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return TrainingPresetConfig.model_validate(data)

    @classmethod
    def save_preset_yaml(cls, preset: TrainingPresetConfig, preset_id_or_path: Optional[str] = None) -> Path:
        if preset_id_or_path and Path(preset_id_or_path).exists():
            path = Path(preset_id_or_path).resolve()
        else:
            path = cls._presets_dir / f"{preset.meta.id}.yaml"

        if not any(e.id == preset.meta.id for e in cls.load_manifest().presets):
            manifest = cls.load_manifest()
            manifest.presets.append({
                "id": preset.meta.id,
                "file": path.name,
                "display_name": preset.meta.display_name or preset.meta.id,
                "level": preset.meta.level,
                "sub_level": preset.meta.sub_level,
                "algorithm": preset.meta.algorithm,
            })
            cls.save_manifest(manifest)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(preset.model_dump(by_alias=True), f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return path.resolve()
