"""Default training config for levels when GameConfig has no reward setup."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import yaml


def _iter_preset_ids_from_manifest() -> Iterable[str]:
    from training.registry import TrainingPresetRegistry

    manifest = TrainingPresetRegistry.load_manifest()
    for entry in manifest.presets:
        yield entry.id


def _iter_preset_ids_from_files() -> Iterable[str]:
    from training.registry import TrainingPresetRegistry

    for path in TrainingPresetRegistry.presets_dir().glob("*.yaml"):
        if path.name == "manifest.yaml":
            continue
        yield str(path)


def _iter_preset_ids() -> Iterable[str]:
    manifest_ids = list(_iter_preset_ids_from_manifest())
    if manifest_ids:
        yield from manifest_ids
        return
    yield from _iter_preset_ids_from_files()


def _load_preset_meta(preset_id_or_path: str):
    from training.registry import TrainingPresetRegistry

    path = TrainingPresetRegistry.resolve_preset_path(preset_id_or_path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("meta", {}), path


def _matching_preset_ids(
    *,
    level: int,
    sub_level: int,
    algorithm: str | None = None,
    obs_type: str | None = None,
) -> list[str]:
    matches = []
    for preset_id in _iter_preset_ids():
        meta, path = _load_preset_meta(preset_id)
        if meta.get("level") != level or meta.get("sub_level") != sub_level:
            continue
        if algorithm is not None and str(meta.get("algorithm", "")).upper() != algorithm.upper():
            continue
        if obs_type is not None and meta.get("obs_type") != obs_type:
            continue
        matches.append(str(meta.get("id") or Path(path).with_suffix("").name))
    return matches


def _expect_single_preset(matches: list[str], description: str) -> str:
    if not matches:
        raise KeyError(f"No YAML preset found for {description}")
    # if len(matches) > 1: # TODO Change function name
    #     raise KeyError(f"Multiple YAML presets found for {description}: {matches}")
    return matches[0]


@lru_cache(maxsize=None)
def get_default_train_cfg(level: int, sub_level: int):
    preset_id = _expect_single_preset(
        _matching_preset_ids(level=level, sub_level=sub_level),
        f"level {level}_{sub_level}",
    )
    from training.loader import TrainingPresetLoader
    return TrainingPresetLoader.load(preset_id).train_cfg


def resolve_preset_id(algorithm: str, level: int, sub_level: int, obs_type: str) -> str:
    return _expect_single_preset(
        _matching_preset_ids(
            algorithm=algorithm,
            level=level,
            sub_level=sub_level,
            obs_type=obs_type,
        ),
        f"algorithm={algorithm}, level={level}_{sub_level}, obs_type={obs_type}",
    )
