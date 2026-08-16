"""Default training config for environments when GameConfig has no reward setup."""

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
    env_id: str,
    algorithm: str | None = None,
    obs_type: str | None = None,
    framework: str | None = None,
) -> list[str]:
    wanted = str(env_id or "").strip()
    matches = []
    for preset_id in _iter_preset_ids():
        meta, path = _load_preset_meta(preset_id)
        if str(meta.get("env_id") or "").strip() != wanted:
            continue
        from training.schema import coerce_framework_algorithm

        coerced = coerce_framework_algorithm(meta)
        meta_fw = str(coerced.get("framework", "SKRL")).upper()
        meta_algo = str(coerced.get("algorithm", "")).upper()
        if algorithm is not None and meta_algo != algorithm.upper():
            continue
        if framework is not None and meta_fw != framework.upper():
            continue
        if obs_type is not None and meta.get("obs_type") != obs_type:
            continue
        matches.append(str(meta.get("id") or Path(path).with_suffix("").name))
    return matches


def _expect_single_preset(matches: list[str], description: str) -> str:
    if not matches:
        raise KeyError(f"No YAML preset found for {description}")
    return matches[0]


@lru_cache(maxsize=None)
def get_default_train_cfg(env_id: str):
    preset_id = _expect_single_preset(
        _matching_preset_ids(env_id=env_id),
        f"env_id={env_id}",
    )
    from training.loader import TrainingPresetLoader
    return TrainingPresetLoader.load(preset_id).train_cfg


def resolve_preset_id(
    algorithm: str,
    env_id: str,
    obs_type: str,
    *,
    framework: str | None = None,
) -> str:
    return _expect_single_preset(
        _matching_preset_ids(
            algorithm=algorithm,
            env_id=env_id,
            obs_type=obs_type,
            framework=framework,
        ),
        f"framework={framework or '*'}, algorithm={algorithm}, "
        f"env_id={env_id}, obs_type={obs_type}",
    )
