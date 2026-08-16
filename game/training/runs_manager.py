from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class CheckpointInfo:
    name: str
    path: Path
    step: Optional[int] = None


@dataclass
class RunInfo:
    name: str
    path: Path
    framework: str
    algorithm: str
    trainer_module: str
    preset_id: str = ""
    policy_module: str = ""
    start_time: str = ""
    num_envs: Optional[int] = None
    env_id: str = ""
    checkpoints: List[CheckpointInfo] = field(default_factory=list)
    latest_checkpoint_step: Optional[int] = None
    has_tensorboard: bool = False

    @property
    def display_label(self) -> str:
        parts = [self.name]
        fw = (self.framework).upper()
        algo = (self.algorithm).upper()
        parts.append(f"[{fw}/{algo}]")
        if self.preset_id:
            parts.append(f"[{self.preset_id}]")
        elif self.env_id:
            parts.append(f"[{self.env_id}]")
        if self.latest_checkpoint_step is not None:
            parts.append(f"step={self.latest_checkpoint_step}")
        return " ".join(parts)


class RunsManager:
    CHECKPOINT_PATTERN = re.compile(r"^(?:agent|model)_(\d+)\.pt$")

    def __init__(self, runs_dir: Optional[Path] = None, project_root: Optional[Path] = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.runs_dir = Path(runs_dir) if runs_dir else self.project_root / "runs"
        if not self.runs_dir.is_absolute():
            self.runs_dir = (self.project_root / self.runs_dir).resolve()
        if not self.runs_dir.exists():
            candidates = [
                self.project_root / "runs",
                self.project_root / "game" / "runs",
                self.project_root.parent / "runs",
            ]
            for candidate in candidates:
                if candidate.exists():
                    self.runs_dir = candidate.resolve()
                    break

    def _framework_folder_names(self) -> set[str]:
        from training.schema import FRAMEWORK_RUN_FOLDER

        return set(FRAMEWORK_RUN_FOLDER.values())

    def _run_parent_dirs(self) -> List[Path]:
        from training.runtime_env import experiment_parent_dirs

        parents = experiment_parent_dirs(project_root=self.project_root)
        if self.runs_dir not in parents:
            parents.append(self.runs_dir)
        return parents

    @staticmethod
    def _looks_like_run_dir(path: Path) -> bool:
        if not path.is_dir():
            return False
        if (path / "checkpoints").is_dir() or (path / "config").is_dir():
            return True
        return any(path.glob("events.out.tfevents.*"))

    def _collect_run_dirs(self) -> List[Path]:
        skip = self._framework_folder_names()
        found: List[Path] = []
        seen: set[Path] = set()

        def add(path: Path) -> None:
            resolved = path.resolve()
            if resolved in seen:
                return
            seen.add(resolved)
            found.append(resolved)

        for parent in self._run_parent_dirs():
            if not parent.is_dir():
                continue
            scan_as_framework_bucket = parent.name in skip
            for entry in parent.iterdir():
                if not entry.is_dir():
                    continue
                if scan_as_framework_bucket:
                    if self._looks_like_run_dir(entry):
                        add(entry)
                    continue
                if entry.name in skip:
                    for child in entry.iterdir():
                        if self._looks_like_run_dir(child):
                            add(child)
                    continue
                if self._looks_like_run_dir(entry):
                    add(entry)
                    continue
                if parent.name.lower() in {"rsl_rl", "rsl-rl"}:
                    for child in entry.iterdir():
                        if child.is_dir() and self._looks_like_run_dir(child):
                            add(child)

        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return found

    def list_runs(self, env_id: Optional[str] = None) -> List[RunInfo]:
        runs: List[RunInfo] = []
        wanted = str(env_id).strip() if env_id else ""
        for entry in self._collect_run_dirs():
            info = self.get_run_info(str(entry))
            if info is None:
                continue
            if wanted and info.env_id != wanted:
                continue
            runs.append(info)
        return runs

    @staticmethod
    def _framework_from_run_path(run_dir: Path) -> Optional[str]:
        from training.schema import FRAMEWORK_RSL_RL, FRAMEWORK_RUN_FOLDER, FRAMEWORK_SKRL

        parts_lower = {part.lower() for part in run_dir.parts}
        rsl_folder = FRAMEWORK_RUN_FOLDER[FRAMEWORK_RSL_RL].lower()
        skrl_folder = FRAMEWORK_RUN_FOLDER[FRAMEWORK_SKRL].lower()
        if rsl_folder in parts_lower or "rsl_rl" in parts_lower:
            return FRAMEWORK_RSL_RL
        if skrl_folder in parts_lower:
            return FRAMEWORK_SKRL
        return None

    @classmethod
    def infer_run_metadata(
        cls,
        run_dir: Path,
        manifest: Optional[Dict[str, Any]] = None,
        *,
        model_obs_type: str = "state_based",
    ) -> Dict[str, Any]:
        from training.runtime_env import parse_experiment_name
        from training.schema import coerce_framework_algorithm

        manifest = dict(manifest or {})
        parsed = parse_experiment_name(run_dir.name)
        if parsed is None:
            parsed = parse_experiment_name(run_dir.parent.name)
        if parsed:
            for key in ("framework", "algorithm", "env_id"):
                value = parsed.get(key)
                if value not in (None, "") and manifest.get(key) in (None, ""):
                    manifest[key] = value
        if manifest.get("framework") in (None, ""):
            path_fw = cls._framework_from_run_path(run_dir)
            if path_fw:
                manifest["framework"] = path_fw
        manifest = coerce_framework_algorithm(manifest)

        if (
            (not manifest.get("policy_module") or not manifest.get("preset_id"))
            and manifest.get("env_id")
        ):
            try:
                from training.env_defaults import resolve_preset_id
                from training.registry import TrainingPresetRegistry

                obs_type = manifest.get("obs_type", model_obs_type)
                preset_id = resolve_preset_id(
                    str(manifest["algorithm"]),
                    str(manifest["env_id"]),
                    obs_type,
                    framework=str(manifest["framework"]),
                )
                preset_meta = TrainingPresetRegistry.load_preset_yaml(preset_id).meta
                manifest.setdefault("preset_id", preset_id)
                manifest.setdefault("policy_module", preset_meta.policy_module)
                manifest.setdefault("trainer_module", preset_meta.trainer_module)
                manifest.setdefault("algorithm", preset_meta.algorithm)
                manifest.setdefault("framework", preset_meta.framework)
                manifest.setdefault("env_id", preset_meta.env_id)
            except KeyError:
                pass

        return manifest

    def resolve_run_dir(self, run_name_or_path: str) -> Path:
        candidate = Path(run_name_or_path)
        if candidate.is_absolute() and candidate.exists():
            return candidate.resolve()
        if candidate.exists():
            return candidate.resolve()
        for parent in self._run_parent_dirs():
            run_path = parent / run_name_or_path
            if run_path.exists():
                return run_path.resolve()
            if parent.name == "rsl_rl" and parent.is_dir():
                for nested in parent.iterdir():
                    nested_path = nested / run_name_or_path
                    if nested.is_dir() and nested_path.exists():
                        return nested_path.resolve()
        raise FileNotFoundError(f"Run not found: {run_name_or_path}")

    def get_run_info(self, run_name_or_path: str) -> Optional[RunInfo]:
        try:
            run_dir = self.resolve_run_dir(run_name_or_path)
        except FileNotFoundError:
            return None

        manifest = self._load_run_manifest(run_dir)
        manifest = self.infer_run_metadata(run_dir, manifest)
        checkpoints = self._list_checkpoints(run_dir)
        latest_step = None
        for ckpt in checkpoints:
            if ckpt.step is not None:
                latest_step = max(latest_step or 0, ckpt.step)

        env_id = str(manifest.get("env_id") or "")
        if not env_id:
            from training.runtime_env import parse_experiment_name

            parsed = parse_experiment_name(run_dir.name)
            if parsed and parsed.get("env_id"):
                env_id = str(parsed["env_id"])

        algorithm = manifest.get("algorithm")
        from training.schema import normalize_framework_id

        framework = normalize_framework_id(manifest.get("framework"))

        return RunInfo(
            name=run_dir.name,
            path=run_dir,
            framework=framework,
            algorithm=algorithm,
            preset_id=manifest.get("preset_id", ""),
            policy_module=manifest.get("policy_module", ""),
            trainer_module=manifest.get("trainer_module"),
            start_time=manifest.get("start_time", ""),
            num_envs=manifest.get("num_envs"),
            env_id=env_id,
            checkpoints=checkpoints,
            latest_checkpoint_step=latest_step,
            has_tensorboard=any(run_dir.rglob("events.out.tfevents.*")),
        )

    def _load_run_manifest(self, run_dir: Path) -> Dict[str, Any]:
        manifest_path = run_dir / "config" / "run_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)

        preset_path = run_dir / "config" / "preset.yaml"
        if preset_path.exists():
            with open(preset_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            meta = data.get("meta", {})
            return {
                "preset_id": meta.get("id", ""),
                "policy_module": meta.get("policy_module", ""),
                "trainer_module": meta.get("trainer_module"),
                "framework": meta.get("framework"),
                "algorithm": meta.get("algorithm"),
                "env_id": meta.get("env_id", ""),
            }
        return {}

    def _list_checkpoints(self, run_dir: Path) -> List[CheckpointInfo]:
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            return []

        checkpoints: List[CheckpointInfo] = []
        for file in sorted(ckpt_dir.glob("*.pt")):
            step = None
            match = self.CHECKPOINT_PATTERN.match(file.name)
            if match:
                step = int(match.group(1))
            checkpoints.append(CheckpointInfo(name=file.stem, path=file.resolve(), step=step))

        checkpoints.sort(key=lambda c: (c.step is None, -(c.step or 0)))
        return checkpoints

    def checkpoint_path(self, run_name_or_path: str, checkpoint: str) -> Path:
        run_info = self.get_run_info(run_name_or_path)
        if run_info is None:
            raise FileNotFoundError(f"Run not found: {run_name_or_path}")

        checkpoint_name = checkpoint.removesuffix(".pt")
        for ckpt in run_info.checkpoints:
            if ckpt.name == checkpoint_name:
                return ckpt.path

        candidate = run_info.path / "checkpoints" / f"{checkpoint_name}.pt"
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint} in {run_info.name}")

    def build_run_manifest(
        # self,
        preset_id: str,
        policy_module: str,
        trainer_module: str,
        algorithm: str,
        num_envs: int,
        env_id: str,
        framework: str,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        manifest = {
            "preset_id": preset_id,
            "policy_module": policy_module,
            "trainer_module": trainer_module,
            "framework": framework,
            "algorithm": algorithm,
            "num_envs": num_envs,
            "env_id": env_id,
            "start_time": datetime.now().isoformat(timespec="seconds"),
        }
        if resume_from:
            manifest["resume_from"] = resume_from
        return manifest

    @staticmethod
    def save_run_artifacts(
        config_dir: Path,
        preset_data: Dict[str, Any],
        manifest: Dict[str, Any],
        model_cfg,
        train_cfg,
        environment_cfg: Dict[str, Any],
    ) -> None:
        config_dir.mkdir(parents=True, exist_ok=True)

        preset_path = config_dir / "preset.yaml"
        with open(preset_path, "w", encoding="utf-8") as f:
            yaml.dump(preset_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        manifest_path = config_dir / "run_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        from rl_framework.skrl_script.trainer_base import Trainer_base
        trainer_base = Trainer_base()
        trainer_base.save_config_pickle(model_cfg, train_cfg, environment_cfg, str(config_dir))

    def launch_tensorboard(self, run_name_or_path: str, port: int = 6006) -> subprocess.Popen:
        from training.runtime_env import resolve_tensorboard_command

        run_dir = self.resolve_run_dir(run_name_or_path)
        cmd = resolve_tensorboard_command(str(run_dir), port)
        return subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
