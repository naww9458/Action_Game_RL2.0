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
    algorithm: str = "PPO"
    preset_id: str = ""
    policy_module: str = ""
    trainer_module: str = "skrl_script.trainer_PPO"
    start_time: str = ""
    num_envs: Optional[int] = None
    level: Optional[int] = None
    sub_level: Optional[int] = None
    checkpoints: List[CheckpointInfo] = field(default_factory=list)
    latest_checkpoint_step: Optional[int] = None
    has_tensorboard: bool = False

    @property
    def display_label(self) -> str:
        parts = [self.name]
        if self.algorithm:
            parts.append(f"[{self.algorithm}]")
        if self.preset_id:
            parts.append(f"[{self.preset_id}]")
        elif self.level is not None and self.sub_level is not None:
            parts.append(f"[Level{self.level}-{self.sub_level}]")
        if self.latest_checkpoint_step is not None:
            parts.append(f"step={self.latest_checkpoint_step}")
        return " ".join(parts)


class RunsManager:
    CHECKPOINT_PATTERN = re.compile(r"^agent_(\d+)\.pt$")
    RUN_LEVEL_PATTERN = re.compile(r"_Level(\d+)-(\d+)$", re.IGNORECASE)

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

    def list_runs(self, level: Optional[int] = None, sub_level: Optional[int] = None) -> List[RunInfo]:
        if not self.runs_dir.exists():
            return []

        runs: List[RunInfo] = []
        for entry in sorted(self.runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not entry.is_dir():
                continue
            info = self.get_run_info(entry.name)
            if info is None:
                continue
            if level is not None and info.level != level:
                continue
            if sub_level is not None and info.sub_level != sub_level:
                continue
            runs.append(info)
        return runs

    @classmethod
    def parse_level_from_run_name(cls, run_name: str) -> tuple[Optional[int], Optional[int]]:
        match = cls.RUN_LEVEL_PATTERN.search(run_name)
        if not match:
            return None, None
        return int(match.group(1)), int(match.group(2))

    @classmethod
    def infer_run_metadata(
        cls,
        run_dir: Path,
        manifest: Optional[Dict[str, Any]] = None,
        *,
        model_obs_type: str = "state_based",
    ) -> Dict[str, Any]:
        manifest = dict(manifest or {})
        run_name = run_dir.name

        level = manifest.get("level")
        sub_level = manifest.get("sub_level")
        if level is None or sub_level is None:
            parsed_level, parsed_sub = cls.parse_level_from_run_name(run_name)
            level = level if level is not None else parsed_level
            sub_level = sub_level if sub_level is not None else parsed_sub
        if level is not None:
            manifest.setdefault("level", level)
        if sub_level is not None:
            manifest.setdefault("sub_level", sub_level)

        algorithm = manifest.get("algorithm")
        if not algorithm:
            if "APG" in run_name and "PPO" not in run_name:
                algorithm = "APG"
            elif "PPO" in run_name:
                algorithm = "PPO"
            else:
                algorithm = "PPO"
            manifest["algorithm"] = algorithm

        algo = str(manifest["algorithm"]).upper()
        if not manifest.get("trainer_module"):
            manifest["trainer_module"] = (
                "skrl_script.trainer_APG" if algo == "APG" else "skrl_script.trainer_PPO"
            )

        if (
            (not manifest.get("policy_module") or not manifest.get("preset_id"))
            and manifest.get("level") is not None
            and manifest.get("sub_level") is not None
        ):
            try:
                from training.level_defaults import resolve_preset_id
                from training.registry import TrainingPresetRegistry

                obs_type = manifest.get("obs_type", model_obs_type)
                preset_id = resolve_preset_id(
                    algo,
                    int(manifest["level"]),
                    int(manifest["sub_level"]),
                    obs_type,
                )
                preset_meta = TrainingPresetRegistry.load_preset_yaml(preset_id).meta
                manifest.setdefault("preset_id", preset_id)
                manifest.setdefault("policy_module", preset_meta.policy_module)
                manifest.setdefault("trainer_module", preset_meta.trainer_module)
                manifest.setdefault("algorithm", preset_meta.algorithm)
            except KeyError:
                pass

        return manifest

    def resolve_run_dir(self, run_name_or_path: str) -> Path:
        candidate = Path(run_name_or_path)
        if candidate.is_absolute() and candidate.exists():
            return candidate.resolve()
        if candidate.exists():
            return candidate.resolve()
        run_path = self.runs_dir / run_name_or_path
        if run_path.exists():
            return run_path.resolve()
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

        level = manifest.get("level")
        sub_level = manifest.get("sub_level")
        if level is None or sub_level is None:
            parsed_level, parsed_sub = self.parse_level_from_run_name(run_dir.name)
            level = level if level is not None else parsed_level
            sub_level = sub_level if sub_level is not None else parsed_sub

        algorithm = manifest.get("algorithm", "PPO")

        return RunInfo(
            name=run_dir.name,
            path=run_dir,
            algorithm=algorithm,
            preset_id=manifest.get("preset_id", ""),
            policy_module=manifest.get("policy_module", ""),
            trainer_module=manifest.get("trainer_module", "skrl_script.trainer_PPO"),
            start_time=manifest.get("start_time", ""),
            num_envs=manifest.get("num_envs"),
            level=level,
            sub_level=sub_level,
            checkpoints=checkpoints,
            latest_checkpoint_step=latest_step,
            has_tensorboard=any(run_dir.glob("events.out.tfevents.*")),
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
                "trainer_module": meta.get("trainer_module", "skrl_script.trainer_PPO"),
                "algorithm": meta.get("algorithm", "PPO"),
                "level": meta.get("level"),
                "sub_level": meta.get("sub_level"),
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
        level: int,
        sub_level: int,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        manifest = {
            "preset_id": preset_id,
            "policy_module": policy_module,
            "trainer_module": trainer_module,
            "algorithm": algorithm,
            "num_envs": num_envs,
            "level": level,
            "sub_level": sub_level,
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
        level_cfg: Dict[str, Any],
    ) -> None:
        config_dir.mkdir(parents=True, exist_ok=True)

        preset_path = config_dir / "preset.yaml"
        with open(preset_path, "w", encoding="utf-8") as f:
            yaml.dump(preset_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        manifest_path = config_dir / "run_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        from skrl_script.trainer_base import Trainer_base
        trainer_base = Trainer_base()
        trainer_base.save_config_pickle(model_cfg, train_cfg, level_cfg, str(config_dir))

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
