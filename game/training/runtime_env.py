from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_game_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _has_project_assets(root: Path) -> bool:
    return (root / "Action_Game_RL_Assets" / "assets").is_dir()


def get_project_root() -> Path:
    game_dir = get_game_dir()
    cwd = Path.cwd()
    if _has_project_assets(cwd):
        return cwd.resolve()
    if _has_project_assets(game_dir.parent):
        return game_dir.parent.resolve()
    return game_dir.parent.resolve()


def ensure_runtime_env() -> tuple[Path, Path]:
    """Add game/ to sys.path and chdir to project root so Action_Game_RL_Assets/assets/ resolves."""
    game_dir = get_game_dir()
    project_root = get_project_root()
    if str(game_dir) not in sys.path:
        sys.path.insert(0, str(game_dir))
    os.chdir(project_root)
    return game_dir, project_root


# ``YY-MM-DD_HH-MM-SS-ffffff_{framework}_{algorithm}_{env_id}[_N]``
# Algorithm is a known token (PPO/APG) so slugs with underscores stay intact.
_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9_]+")
_EXPERIMENT_NAME_PATTERN = None


def _experiment_name_pattern() -> re.Pattern[str]:
    global _EXPERIMENT_NAME_PATTERN
    if _EXPERIMENT_NAME_PATTERN is None:
        from training.schema import FRAMEWORK_ALGORITHMS

        algos = "|".join(
            sorted({a for group in FRAMEWORK_ALGORITHMS.values() for a in group}, key=len, reverse=True)
        )
        _EXPERIMENT_NAME_PATTERN = re.compile(
            r"^(?P<timestamp>\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d+)"
            rf"_(?P<framework>.+)_(?P<algorithm>{algos})_"
            r"(?P<env_id>[A-Za-z][A-Za-z0-9_]*)"
            r"(?:_(?P<seq>\d+))?$",
            re.IGNORECASE,
        )
    return _EXPERIMENT_NAME_PATTERN


def framework_runs_dir(framework: str, project_root: Optional[Path] = None) -> Path:
    """Return ``<project>/runs/<SKRL|RSL-rl>`` for the given framework."""
    from training.schema import framework_run_folder_name

    root = Path(project_root) if project_root else get_project_root()
    return (root / "runs" / framework_run_folder_name(framework)).resolve()


def experiment_parent_dirs(
    framework: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> list[Path]:
    """Directories that may contain run folders (new layout plus legacy)."""
    from training.schema import FRAMEWORK_RUN_FOLDER

    root = Path(project_root) if project_root else get_project_root()
    dirs = [framework_runs_dir(fw, root) for fw in FRAMEWORK_RUN_FOLDER]
    legacy = [root / "runs", root / "logs" / "rsl_rl"]
    if framework:
        target = framework_runs_dir(framework, root)
        dirs = [target] + [d for d in dirs if d != target]
    for path in legacy:
        resolved = path.resolve()
        if resolved not in dirs:
            dirs.append(resolved)
    return dirs


def uniquify_experiment_name(
    name: str,
    parent_dirs: Optional[list[Path]] = None,
) -> str:
    """If ``name`` already exists under any parent, append ``_1``, ``_2``, ..."""
    parents = parent_dirs if parent_dirs is not None else experiment_parent_dirs()

    def taken(candidate: str) -> bool:
        return any((parent / candidate).exists() for parent in parents)

    if not taken(name):
        return name
    seq = 1
    while taken(f"{name}_{seq}"):
        seq += 1
    return f"{name}_{seq}"


def _safe_env_slug(stem: str) -> str:
    slug = _SLUG_UNSAFE.sub("_", str(stem or "")).strip("_")
    if slug and slug[0].isalpha():
        return slug
    return f"env_{slug}" if slug else "env"


def experiment_env_slug(env_id: str) -> str:
    """Sanitize an environment id for use in a run folder name."""
    return _safe_env_slug(env_id)


def make_experiment_name(
    env_id: str,
    algorithm: str,
    framework: str,
    *,
    parent_dirs: Optional[list[Path]] = None,
) -> str:
    timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
    fw = str(framework).upper()
    algo = str(algorithm).upper()
    slug = experiment_env_slug(env_id)
    base = f"{timestamp}_{fw}_{algo}_{slug}"
    parents = parent_dirs if parent_dirs is not None else experiment_parent_dirs(fw)
    return uniquify_experiment_name(base, parents)


def parse_experiment_name(name: str) -> Optional[dict]:
    """Inverse of ``make_experiment_name``."""
    match = _experiment_name_pattern().fullmatch(str(name))
    if not match:
        return None
    seq = match.group("seq")
    return {
        "timestamp": match.group("timestamp"),
        "framework": match.group("framework").upper(),
        "algorithm": match.group("algorithm").upper(),
        "env_id": match.group("env_id"),
        "seq": int(seq) if seq is not None else None,
    }


def resolve_tensorboard_command(logdir: str, port: int) -> list[str]:
    scripts_dir = Path(sys.executable).parent
    for candidate in (scripts_dir / "tensorboard.exe", scripts_dir / "tensorboard"):
        if candidate.exists():
            return [str(candidate), "--logdir", logdir, "--port", str(port)]

    import shutil
    for name in ("tensorboard", "tensorboard.exe"):
        exe = shutil.which(name)
        if exe:
            return [exe, "--logdir", logdir, "--port", str(port)]

    return [sys.executable, "-m", "tensorboard.main", "--logdir", logdir, "--port", str(port)]
