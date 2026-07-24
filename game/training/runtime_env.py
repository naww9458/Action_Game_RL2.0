from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path


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


def make_experiment_name(level: int, sub_level: int, algorithm: str = "PPO") -> str:
    timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
    return f"{timestamp}_{algorithm}_Level{level}-{sub_level}"


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
