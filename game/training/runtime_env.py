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


# ``YY-MM-DD_HH-MM-SS-ffffff_{framework}_{algorithm}_Level{level}-{sub_level}[_N]``
# Framework may contain underscores (e.g. RSL_RL); algorithm is the last token
# before ``_Level``. An optional ``_N`` suffix disambiguates name collisions.
EXPERIMENT_NAME_PATTERN = re.compile(
    r"^(?P<timestamp>\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d+)"
    r"_(?P<framework>.+)_(?P<algorithm>[A-Za-z0-9]+)"
    r"_Level(?P<level>\d+)-(?P<sub_level>\d+)"
    r"(?:_(?P<seq>\d+))?$"
)


def _experiment_parent_dirs() -> list[Path]:
    root = get_project_root()
    return [root / "runs", root / "logs" / "rsl_rl"]


def uniquify_experiment_name(
    name: str,
    parent_dirs: Optional[list[Path]] = None,
) -> str:
    """If ``name`` already exists under any parent, append ``_1``, ``_2``, ..."""
    parents = parent_dirs if parent_dirs is not None else _experiment_parent_dirs()

    def taken(candidate: str) -> bool:
        return any((parent / candidate).exists() for parent in parents)

    if not taken(name):
        return name
    seq = 1
    while taken(f"{name}_{seq}"):
        seq += 1
    return f"{name}_{seq}"


def make_experiment_name(
    level: int,
    sub_level: int,
    algorithm: str,
    framework: str,
    *,
    parent_dirs: Optional[list[Path]] = None,
) -> str:
    timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
    fw = str(framework).upper()
    algo = str(algorithm).upper()
    base = f"{timestamp}_{fw}_{algo}_Level{level}-{sub_level}"
    return uniquify_experiment_name(base, parent_dirs)


def parse_experiment_name(name: str) -> Optional[dict]:
    """Inverse of ``make_experiment_name``. Returns None when the name does not match."""
    match = EXPERIMENT_NAME_PATTERN.fullmatch(str(name))
    if not match:
        return None
    seq = match.group("seq")
    return {
        "timestamp": match.group("timestamp"),
        "framework": match.group("framework").upper(),
        "algorithm": match.group("algorithm").upper(),
        "level": int(match.group("level")),
        "sub_level": int(match.group("sub_level")),
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
