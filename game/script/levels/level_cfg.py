
import yaml
import json

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path

from script.role.controller_utils import (
    normalize_controller,
    parse_controller_override,
)
from script.role.player import PlayerModel
from script.role.platform import PlatformModel
from script.role.entity import EntityModel
from script.role.tool import ToolModel
from script.role.ability_generated_object import AbilityGeneratedObjectModel

from script.simulate.solvers.base_solver import SolverRegistry
from script.simulate.solvers.xpbd import XPBDSolverModel

SolverConfig = SolverRegistry.get_solver_union()


def apply_player_controller_overrides(
    data: Dict[str, Any], controllers: List[str]
) -> None:
    """Override per-player ``controller`` fields from a training preset."""
    if not controllers:
        return

    players = data.get("player_configs") or []
    if not players:
        raise ValueError("Cannot apply player controller overrides: player_configs is empty.")
    if len(controllers) != len(players):
        raise ValueError(
            f"player controller override count ({len(controllers)}) "
            f"does not match player_configs count ({len(players)})."
        )

    for item, raw in zip(players, controllers):
        if not isinstance(item, dict):
            continue
        item["controller"] = normalize_controller(parse_controller_override(raw))


# --- 環境與總配置 ---
class EnvironmentConfig(BaseModel):
    space_xyz: List[float] = [20, 20, 20]
    interval_distance: float = 5.0
    gravity: List[float] = [0, 0, -9.8]
    damping: List[float] = [0.1, 0.1]
    fps_action: int = 30
    sub_steps: int = 6
    ground_shape_friction: float = 0.5
    solver_config: SolverConfig = XPBDSolverModel()

class LevelConfig(BaseModel):
    level_class: Optional[str] = None
    environment_configs: EnvironmentConfig = EnvironmentConfig()
    player_configs: List[PlayerModel] = []
    platform_configs: List[PlatformModel] = []
    entity_configs: Dict[str, EntityModel] = {}
    tool_configs: List[ToolModel] = []
    ability_generated_object_configs: Dict[str, AbilityGeneratedObjectModel] = {}

    @classmethod
    def load(
        cls,
        config_path: Path,
        overrides: Dict[str, Any] = None,
        player_controllers: List[str] | None = None,
    ) -> 'LevelConfig':
        base_dir = Path(__file__).parent.resolve()
        path = Path(base_dir / config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file did not exist! File path: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            # 自動處理 YAML/JSON 格式
            data = yaml.safe_load(f) if path.suffix in ['.yaml', '.yml'] else json.load(f)

        if data:
            if player_controllers:
                apply_player_controller_overrides(data, player_controllers)
            if overrides:
                for key, value in overrides.items():
                    if value is not None:
                        data[key] = value

        return cls.model_validate(data)
