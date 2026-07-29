
import yaml
import json

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path

from script.role.controller_utils import (
    CONTROLLER_CHOICES,
    infer_controller_from_legacy_name,
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


def _migrate_item_role_id_to_name(item: Dict[str, Any]) -> None:
    legacy_id = item.pop("role_id", None)
    if legacy_id and not item.get("name"):
        item["name"] = legacy_id


def _migrate_role_id_to_name(data: Dict[str, Any]) -> None:
    if not data:
        return

    for section in ("player_configs", "platform_configs"):
        for item in data.get(section, []) or []:
            if isinstance(item, dict):
                _migrate_item_role_id_to_name(item)

    for item in (data.get("entity_configs") or {}).values():
        if isinstance(item, dict):
            _migrate_item_role_id_to_name(item)

    for item in (data.get("tool_configs") or {}).values():
        if isinstance(item, dict):
            _migrate_item_role_id_to_name(item)

    for item in (data.get("ability_generated_object_configs") or {}).values():
        if isinstance(item, dict):
            _migrate_item_role_id_to_name(item)


def _migrate_player_controller_fields(data: Dict[str, Any]) -> None:
    for item in data.get("player_configs", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("controller") in CONTROLLER_CHOICES:
            continue
        legacy = item.get("role_id") or item.get("name") or ""
        item["controller"] = infer_controller_from_legacy_name(legacy)


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
    tool_configs: Dict[str, ToolModel] = {}
    ability_generated_object_configs: Dict[str, AbilityGeneratedObjectModel] = {}

    @classmethod
    def load(
        cls,
        config_path: Path,
        overrides: Dict[str, Any] = None,
        player_controllers: List[str] | None = None,
    ) -> 'LevelConfig':
        data = None

        base_dir = Path(__file__).parent.resolve()
        path = Path(base_dir / config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file did not exist! File path: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            # 這裡會自動處理 YAML/JSON 格式
            data = yaml.safe_load(f) if path.suffix in ['.yaml', '.yml'] else json.load(f)
            if data:
                # 處理 JSON 中特殊的 space_x/y/z 轉為 space_xyz 的邏輯（如果有的話）
                if "environment_configs" in data:
                    env = data["environment_configs"]
                    if "space_x" in env and "space_xyz" not in env:
                        env["space_xyz"] = [env.pop("space_x"), env.pop("space_y"), env.pop("space_z")]

        if data:
            _migrate_player_controller_fields(data)
            _migrate_role_id_to_name(data)
            if player_controllers:
                apply_player_controller_overrides(data, player_controllers)
            if overrides:
                for key, value in overrides.items():
                    if value is not None:
                        data[key] = value
            if "player_configs" in data:
                for item in data.get("player_configs", []) or []:
                    if not isinstance(item, dict):
                        continue
                    obj = item.get("object")
                    if isinstance(obj, dict) and obj.get("type") == "usd_unitree_g1":
                        obj["type"] = "usd"
                        obj.setdefault("use_mujoco_policy_init", True)

        return cls.model_validate(data)
