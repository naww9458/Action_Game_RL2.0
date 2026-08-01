import os
import sys
import importlib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from script.levels.levels import Level_Default, Levels
from script.levels.level_cfg import *
from script.game_config import GameConfig

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from script.game import Game
    from script.levels.levels import Levels

from typing import Type


def _import_class(class_path: str) -> Type:
    module_name, _, class_name = class_path.replace(":", ".").rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"Invalid level_class path: {class_path}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _resolve_level_class(level: int, sub_level: int, config_obj: LevelConfig) -> Type:
    if config_obj.level_class:
        return _import_class(config_obj.level_class)

    module_name = f"script.levels.level{level}.level{level}_{sub_level}"
    class_name = f"Level{level}_{sub_level}"
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
    except AttributeError:
        pass

    print(f"No level class found for Level {level}_{sub_level}, automatically changed to parent class Level_Default")
    return Level_Default

def get_level(level: int, 
              sub_level: int,
              game: 'Game',
              level_config_path,
              player_controllers: list[str] | None = None,
              **runtime_overrides
) -> Levels:
    """
    獲取關卡實例，具備自動配置加載與校驗功能。
    """

    config_obj = None
    
    # 1. 基礎校驗
    if level < 4:
        raise ValueError(f"Levels 1-3 are deprecated. Requested: {level}")

    # 2. 配置路徑處理
    if level_config_path is not None:
        config_path = level_config_path
    else:
        config_path = f"level{level}/level_{level}_{sub_level}_default_cfg.yaml"
    
    # 3. 加載與解析配置 (使用 Pydantic)
    try:
        # 將傳入的 runtime_overrides (如 player_configs) 傳給模型
        print("config_path: ", config_path)
        config_obj = LevelConfig.load(
            config_path,
            overrides=runtime_overrides,
            player_controllers=player_controllers,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        # The config is required downstream (level_class resolution, GameConfig,
        # environment params). Swallowing it would only produce a confusing
        # 'NoneType' AttributeError later, so surface the real error instead.
        raise e

    level_class = _resolve_level_class(level, sub_level, config_obj)

    # 4. 初始化全局 GameConfig (單例更新)
    env_data = config_obj.environment_configs.model_dump() # Convert to dict
    try:
        GameConfig.init_from_configs(env_data)
    except AttributeError:
        print("Warning: GameConfig attributes are already initialized and immutable.")

    # 5. 更新物理引擎參數
    if game and hasattr(game, 'physics_manager'):
        game.physics_manager.set_env_params(
            tuple(config_obj.environment_configs.gravity), 
            config_obj.environment_configs.damping
        )

    # 6. 返回實例化對象
    # 注意：這裡直接傳遞模型對象或其字典，建議 Level 類也改為接收模型
    return level_class(
        game=game,
        level_configs=config_obj.model_dump()
    )