from typing import Type # 記得導入 Type

class FrozenClass(type):
    def __setattr__(cls, key, value):
        # 如果屬性已經存在（且不是內部的私有變量），則禁止修改
        if hasattr(cls, key) and not key.startswith("_"):
            raise AttributeError(f"GameConfig 屬性 '{key}' 已鎖定，不可二次修改")
        super().__setattr__(key, value)

class GameConfig(metaclass=FrozenClass):
    # 僅定義類型註解（Type Hinting），不賦予初始值
    # 這樣在調用 init_from_configs 之前，存取這些變數會直接報錯 (AttributeError)
    space_x: int
    space_y: int
    space_z: int
    interval_distance: int
    GRAVITY: tuple
    DAMPING: tuple
    FPS_ACTION: int
    SUB_STEPS: int
    GROUND_FRICTION: float
    SOLVER_CONFIG: dict

    DEVICE: str

    ACTION_SPACE_CONFIG: dict
    ACTION_SHAPE_OFFSET: int

    NUM_PLAYERS: int 
    NUM_OBJECTS_TOTAL: int

    SEED: int
    requires_grad: bool

    reward_components: list
    reward_components_diff: list
    reward_parameters: dict

    Enable_World_Offset: bool = True # TODO 應該想一個方法變成開關，目前是寫死狀態

    @classmethod
    def init_from_configs(cls, env_cfg: dict):
        """
        從配置字典初始化。如果缺少任何必要參數，直接拋出異常。
        """
        required_env_keys = ["space_xyz", "gravity", "damping"]
        for key in required_env_keys:
            if key not in env_cfg:
                raise ValueError(f"配置錯誤：在 environment_configs 中找不到必要參數 '{key}'")

        try:
            cls.space_x = int(env_cfg["space_xyz"][0])
            cls.space_y = int(env_cfg["space_xyz"][1])
            cls.space_z = int(env_cfg["space_xyz"][2])
            cls.interval_distance = int(env_cfg["interval_distance"])
            cls.GRAVITY = tuple(env_cfg["gravity"])
            cls.DAMPING = tuple(env_cfg["damping"])
            cls.FPS_ACTION = float(env_cfg["fps_action"])
            cls.SUB_STEPS = int(env_cfg["sub_steps"])
            cls.GROUND_FRICTION = int(env_cfg["ground_shape_friction"])
            cls.SOLVER_CONFIG = env_cfg["solver_config"]
        except (TypeError, ValueError) as e:
            raise ValueError(f"配置錯誤：參數類型不正確或格式錯誤。細節: {e}")

        print(f"[Config] 全域參數載入成功：{cls.space_x}x{cls.space_y}x{cls.space_z}, Gravity: {cls.GRAVITY}")


