import yaml  # 修改：由 json 改為 yaml
import os
import warp as wp
import torch

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from abc import ABC, abstractmethod
from dataclasses import dataclass
from script.role.abilities.key_mapping import KeyMapping
from script.game_config import GameConfig
from script.role.abilities.abilities_cfg import AbilitiesConfig  # 導入 Pydantic 模型

from script.role.abilities.articulation_control_config.profile_registry import (
    find_player_config_for_ability,
    resolve_articulation_player_pattern,
    resolve_control_policy_version,
)

if TYPE_CHECKING:
    import numpy as np
    # 將導致循環導入的 import 語句移到這裡
    from script.levels.levels import Levels
    from script.simulate.physics_manager import PhysicsManager
    from script.role.bodies.articulation_body import ArticulationBody
    from script.role.bodies.deformable_body import DeformableBody

    from script.role.ability_generated_object import AbilityGeneratedObject
    

@dataclass
class PatternViewContext:
    pattern: str
    view_idx: int = -1
    count_per_world: int = 0
    bodies_per_object: int = 0

    @property
    def valid(self) -> bool:
        return self.view_idx >= 0


class Ability(ABC):
    _default_configs: AbilitiesConfig = None  # 用於緩存配置的類變量 (Pydantic Model)
    _fps = None  # 用於儲存 FPS 的類變量
    physics_manager: 'PhysicsManager' = None

    articulation_body: 'ArticulationBody' = None
    deformable_body: 'DeformableBody' = None


    # 這裡存放所有具體的 Ability 子類 (類別本身，不是實例)
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # 如果不是抽象類，就自動註冊
        import inspect

        if not inspect.isabstract(cls):
            Ability._registry[cls.__name__] = cls

    @abstractmethod
    def __init__(self, ability_name: str):
        self.ability_name = ability_name
        self.ability_generated_object: AbilityGeneratedObject = None
        self.ability_generated_object_name = None
        self.generated_object_pattern = None
        self.index_ability_generated_object_list: list = None
        self.index_ability_generated_object_gpu: list = None

        self.owner_mapping_gpu: wp.array
        self.owner_list_gpu: wp.array
        self.enemy_list_gpu: wp.array
        self.cooldown_ability_owners: wp.array
        self.player_to_bullet_map_gpu: wp.array

        self._pattern_view_cache: Dict[str, PatternViewContext] = {}
        self._human_view_ctx: Optional[PatternViewContext] = None
        self._rl_view_ctx: Optional[PatternViewContext] = None
        self._bot_view_ctx: Optional[PatternViewContext] = None
        self._primary_view_ctx: Optional[PatternViewContext] = None

        # 確保只被初始化一次
        if Ability._default_configs is None:
            Ability._initialize_class_assets()
        
        # 從 Pydantic RootModel 中獲取特定能力的配置
        abilities_configs = Ability._default_configs.root.get(self.ability_name)

        if Ability._fps is None:
            Ability._fps = GameConfig.FPS_ACTION  # Default to 60 FPS if not specified

        # Force 的意思是能力基於施加力來實現
        # Speed 的意思是能力基於直接修改速度來實現
        # Speed 和 Force 只能二選一
        # Cooldown 是該能力的冷卻時間，單位是秒
        if abilities_configs:
            self.force = wp.float32(abilities_configs.force)
            self.speed = wp.float32(abilities_configs.speed)
            self.cooldown = int(abilities_configs.cooldown * Ability._fps)  # Default cooldown of 1 second
            
            # 將 action_space 轉回 dict 以保持與原有 get_action_spec 邏輯相容
            self.action_space = abilities_configs.action_space.model_dump(exclude_none=True)
            self.action_shape = self.action_space.get("shape")
            self.action_shape_offset: int = None
        else:
            # 即使配置已加載，仍需處理找不到特定能力配置的情況
            dir_path = os.path.dirname(os.path.realpath(__file__))
            config_path = os.path.join(dir_path, './abilities_default_cfg.yaml')
            raise ValueError(f"Default config for ability '{self.ability_name}' not found in {config_path}")

    @abstractmethod
    def human_control_interface(self, keyboard_keys, mouse_buttons, look_pitch, look_yaw, index_player: int, current_game_step, **kwargs):
        """
        提供給 HumanControl 使用的接口方法，讓玩家能夠通過鍵盤/滑鼠輸入來控制此能力。
        這個方法目的在於將玩家的輸入轉換為能力所需的 action_value。
        每個能力需要的轉換方式都不一樣，
        函數會返回 action_value 而不是直接呼叫 action 因為 action 需要在遊戲主循環中被呼叫，
        而 human_control_interface 只是負責處理輸入並產生 action_value。

        子類需要覆寫此方法以實現自定義的輸入邏輯。
        """
        pass

    @abstractmethod
    def rl_action(self, actions, **kwargs):
        pass

    @abstractmethod
    def bot_action(self, **kwargs):
        """
        提供給 Bot 使用的接口方法。
        主要目的在於提供一個合格的訓練假人讓 RL 模型熟悉遊戲環境。

        子類需要覆寫此方法以實現自定義的 AI 邏輯。
        """
        pass
    
    @abstractmethod
    def update_index_bot(self, index_rl_players_gpu, num_rl_players, index_bot_players_gpu, num_bot_players):
        self.index_rl_players_gpu = index_rl_players_gpu
        self.num_rl_players = num_rl_players
        self.index_bot_players_gpu = index_bot_players_gpu
        self.num_bot_players = num_bot_players

    @abstractmethod
    def reset(self):
        pass

    def _is_pressed(self, kb_list, ms_list, kb_state: 'np.ndarray', ms_state):
        # 鍵盤檢查：只要 kb_list 和 kb_state 有共同元素，就回傳 True

        if kb_state is not None and len(kb_list) >= 1:
            # isdisjoint 檢查兩者是否「完全沒有」交集，取反則代表「有交集」
            if (kb_state[kb_list] > 0).any():
                return 1

        # 滑鼠檢查 (保持原樣或同步改為 set)
        if ms_state is not None:
            for m in ms_list:
                if ms_state[m]: return 1
        return 0
    
    def update_cooldown(self):
        """
        每幀調用，減少冷卻
        """
        # TODO 目前是掃描所有技能實例逐個更新，未來測試把 cooldown_ability_owners 換成 2D array 使用 p_idx, s_idx = wp.tid() 一次性更新

        wp.launch(
            kernel=self.update_cooldown_gpu,
            dim=self.num_objects_total, 
            inputs=[
                self.cooldown_ability_owners,
            ],
            device=self.physics_manager.device
        )

    @wp.kernel
    def update_cooldown_gpu(
        cooldown_ability_owners: wp.array(dtype=wp.int32), 
    ):
        tid = wp.tid()

        if cooldown_ability_owners[tid] > 0:
            cooldown_ability_owners[tid] -= 1

    @classmethod
    def _initialize_class_assets(cls):
        """一次性初始化配置、按鍵映射與動態 Action 類"""
        print("Initializing Global Ability Assets...")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, './abilities_default_cfg.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_data = yaml.safe_load(f)
        
        # 使用 Pydantic V2 的 model_validate 來驗證 YAML 數據
        cls._default_configs = AbilitiesConfig.model_validate(raw_data)
        cls._fps = GameConfig.FPS_ACTION 

    def configure_from_object(self, object_config: dict, player_config: dict | None = None) -> None:
        self.pattern = resolve_articulation_player_pattern(object_config, player_config)
        self.control_policy_version = resolve_control_policy_version(object_config)

    def _view_ctx(self, controller: str) -> Optional[PatternViewContext]:
        """Return cached per-controller view context, or None if missing/invalid."""
        ctx = {
            "human": self._human_view_ctx,
            "rl": self._rl_view_ctx,
            "bot": self._bot_view_ctx,
        }.get(controller.lower())
        if ctx is None or not ctx.valid:
            return None
        return ctx

    def resolve_pattern_view(self, pattern: str) -> PatternViewContext:
        """Resolve and cache articulation-body view metadata for a pattern id."""
        cached = self._pattern_view_cache.get(pattern)
        if cached is not None:
            return cached

        ctx = PatternViewContext(pattern=pattern)
        body = self.articulation_body
        if body is not None and hasattr(body, "patterns"):
            ctx.view_idx = next(
                (i for i, p in enumerate(body.patterns) if p == pattern), -1
            )
            if ctx.view_idx >= 0:
                view = body.views[ctx.view_idx]
                ctx.count_per_world = int(view.count_per_world)
                bodies_per_world = len(body.patterns_local_indices[pattern])
                if ctx.count_per_world > 0:
                    ctx.bodies_per_object = bodies_per_world // ctx.count_per_world

        self._pattern_view_cache[pattern] = ctx
        return ctx

    def cache_action_pattern_views(self, owner_indices: List[int]) -> None:
        """Cache per-controller pattern/view metadata after setup indices are ready."""
        from script.role.base_role import BaseRole
        from script.role.controller_utils import normalize_controller

        by_controller: Dict[str, PatternViewContext] = {}
        for owner_idx in owner_indices or []:
            params = BaseRole._object_game_params[owner_idx]
            controller = normalize_controller(
                params.get("controller"),
            )
            if controller in by_controller:
                continue
            pattern = str(params.get("runtime_pattern") or self.pattern)
            by_controller[controller] = self.resolve_pattern_view(pattern)

        self._human_view_ctx = by_controller.get("Human")
        self._rl_view_ctx = by_controller.get("RL")
        self._bot_view_ctx = by_controller.get("Bot")

        if getattr(self, "pattern", None):
            self._primary_view_ctx = self.resolve_pattern_view(self.pattern)

    def configure_from_player_configs_post_indices(self, level: "Levels") -> None:
        try:
            ability_idx = level.players.abilities_instance_list.index(self)
            owners = level.players.abilities_owner_list[ability_idx]
        except ValueError:
            owners = []
        self.cache_action_pattern_views(owners)

    def configure_from_generated_object_config(
        self, object_key: str, config: Dict[str, Any]
    ) -> None:
        """Override in abilities that bind to ability_generated_object_configs entries."""
        pass

    def configure_from_player_configs(
        self, player_configs: List[Dict[str, Any]], level: "Levels"
    ) -> None:
        from script.role.objects.object_template.loader import ensure_object_templates_registered

        ensure_object_templates_registered()

        matched_config = find_player_config_for_ability(player_configs, self.__class__.__name__)
        matched_object = dict(matched_config.get("object") or {})
        self.configure_from_object(matched_object, matched_config)
        self.policy_device = matched_object.get("policy_device") or str(
            wp.device_to_torch(self.physics_manager.device)
        )
        self._torch_device = torch.device(self.policy_device)
        self._configured = True

    def setup_keymapping(self, ability_name):
        # 從 RootModel 中提取對應能力的 key 配置
        ability_detail = self._default_configs.root.get(ability_name)
        if ability_detail:
            # 將 Pydantic 模型轉回 dict 以供 KeyMapping.get 使用
            self.control_keys = KeyMapping.get(ability_detail.key.model_dump())

    def setup_cooldown(self, num_objects_total, owners_ability_list):
        # 直接在 GPU 上創建並填充 -1.0 (極快)
        self.num_objects_total = num_objects_total
        self.cooldown_ability_owners = wp.full(shape=num_objects_total, value=-1, dtype=wp.int32)

        # 將擁有者的索引轉成 wp.array (如果它還不是的話)
        owners_wp = wp.array(owners_ability_list, dtype=wp.int32)

        # 執行一個極簡的 Kernel 把對應位置改為 0
        wp.launch(
            kernel=set_indices_to_value_kernel,
            dim=len(owners_ability_list),
            inputs=[self.cooldown_ability_owners, owners_wp, 0],
            device=self.physics_manager.device
        )

    def setup_player_to_env_mapping(self, index_role_offset_env_gpu, num_role_each_env):
        self.index_player_offset_env_gpu = index_role_offset_env_gpu
        self.num_player_each_env = num_role_each_env

    def get_action_spec(self) -> dict:
        """
        返回該能力的動作空間規格說明 (Action Space Specification)。
        
        此規格定義了客戶端（人類或 AI 代理）在調用此能力時，必須提供的 `action_value` 
        數據結構。此格式直接兼容 Gymnasium (OpenAI Gym) 的空間定義，
        並可由 Ray RLlib 自動解析為模型輸出頭。

        Returns:
            dict: 包含動作空間定義的字典。結構如下：
                {
                    "type": str,      # 空間類型，例如 "dict", "box", "discrete"
                    "spaces": dict,    # 當 type 為 "dict" 時，定義子空間的映射
                    "description": str # (可選) 該能力的物理意義描述
                }

        Data Structure Detail (Example: Composite Ability):
            若返回 "type": "dict"，`action_value` 應為一個字典，包含以下子項：
            
            1. "direction" (Box Space):
                - 類型: 連續型數值向量 (Continuous)。
                - 物理意義: 控制能力施放的方向。
                - 數值範圍: [-3.14, 3.14] (弧度制，對應 -180° 到 180°)。
                - AI 建議: RL 模型將使用高斯分佈 (Gaussian) 進行採樣。

            2. "action" (Discrete Space):
                - 類型: 離散型類別 (Categorical)。
                - 物理意義: 觸發開關。0 代表不執行，1 代表執行/觸發。
                - 選項數量 (n): 2。
                - AI 建議: RL 模型將使用分類分佈 (Categorical/Softmax) 進行採樣。

        Example Output Mapping:
            >>> spec = ability.get_action_spec()
            >>> print(spec["spaces"]["direction"]["range"])
            [-3.14, 3.14]

        Note:
            在 Ray RLlib 環境中，此規格將被轉換為 `gym.spaces.Dict`，
            確保策略網絡 (Policy Network) 的輸出維度與遊戲邏輯完美對齊。
        """
        return self.action_space

    def get_name(self):
        return self.__class__.__name__


@wp.kernel
def set_indices_to_value_kernel(cooldown_ability_owners: wp.array(dtype=wp.int32), 
                                indices: wp.array(dtype=wp.int32), 
                                value: wp.int32):
    tid = wp.tid()
    idx = indices[tid]
    cooldown_ability_owners[idx] = value




