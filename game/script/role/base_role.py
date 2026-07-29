import random
import math
import warp as wp
import numpy as np

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from script.role.abilities import get_shared_ability
from script.role.abilities.ability import Ability
# from role.abilities import *  # Import all abilities 


from script.role.objects.base_object import ObjectRegistry
import script.role.objects  # noqa: F401 — register all object handlers
from script.role.objects.rigid_box import RigidBoxModel

from script.role.bodies.articulation_body import ArticulationBody
from script.role.bodies.deformable_body import DeformableBody

from script.game_config import GameConfig
from script.role.abilities.articulation_control_config.profile_registry import (
    compose_runtime_pattern,
)
from script.role.controller_utils import normalize_controller

from typing import TYPE_CHECKING, Type, Dict, List, Any, Literal, Union, Optional, Annotated, Tuple
if TYPE_CHECKING:
    from script.simulate.physics_manager import PhysicsManager


RandomizableValue = Union[float, List[float]]

# 使用 Annotated 進行自動判別
ObjectConfig = ObjectRegistry.get_object_union()

class BaseRoleModel(BaseModel):
    type: Literal["base"] = "base"
    name: str = ""
    color: List[int] = [200, 200, 200]
    default_position: List[RandomizableValue] = [0.0, 0.0, 0.0]
    default_rotation: List[RandomizableValue] = [0.0, 0.0]
    default_velocity: List[float] = [0.0, 0.0, 0.0]
    default_angular_velocity: List[float] = [0.0, 0.0, 0.0]
    possess_offset: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    abilities: list[str] = []

    object: ObjectConfig = RigidBoxModel() # 預設剛體正方體


class BaseRole(ABC):
    _num_env: int
    _num_objects_env: int = 0
    _num_objects_total: int = 0
    _physics_manager: 'PhysicsManager'

    _articulation_body: ArticulationBody
    _deformable_body: DeformableBody

    _xy_offset_env = None
    _xy_offset_env_gpu = None
    _object_game_params: List[Dict[str, Any]] = [] # 儲存所有物件的非物理數據

    _name_list: List[str] = []

    role_key = "BASE"
    model_cls: Type[BaseRoleModel] = BaseRoleModel
    path = "role_configs"
    container = "unknow"
    role_type_id: int = -1  # 用於 GPU kernel 的 ID

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        RoleRegistry.register(cls)


    @abstractmethod
    def __init__(self, is_add_to_mesh=False):
        self.is_add_to_mesh = is_add_to_mesh
        self.index_role_offset_env_list = []
        self.num_role_each_env = 0
        self.num_total_object_role = 0
        self.index_obj_role_to_env_mapping = [] # For record which environment the object belongs to. 
        self.index_obj_role = [] # Includes all indexes for a type of Role in the physics engine.

        if BaseRole._xy_offset_env is None and GameConfig.Enable_World_Offset:
            BaseRole._xy_offset_env = generate_grid_coordinates(self._num_env, (GameConfig.space_x + GameConfig.interval_distance, GameConfig.space_y + GameConfig.interval_distance))
            BaseRole._xy_offset_env_gpu = wp.array(data=BaseRole._xy_offset_env, dtype=wp.vec2, device=self._physics_manager.device)
        elif BaseRole._xy_offset_env is None:
            BaseRole._xy_offset_env = [[0.0, 0.0] for _ in range(self._num_env)]
            BaseRole._xy_offset_env_gpu = wp.array(data=BaseRole._xy_offset_env, dtype=wp.vec2, device=self._physics_manager.device)

        self.abilities_name_index_dict: Dict[str, int] = {}
        self.abilities_instance_list: List[Ability] = []
        self.abilities_owner_list: List[List[int]] = [] 

        # For Setup only
        self.is_ability_generated_object_type_changed = False
        self.current_ability_generated_object_type = ""
        self.ability_generated_object_offset_X = 0.0
        self.ability_generated_object_offset_X_prev = 0.0


        self.index_role_offset_env_gpu = wp.array(data=[0], dtype=wp.int32, device=self._physics_manager.device)
        self.index_obj_role_to_env_mapping_gpu = wp.array(data=[0], dtype=wp.int32, device=self._physics_manager.device)
        self.index_obj_role_gpu = wp.array(data=[0], dtype=wp.int32, device=self._physics_manager.device)

    def setup(self, configs: Union[List, Dict], collision_group: Optional[int] = None):
        self.num_object_created_for_setup = None # for calculate offset

        self.index_object_sort_object_type_dict: Dict[str, List[int]] = {}
        self.default_expired_step_list = []

        num_env_obj = 0

        _collision_group = collision_group if collision_group is not None else 1 # TODO Hard code

        if isinstance(configs, list):
            for config in configs:
                self.add_to_physics_manager(collision_group=_collision_group, **config)
                num_env_obj += 1
                self._name_list.append(config.get("name", ""))

        elif isinstance(configs, dict):
            self.current_ability_generated_object_type = ""
            self.ability_generated_object_offset_X = 0.0
            self.ability_generated_object_offset_X_prev = 0.0
            for key, config in configs.items():
                ability_class_name = config["ability_class_name"]

                if self.current_ability_generated_object_type != key:
                    self.current_ability_generated_object_type = key
                    self.is_ability_generated_object_type_changed = True

                if key not in self.index_object_sort_object_type_dict:
                    self.index_object_sort_object_type_dict[key] = []
                    ability = get_shared_ability(ability_class_name)
                    ability.configure_from_generated_object_config(key, config)
                    self.abilities_instance_list.append(ability)

                default_expired_step = config["default_expired_step"] * GameConfig.FPS_ACTION 
                for n in range(config["quantity"]):
                    self.num_object_created_for_setup = n

                    index = self.add_to_physics_manager(collision_group=_collision_group, **config)
                    num_env_obj += 1

                    self._name_list.append(key)
                    self.index_object_sort_object_type_dict[key].append(index)
                    self.default_expired_step_list.append(default_expired_step)

        self.num_role_each_env = num_env_obj

    def add_to_physics_manager(self, 
                               collision_group: int,
                               type: str,
                               color: List[int],
                               default_position: List[Any],
                               default_rotation: List[Any],
                               default_velocity: List[Any],
                               default_angular_velocity: List[Any],  # TODO
                               object: Dict[str, Any],
                               health: float = -1.0,
                               abilities: List[str] = [],
                               team_id: int = -1,
                               name: str = "",
                               **kwargs
                              ):
        """
        處理新的嵌套 config 結構並解構為 PhysicsManager 所需的參數
        """

        controller = kwargs.get("controller")

        job_pattern = object.get("pattern", "default")
        possess_offset_raw = kwargs.get("possess_offset", object.get("possess_offset"))
        from script.role.abilities.articulation_control_config.joint_config_registry import (
            resolve_possess_offset_for_pattern,
        )

        resolved = None
        if possess_offset_raw is None:
            resolved = resolve_possess_offset_for_pattern(
                job_pattern,
                object.get("control_task"),
                object_config=object,
            )
            possess_offset = list(resolved) if resolved is not None else [0.0, 0.0, 0.0]
        else:
            possess_offset = [float(v) for v in possess_offset_raw[:3]]
            while len(possess_offset) < 3:
                possess_offset.append(0.0)

        runtime_pattern = f"{type}_{job_pattern}"
        if type == "player":
            controller = normalize_controller(controller)
            runtime_pattern = compose_runtime_pattern(controller, type, job_pattern)
        label = runtime_pattern
        role_object_id = BaseRole._num_objects_env
        data, size = self._physics_manager.add_shape(
            label=label,
            object_config=object,
            collision_group=collision_group,
            pos=default_position,
            role_object_id=role_object_id,
        )
        if isinstance(data, dict):
            path_body_map = data.get("path_body_map") or data.get("_path_body_map")
            if path_body_map:
                anchor_offset = resolve_possess_offset_for_pattern(
                    job_pattern,
                    object.get("control_task"),
                    object_config=object,
                    path_body_map=path_body_map,
                )
                if anchor_offset is not None:
                    possess_offset = list(anchor_offset)
        # TODO
        # self._physics_manager.add_shape 返回的索引不一定是索引。
        # 如果加載的是 usd 物件，那麽返回的可能就是 add_usd 返回的關於模型的信息的字典數據，
        # 因此這裏的 index 則換成當前已添加的物件數量作爲索引
        index = BaseRole._num_objects_env

        # ==========================================================================================================================================================
        # Position
        # 支持 [min, max] 列表或單個數值

        # 如果是能力生成的物件，處理排列偏移
        if isinstance(self.num_object_created_for_setup, int):
            if self.is_ability_generated_object_type_changed:
                self.is_ability_generated_object_type_changed = False
                self.ability_generated_object_offset_X += self.ability_generated_object_offset_X_prev

            # 使用形狀的主尺寸作為偏移基準
            main_dim = size[0]
            offset_X = ((main_dim * 2.0) * (self.num_object_created_for_setup * 1.1))

            # 更新 X 座標範圍
            pos_x = default_position[0]
            if isinstance(pos_x, tuple):
                default_position[0] = (pos_x[0] + self.ability_generated_object_offset_X + offset_X, 
                                              pos_x[1] + self.ability_generated_object_offset_X + offset_X)
            else:
                default_position[0] = pos_x + self.ability_generated_object_offset_X + offset_X

            self.ability_generated_object_offset_X_prev = offset_X + (main_dim * 2.0)

        position = [0.0, 0.0, 0.0]
        default_position_tuple = [None, None, None]
        for i, pos in enumerate(default_position):
            if isinstance(pos, (tuple, list)):
                default_position_tuple[i] = pos
                position[i] = random.uniform(pos[0], pos[1])
            elif isinstance(pos, (int, float)):
                default_position_tuple[i] = (float(pos), float(pos))
                position[i] = float(pos)

        # ==========================================================================================================================================================
        # Rotation
        rot = np.pi / 180.0
        default_rotation_tuple = []
        # 目前內核主要處理 Yaw 和 Pitch (2個值)
        for i in range(2):
            val = default_rotation[i]
            if isinstance(val, (list, tuple)) and len(val) == 2:
                default_rotation_tuple.append((val[0] * rot, val[1] * rot))
            else:
                default_rotation_tuple.append((float(val) * rot, float(val) * rot))

        # ==========================================================================================================================================================
        
        if object["type"].startswith("rigid_"):
            self._articulation_body.add_object(label=label, index=index, default_position=default_position_tuple, default_rotation=default_rotation_tuple)

        elif object["type"].startswith("soft_"):
            self._deformable_body.add_object(label=label, index=index, particle_index=data, default_position=default_position_tuple, default_rotation=default_rotation_tuple)

        else: # TODO
            self._articulation_body.add_object(label=label, index=index, default_position=default_position_tuple, default_rotation=default_rotation_tuple)

        self.index_obj_role.append(index)

        # 5. 註冊到物理引擎
        object_key = object['type']
        BaseRole._num_objects_env += 1
        BaseRole._object_game_params.append({
            "pattern": job_pattern,
            "runtime_pattern": runtime_pattern,
            "name": name,
            "controller": controller,
            "shape_key": object_key, 
            "health": health, 
            "team_id": team_id, 
            "color": color,
            "possess_offset": possess_offset,
        })


        if self.is_add_to_mesh:
            self._physics_manager.add_mesh(object_config=object, position=position)

        # 6. 能力系統處理（關節體控制按 role+robot pattern 分實例）
        role_type = str(type or "player")
        for ability_name in abilities:
            ability_cls = Ability._registry.get(ability_name)
            share_key = None
            if ability_cls is not None:
                share_key = ability_cls.share_scope(
                    object_config=object,
                    role_type=role_type,
                )
            registry_key = (
                f"{ability_name}@{share_key}" if share_key else ability_name
            )
            if registry_key not in self.abilities_name_index_dict:
                ability = get_shared_ability(ability_name, share_key=share_key)

                self.abilities_name_index_dict[registry_key] = len(
                    self.abilities_instance_list
                )
                self.abilities_instance_list.append(ability)
                self.abilities_owner_list.append([index])
            else:
                self.abilities_owner_list[
                    self.abilities_name_index_dict[registry_key]
                ].append(index)

        return index

    @classmethod
    def get_possess_offsets(cls) -> List[Tuple[float, float, float]]:
        return [
            tuple(params.get("possess_offset", (0.0, 0.0, 0.0)))
            for params in cls._object_game_params
        ]

    def set_ability_class(self, class_name: str):
        return get_shared_ability(class_name)

    def get_player_abilities(self, index_player: int): # This function use for human player control interface, not RL
        index_abilities = []
        for i, owner_list in enumerate(self.abilities_owner_list):
            if index_player in owner_list:
                index_abilities.append(i)
        return index_abilities

    @classmethod
    def physics_index_match_to_role(cls, role_objects: list["BaseRole"], num_env):
        """
        修復關鍵：一次性擴展類別變數，對齊交錯佈局 [Env0, Env1, Env2...]
        """
        # 1. 備份 Env 0 的基礎資料
        base_params = cls._object_game_params.copy()
        base_name_list = cls._name_list.copy()
        num_object_env = cls._num_objects_env


        # 2. 清空並根據環境數量重新構建（Interleaved Expansion）
        cls._object_game_params = []
        cls._name_list = []

        for i in range(num_env):
            
            # 複製非物理參數
            cls._object_game_params.extend(base_params)
            cls._name_list.extend(base_name_list)

        # 3. 處理每個角色物件實體的索引映射
        for obj in role_objects:
            # 找出該 Role 在 Env 0 內的原始物理索引
            base_indices = [idx for idx in obj.index_obj_role if idx < num_object_env]
            num_role_per_env = len(base_indices)

            if num_role_per_env <= 0:
                continue

            all_indices = []
            index_obj_role_to_env_mapping = []
            index_role_offset_env_list = []

            # --- 處理 abilities_owner_list ---
            # 備份並初始化，確保它是個巢狀結構
            has_abilities = hasattr(obj, 'abilities_owner_list') and obj.abilities_owner_list
            if has_abilities:
                # 備份原始結構
                base_abilities_owner_list = [sub_list.copy() for sub_list in obj.abilities_owner_list]
                # 初始化為空列表 (假設每個能力對應一個 list)
                obj.abilities_owner_list = [[] for _ in base_abilities_owner_list]

            # --- 備份並處理 index_object_sort_object_type_dict ---
            # 如果字典存在且不為空，則進行處理
            has_sort_dict = hasattr(obj, 'index_object_sort_object_type_dict') and obj.index_object_sort_object_type_dict
            if has_sort_dict:
                base_sort_dict = {k: v.copy() for k, v in obj.index_object_sort_object_type_dict.items()}
                obj.index_object_sort_object_type_dict = {k: [] for k in base_sort_dict.keys()}


            for i in range(num_env):
                # 物理引擎交錯偏移
                current_offset = i * num_object_env
                # 該物件在自身的陣列（如 health）中的偏移
                index_role_offset_env_list.append(current_offset + base_indices[0])

                # 更新物理索引
                new_indices = [idx + current_offset for idx in base_indices]
                all_indices.extend(new_indices)
                
                # 更新環境映射
                index_obj_role_to_env_mapping.extend([i] * num_role_per_env)

                # 更新能力擁有者索引
                if has_abilities:
                    for idx, sub_list in enumerate(base_abilities_owner_list):
                        offset_owners = [o + current_offset for o in sub_list]
                        obj.abilities_owner_list[idx].extend(offset_owners)

                # 更新類型排序字典
                if has_sort_dict:
                    for key, val_list in base_sort_dict.items():
                        offset_vals = [v + current_offset for v in val_list]
                        obj.index_object_sort_object_type_dict[key].extend(offset_vals)

            obj.default_expired_step_list *= num_env
                
            # 更新物件實體屬性
            obj.index_obj_role = all_indices
            obj.index_obj_role_to_env_mapping = index_obj_role_to_env_mapping
            obj.index_role_offset_env_list = index_role_offset_env_list
            obj.num_total_object_role = len(all_indices)

            # 更新 GPU 陣列
            obj.index_obj_role_gpu = wp.array(data=obj.index_obj_role, dtype=wp.int32, device=obj._physics_manager.device)
            obj.index_obj_role_to_env_mapping_gpu = wp.array(data=obj.index_obj_role_to_env_mapping, dtype=wp.int32, device=obj._physics_manager.device)
            obj.index_role_offset_env_gpu = wp.array(data=obj.index_role_offset_env_list, dtype=wp.int32, device=obj._physics_manager.device)

        cls._articulation_body.finalize_position_ranges(world_xy_offset_list=cls._xy_offset_env, device=cls._physics_manager.device, num_env=num_env)
        cls._deformable_body.finalize_position_ranges(world_xy_offset_list=cls._xy_offset_env, device=cls._physics_manager.device, num_env=num_env)

def generate_grid_coordinates(n: int, offset: tuple[int, int]) -> list[list[int, int]]:
    if n <= 0: return []
    cols = max(1, math.isqrt(n))
    dx, dy = offset
    result = []
    for i in range(n):
        row_index = i % cols
        col_index = i // cols
        x = float(row_index * dx)
        y = float(col_index * dy)
        result.append([x, y])
    return result






class RoleRegistry:
    _registry: Dict[str, Type['BaseRole']] = {}

    @classmethod
    def register(cls, role_cls: Type['BaseRole']):
        if role_cls.role_key and role_cls.role_key != "BASE":
            cls._registry[role_cls.role_key] = role_cls
            # print(f"[*] 角色註冊成功: {role_cls.role_key}")

    @classmethod
    def get_shape_union(cls):
        """
        核心邏輯：從註冊表中提取所有 model_cls，動態生成 Union 類型
        """
        models = [handler.model_cls for handler in cls._registry.values()]
        if not models:
            # 防止註冊表為空時報錯，至少給一個基礎模型
            return BaseRoleModel
        
        return Annotated[Union[tuple(models)], Field(discriminator="type")]

    @classmethod
    def get_handler(cls, key: str):
        return cls._registry.get(key)

    @classmethod
    def get_all_keys(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def get_all_models(cls) -> List[str]:
        models = [handler.model_cls for handler in cls._registry.values()]
        return models

    @classmethod
    def get_all_infos(cls) -> dict:
        infos = {}
        for key, handler in cls._registry.items():
            infos[key] = {"model": handler.model_cls, "path": handler.path, "container": handler.container}

        return infos


