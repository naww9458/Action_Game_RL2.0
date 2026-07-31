
from __future__ import annotations

import warp as wp

from typing import Literal
from script.role.base_role import BaseRole, BaseRoleModel

class AbilityGeneratedObjectModel(BaseRoleModel):
    type: Literal["ability_generated_object"] = "ability_generated_object"
    name: str = "New_Ability_Generated_Object"
    # 此類物件會依 quantity 批量生成，dict key 即為「物件子角色」
    # （object_sub_role，例如子彈類型），非唯一的物件 ID。
    ability_class_name: str = "DefaultAbility"
    default_expired_step: float = 100.0
    quantity: int = 1


class AbilityGeneratedObject(BaseRole):
    role_key = "ability_generated_object"
    model_cls = AbilityGeneratedObjectModel
    path = "ability_generated_object_configs"
    container = "dict"


    def __init__(self, configs, **kwargs):
        super().__init__(**kwargs)

        # dict key 即為物件子角色（object_sub_role），無需額外欄位同步

        self.expired_steps = None 
        self.owner_mapping: dict[list[list[int]]] = {}
        self.owner_list: dict[list[int]] = {}   # 誰擁有這 個/些 物件 (比如子彈)

        # TODO 這兩個應該移動到 Player 而不是這裏
        self.team_list: dict[list[int]] = {}    # TODO 誰不是這 個/些 物件 (比如子彈) 的擁有者但也無法被傷害
        self.enemy_list: dict[list[int]] = {}   # 誰可以被這 個/些 物件 (比如子彈) 傷害

        self.setup(configs=configs)

    def setup(self, configs):
        super().setup(configs)

    def update_owner(self, num_object_total: int, num_players_each_env: int, index_players_offset_env_list: list):

        if len(index_players_offset_env_list) <= 0:
            return

        # 初始化生命週期陣列 (所有生成物共用)
        self.expired_steps = wp.zeros(shape=len(self.index_obj_role), dtype=wp.int32, device=self._physics_manager.device)
        self.default_expired_step_list_gpu = wp.array(data=self.default_expired_step_list, dtype=wp.int32, device=self._physics_manager.device)

        # 建立 玩家 -> 物件 的分配表
        current_obj_idx = 0
        num_ability_generated_object_env = self.num_role_each_env
        for env_index in range(self._num_env):
            index_players_offset = index_players_offset_env_list[env_index]

            for obj_in_env_idx in range(num_ability_generated_object_env):
                owner_p_idx = (obj_in_env_idx % num_players_each_env) + index_players_offset
                bullet_idx = self.index_obj_role[current_obj_idx]

                obj_type = self._name_list[bullet_idx]

                if obj_type not in self.owner_mapping:
                    self.owner_mapping[obj_type] = [[] for i in range(num_object_total)]
                    self.owner_list[obj_type] = []
                    self.enemy_list[obj_type] = []

                self._physics_manager.builder.shape_collision_filter_pairs.append((owner_p_idx+1, bullet_idx+1))
                self.owner_mapping[obj_type][owner_p_idx].append(current_obj_idx)
                self.owner_list[obj_type].append(owner_p_idx)

                # --- enemy_list 分配邏輯 ---
                enemies_for_this_obj = []
                for p_idx in range(index_players_offset, index_players_offset + num_players_each_env):
                    if p_idx != owner_p_idx:
                        enemies_for_this_obj.append(p_idx)

                self.enemy_list[obj_type].append(enemies_for_this_obj)
                current_obj_idx += 1

        # --- 為每個 Ability 處理 GPU 陣列 ---
        for ability in self.abilities_instance_list:
            ability.ability_generated_object = self
            ability.index_ability_generated_object_list = self.index_object_sort_object_type_dict[ability.ability_generated_object_name]
            ability.index_ability_generated_object_gpu = wp.array(data=ability.index_ability_generated_object_list, dtype=wp.int32, device=self._physics_manager.device)

            # --- 處理 owner_mapping (2D: 玩家索引 -> 多個子彈的 Local Index) ---
            raw_mapping = self.owner_mapping[ability.ability_generated_object_name]
            max_len_map = max([len(sublist) for sublist in raw_mapping]) if raw_mapping else 0
            if max_len_map == 0:
                padded_mapping = [[-1] for _ in range(num_object_total)]
            else:
                padded_mapping = [sublist + [-1] * (max_len_map - len(sublist)) for sublist in raw_mapping]

            ability.owner_mapping_gpu = wp.array(data=padded_mapping, dtype=wp.int32, device=self._physics_manager.device, ndim=2)

            # --- 處理 owner_list (1D: 子彈順序 -> 玩家索引) ---
            owners_data = self.owner_list[ability.ability_generated_object_name]
            ability.owner_list_gpu = wp.array(owners_data, dtype=wp.int32, device=self._physics_manager.device)

            # --- 新增：玩家索引 -> 子彈物理索引的反向查找表 (1D) ---
            # 初始化一個大小為 num_object_total，值為 -1 的清單
            player_to_bullet_map = [-1] * num_object_total
            
            # owners_data 裡面存的是 [0, 1, 8, 9, ...] (誰擁有這顆子彈)
            # ability.index_ability_generated_object_list 裡面存的是子彈在 physics 中的索引
            for local_idx, owner_idx in enumerate(owners_data):
                # 取得這顆子彈真正的物理索引
                index = self.index_obj_role[local_idx]
                # 建立對應關係：玩家索引 -> 子彈物理索引
                player_to_bullet_map[owner_idx] = index

            ability.player_to_bullet_map_gpu = wp.array(
                data=player_to_bullet_map, 
                dtype=wp.int32, 
                device=self._physics_manager.device
            )

            # --- 處理 enemy_list ---
            raw_enemy_list = self.enemy_list[ability.ability_generated_object_name]
            max_len_enemy = max([len(sublist) for sublist in raw_enemy_list]) if raw_enemy_list else 0
            if max_len_enemy == 0:
                padded_enemy = [[-1] for _ in range(len(raw_enemy_list))]
            else:
                padded_enemy = [sublist + [-1] * (max_len_enemy - len(sublist)) for sublist in raw_enemy_list]

            ability.enemy_list_gpu = wp.array(data=padded_enemy, dtype=wp.int32, device=self._physics_manager.device, ndim=2)

    def update_lifetimes(self):
        """每幀調用，減少計時器，若時間到則隱藏物件(可選)"""

        wp.launch(
            kernel=self.update_lifetimes_gpu,
            dim=self.num_total_object_role, 
            inputs=[
                self.expired_steps,
                self._physics_manager.reset_mask_gpu,
                self.index_obj_role_gpu,
            ],
            device=self._physics_manager.device
        )

    @wp.kernel
    def update_lifetimes_gpu(
        expired_steps: wp.array(dtype=wp.int32), 
        reset_mask: wp.array(dtype=wp.int32), 
        index_obj_role_gpu: wp.array(dtype=wp.int32), 
    ):
        tid = wp.tid()
        index_b = index_obj_role_gpu[tid]

        if expired_steps[tid] > 0:
            expired_steps[tid] -= 1
        else: 
            reset_mask[index_b] = 1



