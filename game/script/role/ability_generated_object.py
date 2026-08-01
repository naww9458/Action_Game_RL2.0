
from __future__ import annotations

import warp as wp

from typing import List, Literal, Optional
from script.role.base_role import BaseRole, BaseRoleModel

class AbilityGeneratedObjectModel(BaseRoleModel):
    type: Literal["ability_generated_object"] = "ability_generated_object"
    name: str = "New_Ability_Generated_Object"
    # 此類物件會依 quantity 批量生成，dict key 即為「物件子角色」
    # （object_sub_role，例如子彈類型），非唯一的物件 ID。
    ability_class_name: str = "DefaultAbility"
    default_expired_step: float = 100.0
    quantity: int = 1
    # Owner pool: "player" (default) or "tool". Owner-role generated objects
    # get collision filtering against the owner bodies listed below.
    owner_role_type: str = "player"
    collision_filter_owner_bodies: List[str] = []


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

    def update_owner(
        self,
        num_object_total: int,
        num_players_each_env: int,
        index_players_offset_env_list: list,
        num_tools_each_env: int = 0,
        index_tools_offset_env_list: Optional[list] = None,
    ):

        if len(index_players_offset_env_list) <= 0:
            return

        # 初始化生命週期陣列 (所有生成物共用)
        self.expired_steps = wp.zeros(shape=len(self.index_obj_role), dtype=wp.int32, device=self._physics_manager.device)
        self.default_expired_step_list_gpu = wp.array(data=self.default_expired_step_list, dtype=wp.int32, device=self._physics_manager.device)

        # 子彈子角色 -> 綁定 Ability（讀取 owner_role_type / collision_filter_owner_bodies）
        ability_by_type = {
            a.ability_generated_object_name: a for a in self.abilities_instance_list
        }
        index_tools_offset_env_list = index_tools_offset_env_list or [0] * self._num_env
        # 每個 env 內各 owner 類型的已分配計數（支持同一關卡混用 player/tool 子彈）
        env_owner_counters = {"player": [0] * self._num_env, "tool": [0] * self._num_env}

        # 建立 玩家 -> 物件 的分配表
        current_obj_idx = 0
        num_ability_generated_object_env = self.num_role_each_env
        for env_index in range(self._num_env):
            index_players_offset = index_players_offset_env_list[env_index]
            index_tools_offset = int(index_tools_offset_env_list[env_index])

            for obj_in_env_idx in range(num_ability_generated_object_env):
                bullet_idx = self.index_obj_role[current_obj_idx]

                obj_type = self._name_list[bullet_idx]

                ability = ability_by_type.get(obj_type)
                owner_role_type = (
                    str(getattr(ability, "owner_role_type", "player") or "player")
                    if ability is not None
                    else "player"
                )

                if owner_role_type == "tool":
                    num_owners_each_env = int(num_tools_each_env)
                    offset = index_tools_offset
                else:
                    num_owners_each_env = int(num_players_each_env)
                    offset = int(index_players_offset)

                if num_owners_each_env <= 0:
                    raise ValueError(
                        f"Generated object '{obj_type}' has owner_role_type={owner_role_type!r} "
                        f"but the level has 0 such owners per env."
                    )

                # 該環境中依序分配給下一個同類 owner（單一類型時即 round-robin）
                local_owner = env_owner_counters[owner_role_type][env_index]
                owner_p_idx = (local_owner % num_owners_each_env) + offset
                env_owner_counters[owner_role_type][env_index] += 1

                if obj_type not in self.owner_mapping:
                    self.owner_mapping[obj_type] = [[] for i in range(num_object_total)]
                    self.owner_list[obj_type] = []
                    self.enemy_list[obj_type] = []

                if owner_role_type == "tool":
                    # Owner-role object: resolve the real shape indices of the
                    # generated object and the owner bodies (config-driven
                    # ``collision_filter_owner_bodies``) for collision filtering.
                    self._add_owner_collision_filter_pairs(bullet_idx, owner_p_idx, ability)
                else:
                    self._physics_manager.builder.shape_collision_filter_pairs.append((owner_p_idx+1, bullet_idx+1))
                self.owner_mapping[obj_type][owner_p_idx].append(current_obj_idx)
                self.owner_list[obj_type].append(owner_p_idx)

                # --- enemy_list 分配邏輯 ---
                if owner_role_type == "tool":
                    # 工具子彈的敵人是該 env 中的所有玩家
                    enemies_for_this_obj = [
                        index_players_offset + p
                        for p in range(num_players_each_env)
                    ]
                else:
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

    def _shape_range_for_role(self, local_role_id: int) -> Optional[tuple]:
        """Return the (shape_begin, shape_end) range in builder_env for a role object."""
        ranges = getattr(self._physics_manager, "_role_shape_ranges", None) or []
        for begin, end, rid in ranges:
            if int(rid) == int(local_role_id):
                return int(begin), int(end)
        return None

    def _global_shape_indices(
        self,
        role_object_id: int,
        body_suffixes: Optional[list] = None,
    ) -> list:
        """Global (across-env) shape indices for a role object.

        Ground plane occupies shape 0; env ``e`` shapes start at
        ``1 + e * env_shape_count`` (Newton ``add_world`` offsets per world).
        When ``body_suffixes`` is given, only shapes whose body label basename
        matches one of the suffixes are returned.
        """
        physics_manager = self._physics_manager
        builder_env = physics_manager.builder_env
        env_shape_count = int(builder_env.shape_count)
        num_objects_env = int(getattr(self, "_num_objects_env", BaseRole._num_objects_env) or 1)
        env_index = int(role_object_id) // num_objects_env
        local_role_id = int(role_object_id) % num_objects_env

        shape_range = self._shape_range_for_role(local_role_id)
        if shape_range is None:
            return []
        begin, end = shape_range

        out = []
        for local_shape in range(begin, end):
            if body_suffixes:
                body_idx = int(builder_env.shape_body[local_shape])
                if body_idx < 0 or body_idx >= builder_env.body_count:
                    continue
                label = str(builder_env.body_label[body_idx])
                basename = label.rstrip("/").split("/")[-1].lower()
                if not any(
                    s and (basename == str(s).lower() or basename.endswith(str(s).lower()))
                    for s in body_suffixes
                ):
                    continue
            out.append(1 + env_index * env_shape_count + local_shape)
        return out

    def _add_owner_collision_filter_pairs(
        self,
        generated_role_object_id: int,
        owner_role_object_id: int,
        ability,
    ) -> None:
        """Filter collisions between an owner's generated object and its bodies.

        Unlike single-shape player owners (where ``role_id + 1`` coincides with
        the shape index), multi-body owners resolve the real shape indices of the
        generated object and of the owner bodies named in
        ``ability.collision_filter_owner_bodies``.
        """
        body_suffixes = list(getattr(ability, "collision_filter_owner_bodies", None) or [])
        if not body_suffixes:
            return
        generated_shapes = self._global_shape_indices(int(generated_role_object_id))
        owner_shapes = self._global_shape_indices(
            int(owner_role_object_id), body_suffixes=body_suffixes
        )
        if not generated_shapes or not owner_shapes:
            return
        builder = self._physics_manager.builder
        for gs in generated_shapes:
            for os in owner_shapes:
                builder.shape_collision_filter_pairs.append((min(gs, os), max(gs, os)))

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



