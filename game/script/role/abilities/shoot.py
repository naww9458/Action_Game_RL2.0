import warp as wp
import math

from .ability import Ability
from script.game_config import GameConfig
from script.role.abilities.articulation_control_config.profile_registry import (
    resolve_ability_generated_object_pattern,
)
from utils.warp_math import sigmoid

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from script.levels.levels import Levels
    from script.role.player import Player
    from script.role.ability_generated_object import AbilityGeneratedObject

class Shoot(Ability):
    """
    Bot 的射擊頻率是隨機化的，具體冷卻時間為 cooldown +- 50% (範圍内隨機)
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        self.seeds = None
        self.offset = None
        self.fire_intent_buffer = None
        self._generated_object_view_ctx = None

    def configure_from_generated_object_config(self, object_key: str, config: dict) -> None:
        object_config = dict(config.get("object") or {})
        self.ability_generated_object_name = object_key
        self.generated_object_pattern = resolve_ability_generated_object_pattern(object_config)

    def configure_from_player_configs_post_indices(self, level: "Levels") -> None:
        super().configure_from_player_configs_post_indices(level)
        pattern = self.generated_object_pattern
        self._generated_object_view_ctx = (
            self.resolve_pattern_view(pattern) if pattern else None
        )

    def _shoot_ctx(self, controller: str):
        player_ctx = self._view_ctx(controller)
        bullet_ctx = self._generated_object_view_ctx
        if player_ctx is None or bullet_ctx is None or not bullet_ctx.valid:
            return None
        return player_ctx, bullet_ctx

    def human_control_interface(self, keyboard_keys, mouse_buttons, index_human_player_gpu: wp.array, **kwargs):
        if not getattr(self, "_configured", False):
            return
        
        physics_manager = self.physics_manager

        if self._is_pressed(self._keyboard_action, self._mouse_action, keyboard_keys, mouse_buttons) == 1:
            ctxs = self._shoot_ctx("human")
            if ctxs is None:
                return
            player_ctx, bullet_ctx = ctxs
            pattern = player_ctx.pattern

            wp.launch(
                kernel=self.human_action_gpu,
                dim=1, 
                inputs=[
                    self.articulation_body.control_mask_gpus[self.generated_object_pattern],
                    self.articulation_body.control_pos_gpus[self.generated_object_pattern],
                    self.articulation_body.control_rot_gpus[self.generated_object_pattern],
                    self.articulation_body.control_vel_gpus[self.generated_object_pattern],
                    physics_manager.state_0.body_q,
                    index_human_player_gpu,
                    self.owner_mapping_gpu,
                    self.ability_generated_object.expired_steps,
                    self.ability_generated_object.index_obj_role_gpu,
                    self.ability_generated_object.default_expired_step_list_gpu,
                    self.cooldown_ability_owners,
                    self.speed,
                    self.cooldown,
                    
                    self.articulation_body.view_object_indices_gpus[pattern],
                    self.articulation_body.view_body_local_indices_gpus[pattern],
                    player_ctx.count_per_world,
                    player_ctx.bodies_per_object,
                    
                    self.articulation_body.view_object_indices_gpus[self.generated_object_pattern],
                    bullet_ctx.count_per_world,
                    
                    self.articulation_body.num_objects_env,
                    self.articulation_body.num_rigid_bodies_env,
                ],
                device=physics_manager.device
            )

    @wp.kernel
    def human_action_gpu(
        bullet_control_mask: wp.array3d(dtype=wp.int32),
        bullet_control_pos: wp.array3d(dtype=wp.vec3),
        bullet_control_rot: wp.array3d(dtype=wp.quat),
        bullet_control_vel: wp.array3d(dtype=wp.vec3),
        body_q: wp.array(dtype=wp.transform), 
        index_human_player_gpu: wp.array(dtype=wp.int32),
        owner_mapping_gpu: wp.array(dtype=wp.int32, ndim=2),
        expired_steps: wp.array(dtype=wp.int32),
        index_obj_role_gpu: wp.array(dtype=wp.int32),
        default_expired_step_list_gpu: wp.array(dtype=wp.int32),
        cooldown_ability_owners: wp.array(dtype=wp.int32),
        speed: wp.float32,
        cooldown: wp.int32,

        player_view_object_indices: wp.array(dtype=int),
        player_view_body_local_indices: wp.array(dtype=int),
        player_count_per_world: int,
        player_bodies_per_object: int,

        bullet_view_object_indices: wp.array(dtype=int),
        bullet_count_per_world: int,

        num_objects_env: int,
        num_body_object_env: int,
    ):
        tid = wp.tid()
        index_player = index_human_player_gpu[tid]

        if cooldown_ability_owners[index_player] != 0:
            return

        available_bullet_local_idx = wp.int32(-1)
        num_bullets_per_player = owner_mapping_gpu.shape[1] 
        for i in range(num_bullets_per_player):
            local_idx = owner_mapping_gpu[index_player, i]

            if local_idx == -1:
                continue

            if expired_steps[local_idx] <= 0:
                available_bullet_local_idx = local_idx
                break

        if available_bullet_local_idx == -1:
            return 

        # Player transformation lookup
        p_world = index_player // num_objects_env
        p_local_idx = index_player % num_objects_env

        p_obj_idx = wp.int32(-1)
        for i in range(player_count_per_world):
            if player_view_object_indices[i] == p_local_idx:
                p_obj_idx = i
                break

        if p_obj_idx == -1:
            return

        p_local_tid = p_obj_idx * player_bodies_per_object
        p_local_body_idx = player_view_body_local_indices[p_local_tid]
        p_global_body_idx = p_world * num_body_object_env + p_local_body_idx

        # 取得 Bot 自己的資訊
        tf = body_q[p_global_body_idx]
        bot_pos = tf.p 
        bot_rot = tf.q 

        x = bot_rot[0]
        y = bot_rot[1]
        z = bot_rot[2]
        w = bot_rot[3]
        # 從四元數轉換旋轉矩陣的第一行 (Local X axis)
        forward = wp.vec3(
            1.0 - 2.0 * (y**2.0 + z**2.0),
            2.0 * (x * y + w * z),
            -2.0 * (x * z - w * y)
        )

        # 計算子彈初始位置與速度
        bullet_pos = wp.vec3(bot_pos[0], bot_pos[1], bot_pos[2])
        bullet_vel = wp.vec3(forward[0] * speed, forward[1] * speed, forward[2] * speed)

        # Bullet mapping
        bullet_idx = index_obj_role_gpu[available_bullet_local_idx]
        b_world = bullet_idx // num_objects_env
        b_local_idx = bullet_idx % num_objects_env

        b_obj_idx = wp.int32(-1)
        for i in range(bullet_count_per_world):
            if bullet_view_object_indices[i] == b_local_idx:
                b_obj_idx = i
                break

        if b_obj_idx == -1:
            return

        # 套用物理變更
        cooldown_ability_owners[index_player] = cooldown
        expired_steps[available_bullet_local_idx] = default_expired_step_list_gpu[available_bullet_local_idx]

        bullet_control_pos[b_world, b_obj_idx, 0] = bullet_pos
        bullet_control_rot[b_world, b_obj_idx, 0] = bot_rot
        bullet_control_vel[b_world, b_obj_idx, 0] = bullet_vel
        bullet_control_mask[b_world, b_obj_idx, 0] = bullet_control_mask[b_world, b_obj_idx, 0] | 7


    def rl_action(self, actions, **kwargs):
        if getattr(self, "num_rl_players", 0) <= 0 or not getattr(self, "_configured", False):
            return
            
        physics_manager = self.physics_manager

        ctxs = self._shoot_ctx("rl")
        if ctxs is None:
            return
        player_ctx, bullet_ctx = ctxs
        pattern = player_ctx.pattern

        wp.launch(
            kernel=self.rl_action_gpu,
            dim=self.num_rl_players, 
            inputs=[
                self.articulation_body.control_mask_gpus[self.generated_object_pattern],
                self.articulation_body.control_pos_gpus[self.generated_object_pattern],
                self.articulation_body.control_rot_gpus[self.generated_object_pattern],
                self.articulation_body.control_vel_gpus[self.generated_object_pattern],
                physics_manager.state_0.body_q,
                actions,
                self.action_shape_offset,
                self.fire_intent_buffer, 
                self.index_rl_players_gpu,
                self.owner_mapping_gpu,
                self.ability_generated_object.expired_steps,
                self.ability_generated_object.index_obj_role_gpu,
                self.ability_generated_object.default_expired_step_list_gpu,
                self.cooldown_ability_owners,
                self.speed,
                self.cooldown,
                
                self.articulation_body.view_object_indices_gpus[pattern],
                self.articulation_body.view_body_local_indices_gpus[pattern],
                player_ctx.count_per_world,
                player_ctx.bodies_per_object,
                
                self.articulation_body.view_object_indices_gpus[self.generated_object_pattern],
                bullet_ctx.count_per_world,
                
                self.articulation_body.num_objects_env,
                self.articulation_body.num_rigid_bodies_env,
            ],
            device=physics_manager.device
        )

    @wp.kernel
    def rl_action_gpu(
        bullet_control_mask: wp.array3d(dtype=wp.int32),
        bullet_control_pos: wp.array3d(dtype=wp.vec3),
        bullet_control_rot: wp.array3d(dtype=wp.quat),
        bullet_control_vel: wp.array3d(dtype=wp.vec3),
        body_q: wp.array(dtype=wp.transform), 
        actions: wp.array2d(dtype=wp.float32), 
        action_shape_offset: wp.int32,
        fire_intent_buffer: wp.array(dtype=wp.float32), 
        index_bot_players_gpu: wp.array(dtype=wp.int32),
        owner_mapping_gpu: wp.array(dtype=wp.int32, ndim=2),
        expired_steps: wp.array(dtype=wp.int32),
        index_obj_role_gpu: wp.array(dtype=wp.int32),
        default_expired_step_list_gpu: wp.array(dtype=wp.int32),
        cooldown_ability_owners: wp.array(dtype=wp.int32),
        speed: wp.float32,
        cooldown: wp.int32,

        player_view_object_indices: wp.array(dtype=int),
        player_view_body_local_indices: wp.array(dtype=int),
        player_count_per_world: int,
        player_bodies_per_object: int,

        bullet_view_object_indices: wp.array(dtype=int),
        bullet_count_per_world: int,

        num_objects_env: int,
        num_body_object_env: int,
    ):
        tid = wp.tid()

        raw_fire_input = actions[tid][action_shape_offset]
        
        # 計算連續的發射意圖
        fire_intent = sigmoid(raw_fire_input * 10.0)
        fire_intent_buffer[tid] = fire_intent
    
        if raw_fire_input < 0.0:   
            return

        index_bot = index_bot_players_gpu[tid]

        if cooldown_ability_owners[index_bot] != 0:
            return

        available_bullet_local_idx = wp.int32(-1)
        num_bullets_per_player = owner_mapping_gpu.shape[1] 
        for i in range(num_bullets_per_player):
            local_idx = owner_mapping_gpu[index_bot, i]
            if local_idx != -1 and expired_steps[local_idx] <= 0:
                available_bullet_local_idx = local_idx
                break

        if available_bullet_local_idx == -1:
            return 

        # Player transformation lookup
        p_world = index_bot // num_objects_env
        p_local_idx = index_bot % num_objects_env

        p_obj_idx = wp.int32(-1)
        for i in range(player_count_per_world):
            if player_view_object_indices[i] == p_local_idx:
                p_obj_idx = i
                break

        if p_obj_idx == -1:
            return

        p_local_tid = p_obj_idx * player_bodies_per_object
        p_local_body_idx = player_view_body_local_indices[p_local_tid]
        p_global_body_idx = p_world * num_body_object_env + p_local_body_idx

        # 獲取發射者狀態
        tf = body_q[p_global_body_idx]
        bot_pos = tf.p 
        bot_rot = tf.q 

        x = bot_rot[0]
        y = bot_rot[1]
        z = bot_rot[2]
        w = bot_rot[3]
        forward = wp.vec3(
            1.0 - 2.0 * (y**2.0 + z**2.0),
            2.0 * (x * y + w * z),
            -2.0 * (x * z - w * y)
        )

        bullet_pos = wp.vec3(bot_pos[0], bot_pos[1], bot_pos[2])
        bullet_vel = wp.vec3(forward[0] * speed, forward[1] * speed, forward[2] * speed)

        # Bullet mapping
        bullet_idx = index_obj_role_gpu[available_bullet_local_idx]
        b_world = bullet_idx // num_objects_env
        b_local_idx = bullet_idx % num_objects_env

        b_obj_idx = wp.int32(-1)
        for i in range(bullet_count_per_world):
            if bullet_view_object_indices[i] == b_local_idx:
                b_obj_idx = i
                break

        if b_obj_idx == -1:
            return

        cooldown_ability_owners[index_bot] = cooldown
        expired_steps[available_bullet_local_idx] = default_expired_step_list_gpu[available_bullet_local_idx]

        bullet_control_pos[b_world, b_obj_idx, 0] = bullet_pos
        bullet_control_rot[b_world, b_obj_idx, 0] = bot_rot
        bullet_control_vel[b_world, b_obj_idx, 0] = bullet_vel
        bullet_control_mask[b_world, b_obj_idx, 0] = bullet_control_mask[b_world, b_obj_idx, 0] | 7


    def bot_action(self, **kwargs):
        if getattr(self, "num_bot_players", 0) <= 0 or not getattr(self, "_configured", False):
            return
            
        physics_manager = self.physics_manager

        ctxs = self._shoot_ctx("bot")
        if ctxs is None:
            return
        player_ctx, bullet_ctx = ctxs
        pattern = player_ctx.pattern

        wp.launch(
            kernel=self.bot_action_gpu,
            dim=self.num_bot_players, 
            inputs=[
                self.articulation_body.control_mask_gpus[self.generated_object_pattern],
                self.articulation_body.control_pos_gpus[self.generated_object_pattern],
                self.articulation_body.control_rot_gpus[self.generated_object_pattern],
                self.articulation_body.control_vel_gpus[self.generated_object_pattern],
                physics_manager.state_0.body_q,
                self.index_bot_players_gpu,
                self.owner_mapping_gpu,
                self.ability_generated_object.expired_steps,
                self.ability_generated_object.index_obj_role_gpu,
                self.ability_generated_object.default_expired_step_list_gpu,
                self.cooldown_ability_owners,
                
                self.seeds,
                self.offset,
                self.speed, 
                self.cooldown,
                
                self.articulation_body.view_object_indices_gpus[pattern],
                self.articulation_body.view_body_local_indices_gpus[pattern],
                player_ctx.count_per_world,
                player_ctx.bodies_per_object,
                
                self.articulation_body.view_object_indices_gpus[self.generated_object_pattern],
                bullet_ctx.count_per_world,
                
                self.articulation_body.num_objects_env,
                self.articulation_body.num_rigid_bodies_env,
            ],
            device=physics_manager.device
        )

    @wp.kernel
    def bot_action_gpu(
        bullet_control_mask: wp.array3d(dtype=wp.int32),
        bullet_control_pos: wp.array3d(dtype=wp.vec3),
        bullet_control_rot: wp.array3d(dtype=wp.quat),
        bullet_control_vel: wp.array3d(dtype=wp.vec3),
        body_q: wp.array(dtype=wp.transform), 
        index_bot_players_gpu: wp.array(dtype=wp.int32),
        owner_mapping_gpu: wp.array(dtype=wp.int32, ndim=2),
        expired_steps: wp.array(dtype=wp.int32),
        index_obj_role_gpu: wp.array(dtype=wp.int32),
        default_expired_step_list_gpu: wp.array(dtype=wp.int32),
        cooldown_ability_owners: wp.array(dtype=wp.int32),
        
        seeds: wp.array(dtype=wp.int32),
        offset: wp.array(dtype=wp.int32),
        speed: wp.float32,
        cooldown: wp.int32,

        player_view_object_indices: wp.array(dtype=int),
        player_view_body_local_indices: wp.array(dtype=int),
        player_count_per_world: int,
        player_bodies_per_object: int,

        bullet_view_object_indices: wp.array(dtype=int),
        bullet_count_per_world: int,

        num_objects_env: int,
        num_body_object_env: int,
    ):
        tid = wp.tid()
        index_bot = index_bot_players_gpu[tid]

        if cooldown_ability_owners[index_bot] != 0:
            return

        available_bullet_local_idx = wp.int32(-1)
        num_bullets_per_player = owner_mapping_gpu.shape[1] 
        for i in range(num_bullets_per_player):
            local_idx = owner_mapping_gpu[index_bot, i]

            if local_idx == -1:
                continue

            if expired_steps[local_idx] <= 0:
                available_bullet_local_idx = local_idx
                break

        if available_bullet_local_idx == -1:
            return 

        # Player transformation lookup
        p_world = index_bot // num_objects_env
        p_local_idx = index_bot % num_objects_env

        p_obj_idx = wp.int32(-1)
        for i in range(player_count_per_world):
            if player_view_object_indices[i] == p_local_idx:
                p_obj_idx = i
                break

        if p_obj_idx == -1:
            return

        p_local_tid = p_obj_idx * player_bodies_per_object
        p_local_body_idx = player_view_body_local_indices[p_local_tid]
        p_global_body_idx = p_world * num_body_object_env + p_local_body_idx

        # 取得 Bot 自己的資訊
        tf = body_q[p_global_body_idx]
        bot_pos = tf.p 
        bot_rot = tf.q 

        x = bot_rot[0]
        y = bot_rot[1]
        z = bot_rot[2]
        w = bot_rot[3]
        forward = wp.vec3(
            1.0 - 2.0 * (y**2.0 + z**2.0),
            2.0 * (x * y + w * z),
            -2.0 * (x * z - w * y)
        )

        bullet_pos = wp.vec3(bot_pos[0], bot_pos[1], bot_pos[2])
        bullet_vel = wp.vec3(forward[0] * speed, forward[1] * speed, forward[2] * speed)

        # Bullet mapping
        bullet_idx = index_obj_role_gpu[available_bullet_local_idx]
        b_world = bullet_idx // num_objects_env
        b_local_idx = bullet_idx % num_objects_env

        b_obj_idx = wp.int32(-1)
        for i in range(bullet_count_per_world):
            if bullet_view_object_indices[i] == b_local_idx:
                b_obj_idx = i
                break

        if b_obj_idx == -1:
            return

        rng = wp.rand_init(seeds[tid], offset=offset[tid])
        cooldown_ability_owners[index_bot] = cooldown + wp.randi(rng, 0, cooldown/2)
        offset[tid] += 1
        expired_steps[available_bullet_local_idx] = default_expired_step_list_gpu[available_bullet_local_idx]

        bullet_control_pos[b_world, b_obj_idx, 0] = bullet_pos
        bullet_control_rot[b_world, b_obj_idx, 0] = bot_rot
        bullet_control_vel[b_world, b_obj_idx, 0] = bullet_vel
        bullet_control_mask[b_world, b_obj_idx, 0] = bullet_control_mask[b_world, b_obj_idx, 0] | 7


    def setup_keymapping(self):
        super().setup_keymapping(self.__class__.__name__)
        self._keyboard_action = self.control_keys["keyboard"].get("action", [])
        self._mouse_action = self.control_keys["mouse"].get("action", [])

    def update_index_bot(self, index_rl_players_gpu, num_rl_players, index_bot_players_gpu, num_bot_players):
        super().update_index_bot(index_rl_players_gpu=index_rl_players_gpu, num_rl_players=num_rl_players, index_bot_players_gpu=index_bot_players_gpu, num_bot_players=num_bot_players)
        self.fire_intent_buffer = wp.zeros(self.num_rl_players, dtype=wp.float32, device=self.physics_manager.device)

        self.num_bullets = len(self.index_ability_generated_object_gpu)
        self.hitted_bullet = wp.zeros(self.num_bullets, dtype=wp.int32, device=self.physics_manager.device)

        import numpy as np
        seed = GameConfig.SEED
        seeds_np = np.arange(seed, seed+self.num_bot_players + 1, dtype=np.int32)
        self.seeds = wp.array(seeds_np, dtype=wp.int32, device=self.physics_manager.device)
        self.offset = wp.zeros(shape=self.num_bot_players, dtype=wp.int32, device=self.physics_manager.device)
        
    def reset(self):
        return super().reset()