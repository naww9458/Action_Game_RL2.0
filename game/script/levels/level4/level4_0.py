import torch
import warp as wp
import numpy as np

from script.game_config import GameConfig
from utils.warp_math import sigmoid, safe_length, safe_normalize, calculate_ballistic_aim_dir_move

try:
    from levels.rewards.reward_calculator import RewardCalculator
    from levels.rewards.game_end_reward import ShootingGameTerminated
    from training.level_defaults import get_default_train_cfg
    from levels.levels import Levels
except ImportError:
    from script.levels.rewards.reward_calculator import RewardCalculator
    from script.levels.rewards.game_end_reward import ShootingGameTerminated
    from training.level_defaults import get_default_train_cfg
    from script.levels.levels import Levels

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from script.simulate.physics_manager import PhysicsManager
    # 將導致循環導入的 import 語句移到這裡
    pass


class Level4_0(Levels):
    """
    Level 4.0: Basic setup with two dynamic body.

    One bot and one player.
    """
    def __init__(self, 
                 **kwargs
                ):
        super().__init__(**kwargs)

        self.obs_dim = 18

        self.window_size = (GameConfig.space_x, GameConfig.space_y, GameConfig.space_z)

        self.max_dist = np.linalg.norm(self.window_size)

        self.velocity_scale = 8
        # self.ang_vel_scale = players[0].abilities["Turning_topdown_viewing_angle"].speed + 1 # 加 1 避免角色因为撞擊導致角速度超過最大值
        self.ang_vel_scale = 3.5

    def setup(self):
        super().setup()

        pattern = self.resolve_player_pattern()
        
        # Set initial random velocity after setup
        self.reset_env(self.game.terminated, self.game.current_step)
        self.physics_manager.simulate()

        self.obs_buf_gpu: wp.array # assign value after assign player
        self.obs_buf_gpu = wp.zeros((self.players.num_rl_players, self.obs_dim), dtype=float, device=GameConfig.DEVICE)

        try:
            reward_components_cls = GameConfig.reward_components
            reward_components_diff_cls = GameConfig.reward_components_diff

        except AttributeError as e:
            print(f"\n\033[38;5;196mNo specify reward_components, using default rewards.\033[0m")
            train_cfg = get_default_train_cfg(4, 0)
            reward_components_cls = train_cfg.reward_components
            reward_components_diff_cls = train_cfg.reward_components_diff
            GameConfig.reward_parameters = train_cfg.reward_parameters


        reward_components = []
        for cls in reward_components_cls:
            rc = cls(
                device=GameConfig.DEVICE,
                abilities_objects=self.abilities_objects,
                num_max_players=self.players.num_total_object_role,
                articulation_body=self.articulation_body,
                deformable_body=self.deformable_body,
                reward_parameters=GameConfig.reward_parameters,
                pattern=pattern,
            )
            if hasattr(rc, "bind_level"):
                rc.bind_level(self)
            reward_components.append(rc)

        reward_components_diff = []
        for cls in reward_components_diff_cls:
            reward_components_diff.append(
                cls(
                    device=GameConfig.DEVICE,
                    abilities_objects=self.abilities_objects,
                    num_max_players=self.players.num_total_object_role,
                    articulation_body=self.articulation_body,
                    deformable_body=self.deformable_body,
                    reward_parameters=GameConfig.reward_parameters,
                )
            )


        shooting_game_end = ShootingGameTerminated(
            device=GameConfig.DEVICE,
            articulation_body=self.articulation_body,
            deformable_body=self.deformable_body, 
            reward_parameters=GameConfig.reward_parameters
        )

        self.reward_calculator = RewardCalculator(
            level=self,
            terminated=self.game.terminated,
            reward_components=reward_components,
            reward_components_diff=reward_components_diff,
            episode_end_detector=shooting_game_end,
        )

        self.shoot_cooldown_ability_owners = None
        self.bullet_speed = 0.0
        for ability in self.players.abilities_instance_list: 
            if ability.__class__.__name__.lower() == "shoot":
                self.bullet_shape_ids_gpu = ability.index_ability_generated_object_gpu
                self.owner_map_bullet_gpu = ability.owner_mapping_gpu
                self.bullet_map_owner_gpu = ability.owner_list_gpu
                self.shoot_cooldown_ability_owners = ability.cooldown_ability_owners
                self.bullet_speed = ability.speed
                self.hitted_bullet = ability.hitted_bullet
                self.num_bullets = ability.num_bullets
                break
        
        self.gravity = self.physics_manager.gravity[2]
        # self.debug_values = wp.zeros(shape=10, dtype=wp.int32, device=self.physics_manager.device)


        if self.players.num_total_object_role > 2 and self.players.num_total_object_role < 1:
            print(f"\n\033[38;5;196mLevel 4 only supports 1 or 2 players in RL.196m\033[0m")

        return self.players, self.platforms, self.entities, self.abilities_objects, self.reward_calculator

    def action(self):
        """
        shape state changes in the game
        """

        pass

    def update_game_status(self, physics_manager: 'PhysicsManager', reward_calculator: RewardCalculator, num_env: wp.int32, current_step: wp.array):
        """
        shape state changes in the game
        """
        # 處理子彈命中等碰撞邏輯，更新角色血量等狀態，判斷游戲是否結束，然後才是獎勵計算
        wp.launch(
            kernel=self.update_game_status_gpu,
            dim=self.num_bullets,
            inputs=[
                physics_manager.collision_matrix,
                physics_manager.ground_contact_flags,
                self.num_objects_total,

                self.bullet_shape_ids_gpu,
                self.bullet_map_owner_gpu,
                self.abilities_objects.expired_steps,
                self.abilities_objects.default_expired_step_list_gpu,

                self._index_obj_to_env_mapping_gpu,
                reward_calculator.index_player_offset_env_gpu,
                reward_calculator.index_platform_offset_env_gpu,

                self.players.num_role_each_env,
                self.platforms.num_role_each_env,
                reward_calculator.player_health,
                
                self.hitted_bullet,
                
                num_env,
                current_step,

                # debug
                # self.debug_values,
            ],
            device=GameConfig.DEVICE
        )

        # print("self.abilities_objects.expired_steps: ", self.abilities_objects.expired_steps)
        # print("self.debug_values.numpy(): ", self.debug_values.numpy())

    def reset_env(self, terminated, current_step):
        """
        Reset the env to its initial state.
        """
        super().reset_env(terminated=terminated, current_step=current_step)

    def _get_observation_state_based(self) -> torch.Tensor:

        # # ==========================================================================================================================================
        # # The following is for debugging situations that throw a "non-numeric error," 
        # # which is a CPU operation and must be used when CUDA graph are disabled or the function is moved out of the CUDA graph area.
        # body_q = wp.to_torch(self.physics_manager.state_0.body_q)
        # body_qd = wp.to_torch(self.physics_manager.state_0.body_qd)

        # if not torch.isfinite(body_q).all():
        #     print("body_q NAN")
        #     raise RuntimeError()

        # if not torch.isfinite(body_qd).all():
        #     print("body_qd NAN")
        #     raise RuntimeError()
        # # ==========================================================================================================================================

        wp.launch(
            kernel=self.compute_observations_kernel,
            dim=self.players.num_rl_players,
            inputs=[
                self.physics_manager.state_0.body_q,
                self.physics_manager.state_0.body_qd,
                self.game.current_step,
                self.game.max_episode_step,
                self.reward_calculator.player_health,
                self.shoot_cooldown_ability_owners,
                self.players.index_rl_players_gpu,
                self._index_obj_to_env_mapping_gpu,
                self.reward_calculator.index_player_offset_env_gpu,
                self.reward_calculator.num_players_each_env,
                self.velocity_scale,
                self.ang_vel_scale,
                self.max_dist,
                self.bullet_speed,
                self.gravity,
                self.obs_buf_gpu,
            ],
            device=GameConfig.DEVICE,
        )

        obs_torch = wp.to_torch(self.obs_buf_gpu)
        # print("obs_torch: ", obs_torch)

        # if not torch.isfinite(obs_torch).all():
        #     print("obs_torch NAN")
        #     raise RuntimeError()
        
        return obs_torch

    def _get_observation_image_based(self) -> torch.Tensor:
        pass


    @wp.kernel
    def compute_observations_kernel(
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector),
        current_step: wp.array(dtype=wp.int32),
        max_episode_step: wp.int32,
        player_health: wp.array(dtype=wp.float32),
        cooldowns: wp.array(dtype=wp.int32),
        index_rl_players_gpu: wp.array(dtype=wp.int32),
        _index_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        index_player_offset_env_gpu: wp.array(dtype=wp.int32),
        num_players_each_env: wp.int32,
        velocity_scale: wp.float32,
        ang_vel_scale: wp.float32,
        max_dist: wp.float32,
        bullet_speed: wp.float32,    
        gravity: wp.float32,             
        obs_out: wp.array(dtype=wp.float32, ndim=2)
    ):
        tid = wp.tid()
        
        self_idx = index_rl_players_gpu[tid]
        # TODO: 根據邏輯修改
        enemy_idx = self_idx + 1  
        index_env = _index_obj_to_env_mapping_gpu[self_idx]

        # 獲取自身狀態
        self_t = body_q[self_idx]
        self_pos = wp.transform_get_translation(self_t)
        self_quat = wp.transform_get_rotation(self_t)
        
        self_vel_spatial = body_qd[self_idx]
        self_ang_vel = wp.vec3(self_vel_spatial[3], self_vel_spatial[4], self_vel_spatial[5])
        
        enemy_t = body_q[enemy_idx]
        enemy_pos = wp.transform_get_translation(enemy_t)
        
        # --- 彈道計算 (Ballistic Calculation) ---
        
        # 1. 計算相對位移
        diff = enemy_pos - self_pos
        
        # 獲取世界座標下的理想瞄準方向
        target_vel = body_qd[enemy_idx]
        tv = wp.vec3(target_vel[0], target_vel[1], target_vel[2])
        aim_dir_world = calculate_ballistic_aim_dir_move(diff, tv, bullet_speed, gravity)

        # 安全的局部瞄準向量歸一化
        aim_dir_local = wp.quat_rotate_inv(self_quat, aim_dir_world)
        aim_dir_local_norm = safe_normalize(aim_dir_local)

        # --- 計算角度誤差 ---
        aim_x = aim_dir_local_norm[0]
        aim_y = aim_dir_local_norm[1]

        # 防止 atan2(0,0) 的奇異點 NaN 梯度
        yaw_error = wp.atan2(aim_y, aim_x + 1e-8)   
        pitch_error = wp.asin(wp.clamp(aim_dir_local_norm[2], -0.999, 0.999))

        # --- 填充 Observation ---
        # 角速度
        obs_out[tid, 0] = wp.tanh(self_ang_vel[0] / ang_vel_scale)
        obs_out[tid, 1] = wp.tanh(self_ang_vel[1] / ang_vel_scale)
        obs_out[tid, 2] = wp.tanh(self_ang_vel[2] / ang_vel_scale)

        # Yaw Error (弧度，建議除以 PI 歸一化到 -1~1)
        obs_out[tid, 3] = yaw_error / 3.1415926
        # Pitch Error (弧度，建議除以 PI/2 歸一化到 -1~1)
        obs_out[tid, 4] = pitch_error / 1.5707963

        # 冷卻
        obs_out[tid, 5] = 1.0 if cooldowns[self_idx] <= 0 else 0.0
        # 進度
        obs_out[tid, 6] = wp.float32(current_step[index_env]) / wp.float32(max_episode_step)
        # 敵人血量
        obs_out[tid, 7] = wp.float32(player_health[enemy_idx]) / 5.0

        # 安全的總距離計算
        dist = safe_length(diff)
        obs_out[tid, 8] = wp.tanh(dist / max_dist)

        # 安全的相對方向向量
        rel_dir_local = safe_normalize(wp.quat_rotate_inv(self_quat, diff))
        obs_out[tid, 9] = rel_dir_local[0]
        obs_out[tid, 10] = rel_dir_local[1]
        obs_out[tid, 11] = rel_dir_local[2]
        # 相對速度 (Local)
        enemy_v = body_qd[enemy_idx]
        self_v = body_qd[self_idx]
        rel_v_world = wp.vec3(enemy_v[0]-self_v[0], enemy_v[1]-self_v[1], enemy_v[2]-self_v[2])
        rel_v_local = wp.quat_rotate_inv(self_quat, rel_v_world)
        obs_out[tid, 12] = wp.tanh(rel_v_local[0] / velocity_scale)
        obs_out[tid, 13] = wp.tanh(rel_v_local[1] / velocity_scale)
        obs_out[tid, 14] = wp.tanh(rel_v_local[2] / velocity_scale)
        # 理想瞄準向量 (Local)
        obs_out[tid, 15] = aim_dir_local_norm[0]
        obs_out[tid, 16] = aim_dir_local_norm[1]
        obs_out[tid, 17] = aim_dir_local_norm[2]

    @wp.kernel
    def update_game_status_gpu(
        role_contact_matrix: wp.array(dtype=wp.int32),
        ground_contact_flags: wp.array(dtype=wp.int32),
        num_objects_total: wp.int32,

        bullet_shape_ids_gpu: wp.array(dtype=wp.int32),
        bullet_owner_gpu: wp.array(dtype=wp.int32),
        expired_steps: wp.array(dtype=wp.int32),
        default_expired_step_list_gpu: wp.array(dtype=wp.int32),

        _index_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        index_player_offset_env_gpu: wp.array(dtype=wp.int32),
        index_platform_offset_env_gpu: wp.array(dtype=wp.int32),

        num_player_each_env: wp.int32,
        num_platform_each_env: wp.int32,
        player_health: wp.array(dtype=wp.float32),

        hitted_bullet: wp.array(dtype=wp.int32),

        num_env: wp.int32,
        current_step: wp.array(dtype=wp.int32),


        # debug
        # debug_values: wp.array(dtype=wp.int32),
    ):
        bullet_idx = wp.tid()

        if bullet_idx < num_env: 
            current_step[bullet_idx] += 1

        b_shape_id = bullet_shape_ids_gpu[bullet_idx]
        b_owner_id = bullet_owner_gpu[bullet_idx]
        index_env = _index_obj_to_env_mapping_gpu[b_shape_id]

        # Reset 檢查
        if expired_steps[bullet_idx] == 0:
            hitted_bullet[bullet_idx] = 0
            return

        if hitted_bullet[bullet_idx] != 0 and hitted_bullet[bullet_idx] != 3:
            # Hitted type 1: disable reward，TODO Hard code
            hitted_bullet[bullet_idx] = 1 
            return

        base_idx = b_shape_id * num_objects_total

        # 地面檢測
        if ground_contact_flags[b_shape_id] == 1:
            expired_steps[bullet_idx] = 0
            hitted_bullet[bullet_idx] = 0
            return
        
        if hitted_bullet[bullet_idx] == 3:
            hitted_bullet[bullet_idx] = 0
        elif expired_steps[bullet_idx] == default_expired_step_list_gpu[bullet_idx]:
            hitted_bullet[bullet_idx] = 3
            return

        # 平台檢測
        offset_index_platform = index_platform_offset_env_gpu[index_env]
        for j in range(num_platform_each_env):
            index_platform = j + offset_index_platform
            if role_contact_matrix[base_idx + index_platform] == 1:
                # Hitted type 1: disable reward，TODO Hard code
                hitted_bullet[bullet_idx] = 1 
                return

        # 玩家檢測 (最關鍵的副作用邏輯)
        offset_index_player = index_player_offset_env_gpu[index_env]
        for i in range(num_player_each_env):
            index_player = i + offset_index_player
            if b_owner_id == index_player:
                continue  # bullet cannot hit its owner

            if role_contact_matrix[base_idx + index_player] == 1:
                # Hitted type 2: reward type 1，TODO Hard code
                hitted_bullet[bullet_idx] = 2 
                # --- 必須使用原子操作防止競態 ---
                # decrease 1 health，TODO Hard code
                wp.atomic_sub(player_health, index_player, 1.0) 
                return
            





