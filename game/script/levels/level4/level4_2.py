import torch
import warp as wp
import numpy as np

from script.game_config import GameConfig
from utils.warp_math import sigmoid, safe_length, safe_normalize, calculate_ballistic_aim_dir_move
from utils.ray_distance_to_blocks import get_ray_distance_to_blocks

try:
    from levels.rewards.reward_calculator import RewardCalculator
    from levels.rewards.game_end_reward import ShootingGameTerminated
    from levels.rewards.player_reward import PlayerShotReward2, PlayerFaceToTargetReward1, PlayerFaceToTargetReward2
    from levels.rewards.player_reward_diff import PlayerFaceToTargetReward1_diff, BulletTrajectoryEvasionReward_diff, ProximityPenaltyReward_diff
    from training.level_defaults import get_default_train_cfg
    from levels.levels import Levels
    from levels.level4.level4_0 import Level4_0
except ImportError:
    from script.levels.rewards.reward_calculator import RewardCalculator
    from script.levels.rewards.game_end_reward import ShootingGameTerminated
    from script.levels.rewards.player_reward import PlayerShotReward2, PlayerFaceToTargetReward1, PlayerFaceToTargetReward2
    from script.levels.rewards.player_reward_diff import PlayerFaceToTargetReward1_diff, BulletTrajectoryEvasionReward_diff, ProximityPenaltyReward_diff
    from training.level_defaults import get_default_train_cfg
    from script.levels.levels import Levels
    from script.levels.level4.level4_0 import Level4_0

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from script.simulate.physics_manager import PhysicsManager
    # 將導致循環導入的 import 語句移到這裡
    pass


class Level4_1(Level4_0):
    """
    Level 4_1: Basic setup with a dynamic body and a static kinematic body.

    Two players are introduced, each with their own dynamic body.
    """
    def __init__(self, 
                 **kwargs
                ):
        super().__init__(**kwargs)
        self.obs_dim = 42

        # debug
        # self.debug_value1 = wp.array(shape=30, dtype=wp.int32, device=GameConfig.DEVICE)


    def setup(self):
        Levels.setup(self=self)
        
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
            print(f"\n\033[38;5;No specify reward_components, using default rewards.196m\033[0m")
            # reward_components_cls = [PlayerShotReward2]
            reward_components_cls = [
                    # PlayerShotReward2, 
                    # PlayerFaceToTargetReward1, 
                    # PlayerFaceToTargetReward2, 
                    # PlayerFaceToTargetReward1_diff, 
                    # BulletTrajectoryEvasionReward_diff, 
                    # ProximityPenaltyReward_diff, 
            ]
            reward_components_diff_cls = []
            GameConfig.reward_parameters = get_default_train_cfg(4, 1).reward_parameters

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
                self.index_player_to_bullet_map_gpu = ability.player_to_bullet_map_gpu
                self.shoot_cooldown_ability_owners = ability.cooldown_ability_owners
                self.bullet_speed = ability.speed
                self.hitted_bullet = ability.hitted_bullet
                self.num_bullets = ability.num_bullets
                break
        
        self.gravity = self.physics_manager.gravity[2]
        # self.debug_values = wp.zeros(shape=10, dtype=wp.int32, device=GameConfig.DEVICE)

        if self.players.num_total_object_role > 2 and self.players.num_total_object_role < 1:
            print(f"\n\033[38;5;196mLevel 4 only supports 1 or 2 players in RL.196m\033[0m")

        return self.players, self.platforms, self.entities, self.abilities_objects, self.reward_calculator

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

        # print("self.entities.index_role_offset_env_gpu: ", self.entities.index_role_offset_env_gpu)
        # print("len(self.entities.index_role_offset_env_gpu): ", len(self.entities.index_role_offset_env_gpu.numpy()))
        # print("self.entities.num_role_each_env: ", self.entities.num_role_each_env)
        # print("len(self.entities.num_role_each_env): ", len(self.entities.num_role_each_env.numpy()))

        wp.launch(
            kernel=self.compute_observations_kernel,
            dim=self.players.num_rl_players,
            inputs=[
                self.physics_manager.state_0.body_q,
                self.physics_manager.state_0.body_qd,
                self.physics_manager.body_size_gpu,
                self.game.current_step,
                self.game.max_episode_step,
                self.reward_calculator.player_health,
                self.shoot_cooldown_ability_owners,
                self.players.index_rl_players_gpu,
                self._index_obj_to_env_mapping_gpu,

                self.reward_calculator.index_player_offset_env_gpu,
                self.reward_calculator.index_platform_offset_env_gpu,
                self.reward_calculator.num_players_each_env,
                self.reward_calculator.num_platform_each_env,

                self.index_player_to_bullet_map_gpu,
                self.hitted_bullet,

                self.entities.index_role_offset_env_gpu,
                self.entities.num_role_each_env,

                self.owner_map_bullet_gpu,

                self.velocity_scale,
                self.ang_vel_scale,
                self.max_dist,

                self.bullet_speed,
                self.gravity,
                self.obs_buf_gpu,

                # debug
                # self.debug_value1
            ],
            device=GameConfig.DEVICE,
        )

        obs_torch = wp.to_torch(self.obs_buf_gpu)
        # print("obs_torch: ", obs_torch)

        # if not torch.isfinite(obs_torch).all():
        #     print("obs_torch NAN")
        #     raise RuntimeError()
        
        return obs_torch

    @wp.kernel
    def compute_observations_kernel(
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector),
        body_size_gpu: wp.array(dtype=wp.vec3),
        current_step: wp.array(dtype=wp.int32),
        max_episode_step: wp.int32,
        player_health: wp.array(dtype=wp.float32),
        cooldowns: wp.array(dtype=wp.int32),
        index_rl_players_gpu: wp.array(dtype=wp.int32),
        _index_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        
        env_players_index_offset: wp.array(dtype=wp.int32), 
        env_platforms_index_offset: wp.array(dtype=wp.int32),

        num_players_each_env: wp.int32,
        num_platforms_each_env: wp.int32,

        index_player_to_bullet_map_gpu: wp.array(dtype=wp.int32),
        hitted_bullet: wp.array(dtype=wp.int32),

        index_entities_offset_env_gpu: wp.array(dtype=wp.int32), 
        num_entities_each_env: wp.int32,

        owner_map_bullet_gpu: wp.array2d(dtype=wp.int32), 

        velocity_scale: float,
        ang_vel_scale: float,
        max_dist: float,

        bullet_speed: float,    
        gravity: float,             
        obs_out: wp.array(dtype=float, ndim=2),

        # debug_value1: wp.array(dtype=wp.int32), 
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

        # --- 敵方子彈的距離、方向和速度 (接續你的代碼) ---
        
        # 取得敵方子彈在物理引擎中的索引
        # 假設 index_player_to_bullet_map_gpu[enemy_idx] 指向該敵人發射的子彈
        # bot 必須要有 shoot 技能，不然 index_player_to_bullet_map_gpu 中對應的索引的值會是 -1 然後導致非法内存存取
        bullet_idx = index_player_to_bullet_map_gpu[enemy_idx]
        if bullet_idx >= 0: 
            # 初始化子彈相關觀測值 (預設為 0，若子彈不存在/未激活則維持 0)
            b_rel_dir_local = wp.vec3(0.0, 0.0, 0.0)
            # 預設距離極遠
            b_dist_norm = 1.0  
            b_rel_v_local = wp.vec3(0.0, 0.0, 0.0)

            bullet_t = body_q[bullet_idx]
            bullet_pos = wp.transform_get_translation(bullet_t)
            
            # 1. 計算相對距離與方向
            b_diff = bullet_pos - self_pos
            b_dist = safe_length(b_diff)
            
            # 歸一化距離 (tanh 映射到 0~1)
            b_dist_norm = wp.tanh(b_dist / max_dist)
            
            # 局部方向向量
            b_rel_dir_local = safe_normalize(wp.quat_rotate_inv(self_quat, b_diff))
            
            # 2. 計算相對速度
            # 取得子彈速度 (假設線性速度在前三個分量)
            b_v_spatial = body_qd[bullet_idx]
            bullet_vel = wp.vec3(b_v_spatial[0], b_v_spatial[1], b_v_spatial[2])
            
            # 玩家自身速度 (線性)
            self_v_lin = wp.vec3(self_vel_spatial[0], self_vel_spatial[1], self_vel_spatial[2])
            
            # 世界座標系下的相對速度
            b_rel_v_world = bullet_vel - self_v_lin
            
            # 轉換到玩家局部座標系
            b_rel_v_local = wp.quat_rotate_inv(self_quat, b_rel_v_world)

            speed_val = wp.length(bullet_vel)

            # TODO 代表每個玩家必定只能有一個子彈
        
            bullet_local_idx = owner_map_bullet_gpu[enemy_idx][0]
            if speed_val > 10 and bullet_local_idx >= 0 and (hitted_bullet[bullet_local_idx] == 0 or hitted_bullet[bullet_local_idx] == 3): 
                obs_out[tid, 18] = 1.0 
                obs_out[tid, 19] = b_dist_norm
                
                # 27, 28, 29: 子彈相對方向 (Local)
                obs_out[tid, 20] = b_rel_dir_local[0]
                obs_out[tid, 21] = b_rel_dir_local[1]
                obs_out[tid, 22] = b_rel_dir_local[2]
                
                # 30, 31, 32: 子彈相對速度 (Local, 使用 velocity_scale 歸一化)
                obs_out[tid, 23] = wp.tanh(b_rel_v_local[0] / velocity_scale)
                obs_out[tid, 24] = wp.tanh(b_rel_v_local[1] / velocity_scale)
                obs_out[tid, 25] = wp.tanh(b_rel_v_local[2] / velocity_scale)

            else:
                obs_out[tid, 18] = 0.0 
                obs_out[tid, 19] = 1.0
                
                # 27, 28, 29: 子彈相對方向 (Local)
                obs_out[tid, 20] = 0.0
                obs_out[tid, 21] = 0.0
                obs_out[tid, 22] = 0.0
                
                # 30, 31, 32: 子彈相對速度 (Local, 使用 velocity_scale 歸一化)
                obs_out[tid, 23] = 0.0
                obs_out[tid, 24] = 0.0
                obs_out[tid, 25] = 0.0
                
        else:
            obs_out[tid, 18] = 0.0 
            obs_out[tid, 19] = 1.0
            
            # 27, 28, 29: 子彈相對方向 (Local)
            obs_out[tid, 20] = 0.0
            obs_out[tid, 21] = 0.0
            obs_out[tid, 22] = 0.0
            
            # 30, 31, 32: 子彈相對速度 (Local, 使用 velocity_scale 歸一化)
            obs_out[tid, 23] = 0.0
            obs_out[tid, 24] = 0.0
            obs_out[tid, 25] = 0.0

        
        # --- 新增：計算牆壁向量 (26-29) ---
        index_env = _index_obj_to_env_mapping_gpu[self_idx]
        plat_offset = env_platforms_index_offset[index_env]
        n_plats = num_platforms_each_env
        
        min_dist = wp.float32(1e6)
        closest_vec_local = wp.vec3(0.0, 0.0, 0.0)

        # 四個墻壁的數據
        for i in range(n_plats):
            offset = i * n_plats

            p_idx = plat_offset + i
            p_pos = wp.transform_get_translation(body_q[p_idx])
            
            # 使用 AABB 距離計算
            size = body_size_gpu[p_idx]
            half_size = size * 0.5
            diff_vec = p_pos - self_pos
            
            dx = wp.max(0.0, wp.abs(diff_vec[0]) - half_size[0])
            dy = wp.max(0.0, wp.abs(diff_vec[1]) - half_size[1])
            dz = wp.max(0.0, wp.abs(diff_vec[2]) - half_size[2])
            dist = wp.sqrt(dx*dx + dy*dy + dz*dz + 1e-8)
            
            closest_vec_local = wp.quat_rotate_inv(self_quat, diff_vec)

            # 寫入 Observation (正規化)
            # 若沒有牆壁，min_dist 將保持 1e6，tanh 後趨近 1.0 (代表安全)
            obs_out[tid, 26 + offset] = wp.tanh(dist / max_dist) 
            # 將方向向量歸一化並放入 (使用 max_dist 正規化以保持梯度一致性)
            obs_out[tid, 27 + offset] = closest_vec_local[0] / max_dist
            obs_out[tid, 28 + offset] = closest_vec_local[1] / max_dist
            obs_out[tid, 29 + offset] = closest_vec_local[2] / max_dist


        # # --- 計算射線距離 (Ray Distance) ---
        # self_t = body_q[self_idx]
        # self_pos = wp.transform_get_translation(self_t)
        # self_quat = wp.transform_get_rotation(self_t)

        # # --- 填充 Observation 18-25: 使用工具函數 ---
        # chest_height = 0.5 
        # block_radius = 0.6
        # chest_pos = self_pos + wp.vec3(0.0, 0.0, chest_height)
        # index_entities_offset_env = index_entities_offset_env_gpu[index_env]

        # fwd_world = wp.quat_rotate(self_quat, wp.vec3(1.0, 0.0, 0.0))
        
        # # 2. 將 Forward 投影到地平面 (XY 平面)，並歸一化
        # # 這樣就得到了一個只代表「左右轉向」的水平向量
        # fwd_flat = wp.normalize(wp.vec3(fwd_world[0], fwd_world[1], 0.0))
        
        # # 3. 計算 Yaw 角度 (atan2)
        # yaw = wp.atan2(fwd_flat[1], fwd_flat[0])
        
        # # 4. 現在我們自己構建旋轉矩陣/向量，只包含這個 Yaw
        # # 我們不再使用 self_quat，而是使用這個僅包含 Yaw 的旋轉邏輯
        
        # chest_height = 0.1 
        # block_radius = 0.6
        # chest_pos = self_pos + wp.vec3(0.0, 0.0, chest_height)

        # for i in range(8):
        #     # 射線相對於角色的角度
        #     angle = float(i) * 0.78539816339 
            
        #     # 在水平面上，射線的方向 = (角色Yaw + 射線偏移角)
        #     ray_angle = yaw + angle
            
        #     # 轉換為世界空間的方向向量
        #     world_ray_dir = wp.vec3(wp.cos(ray_angle), wp.sin(ray_angle), 0.0)
            
        #     # 呼叫工具函數 (現在 world_ray_dir 永遠是水平的)
        #     dist = get_ray_distance_to_blocks(
        #         chest_pos, 
        #         world_ray_dir, 
        #         index_entities_offset_env, 
        #         num_entities_each_env, 
        #         body_q, 
        #         max_dist, 
        #         block_radius
        #     )
            
        #     obs_out[tid, 26 + i] = wp.clamp(dist / max_dist, 0.0, 1.0)






