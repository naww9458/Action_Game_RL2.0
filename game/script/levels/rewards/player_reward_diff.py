import math
import warp as wp
import numpy as np

from script.game_config import GameConfig
from script.levels.rewards.reward_calculator import RewardComponent
from utils.warp_math import sigmoid, calculate_ballistic_aim_dir_move  # 確保導入了你定義的 sigmoid

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from script.simulate.physics_manager import PhysicsManager
    from script.role.base_role import BaseRole
    from script.role.player import Player
    from script.role.platform import Platform
    from script.role.ability_generated_object import AbilityGeneratedObject


class PlayerFaceToTargetReward1_diff(RewardComponent):
    """
    可微化改造後的面向獎勵：
    1. 使用 Sigmoid 替代硬性 FOV 邊界，提供連續梯度。
    2. 解決了準星在目標邊緣時梯度消失(Gradient Vanishing)的問題。
    """

    def __init__(self, device, num_max_players, abilities_objects: 'AbilityGeneratedObject', **kwargs):
        super().__init__(**kwargs)

        self.device = device

        # 原始參數
        self.max_reward_fov_degrees = self.params["max_reward_fov_degrees"]
        self.min_reward_fov_degrees = self.params["min_reward_fov_degrees"]
        self.max_dist = self.params["max_dist"]

        self.face_to_target_reward = self.params["face_to_target_reward"]
        
        for ability in abilities_objects.abilities_instance_list:
            if ability.ability_name.lower() == "shoot":
                self.bullet_speed = ability.speed
                break

        self.gravity = GameConfig.GRAVITY[2]

        # 衰減參數 (作用於 max_reward_fov_degrees)
        self.decrease_starting_step = self.params["decrease_starting_step"]
        self.max_reward_fov_degrees_final = self.params["max_reward_fov_degrees_final"]
        self.decrease_fov_speed = self.params["decrease_fov_speed"]

        # 訓練步數計數器
        self.current_training_step_gpu = wp.zeros(shape=num_max_players, dtype=wp.int32, device=device)

        # 距離轉平方
        self.max_dist_sq = self.max_dist * self.max_dist

        # # debug
        # self.debug_value1 = wp.zeros(shape=10, dtype=wp.int32, device=self.device)
        # self.debug_value2 = wp.zeros(shape=10, dtype=wp.int32, device=self.device)
        # self.debug_value3 = wp.zeros(shape=10, dtype=wp.float32, device=self.device)

    def calculate(self, num_players, physics_manager: 'PhysicsManager', player_shape_ids_gpu: wp.array, is_rl_player_mask_gpu: wp.array, index_player_obj_to_env_mapping_gpu: wp.array, env_players_index_offset: wp.array, num_players_each_env: wp.int32, player_health: wp.array, step_total_rewards: wp.array, **kwargs):

        mesh_id = physics_manager.mesh.id 

        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_players,
            inputs=[
                physics_manager.state_0.body_q,
                physics_manager.state_0.body_qd,
                player_shape_ids_gpu,
                is_rl_player_mask_gpu,
                index_player_obj_to_env_mapping_gpu,
                env_players_index_offset,
                num_players_each_env,

                player_health,
                step_total_rewards,
                mesh_id,
                physics_manager.body_size_gpu,
                self.max_dist_sq,
                self.face_to_target_reward,

                self.current_training_step_gpu,
                self.max_reward_fov_degrees,       
                self.max_reward_fov_degrees_final, 
                self.decrease_starting_step,
                self.decrease_fov_speed,
                self.bullet_speed,
                self.gravity,

                # # debug
                # self.debug_value1,
                # self.debug_value2,
                # self.debug_value3,
            ],
            device=self.device
        )
        # print("debug_value1: ", self.debug_value1)
        # print("debug_value2: ", self.debug_value2)
        # print("debug_value3: ", self.debug_value3.numpy()[0], self.debug_value3.numpy()[2])
        # print("is_rl_player_mask_gpu: ", is_rl_player_mask_gpu)
        
        # if step_total_rewards.to("cpu").numpy()[0] > 0:
        #     print("step_total_rewards: ", step_total_rewards.to('cpu').numpy()[0])  # 調試輸出

    @wp.kernel
    def calculate_gpu(
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector),
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        is_rl_player_mask_gpu: wp.array(dtype=wp.int32),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        env_players_index_offset: wp.array(dtype=wp.int32),
        num_players_each_env: wp.int32,

        player_health: wp.array(dtype=wp.float32),
        step_total_rewards: wp.array(dtype=wp.float32),
        mesh_id: wp.uint64,
        body_size_gpu: wp.array(dtype=wp.vec3),
        max_dist_sq: wp.float32,
        face_to_target_reward: wp.float32,
        current_training_step_gpu: wp.array(dtype=wp.int32),
        max_reward_fov_degrees: wp.float32,
        max_reward_fov_degrees_final: wp.float32,
        decrease_starting_step: wp.int32,
        decrease_fov_speed: wp.float32,
        bullet_speed: float, 
        gravity: float,


        # # debug
        # debug_value1: wp.array(dtype=wp.int32),
        # debug_value2: wp.array(dtype=wp.int32),
        # debug_value3: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        my_shape_id = player_shape_ids_gpu[tid]

        # Warp AD: avoid early return. Mask inactive threads instead.
        alive = wp.where(player_health[my_shape_id] > 0.0, 1.0, 0.0)
        is_rl = wp.where(is_rl_player_mask_gpu[tid] != -1, 1.0, 0.0)
        active = alive * is_rl

        my_tf = body_q[my_shape_id]
        my_pos = wp.transform_get_translation(my_tf)
        my_rot = wp.transform_get_rotation(my_tf)

        # TODO Hardcode 現在硬編碼，之後添加一個包含所有敵方目標索引的陣列
        # 沒必要算誰最近，而是把所有的叠加起來并用極高的指數衰減讓不會出現瞄準兩個目標中間大於完美瞄準一個目標的情況
        target_pos = wp.transform_get_translation(body_q[my_shape_id + 1])

        # --- 連續向量計算 (Differentiable Path) ---
        diff = target_pos - my_pos
        target_vel = body_qd[my_shape_id + 1]
        tv = wp.vec3(target_vel[0], target_vel[1], target_vel[2])
        to_target_dir = calculate_ballistic_aim_dir_move(diff, tv, bullet_speed, gravity)

        # 獲取玩家前方向量 (X-Forward)
        forward_dir = wp.quat_rotate(my_rot, wp.vec3(1.0, 0.0, 0.0))
        dot_prod = wp.dot(forward_dir, to_target_dir)

        # --- 動態衰減邏輯 (常量計算，不影響梯度) ---
        step = current_training_step_gpu[tid]
        current_training_step_gpu[tid] = step + 1
        current_max_fov = max_reward_fov_degrees
        if step > decrease_starting_step:
            decay = wp.float32(step - decrease_starting_step) * decrease_fov_speed
            current_max_fov = wp.max(max_reward_fov_degrees_final, max_reward_fov_degrees - decay)

        # APG-safe facing score:
        # - Do NOT use acos(clamp(dot)): derivative blows up / zeros at |dot|=1.
        # - Do NOT use exp(-k * angle) with max_fov≈1°: away from the target the
        #   reward and ∂R/∂q both underflow to ~0, so torque/action grads vanish.
        # Wide basin from cosine similarity (always non-zero slope except exactly
        # opposite), plus a soft peak near the configured FOV using (1 - dot).
        coarse_score = 0.5 * (1.0 + dot_prod)
        target_fov_rad = current_max_fov * 3.1415926 / 180.0
        # 1-cos(θ) ≈ θ²/2; scale so the peak width tracks max FOV without underflow.
        fine_k = 1.0 / wp.max(1.0 - wp.cos(target_fov_rad), 1e-3)
        fine_score = wp.exp(-fine_k * (1.0 - dot_prod))
        facing_score = 0.5 * coarse_score + 0.5 * fine_score


        # # --- 射線檢測 (保持離散遮斷，但不影響旋轉梯度) ---
        # is_blocked = wp.bool(False)
        # # 眼睛位置偏移
        # eye_offset = wp.vec3(0.0, 0.0, body_size_gpu[my_shape_id][2] * 0.5) 
        # ray_start = my_pos + eye_offset
        # ray_end = target_pos + eye_offset
        # ray_vec = ray_end - ray_start
        # ray_dist = wp.length(ray_vec)

        # ray_dir = ray_vec / ray_dist 
        # query = wp.mesh_query_ray(mesh_id, ray_start, ray_dir, ray_dist)
        # if query.face != -1 and query.t > 0.0001 and query.t < ray_dist * 0.99:
        #    is_blocked = wp.bool(True)
        
        # if ray_dist > 0.001:
        #     query = wp.mesh_query_ray(mesh_id, ray_start, ray_vec / ray_dist, ray_dist)
        #     # 如果射線撞到了物體，且不是剛好撞到目標本身 (t < ray_dist * 0.99)
        #     if query.face != -1 and query.t < ray_dist * 0.95:
        #         is_blocked = True

        # # --- 最終獎勵累積 ---
        # if not is_blocked:
        #     # 獎勵數值現在是關於旋轉四元數 my_rot 的連續函數
        #     # 梯度可以通過 facing_score -> dot_prod -> forward_dir -> my_rot 完美傳導

        current_step_reward = face_to_target_reward * facing_score * active
        wp.atomic_add(step_total_rewards, my_shape_id, current_step_reward)

    def reset(self, **kwargs):
        pass


class BulletTrajectoryEvasionReward_diff(RewardComponent):
    """
    1.以己方子彈的視角/索引開始
    2.通過自己子彈可傷害角色陣列獲取敵方玩家的索引
    3.再通過敵方玩家的索引獲取敵方子彈的索引
    4.然後通過索引獲取需要的坐標，速度數據計算獎勵
    5.最後通過己方子彈的索引獲取自己角色索引然後給予獎勵

    軌跡避險獎勵：
    1. 僅當玩家位於子彈前進方向的『威脅圓柱體』內時給予懲罰。
    2. 如果玩家正在遠離彈道中心線（增加垂直距離），給予規避獎勵。
    3. 忽略子彈與玩家的歐幾里得距離變化，只看垂直偏移。
    """
    
    def __init__(self, device, abilities_objects: 'AbilityGeneratedObject', **kwargs):
        super().__init__(**kwargs)
        self.device = device
        
        # 參數設置
        self.threat_radius = self.params["threat_radius"]        # 彈道威脅半徑（圓柱體粗細）
        self.evasion_penalty = self.params["evasion_penalty"]    # 處於威脅區的每幀懲罰
        self.evasion_bonus_scale = self.params["evasion_bonus"]  # 正在遠離彈道的獎勵權重

        
        self.window_size = (GameConfig.space_x, GameConfig.space_y, GameConfig.space_z)
        self.max_threat_dist = np.linalg.norm(self.window_size)

        for ability in abilities_objects.abilities_instance_list:
            if ability.ability_name.lower() == "shoot":
                self.bullet_shape_ids_gpu = ability.index_ability_generated_object_gpu
                self.bullet_owner_gpu = ability.owner_list_gpu
                self.bullet_enemy_gpu = ability.enemy_list_gpu
                self.player_to_bullet_map_gpu = ability.player_to_bullet_map_gpu
                self.hitted_bullet = ability.hitted_bullet

        self.num_bullets = len(self.bullet_shape_ids_gpu)

        self.debug_value_prev = 0.0
        # self.debug_value1 = wp.zeros(shape=self.num_bullets, dtype=wp.int32, device=device)
        # self.debug_value2 = wp.zeros(shape=self.num_bullets, dtype=wp.float32, device=device)

    def calculate(self, physics_manager, is_rl_player_mask_gpu, index_player_obj_to_env_mapping_gpu, 
                  env_players_index_offset, num_players_each_env, step_total_rewards, **kwargs):

        wp.launch(
            kernel=self.calculate_trajectory_evasion_gpu,
            dim=self.num_bullets,
            inputs=[
                physics_manager.state_0.body_q,
                physics_manager.state_0.body_qd,
                self.bullet_owner_gpu,
                self.bullet_enemy_gpu,
                self.hitted_bullet,
                self.player_to_bullet_map_gpu,

                is_rl_player_mask_gpu,
                index_player_obj_to_env_mapping_gpu,
                env_players_index_offset,
                num_players_each_env,
                step_total_rewards,
                self.threat_radius,
                self.evasion_penalty,
                self.evasion_bonus_scale,
                self.max_threat_dist,

                # self.debug_value1,
                # self.debug_value2,
            ],
            device=self.device
        )

        # if self.debug_value.to("cpu").numpy()[0] != 0 and self.debug_value.to("cpu").numpy()[0] != self.debug_value_prev:
        #     print("Bullet Evasion Debug Values:", self.debug_value.to("cpu").numpy()[0])  # 調試輸出
        #     self.debug_value_prev = self.debug_value.to("cpu").numpy()[0]

        # if step_total_rewards.to("cpu").numpy()[0] > 0:
        #     print("step_total_rewards: ", step_total_rewards.to('cpu').numpy()[0])  # 調試輸出

    @wp.kernel
    def calculate_trajectory_evasion_gpu(
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector), 
        bullet_owner_gpu: wp.array(dtype=wp.int32),
        bullet_enemy_gpu: wp.array(dtype=wp.int32, ndim=2),
        hitted_bullet: wp.array(dtype=wp.int32),
        player_to_bullet_map_gpu: wp.array(dtype=wp.int32),

        is_rl_player_mask_gpu: wp.array(dtype=wp.int32),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        env_players_index_offset: wp.array(dtype=wp.int32),
        num_players_each_env: wp.int32,
        step_total_rewards: wp.array(dtype=wp.float32), 
        threat_radius: wp.float32,
        evasion_penalty: wp.float32,
        evasion_bonus_scale: wp.float32,
        max_threat_dist: wp.float32,

        # debug_value1: wp.array(dtype=wp.int32),
        # debug_value2: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        # if is_rl_player_mask_gpu[tid] == -1:
        #     return

        index_self_player = bullet_owner_gpu[tid]
        index_enemy_player = bullet_enemy_gpu[tid][0] # TODO Hard code
        b_id = player_to_bullet_map_gpu[index_enemy_player]

        if hitted_bullet[tid] != 0: return

        # 獲取狀態
        b_pos = wp.transform_get_translation(body_q[b_id])
        b_vel = wp.vec3(body_qd[b_id][0], body_qd[b_id][1], body_qd[b_id][2])

        p_pos = wp.transform_get_translation(body_q[index_self_player])
        p_vel = wp.vec3(body_qd[index_self_player][0], body_qd[index_self_player][1], body_qd[index_self_player][2])

        b_speed = wp.length(b_vel)
        if b_speed < 10: return
        b_dir = b_vel / b_speed

        # 投影距離
        rel_pos = p_pos - b_pos
        proj_len = wp.dot(rel_pos, b_dir)
        
        # 激活區域 mask
        mask = wp.clamp(proj_len / 0.1, 0.0, 1.0) * wp.clamp((max_threat_dist - proj_len) / 0.1, 0.0, 1.0)
        
        # 計算垂直偏移向量與距離
        perp_vec = rel_pos - b_dir * proj_len
        perp_dist = wp.length(perp_vec)
        
        # 防止除以零且確保在威脅區內
        if perp_dist < 1e-4: return 
        perp_dir = perp_vec / perp_dist

        # 【核心修改點】：只提取橫向速度 (Lateral Velocity)
        # 1. 計算速度在彈道方向上的投影 (平行速度分量)
        vel_parallel_mag = wp.dot(p_vel, b_dir)
        vel_parallel = vel_parallel_mag * b_dir
        
        # 2. 從總速度中扣除平行分量，得到純橫向速度
        vel_lateral = p_vel - vel_parallel
        
        # 3. 計算橫向速度在遠離彈道方向上的分量
        # 只有當玩家橫向移動的方向是「遠離彈道中心線」時，才會獲得正向獎勵
        away_velocity = wp.dot(vel_lateral, perp_dir)
        
        reward_multiplier = sigmoid(away_velocity - 2.0)
        # 懲罰項（保持不變）
        threat_intensity = wp.max(0.0, 1.0 - (perp_dist / threat_radius))
        reward_p = evasion_penalty * threat_intensity * mask
        
        # 獎勵項：強制只獎勵「遠離彈道的橫向移動」
        reward_p += wp.max(0.0, reward_multiplier) * evasion_bonus_scale * mask * threat_intensity

        step_total_rewards[index_self_player] += reward_p

    def reset(self, **kwargs):
        pass


class ProximityPenaltyReward_diff(RewardComponent):
    """
    指數級接近懲罰：
    1. 機器人接近懲罰：防止模型貼臉換血，強迫維持作戰距離。
    2. 邊界接近懲罰：防止模型卡在牆角或試圖衝出邊界。
    
    公式：Penalty = -scale * exp(alpha * (threshold - distance))
    當 distance < threshold 時，懲罰隨距離縮短而指數級激增。
    """
    
    def __init__(self, device, num_max_players, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.num_max_players = num_max_players

        # --- 機器人排斥參數 ---
        self.robot_threshold = self.params["robot_proximity_threshold"]  # 多少米內開始觸發
        self.robot_scale = self.params["robot_proximity_scale"]          # 基礎係數
        self.robot_alpha = self.params["robot_proximity_alpha"]          # 指數增長率

        # --- 環境邊界參數 (假設地圖是 AABB 矩形) ---
        self.wall_threshold = self.params["wall_proximity_threshold"]    # 離牆2米開始懲罰
        self.wall_scale = self.params["wall_proximity_scale"]
        self.wall_alpha = self.params["wall_proximity_alpha"]

        # # debug
        # self.debug_value1 = wp.zeros(shape=10, dtype=wp.int32, device=self.device)
        # self.debug_value2 = wp.zeros(shape=10, dtype=wp.int32, device=self.device)
        # self.debug_value3 = wp.zeros(shape=10, dtype=wp.float32, device=self.device)

    def calculate(self, num_players, physics_manager: 'PhysicsManager', player_shape_ids_gpu, 
                  index_player_obj_to_env_mapping_gpu, env_players_index_offset, env_platforms_index_offset, 
                  num_players_each_env, num_platforms_each_env, player_health, step_total_rewards, **kwargs):

        wp.launch(
            kernel=self.calculate_proximity_gpu,
            dim=num_players,
            inputs=[
                physics_manager.state_0.body_q,
                physics_manager.body_size_gpu,
                player_shape_ids_gpu,
                index_player_obj_to_env_mapping_gpu,

                env_players_index_offset,
                env_platforms_index_offset,

                num_players_each_env,
                num_platforms_each_env,

                player_health,
                step_total_rewards,
                # 機器人避障參數
                self.robot_threshold,
                self.robot_scale,
                self.robot_alpha,
                # 牆體避障參數
                self.wall_threshold,
                self.wall_scale,
                self.wall_alpha,

                # # debug
                # self.debug_value1,
                # self.debug_value3,
            ],
            device=self.device
        )

        # print("self.debug_value1: ", self.debug_value1)
        # print("self.debug_value3: ", self.debug_value3.numpy()[2])

        # if step_total_rewards.to("cpu").numpy()[0] < 0:
        #     print("Proximity Penalty Applied, step_total_rewards: ", step_total_rewards.to('cpu').numpy()[0])  # 調試輸出

    @wp.kernel
    def calculate_proximity_gpu(
        body_q: wp.array(dtype=wp.transform),
        body_size_gpu: wp.array(dtype=wp.vec3),
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),

        env_players_index_offset: wp.array(dtype=wp.int32), 
        env_platforms_index_offset: wp.array(dtype=wp.int32),

        num_players_each_env: wp.int32,
        num_platforms_each_env: wp.int32,

        player_health: wp.array(dtype=wp.float32),
        step_total_rewards: wp.array(dtype=wp.float32),
        
        r_threshold: wp.float32,
        r_scale: wp.float32,
        r_alpha: wp.float32,
        
        w_threshold: wp.float32,
        w_scale: wp.float32,
        w_alpha: wp.float32,

        # # debug
        # debug_value1: wp.array(dtype=wp.int32), 
        # debug_value3: wp.array(dtype=wp.float32), 
        
    ):
        tid = wp.tid()

        my_id = player_shape_ids_gpu[tid]
        if player_health[my_id] <= 0: return

        my_pos = wp.transform_get_translation(body_q[my_id])
        
        env_idx = index_player_obj_to_env_mapping_gpu[tid]
        total_penalty = wp.float32(0.0)
        
        # --- 1. 機器人與機器人間的距離懲罰 ---
        offset = env_players_index_offset[env_idx]

        for i in range(num_players_each_env):
            other_id = offset + i
            if other_id == my_id or player_health[other_id] <= 0:
                continue
            
            other_pos = wp.transform_get_translation(body_q[other_id])
            
            dist = wp.length(my_pos - other_pos)

            # if dist < r_threshold:
            # 指數懲罰公式: -scale * exp(alpha * (threshold - dist))
            # 當 dist=threshold 時, penalty = -scale
            # 當 dist=0 時, penalty = -scale * exp(alpha * threshold) -> 非常巨大
            p = r_scale * wp.exp(r_alpha * (r_threshold - dist))
            total_penalty -= p

        # --- 2. 動態牆體/平台接近懲罰 ---
        plat_offset = env_platforms_index_offset[env_idx]
        
        for i in range(num_platforms_each_env):
            # 獲取平台索引
            plat_id = plat_offset + i
            plat_pos = wp.transform_get_translation(body_q[plat_id])
            
            # 獲取平台尺寸 (假設 body_size 為盒子全尺寸, 需要除以 2 得到半寬)
            size = body_size_gpu[plat_id]
            half_size = size
            
            # 計算點到 AABB 的最短距離
            dx = wp.max(0.0, wp.abs(my_pos[0] - plat_pos[0]) - half_size[0])
            dy = wp.max(0.0, wp.abs(my_pos[1] - plat_pos[1]) - half_size[1])
            dz = wp.max(0.0, wp.abs(my_pos[2] - plat_pos[2]) - half_size[2])
            dist = wp.sqrt(dx*dx + dy*dy + dz*dz + 1e-8)
            
            # if dist < w_threshold:
            wall_p = w_scale * wp.exp(w_alpha * (w_threshold - dist))
            total_penalty -= wall_p

        wp.atomic_add(step_total_rewards, my_id, total_penalty)

    def reset(self, **kwargs):
        pass


class PlayerTestGradReward_diff(RewardComponent):
    """
    注意：此獎勵函數僅僅用於測試物理引擎的微分功能，和實際訓練完全沒有關係
    """

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        

    def calculate(self, num_players, physics_manager: 'PhysicsManager', actions: wp.array2d, is_rl_player_mask_gpu: wp.array, index_player_obj_to_env_mapping_gpu: wp.array, player_shape_ids_gpu: wp.array, num_players_each_env: wp.int32, step_total_rewards: wp.array, **kwargs):
        # 注意：actions 必須從外層傳進來，這是梯度的來源
        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_players,
            inputs=[
                actions, # TODO if action is None, will throw illegal memory access error

                physics_manager.state_0.body_q,
                physics_manager.state_0.body_qd,
                is_rl_player_mask_gpu,
                player_shape_ids_gpu,
                step_total_rewards,
            ],
            device=self.device
        )

    @wp.kernel
    def calculate_gpu(
        actions: wp.array(dtype=wp.float32, ndim=2),

        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector),
        is_rl_player_mask_gpu: wp.array(dtype=wp.int32),
        player_shape_ids_gpu: wp.array(dtype=wp.int32),

        step_total_rewards: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        index_rl_action = is_rl_player_mask_gpu[tid]
        player_index = player_shape_ids_gpu[tid]

        # # --- 計算發射意圖權重 ---
        # # raw_fire_input 是 80 * 3，屬於 RL agent 的 shooter_idx 是 0，2，4，6，8... 160 因此導致非法内存存取，shooter_idx 1，3，5，7，9... 是 Bot Player
        # if index_rl_action != -1: 
        #     raw_fire_input = actions[index_rl_action, 0] # + actions[index_rl_action, 1] + actions[index_rl_action, 2] + actions[index_rl_action, 3] 
        # else:
        #     raw_fire_input = 1.0
        

        # my_tf = body_q[player_index]
        # # my_pos = wp.transform_get_translation(my_tf)

        # 坐標 + 旋轉
        # d1 = my_tf[0] + my_tf[1] + my_tf[2] + my_tf[3] + my_tf[4] + my_tf[5] + my_tf[6] 

        my_v = body_qd[player_index]

        # 綫速度 + 角速度
        # d2 = my_v[0] + my_v[1] + my_v[2] + my_v[3] + my_v[4] + my_v[5] 

        # 角速度
        d2 = my_v[3] + my_v[4] + my_v[5] 

        # raw_fire_input = d1 + d2
        raw_fire_input = d2


        wp.atomic_add(step_total_rewards, player_index, raw_fire_input)

    def reset(self, **kwargs):
        pass




