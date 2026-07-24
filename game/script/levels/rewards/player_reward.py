import math
import warp as wp
import numpy as np

from script.game_config import GameConfig
from script.levels.rewards.reward_calculator import RewardComponent
from utils.warp_math import calculate_ballistic_aim_dir_move

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from script.simulate.physics_manager import PhysicsManager
    from script.role.base_role import BaseRole
    from script.role.player import Player
    from script.role.platform import Platform
    from script.role.ability_generated_object import AbilityGeneratedObject

class PlayerShotReward2(RewardComponent):
    """
    1. 判斷 hitted_bullet == 1: 退出計算
    2. 判斷 hitted_bullet == 2: 給予 hit_reward + 剩餘接近獎勵
    3. 判斷 hitted_bullet == 3: 新增：計算射擊精度獎勵（彈道匹配），抵消 shoot_penalty
    4. 默認：計算子彈接近敵人的動態獎勵
    """
    
    def __init__(self, device, abilities_objects: 'AbilityGeneratedObject', **kwargs):
        super().__init__(**kwargs)
        self.device = device
        
        # 獎勵參數
        self.hit_reward = self.params["shoot_hit_reward"]
        self.being_hit_penalty = self.params["shoot_being_hit_penalty"]
        self.shoot_penalty = self.params["shoot_penalty"] 
        self.total_approach_budget = self.params["shoot_total_approach_budget"]
        
        # --- 新增參數 ---
        self.gravity = self.params.get("gravity", -98.1)
        self.accuracy_bonus_scale = self.params.get("accuracy_bonus_scale", 4.0) 
        
        for ability in abilities_objects.abilities_instance_list:
            if ability.ability_name.lower() == "shoot":
                self.bullet_shape_ids_gpu = ability.index_ability_generated_object_gpu
                self.bullet_owner_gpu = ability.owner_list_gpu
                self.hitted_bullet = ability.hitted_bullet

        self.num_bullets = len(self.bullet_shape_ids_gpu)
        if self.num_bullets > 0:
            self.bullet_spent_budget = wp.zeros(self.num_bullets, dtype=wp.float32, device=self.device)
        else:
            raise ValueError(f"Number of bullets must more than one!")

    def calculate(self, physics_manager: 'PhysicsManager', is_rl_player_mask_gpu: wp.array, _index_obj_to_env_mapping_gpu: wp.array, env_players_index_offset: wp.array, num_players_each_env: wp.int32, step_total_rewards: wp.array, **kwargs):

        wp.launch(
            kernel=self.calculate_gpu,
            dim=self.num_bullets,
            inputs=[
                physics_manager.state_0.body_q,
                physics_manager.state_0.body_qd,

                self.bullet_shape_ids_gpu,
                self.bullet_owner_gpu,

                is_rl_player_mask_gpu,
                _index_obj_to_env_mapping_gpu,
                env_players_index_offset, 
                num_players_each_env,
                self.bullet_spent_budget,

                step_total_rewards,
                self.hitted_bullet,
                self.hit_reward,
                self.being_hit_penalty,
                self.shoot_penalty,
                self.total_approach_budget,

                self.gravity,
                self.accuracy_bonus_scale
            ],
            device=self.device
        )

        # # Debug 輸出 (可選)
        # reward_cpu = step_total_rewards.numpy()
        # if reward_cpu.sum() != 0.0:
        #     print("PlayerShotReward triggered:")
        # # print("self.debug_values_query_face: ", self.debug_values_query_face.numpy())
        # # print("self.debug_values_is_blocked: ", self.debug_values_is_blocked.numpy())
        # # print("self.debug_values_obj_size: ", self.debug_values_obj_size.numpy()[0])
        #     print(f"Step rewards: {reward_cpu}")

    @wp.kernel
    def calculate_gpu(
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector), 

        bullet_shape_ids_gpu: wp.array(dtype=wp.int32),
        bullet_owner_gpu: wp.array(dtype=wp.int32),

        is_rl_player_mask_gpu: wp.array(dtype=wp.int32),
        _index_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        env_players_index_offset: wp.array(dtype=wp.int32),
        num_players_each_env: wp.int32,
        bullet_spent_budget: wp.array(dtype=wp.float32), 

        step_total_rewards: wp.array(dtype=wp.float32),
        hitted_bullet: wp.array(dtype=wp.int32),
        hit_reward: wp.float32,
        being_hit_penalty: wp.float32,
        shoot_penalty: wp.float32,
        total_approach_budget: wp.float32,
        gravity: wp.float32,
        accuracy_bonus_scale: wp.float32
    ):
        tid = wp.tid()
        if tid >= bullet_shape_ids_gpu.shape[0]: return

        if hitted_bullet[tid] == 1: 
            bullet_spent_budget[tid] = 0.0
            return

        b_shape_id = bullet_shape_ids_gpu[tid]
        shooter_idx = bullet_owner_gpu[tid]
        # if is_rl_player_mask_gpu[shooter_idx] == -1: return

        b_transform = body_q[b_shape_id]
        b_pos = wp.transform_get_translation(b_transform)
        b_vel = wp.spatial_top(body_qd[b_shape_id])
        b_speed = wp.length(b_vel)

        index_env = _index_obj_to_env_mapping_gpu[shooter_idx]
        index_offset_player = env_players_index_offset[index_env]

        if hitted_bullet[tid] == 3:
            if b_speed < 1e-3: 
                wp.atomic_add(step_total_rewards, shooter_idx, shoot_penalty)
                return
            
            b_dir = b_vel / b_speed
            max_aim_quality = wp.float32(0.0)

            for i in range(num_players_each_env):
                p_shape_id = i + index_offset_player
                if p_shape_id == shooter_idx: continue

                p_pos = wp.transform_get_translation(body_q[p_shape_id])
                diff = p_pos - b_pos
                target_vel = body_qd[p_shape_id]
                tv = wp.vec3(target_vel[0], target_vel[1], target_vel[2])
                ideal_dir = calculate_ballistic_aim_dir_move(diff, tv, b_speed, -gravity)
                    
                aim_quality = wp.dot(b_dir, ideal_dir)
                if aim_quality > max_aim_quality:
                    max_aim_quality = aim_quality

            accuracy_bonus = wp.pow(wp.max(0.0, max_aim_quality), 8.0) * accuracy_bonus_scale
            wp.atomic_add(step_total_rewards, shooter_idx, accuracy_bonus)
            return

        if hitted_bullet[tid] == 2:
            being_hit_player = wp.int32(-1)
            for i in range(num_players_each_env):
                p_shape_id = i + index_offset_player
                if p_shape_id == shooter_idx: continue
                being_hit_player = p_shape_id
                
            remaining_approach = wp.max(0.0, total_approach_budget - bullet_spent_budget[tid])
            wp.atomic_add(step_total_rewards, shooter_idx, hit_reward + remaining_approach)
            wp.atomic_add(step_total_rewards, being_hit_player, being_hit_penalty)
            bullet_spent_budget[tid] = total_approach_budget 
            return 

    def reset(self, **kwargs):
        pass


class PlayerFaceToTargetReward1(RewardComponent):
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
        self.gravity = GameConfig.GRAVITY[2]

        self.decrease_starting_step = self.params.get("decrease_starting_step", 100000)
        self.max_reward_fov_degrees_final = self.params.get("max_reward_fov_degrees_final", 3.0)
        self.decrease_fov_speed = self.params.get("decrease_fov_speed", 0.001)

        # 訓練步數計數器
        self.current_training_step_gpu = wp.zeros(shape=num_max_players, dtype=wp.int32, device=device)

        # 距離轉平方
        self.max_dist_sq = self.max_dist * self.max_dist

        # debug
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
                physics_manager.state_0.body_qd, # 傳入速度狀態
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
                self.min_reward_fov_degrees,       
                self.decrease_starting_step,
                self.decrease_fov_speed,
                self.bullet_speed,
                self.gravity,

                # debug
                # self.debug_value1,
                # self.debug_value2,
                # self.debug_value3,
            ],
            device=self.device
        )
        # print("debug_value1: ", self.debug_value1)
        # print("debug_value2: ", self.debug_value2)
        # print("debug_value3: ", self.debug_value3)

    @wp.kernel
    def calculate_gpu(
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector), # 線性速度資訊
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
        max_fov_deg_start: wp.float32,
        max_fov_deg_final: wp.float32,
        min_fov_deg_static: wp.float32,
        decrease_start_step: wp.int32,
        decrease_speed: wp.float32,
        bullet_speed: float,
        gravity: float,

        # debug
        # debug_value1: wp.array(dtype=wp.int32),
        # debug_value2: wp.array(dtype=wp.int32),
        # debug_value3: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        my_shape_id = player_shape_ids_gpu[tid]

        if player_health[my_shape_id] <= 0:
            return
        if is_rl_player_mask_gpu[tid] == -1:
            return
        
        my_tf = body_q[my_shape_id]
        my_pos = wp.transform_get_translation(my_tf)
        my_rot = wp.transform_get_rotation(my_tf)
        # 獲取自身速度
        mv = body_qd[my_shape_id]
        my_vel = wp.vec3(mv[0], mv[1], mv[2])

        closest_dist_sq = wp.float32(max_dist_sq)
        target_pos = wp.vec3(0.0, 0.0, 0.0)
        target_vel = wp.vec3(0.0, 0.0, 0.0)
        found_target = wp.bool(False)

        index_env = index_player_obj_to_env_mapping_gpu[tid]
        index_offset_player = env_players_index_offset[index_env]

        for i in range(num_players_each_env):
            index_player = i + index_offset_player
            if index_player == my_shape_id or player_health[index_player] <= 0:
                continue
            
            other_pos = wp.transform_get_translation(body_q[index_player])
            diff = other_pos - my_pos
            d_sq = wp.dot(diff, diff)

            if d_sq < closest_dist_sq:
                closest_dist_sq = d_sq
                target_pos = other_pos
                tv = body_qd[index_player]
                target_vel = wp.vec3(tv[0], tv[1], tv[2])
                found_target = wp.bool(True)

        if not found_target:
            return

        # 計算相對速度 (如果需要)
        rel_vel = target_vel - my_vel
        ideal_aim_dir = calculate_ballistic_aim_dir_move(target_pos - my_pos, rel_vel, bullet_speed, gravity)

        forward_dir = wp.quat_rotate(my_rot, wp.vec3(1.0, 0.0, 0.0))
        
        step = current_training_step_gpu[tid]
        current_training_step_gpu[tid] += 1
        current_max_fov = max_fov_deg_start
        if step > decrease_start_step:
            decay = wp.float32(step - decrease_start_step) * decrease_speed
            current_max_fov = wp.max(max_fov_deg_final, max_fov_deg_start - decay)

        max_dot = wp.cos(current_max_fov * 3.1415926 / 180.0)
        min_dot = wp.cos(min_fov_deg_static * 3.1415926 / 180.0)

        # 比較 forward_dir 與 彈道預判方向 ideal_aim_dir
        dot_prod = wp.clamp(wp.dot(forward_dir, ideal_aim_dir), -1.0, 1.0)

        facing_score = wp.float32(0.0)
        if dot_prod >= max_dot:
            facing_score = 1.0
        elif dot_prod > min_dot:
            facing_score = (dot_prod - min_dot) / (max_dot - min_dot)
        else:
            facing_score = 0.0

        # 射線檢測 (保持原本檢測「是否有阻擋」的邏輯)
        is_blocked = wp.bool(False)
        eye_offset = wp.vec3(0.0, 0.0, body_size_gpu[my_shape_id][2] + my_pos[2]) 
        ray_start = my_pos + eye_offset
        ray_end = target_pos + eye_offset
        ray_vec = ray_end - ray_start
        ray_dist = wp.length(ray_vec)
        ray_dir = ray_vec / ray_dist 
        query = wp.mesh_query_ray(mesh_id, ray_start, ray_dir, ray_dist)
        if query.face != -1 and query.t > 0.0001 and query.t < ray_dist * 0.99:
            is_blocked = wp.bool(True)

        if not is_blocked:
            step_total_rewards[my_shape_id] += (face_to_target_reward * facing_score)

    def reset(self, **kwargs):
        pass

class PlayerFaceToTargetReward2(RewardComponent):
    def __init__(self, device, num_max_players, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.num_max_players = num_max_players

        # 參數加載
        self.focus_fov_degrees = self.params.get("focus_fov_degrees")
        self.max_dist = self.params["max_dist"]
        self.face_to_target_reward = self.params["face_to_target_reward"]
        
        # 新增彈道參數
        self.bullet_speed = self.params.get("bullet_speed", 100.0)
        self.gravity = self.params.get("gravity", -9.81)
        
        self.turn_towards_factor = self.params.get("turn_towards_factor", 10.0) 
        self.lost_focus_penalty = self.params.get("lost_focus_penalty", -0.05)   
        self.focus_timeout_steps = self.params.get("focus_timeout_steps", 100) 

        # 衰減參數
        self.current_training_step_gpu = wp.zeros(shape=num_max_players, dtype=wp.int32, device=device)
        self.decrease_starting_step = self.params.get("decrease_starting_step", 100000)
        self.focus_fov_degrees_final = self.params.get("focus_fov_degrees_final", 5.0)
        self.decrease_fov_speed = self.params.get("decrease_fov_speed", 0.001)

        # --- 狀態緩衝區 ---
        self.last_dot_prod = wp.zeros(shape=num_max_players, dtype=wp.float32, device=self.device)
        self.last_dot_prod.fill_(-2.0) 
        self.lost_focus_count = wp.zeros(shape=num_max_players, dtype=wp.int32, device=self.device)

        self.max_dist_sq = self.max_dist * self.max_dist

        # # debug
        # self.debug_value1 = wp.zeros(shape=10, dtype=wp.int32, device=self.device)
        # self.debug_value2 = wp.zeros(shape=10, dtype=wp.int32, device=self.device)
        # self.debug_value3 = wp.zeros(shape=10, dtype=wp.float32, device=self.device)

    def calculate(self, num_players, physics_manager, player_shape_ids_gpu, is_rl_player_mask_gpu, 
                  index_player_obj_to_env_mapping_gpu, env_players_index_offset, num_players_each_env, 
                  player_health, step_total_rewards, **kwargs):

        wp.launch(
            kernel=self.calculate_gpu_improved,
            dim=num_players,
            inputs=[
                physics_manager.state_0.body_q,
                physics_manager.state_0.body_qd, # 傳入速度以計算提前量
                player_shape_ids_gpu,
                is_rl_player_mask_gpu,
                index_player_obj_to_env_mapping_gpu,
                env_players_index_offset,
                num_players_each_env,
                player_health,
                step_total_rewards,
                physics_manager.mesh.id,
                physics_manager.body_size_gpu,
                self.max_dist_sq,
                self.face_to_target_reward,
                self.turn_towards_factor,
                self.lost_focus_penalty,
                self.focus_timeout_steps,
                self.last_dot_prod,
                self.lost_focus_count,
                self.current_training_step_gpu,
                self.focus_fov_degrees,
                self.decrease_starting_step,
                self.focus_fov_degrees_final,
                self.decrease_fov_speed,
                self.bullet_speed, # 彈道參數
                self.gravity,      # 彈道參數

                # # debug
                # self.debug_value1,
                # self.debug_value2,
                # self.debug_value3,
            ],
            device=self.device
        )
        # print("debug_value1: ", self.debug_value1)
        # print("debug_value2: ", self.debug_value2)
        # print("debug_value3: ", self.debug_value3)

    @wp.kernel
    def calculate_gpu_improved(
        body_q: wp.array(dtype=wp.transform),
        body_qd: wp.array(dtype=wp.spatial_vector), # 加入速度數組
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
        turn_towards_factor: wp.float32,
        lost_focus_penalty: wp.float32,
        focus_timeout_steps: wp.int32,
        last_dot_prod_arr: wp.array(dtype=wp.float32),
        lost_focus_count_arr: wp.array(dtype=wp.int32),
        current_training_step_gpu: wp.array(dtype=wp.int32),
        focus_fov_degrees: wp.float32,
        decrease_starting_step: wp.int32,
        focus_fov_degrees_final: wp.float32,
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

        if player_health[my_shape_id] <= 0:
            last_dot_prod_arr[tid] = -2.0  
            lost_focus_count_arr[tid] = 0
            return

        # 動態閾值計算...
        step = current_training_step_gpu[tid]
        current_training_step_gpu[tid] += 1
        current_fov = focus_fov_degrees
        if step > decrease_starting_step:
            decay = wp.float32(step - decrease_starting_step) * decrease_fov_speed
            current_fov = wp.max(focus_fov_degrees_final, focus_fov_degrees - decay)
        dynamic_threshold_dot = wp.cos(current_fov * 3.1415926 / 180.0)

        if is_rl_player_mask_gpu[tid] == -1: return
        
        my_tf = body_q[my_shape_id]
        my_pos = wp.transform_get_translation(my_tf)
        my_rot = wp.transform_get_rotation(my_tf)

        closest_dist_sq = max_dist_sq
        target_pos = wp.vec3(0.0, 0.0, 0.0)
        target_shape_id = wp.int32(-1)
        found_target = wp.bool(False)

        index_env = index_player_obj_to_env_mapping_gpu[tid]
        index_offset_player = env_players_index_offset[index_env]

        for i in range(num_players_each_env):
            index_player = i + index_offset_player
            if index_player == my_shape_id or player_health[index_player] <= 0: continue
            
            other_pos = wp.transform_get_translation(body_q[index_player])
            diff = other_pos - my_pos
            d_sq = wp.dot(diff, diff)
            if d_sq < closest_dist_sq:
                closest_dist_sq = d_sq
                target_pos = other_pos
                target_shape_id = index_player
                found_target = True

        if not found_target:
            last_dot_prod_arr[tid] = -2.0 
            return

        # --- 【修改點】計算彈道射角 ---
        to_target_vec = target_pos - my_pos
        # 從 spatial_vector 提取前 3 個元素作為線速度
        target_vel = wp.vec3(
            body_qd[target_shape_id][0], 
            body_qd[target_shape_id][1], 
            body_qd[target_shape_id][2]
        )
        
        # 替換原有的 to_target_dir (歸一化)
        to_target_dir = calculate_ballistic_aim_dir_move(to_target_vec, target_vel, bullet_speed, gravity)
        
        forward_dir = wp.quat_rotate(my_rot, wp.vec3(1.0, 0.0, 0.0))
        current_dot = wp.clamp(wp.dot(forward_dir, to_target_dir), -1.0, 1.0)

        # 射線檢測保持不變 (檢測是否存在遮擋)
        is_blocked = False
        eye_offset = wp.vec3(0.0, 0.0, body_size_gpu[my_shape_id][2] + my_pos[2]) 
        ray_start = my_pos + eye_offset
        ray_end = target_pos + eye_offset
        ray_vec = ray_end - ray_start
        ray_dist = wp.length(ray_vec)
        # query = wp.mesh_query_ray(mesh_id, ray_start, ray_vec / wp.max(ray_dist, 1e-6), ray_dist)
        query = wp.mesh_query_ray(mesh_id, ray_start, ray_vec / ray_dist, ray_dist)
        if query.face != -1 and query.t > 0.0001 and query.t < ray_dist * 0.99:
            is_blocked = True

            
        reward = 0.0
        last_dot = last_dot_prod_arr[tid]
        if not is_blocked:
            if last_dot > -1.5:
                delta_dot = current_dot - last_dot
                reward += delta_dot * turn_towards_factor

            if current_dot > dynamic_threshold_dot:
                reward += face_to_target_reward
                lost_focus_count_arr[tid] = 0  
            else:
                lost_focus_count_arr[tid] += 1 
        else:
            lost_focus_count_arr[tid] += 1

        if lost_focus_count_arr[tid] > focus_timeout_steps:
            reward += lost_focus_penalty

        step_total_rewards[my_shape_id] += reward
        last_dot_prod_arr[tid] = current_dot

    def reset(self, num_players, terminated, index_player_obj_to_env_mapping_gpu, **kwargs):
        wp.launch(
            kernel=self.reset_gpu,
            dim=num_players,
            inputs=[
                terminated,
                index_player_obj_to_env_mapping_gpu,
                self.last_dot_prod,
            ],
            device=self.device
        )

    @wp.kernel
    def reset_gpu(
        terminated: wp.array(dtype=wp.bool), 
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32), 
        last_dot_prod: wp.array(dtype=wp.float32), 
    ):
        tid = wp.tid()
        index_env = index_player_obj_to_env_mapping_gpu[tid]
        if terminated[index_env] == False:
            return
        last_dot_prod[tid] = -2.0



















