import warp as wp

from script.game_config import GameConfig

from .ability import Ability
from utils.warp_math import calculate_ballistic_aim_dir

class Turning_topdown_viewing_angle(Ability):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        # 控制增益
        self.torque_gain = 100.0   # 扭矩增益，越大則越快達到目標速度
        self.max_torque = 400.0    # 最大扭矩限制

        cfg = Ability._default_configs.root.get(self.ability_name)
        self.torque_gain = wp.float32(float(getattr(cfg, "torque_gain", 100.0)) if cfg else 100.0)
        self.max_torque = wp.float32(float(getattr(cfg, "max_torque", 400.0)) if cfg else 400.0)
        # Bot 瞄準用的彈道子彈速度，獨立於 Shoot 能力，避免跨能力耦合。
        self.bullet_speed = wp.float32(float(getattr(cfg, "bullet_speed", 0.0)) if cfg else 0.0)

        self.gravity = GameConfig.GRAVITY[2]

    def apply_ability_config_overrides(self, overrides) -> None:
        super().apply_ability_config_overrides(overrides)
        if not isinstance(overrides, dict):
            return
        if "bullet_speed" in overrides:
            self.bullet_speed = wp.float32(float(overrides["bullet_speed"]))
        if "torque_gain" in overrides:
            self.torque_gain = wp.float32(float(overrides["torque_gain"]))
        if "max_torque" in overrides:
            self.max_torque = wp.float32(float(overrides["max_torque"]))
            
    def human_control_interface(self, look_yaw, look_pitch, index_human_player_gpu, **kwargs):
        """
        將滑鼠輸入的 look_yaw, look_pitch 傳入 GPU 計算並更新旋轉四元數
        """
        if look_yaw is None or look_pitch is None or not getattr(self, "_configured", False):
            return None
        
        ctx = self._view_ctx("human")
        if ctx is None:
            return
        pattern = ctx.pattern

        # 啟動 Kernel
        wp.launch(
            kernel=self.human_control_rotation_gpu,
            dim=1, 
            inputs=[
                self.articulation_body.control_mask_gpus[pattern],  
                self.articulation_body.control_rot_gpus[pattern],  # 目標：儲存旋轉的 buffer (dtype=wp.quat)
                index_human_player_gpu,             # 目標索引
                self.cooldown_ability_owners,
                look_yaw,                 # 純量輸入 (float)
                look_pitch,               # 純量輸入 (float)
                
                self.articulation_body.view_object_indices_gpus[pattern],
                self.articulation_body.num_objects_env,
                ctx.count_per_world,
            ],
            device=self.physics_manager.device
        )

    @wp.kernel
    def human_control_rotation_gpu(
        control_mask: wp.array3d(dtype=wp.int32),
        control_rot: wp.array3d(dtype=wp.quat), # 輸出：旋轉 Buffer
        index_human_player_gpu: wp.array(dtype=wp.int32), # 參數：玩家索引
        cooldown_ability_owners: wp.array(dtype=wp.int32),
        look_yaw: float,                  # 參數：Yaw 角度 (Degree)
        look_pitch: float,                # 參數：Pitch 角度 (Degree)

        view_object_indices: wp.array(dtype=int),
        num_objects_env: int,
        count_per_world: int,
    ):
        player_idx = index_human_player_gpu[0]
        if cooldown_ability_owners[player_idx] != 0:
            return
            
        world = player_idx // num_objects_env
        local_idx = player_idx % num_objects_env

        obj_idx = wp.int32(-1)
        for i in range(count_per_world):
            if view_object_indices[i] == local_idx:
                obj_idx = i
                break

        if obj_idx == -1:
            return
        
        deg_to_rad = 0.01745329251 # PI / 180.0
        
        yaw_rad = look_yaw * deg_to_rad
        pitch_rad = look_pitch * deg_to_rad

        yaw_half = yaw_rad * 0.5
        pitch_half = pitch_rad * 0.5

        cy = wp.cos(yaw_half)
        sy = wp.sin(yaw_half)
        cp = wp.cos(pitch_half)
        sp = wp.sin(pitch_half)

        cr = 1.0 # cos(0)
        sr = 0.0 # sin(0)

        qw = cy * cp * cr + sy * sp * sr
        qx = cy * cp * sr - sy * sp * cr
        qy = cy * sp * cr + sy * cp * sr
        qz = sy * cp * cr - cy * sp * sr

        control_mask[world, obj_idx, 0] = control_mask[world, obj_idx, 0] | 2
        control_rot[world, obj_idx, 0] = wp.quat(qx, qy, qz, qw)

    def rl_action(self, actions, **kwargs):
        if getattr(self, "num_rl_players", 0) <= 0 or not getattr(self, "_configured", False):
            return

        ctx = self._view_ctx("rl")
        if ctx is None:
            return
        pattern = ctx.pattern
        bodies_per_object = ctx.bodies_per_object
        
        action_shape_offset = self.action_shape_offset if self.action_shape_offset is not None else 0

        wp.launch(
            kernel=self.rl_action_gpu,
            dim=self.num_rl_players, 
            inputs=[
                self.articulation_body.control_mask_gpus[pattern], 
                self.articulation_body.control_rot_gpus[pattern], 
                self.articulation_body.control_omega_gpus[pattern], 
                self.articulation_body.control_torque_gpus[pattern], 
                self.physics_manager.state_0.body_q, 
                self.physics_manager.state_0.body_qd, 
                self.index_rl_players_gpu,
                self.cooldown_ability_owners,
                actions,
                action_shape_offset,
                self.speed,
                self.torque_gain,
                self.max_torque,
                
                self.articulation_body.view_object_indices_gpus[pattern],
                self.articulation_body.num_objects_env,
                ctx.count_per_world,
                self.articulation_body.view_body_local_indices_gpus[pattern],
                self.articulation_body.num_rigid_bodies_env,
                bodies_per_object,
            ],
            device=self.physics_manager.device
        )

    @wp.kernel
    def rl_action_gpu(
        control_mask: wp.array3d(dtype=wp.int32),
        control_rot: wp.array3d(dtype=wp.quat), 
        control_omega: wp.array3d(dtype=wp.vec3), 
        control_torque: wp.array3d(dtype=wp.vec3),          
        body_q: wp.array(dtype=wp.transform),         
        body_qd: wp.array(dtype=wp.spatial_vector),   
        index_rl_players_gpu: wp.array(dtype=wp.int32),
        cooldown_ability_owners: wp.array(dtype=wp.int32),
        actions: wp.array2d(dtype=wp.float32),
        action_shape_offset: int,
        speed: float, 
        torque_gain: float,     # 基礎增益
        max_torque: float,
        
        view_object_indices: wp.array(dtype=wp.int32),
        num_objects_env: int,
        count_per_world: int,
        view_body_local_indices: wp.array(dtype=wp.int32),
        num_body_object_env: int,
        bodies_per_object: int,
    ):
        tid = wp.tid()
        player_idx = index_rl_players_gpu[tid]

        # Warp AD: early return / break in kernels zeros grads for the whole launch.
        # RL pattern is (num_envs, 1, 1) with tid == world for one RL agent per env.
        active = wp.where(cooldown_ability_owners[player_idx] == 0, 1.0, 0.0)

        local_body_idx = view_body_local_indices[0]
        global_body_idx = tid * num_body_object_env + local_body_idx
        
        tf = body_q[global_body_idx]
        q = tf.q
        qd = body_qd[global_body_idx]
        
        # 世界坐標系下的角速度
        curr_world_omega = wp.vec3(qd[3], qd[4], qd[5])
        # 轉為局部角速度 (Local)
        curr_local_omega = wp.quat_rotate_inv(q, curr_world_omega)
        
        # Roll 控制（穩定 X 軸）
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        curr_roll_rad = wp.atan2(sinr_cosp, cosr_cosp)

        target_w_yaw = actions[tid, action_shape_offset] * speed
        target_w_pitch = actions[tid, action_shape_offset + 1] * speed
        target_w_roll = -curr_roll_rad * 5.0  

        # Yaw 對抗地面摩擦，使用更高剛度
        yaw_stiffness = 2.0  
        
        torque_x = (target_w_roll - curr_local_omega[0]) * torque_gain
        torque_y = (target_w_pitch - curr_local_omega[1]) * torque_gain
        torque_z = (target_w_yaw - curr_local_omega[2]) * torque_gain * yaw_stiffness

        # Soft limits (tanh): hard wp.clamp saturates and zeros APG gradients.
        limit_pitch = max_torque * 0.1
        limit_yaw = max_torque
        limit_roll = max_torque

        torque_x = limit_roll * wp.tanh(torque_x / wp.max(limit_roll, 1e-6))
        torque_y = limit_pitch * wp.tanh(torque_y / wp.max(limit_pitch, 1e-6))
        torque_z = limit_yaw * wp.tanh(torque_z / wp.max(limit_yaw, 1e-6))

        local_torque = wp.vec3(torque_x, torque_y, torque_z)

        # --- 轉回世界坐標並輸出 ---
        world_torque = wp.quat_rotate(q, local_torque)

        # Index with tid only — runtime (world, obj_idx) scatters drop value grads in Warp AD.
        # TODO 一個環境只能有一個 RL Agent
        control_torque[tid, 0, 0] += world_torque * active

    def bot_action(self, index_obj_to_env_mapping_gpu, **kwargs) -> float:
        if getattr(self, "num_bot_players", 0) <= 0 or not getattr(self, "_configured", False):
            return

        ctx = self._view_ctx("bot")
        if ctx is None:
            return
        pattern = ctx.pattern
        bodies_per_object = ctx.bodies_per_object

        wp.launch(
            kernel=self.bot_action_gpu,
            dim=self.num_bot_players, 
            inputs=[
                self.articulation_body.control_mask_gpus[pattern],        # 用來標記需要更新旋轉
                self.articulation_body.control_rot_gpus[pattern],         # 用來寫入新的旋轉
                self.physics_manager.state_0.body_q,  # 用來讀取當前所有物件的位置
                self.index_bot_players_gpu,     # Bot 的索引列表
                self.index_player_offset_env_gpu, 
                self.num_player_each_env,
                index_obj_to_env_mapping_gpu, # 每個物件對應的環境 ID
                self.cooldown_ability_owners,
                
                self.bullet_speed, 
                self.gravity,
                
                self.articulation_body.view_object_indices_gpus[pattern],
                self.articulation_body.num_objects_env,
                ctx.count_per_world,
                self.articulation_body.view_body_local_indices_gpus[pattern],
                self.articulation_body.num_rigid_bodies_env,
                bodies_per_object,
            ],
            device=self.physics_manager.device
        )

    @wp.kernel
    def bot_action_gpu(
        control_mask: wp.array3d(dtype=wp.int32),
        control_rot: wp.array3d(dtype=wp.quat),
        body_q: wp.array(dtype=wp.transform), 
        index_bot_players_gpu: wp.array(dtype=wp.int32),
        index_player_offset_env_gpu: wp.array(dtype=wp.int32),
        num_player_each_env: wp.int32,
        index_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        cooldown_ability_owners: wp.array(dtype=wp.int32),
        bullet_speed: float,
        gravity: float,

        view_object_indices: wp.array(dtype=int),
        num_objects_env: int,
        count_per_world: int,
        view_body_local_indices: wp.array(dtype=int),
        num_body_object_env: int,
        bodies_per_object: int,
    ):
        tid = wp.tid()
        player_idx = index_bot_players_gpu[tid]
        bot_env = index_obj_to_env_mapping_gpu[player_idx]

        if cooldown_ability_owners[player_idx] != 0:
            return
            
        world = player_idx // num_objects_env
        local_idx = player_idx % num_objects_env

        obj_idx = wp.int32(-1)
        for i in range(count_per_world):
            if view_object_indices[i] == local_idx:
                obj_idx = i
                break

        if obj_idx == -1:
            return

        # 定位 Root Body
        local_tid = obj_idx * bodies_per_object
        local_body_idx = view_body_local_indices[local_tid]
        global_body_idx = world * num_body_object_env + local_body_idx
        
        bot_pos = body_q[global_body_idx].p

        index_offset = index_player_offset_env_gpu[bot_env]
        num_target = num_player_each_env

        min_dist_sq = float(1.0e9)   
        found_target = int(0)
        target_pos = wp.vec3(0.0, 0.0, 0.0)

        for i in range(num_target):
            target_i = i + index_offset
            if target_i == player_idx:
                continue

            # Bot 獲取其他所有玩家/目標的座標
            other_pos = body_q[target_i].p
            diff = other_pos - bot_pos
            dist_sq = wp.length_sq(diff)

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                target_pos = other_pos
                found_target = 1  

        if found_target == 1:
            # 1. 計算相對位移
            diff = target_pos - bot_pos
            
            # 2. 調用彈道公式
            # 這會給你一個考慮了重力下墜後，應該射擊的世界座標方向
            aim_dir_world = calculate_ballistic_aim_dir(diff, bullet_speed, gravity)
            
            # 3. 因為是球體，直接讓球體「面向」這個方向
            # 我們假設 Z 軸 (0,0,1) 是世界座標的「上」
            forward = wp.normalize(aim_dir_world)
            # 計算右向量 (Z 軸 cross Forward)
            right = wp.normalize(wp.cross(wp.vec3(0.0, 0.0, 1.0), forward))
            # 重新計算正交的向上向量
            actual_up = wp.cross(forward, right)
            
            # 從矩陣構造四元數 (Warp 內建支援從旋轉矩陣分量構造，或手動構造)
            # 這裡使用最簡單的：先算 Yaw 旋轉，再算 Pitch 旋轉
            yaw = wp.atan2(forward.y, forward.x)
            pitch = -wp.asin(wp.clamp(forward.z, -0.999, 0.999))
            
            q_yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
            q_pitch = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), pitch)

            new_rot = wp.mul(q_yaw, q_pitch)

            # 4. 更新旋轉與操作 Mask
            control_rot[world, obj_idx, 0] = new_rot
            control_mask[world, obj_idx, 0] = control_mask[world, obj_idx, 0] | 2

    def setup_keymapping(self):
        pass
    
    def update_index_bot(self, index_rl_players_gpu, num_rl_players, index_bot_players_gpu, num_bot_players):
        super().update_index_bot(index_rl_players_gpu=index_rl_players_gpu, num_rl_players=num_rl_players, index_bot_players_gpu=index_bot_players_gpu, num_bot_players=num_bot_players)

    def reset(self):
        return super().reset()