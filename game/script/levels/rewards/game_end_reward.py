import math
import warp as wp

from script.game_config import GameConfig
from .reward_calculator import RewardComponent


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from script.simulate.physics_manager import PhysicsManager
    from script.role.base_role import BaseRole
    from script.role.player import Player
    from script.role.platform import Platform
    from script.role.ability_generated_object import AbilityGeneratedObject

class TimeOutTerminated(RewardComponent):
    """
    1. 最簡單的回合結束判斷，時間到了就重置
    """

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device

        self.winner = wp.zeros(shape=GameConfig.NUM_OBJECTS_TOTAL, dtype=wp.int32, device=self.device)

        # # debug
        # # vec2, x存 dist, y存 reward_rate
        # self.debug_values1 = wp.zeros(10, dtype=wp.float32, device=self.device)
        # self.debug_values2 = wp.zeros(10, dtype=wp.float32, device=self.device)
        # self.debug_values3 = wp.zeros(10, dtype=wp.float32, device=self.device)

    def calculate(self, 
                  num_env, 
                  env_players_index_offset: wp.array, 
                  current_step: wp.array, 
                  max_episode_step: int, 
                  terminated: wp.array, 
                  **kwargs):

        # 啟動 Kernel
        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_env,
            inputs=[
                env_players_index_offset,
                current_step,           # (num_env,)
                max_episode_step,

                terminated,              # (num_env,)

                # # debug
                # self.debug_values1,
                # self.debug_values2,
                # self.debug_values3,
            ],
            device=self.device
        )

        # Debug 輸出 (可選)
        # reward_cpu = step_total_rewards.numpy()
        # if reward_cpu.sum() != 0.0:
        #     print("PlayerShotReward triggered:")
        # print("self.debug_values1: ", self.debug_values1.numpy())
        # print("self.debug_values2: ", self.debug_values2.numpy())
        # print("self.debug_values3: ", self.debug_values3.numpy())
        #     print(f"Step rewards: {reward_cpu}")

    @wp.kernel
    def calculate_gpu(
        env_players_index_offset: wp.array(dtype=wp.int32, ndim=1),
        current_step: wp.array(dtype=wp.int32, ndim=1),
        max_episode_step: int,

        terminated: wp.array(dtype=wp.bool, ndim=1),

        # debug
        # debug_values1: wp.array(dtype=wp.float32),
        # debug_values2: wp.array(dtype=wp.float32),
        # debug_values3: wp.array(dtype=wp.float32),
    ):
        env_idx = wp.tid()

        cur_step = current_step[env_idx]
        
        is_terminated = 0

        # 達到最大步數
        if cur_step >= max_episode_step:
            is_terminated = 1

        # 3. 如果游戲結束，計算獎勵
        if is_terminated == 1:
            terminated[env_idx] = True


        # debug_values[env_idx] = winner_bonus

    def reset(self, **kwargs):
        pass



class ShootingGameTerminated(RewardComponent):
    """
    注意：目前子彈接近獎勵沒有射綫檢測，目前沒有障礙物暫時不需要
    
    1. 如果只剩一個玩家或者，回合結束并且活著的人是贏家
    2. 如果游戲達到最大步數，血量最高的人是贏家
    3. 贏家的勝利額外獎勵計算爲 episode_end_reward * (player_health_env[winner] / default_player_health[env_idx][winner]) * ((max_episode_step - current_step_env) / max_episode_step)
    4. 沒達到最大游戲步數存活人數超過一人，沒有變化游戲繼續
    """

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device

        # 獎勵參數
        self.episode_end_reward = wp.float32(self.params["episode_end_reward"])

        self.time_decay_rate = wp.float32(self.params.get("time_decay_rate", 2.0))

        self.winner = wp.zeros(shape=GameConfig.NUM_OBJECTS_TOTAL, dtype=wp.int32, device=self.device)

        # # debug
        # # vec2, x存 dist, y存 reward_rate
        # self.debug_values1 = wp.zeros(10, dtype=wp.float32, device=self.device)
        # self.debug_values2 = wp.zeros(10, dtype=wp.float32, device=self.device)
        # self.debug_values3 = wp.zeros(10, dtype=wp.float32, device=self.device)

    def calculate(self, 
                  num_env, 
                  env_players_index_offset: wp.array, 
                  num_players_each_env: wp.int32, 
                  player_health: wp.array, 
                  default_player_health: wp.array, 
                  current_step: wp.array, 
                  max_episode_step: int, 
                  step_total_rewards: wp.array, 
                  terminated: wp.array, 
                  **kwargs):

        # 啟動 Kernel
        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_env,
            inputs=[
                env_players_index_offset,
                num_players_each_env,
                player_health,          # (num_players,)
                default_player_health,  # (num_players,)
                current_step,           # (num_env,)
                max_episode_step,
                self.episode_end_reward,

                self.time_decay_rate,

                step_total_rewards,     # (num_players,)
                terminated,              # (num_env,)
                self.winner, 

                # # debug
                # self.debug_values1,
                # self.debug_values2,
                # self.debug_values3,
            ],
            device=self.device
        )

        # Debug 輸出 (可選)
        # reward_cpu = step_total_rewards.numpy()
        # if reward_cpu.sum() != 0.0:
        #     print("PlayerShotReward triggered:")
        # print("self.debug_values1: ", self.debug_values1.numpy())
        # print("self.debug_values2: ", self.debug_values2.numpy())
        # print("self.debug_values3: ", self.debug_values3.numpy())
        #     print(f"Step rewards: {reward_cpu}")

    @wp.kernel
    def calculate_gpu(
        env_players_index_offset: wp.array(dtype=wp.int32, ndim=1),
        num_players_each_env: wp.int32,
        player_health: wp.array(dtype=wp.float32, ndim=1),
        default_player_health: wp.array(dtype=wp.float32, ndim=1),
        current_step: wp.array(dtype=wp.int32, ndim=1),
        max_episode_step: int,
        episode_end_reward: wp.float32,

        time_decay_rate: wp.float32,

        step_total_rewards: wp.array(dtype=wp.float32, ndim=1),
        terminated: wp.array(dtype=wp.bool, ndim=1),
        winner: wp.array(dtype=wp.int32, ndim=1),

        # debug
        # debug_values1: wp.array(dtype=wp.float32),
        # debug_values2: wp.array(dtype=wp.float32),
        # debug_values3: wp.array(dtype=wp.float32),
    ):
        env_idx = wp.tid()

        index_player_offset = env_players_index_offset[env_idx] 
        cur_step = current_step[env_idx]
        
        alive_count = wp.int32(0)
        last_alive_idx = wp.int32(-1)
        max_h = wp.float32(-1)
        max_h_idx = wp.int32(-1)

        # 1. 遍歷玩家統計狀態 (在 GPU 上這是非常快的微型循環)
        for p in range(num_players_each_env):
            index_player = p + index_player_offset
            h = player_health[index_player]
            if h > 0:
                alive_count += 1
                last_alive_idx = index_player 
            
            # 用於超時判斷血量最高者
            if h > max_h:
                max_h = h
                max_h_idx = index_player
            
            if h == max_h:
                max_h_idx = -1

        winner_idx = -1
        is_terminated = 0

        # 2. 判斷終止條件
        # 條件 A: 只剩一人或無人存活（決出勝負）
        if alive_count <= 1:
            is_terminated = 1
            # 如果都死了，這裡是 -1，沒人拿獎勵
            winner_idx = last_alive_idx 
        
        # 條件 B: 達到最大步數
        elif cur_step >= max_episode_step:
            is_terminated = 1
            winner_idx = max_h_idx

        # 3. 如果游戲結束，計算獎勵
        if is_terminated == 1:
            terminated[env_idx] = True

            if winner_idx != -1:
                # 獲取贏家血量數據
                h_winner = float(player_health[winner_idx])
                h_default = float(default_player_health[winner_idx])

                # 計算血量比率 (防止除零)
                health_ratio = 1.0
                if h_default > 0.0:
                    health_ratio = h_winner / h_default

                # # 計算時間獎勵係數 (越快獎勵越高)
                # 線性衰減：
                # time_ratio = float(max_episode_step - cur_step) / float(max_episode_step)

                # 指數衰減：
                # progress 從 0 (剛開始) 到 1 (達到最大步數)
                progress = float(cur_step) / float(max_episode_step)
                # 使用 exp(-k * x)，當 progress=0 時值為 1.0，隨時間增加而快速下降
                time_ratio = wp.exp(-time_decay_rate * progress)

                winner_bonus = episode_end_reward * health_ratio * time_ratio
                step_total_rewards[winner_idx] += winner_bonus
                winner[winner_idx] += 1


        # debug_values[env_idx] = winner_bonus

    def reset(self, **kwargs):
        pass



# =============================================================================
# 雙足機器人運動控制終止管理器 (摔倒與超時雙軌檢測 - 已採用 ArticulationView 改進)
# =============================================================================
class G1LocomotionTerminator(RewardComponent):
    """
    玩家數量只能爲 1 個
    """

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        
        self.winner = wp.zeros(shape=GameConfig.NUM_OBJECTS_TOTAL, dtype=wp.int32, device=self.device)

        self.articulation_body = kwargs.get("articulation_body")
        self.pattern = "player_unitree_g1"
        self.view_idx = next((i for i, p in enumerate(self.articulation_body.patterns) if p == self.pattern), -1)
        self.view = self.articulation_body.views[self.view_idx]

        # 🌟 從配置參數讀取跌倒判定閾值 (提供默認值)
        # 骨盆高度低於此值視為摔倒 (G1 預設站立高度約為 0.74m)
        cfg = self.params.get("G1LocomotionTerminator", self.params)
        self.fall_height_threshold = cfg.get("fall_height_threshold", 0.35)
        self.fall_tilt_threshold = cfg.get("fall_tilt_threshold", 0.883)
        self.survival_base = cfg.get("survival_base", 0.0)
        self.survival_factor = cfg.get("survival_factor", 1.0)
        self.max_lin_speed_sq = cfg.get("max_lin_speed_sq", 10000.0)   # ~100 m/s
        self.max_ang_speed_sq = cfg.get("max_ang_speed_sq", 10000.0)   # ~100 rad/s
        self.max_height = cfg.get("max_height", 5.0)
        self.min_height = cfg.get("min_height", -1.0)

    def calculate(self, 
                  num_env, 
                  physics_manager,
                  env_players_index_offset: wp.array, 
                  player_shape_ids_gpu: wp.array,
                  current_step: wp.array, 
                  max_episode_step: int, 
                  step_total_rewards: wp.array,
                  terminated: wp.array, 
                  **kwargs):
        
        pm = physics_manager

        root_tfs = self.view.get_root_transforms(pm.state_0)
        root_vels = self.view.get_root_velocities(pm.state_0)

        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_env,
            inputs=[
                root_tfs,
                root_vels,
                current_step,
                max_episode_step,
                self.fall_height_threshold,
                self.fall_tilt_threshold,
                self.max_lin_speed_sq,
                self.max_ang_speed_sq,
                self.max_height,
                self.min_height,
                self.survival_base,
                self.survival_factor,
                terminated,
                step_total_rewards,
                env_players_index_offset,
                player_shape_ids_gpu,
            ],
            device=self.device
        )

    @wp.kernel
    def calculate_gpu(
        root_tfs: wp.array2d(dtype=wp.transform),
        root_vels: wp.array2d(dtype=wp.spatial_vector),
        current_step: wp.array(dtype=wp.int32, ndim=1),
        max_episode_step: int,
        fall_height_threshold: float,
        fall_tilt_threshold: float,
        max_lin_speed_sq: float,
        max_ang_speed_sq: float,
        max_height: float,
        min_height: float,
        survival_base: float,
        survival_factor: float,
        terminated: wp.array(dtype=wp.bool, ndim=1),
        step_total_rewards: wp.array(dtype=wp.float32),
        env_players_index_offset: wp.array(dtype=wp.int32),
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
    ):
        env_idx = wp.tid()

        my_tf = root_tfs[env_idx, 0]
        my_pos = wp.transform_get_translation(my_tf)
        my_rot = wp.transform_get_rotation(my_tf)
        root_qd = root_vels[env_idx, 0]

        cur_step = current_step[env_idx]
        time_out = (cur_step >= max_episode_step)

        height_too_low = (my_pos[2] < fall_height_threshold)
        height_absurd = (my_pos[2] > max_height or my_pos[2] < min_height)

        inv_rot = wp.quat_inverse(my_rot)
        world_gravity = wp.vec3(0.0, 0.0, -1.0)
        projected_gravity = wp.quat_rotate(inv_rot, world_gravity)
        tilt_err_sq = projected_gravity[0] * projected_gravity[0] + projected_gravity[1] * projected_gravity[1]
        tilted_too_much = (tilt_err_sq > fall_tilt_threshold)

        lin_speed_sq = root_qd[0] * root_qd[0] + root_qd[1] * root_qd[1] + root_qd[2] * root_qd[2]
        ang_speed_sq = root_qd[3] * root_qd[3] + root_qd[4] * root_qd[4] + root_qd[5] * root_qd[5]
        speed_too_high = (lin_speed_sq > max_lin_speed_sq or ang_speed_sq > max_ang_speed_sq)

        state_non_finite = (
            (not wp.isfinite(my_pos[0]))
            or (not wp.isfinite(my_pos[1]))
            or (not wp.isfinite(my_pos[2]))
            or (not wp.isfinite(my_rot[0]))
            or (not wp.isfinite(my_rot[1]))
            or (not wp.isfinite(my_rot[2]))
            or (not wp.isfinite(my_rot[3]))
            or (not wp.isfinite(root_qd[0]))
            or (not wp.isfinite(root_qd[1]))
            or (not wp.isfinite(root_qd[2]))
            or (not wp.isfinite(root_qd[3]))
            or (not wp.isfinite(root_qd[4]))
            or (not wp.isfinite(root_qd[5]))
        )

        has_fallen = (height_too_low or tilted_too_much)
        is_unstable = (state_non_finite or height_absurd or speed_too_high)

        if time_out or has_fallen or is_unstable:
            terminated[env_idx] = True

            if survival_base > 0.0 and survival_factor != 1.0:
                player_idx = env_players_index_offset[env_idx]
                my_shape_id = player_shape_ids_gpu[player_idx]
                steps_survived = float(cur_step)
                survival_bonus = survival_base * wp.pow(survival_factor, steps_survived)
                step_total_rewards[my_shape_id] += survival_bonus

    def reset(self, **kwargs):
        pass


