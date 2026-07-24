# 留著參考，目前的 APG_config.py 已經不使用了

# import pathlib
# import numpy as np
# import torch

# from skrl_script.apg import APG_DEFAULT_CONFIG

# from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG
# from gymnasium import spaces
# from script.levels.rewards.player_reward_diff import PlayerFaceToTargetReward1_diff

# class model_config:
#     model_obs_type = "state_based" # "game_screen", "state_based", "mixed"

#     obs_width = 0
#     obs_height = 0
#     stack_size = 1
#     state_obs_size = 18
#     observation_space = spaces.Box(low=-1, high=1, shape=(state_obs_size,), dtype=np.float32)

#     level = 4
#     sub_level = 0

#     cfg = APG_DEFAULT_CONFIG.copy()
#     cfg["learning_rate"] = 1e-4  # 初始學習率
#     cfg["experiment"]["checkpoint_interval"] = 40  # 每多少此更新保存一個檢查點


# class train_config:
    
#     # SKRL 搭配 PPO 的時候 1 timestep 原本是等於一次環境 step，
#     # 但是這裏爲了方便記錄修改爲 1 timestep 等於 1 epochs

#     max_episode_epochs = 40     # 每多少次更新重置一次環境，代表一個 episode 中有多少 epochs
#     horizon = 3                 # horizon 是指模擬多少步數後計算梯度更新權重，也代表模擬一個 horizon 等於一個 epochs
#     max_episode_step = max_episode_epochs * horizon # 總共模擬多少步數後重置環境 
#     max_episode_step_evaluate = 3000  # 訓練時限制最大步數是爲了保證環境時常統一避免數據污染，評估不受此影響

#     total_epochs = 4000 # 這個訓練進程會持續訓練多少 epochs，總訓練步數約等於 total_epochs * horizon
    
#     cfg_trainer = SEQUENTIAL_TRAINER_DEFAULT_CONFIG.copy()
#     cfg_trainer["timesteps"] = total_epochs * horizon
#     cfg_trainer["enable_namespaces"] = True

#     reward_components = []
#     reward_components_diff = [
#         PlayerFaceToTargetReward1_diff
#     ]
#     reward_parameters = {
#         # 戰鬥相關
#         "shoot_hit_reward": 5,
#         "shoot_being_hit_penalty": -2,
#         "shoot_total_approach_budget": 2,
#         "shoot_penalty": -0.5,
        
#         # 瞄準與距離相關
#         "max_reward_fov_degrees": 1,
#         "min_reward_fov_degrees": 80,
#         "max_dist": 30,
#         "face_to_target_reward": 0.002,

#         "decrease_starting_step": 100000,
#         "max_reward_fov_degrees_final": 1.0,
#         "decrease_fov_speed": 0.0,

#         # 結局獎勵
#         "episode_end_reward": 1000,
#     }

#     # 總訓練回合數

#     seed = 31415926
#     num_agents_each_env = 1
#     player_ids = ["RL_player1", "Bot_player"]