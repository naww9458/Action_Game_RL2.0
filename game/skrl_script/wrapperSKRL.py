import torch
import random
import warp as wp
import numpy as np
import multiprocessing as mp
import gymnasium as gym

from queue import Queue
from gymnasium import spaces
from script.game import Game
from script.simulate.physics_manager import PhysicsManager
from script.custom_viewergl import CustomViewerGL
from script.game_config import GameConfig
from utils.schema_to_gym_space import schema_to_gym_space

from skrl.utils.spaces.torch import flatten_tensorized_space

def set_seed(seed: int):
    # 1. Python built-in random library
    random.seed(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # If using multiple GPUs
    
    # For absolute reproducibility (but will reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 4. NVIDIA Warp
    # Note: Warp randomness is usually controlled within kernels via wp.rand_init(seed, index)
    # The line below mainly affects some internal random behaviors of Warp
    # However, you need to actively pass the seed when calling kernels
    pass

class WarpEnv(gym.Env):
    def __init__(
        self,
        num_envs,
        device,
        model_cfg,
        train_cfg,
        level_config_path,
        is_training,
        step_mode: str,
        enable_window=False,
        window_num_envs=1,
    ):

        # render_mode="window"
        render_mode="headless"
        # render_mode="server"

        if step_mode.lower() not in ["cuda_graph", "differentiation"]:
            raise ValueError(f"step_mode must be CUDA_Graph or Differentiation, current step_mode: {step_mode}")

        self.requires_grad = False
        if step_mode.lower() == "cuda_graph":
            self.step = self.step_CUDA_Graph

        elif step_mode.lower() == "differentiation":
            self.step = self.step_Diff
            self.requires_grad = True

        self.seed = None
        if is_training:
            # To ensure experimental reproducibility, seeds are only used during training. Using seeds outside of training is equivalent to using training data for testing.
            set_seed(train_cfg.seed)
            self.seed = train_cfg.seed
            GameConfig.SEED = train_cfg.seed 

        else:
            render_mode="window" # show total reward, FPS

        
        GameConfig.reward_components = train_cfg.reward_components
        GameConfig.reward_components_diff = train_cfg.reward_components_diff
        GameConfig.reward_parameters = train_cfg.reward_parameters

        event_is_window_setup_ready = mp.Event()
        human_input_queue = Queue(maxsize=1)


        physics_manager = None
        if enable_window:
            render_mode="window" # show total reward, FPS
            viewerGL=CustomViewerGL(
                event_is_window_setup_ready=event_is_window_setup_ready,
                human_input_queue=human_input_queue,
                follow_body_index=0,
                num_envs_display=window_num_envs,
            )
            physics_manager = PhysicsManager(device=device, viewerGL=viewerGL)

        self.model_obs_type = model_cfg.model_obs_type
        player_controllers = train_cfg.player_controllers
        self.game = Game(render_mode=render_mode, 
                         model_obs_type=model_cfg.model_obs_type, 
                         obs_width=model_cfg.obs_width, 
                         obs_height=model_cfg.obs_height, 
                         device=device,
                         physics_manager=physics_manager, 
                         max_episode_step=train_cfg.max_episode_step, 
                         player_configs=None, 
                         platform_configs=None, 
                         environment_configs=None, 
                         level_config_path=level_config_path,
                         num_env=num_envs, 
                         level=model_cfg.level, 
                         sub_level=model_cfg.sub_level, 
                         capture_per_second=None, 
                         requires_grad=self.requires_grad,
                         player_controllers=player_controllers,
                        )

        self.num_agents_each_env = train_cfg.num_agents_each_env

        self.num_envs = num_envs
        self.device = device
        self.dt = self.game.physics_manager.frame_dt
        self.max_speed = 5.0

        self.truncated_tensor = torch.zeros((num_envs, 1), device=GameConfig.DEVICE, dtype=torch.float32)

        # 定義 Gymnasium 空間
        self.observation_space = model_cfg.observation_space
        self.action_space = schema_to_gym_space(GameConfig.ACTION_SPACE_CONFIG)[0] # TODO
        print("self.action_space: ", self.action_space)

        self.state_space = self.observation_space

    def reset(self, seed=None, options=None):
        super().reset(seed=self.seed)

        # get seed via GameConfig
        self.game.reset()
        
        if self.model_obs_type == "game_screen":
            self.obs = self.game._get_observation_game_screen()
            
        elif self.model_obs_type == "state_based":
            self.obs = self.game._get_observation_state_based()

        elif self.model_obs_type == "mixed":
            self.obs = {
                "visual": self.game._get_observation_game_screen(),
                "state": self.game._get_observation_state_based()
            }

        # Gymnasium 要求返回 (observation, info)
        if isinstance(self.obs, dict):
            return flatten_tensorized_space(self.obs), {}
        
        return self.obs, {}

    def render(self):
        self.game.render()

    def step_CUDA_Graph(self, actions: torch.Tensor):
        # --- 斷路器：檢查模型輸出 ---
        # if not torch.isfinite(actions).all():
        #     print("Detected NaN in Model Actions:")
        #     print(actions)
        #     raise RuntimeError("Model produced NaN actions. Stopping to prevent weight corruption.")

        obs, step_total_rewards, terminated = self.game.step_CUDA_Graph(actions=actions)

        rewards = wp.to_torch(step_total_rewards).view(-1, 1)
        terminated_tensor = wp.to_torch(terminated).view(-1, 1)

        # print("step_total_rewards.numpy(): ", step_total_rewards.numpy())

        # if not torch.isfinite(obs).all():
        #     print("OBS NAN detected in env.step()")
        #     print("obs:", obs)
        #     raise RuntimeError("Observation contains NaN or Inf")
        
        # if not torch.isfinite(rewards).all():
        #     print("REWARD NAN")
        #     raise RuntimeError()

        # if torch.abs(rewards).max() > 1000:
        #     print("REWARD EXPLODING:", torch.abs(rewards).max())

        # # 加入 1% 的浮點數容差 (1.01)，並修正無效的小於零檢查
        # if obs.abs().max() > 1.01: 
        #     env_ids, feat_ids = torch.where(obs.abs() > 1.01)
        #     print(f"!!! 檢測到非法觀察值 (超出歸一化範圍 [-1, 1]) !!!")
        #     print(f"環境 ID: {env_ids[:3]}")
        #     print(f"特徵 ID (Index): {feat_ids[:3]}") 
        #     print(f"實際數值: {obs[env_ids[0], feat_ids[0]]}")

        # if not torch.isfinite(rewards).all():
        #     # 找到哪個環境的獎勵炸了
        #     bad_env_idx = torch.where(~torch.isfinite(rewards))[0]
        #     print(f"!!! [Reward NaN] 檢測到非法獎勵值 !!!")
        #     print(f"環境 ID: {bad_env_idx}")
        #     # 強制截斷以防止崩潰，但必須找出 Warp 裡的原因
        #     rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

        if isinstance(obs, dict):
            obs = flatten_tensorized_space(obs)

        return obs, rewards, terminated_tensor, self.truncated_tensor, {}

    def step_Diff(self, actions: torch.Tensor):
        # 接收 game 回傳的 actions_wp (不包含 tape)
        obs, step_total_rewards, step_total_rewards_diff, terminated, actions_wp = self.game.step_Diff(actions=actions)

        rewards = wp.to_torch(step_total_rewards).view(-1, 1)
        terminated_tensor = wp.to_torch(terminated).view(-1, 1)
        
        if isinstance(obs, dict):
            obs = flatten_tensorized_space(obs)

        # 透過 info 將 Warp 層級的陣列傳給外部 Trainer
        info = {
            "actions_wp": actions_wp,
            "rewards_wp": step_total_rewards_diff
        }

        # 最後一個參數從 tape 改為 info
        return obs, rewards, terminated_tensor, self.truncated_tensor, info

    def _get_observations(self):
        # obs will return by game step function

        pass

    def close(self):
        self.game.physics_manager.cleanup() 
        self.game.close()




