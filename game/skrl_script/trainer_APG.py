# This file contains code adapted from:
# https://github.com/Toni-SM/skrl
#
# Modified for Action_Game_RL.
#
# The original project is licensed under the MIT License.
# See the LICENSE file in the project root for details.

import torch
import time
import os
import numpy as np
import warp as wp

# debug
# torch.autograd.set_detect_anomaly(True)

from script.game_config import GameConfig
from skrl_script.algorithm.apg import APG

from skrl_script.trainer_base import Trainer_base
from skrl_script.wrapperSKRL import WarpEnv
from training.runtime_env import ensure_runtime_env, make_experiment_name


class Trainer(Trainer_base):
    def __init__(
        self,
        device,
        num_envs,
        is_training,
        level=None,
        sub_level=None,
        obs_type=None,
        enable_window=False,
        window_num_envs=None,
        checkpoint_path=None,
        loaded_config=None,
        preset_path=None,
        preset_id=None,
    ):
        ensure_runtime_env()

        self.device=device
        self.model_cfg = None
        self.train_cfg = None
        self.level_config_path = None
        self.Policy = None
        self.Value = None
        self._resume_from = checkpoint_path

        if loaded_config is None and preset_id is not None:
            from training.loader import TrainingPresetLoader
            loaded_config = TrainingPresetLoader.load(preset_id)

        if loaded_config is None and level is not None and sub_level is not None and obs_type is not None:
            from training.level_defaults import resolve_preset_id
            from training.loader import TrainingPresetLoader

            preset_key = resolve_preset_id("APG", level, sub_level, obs_type)
            loaded_config = TrainingPresetLoader.load(preset_key)

        if checkpoint_path is not None:
            self.model_cfg, self.train_cfg, self.level_config_path, loaded_from_ckpt = self.load_config_from_checkpoint(checkpoint_path)
            if loaded_config is None:
                loaded_config = loaded_from_ckpt
        elif loaded_config is not None:
            self.model_cfg = loaded_config.model_cfg
            self.train_cfg = loaded_config.train_cfg
        else:
            raise ValueError(
                "Trainer requires loaded_config, preset_id, legacy (level, sub_level, obs_type), or checkpoint_path"
            )

        self.loaded_config = loaded_config
        self.preset_path = preset_path or (str(loaded_config.preset_path) if loaded_config and loaded_config.preset_path else None)
        self.Policy = loaded_config.Policy

        if num_envs is None:
            raise ValueError("num_envs cannot be None !!!")
        self.model_cfg.cfg["num_envs"] = num_envs

        self.seed = self.train_cfg.seed

        self.horizon = self.train_cfg.horizon  # APG 通常使用較短的時域 (16-64步)，防止梯度爆炸和顯存溢出

        self.timestep = 0
        self.timesteps = self.train_cfg.total_epochs

        self.enable_window = enable_window
        self.window_num_envs = window_num_envs if window_num_envs is not None else 1
        self.env = WarpEnv(
            num_envs=num_envs,
            device=device, 
            model_cfg=self.model_cfg,
            train_cfg=self.train_cfg,
            level_config_path=self.level_config_path,
            is_training=is_training,
            step_mode="Differentiation",
            enable_window=self.enable_window,
            window_num_envs=self.window_num_envs,
        )

        if is_training and not checkpoint_path:
            meta = loaded_config.meta
            self.model_cfg.cfg.setdefault("experiment", {})
            self.model_cfg.cfg["experiment"]["experiment_name"] = make_experiment_name(
                meta.level, meta.sub_level, meta.algorithm
            )
        
        # 獲取策略模型 (Actor)
        self.policy = self.Policy(self.env.observation_space, self.env.action_space, device=self.device) # TODO Hard code
        self.agent = APG(
            models={"policy": self.policy}, 
            memory=None, 
            cfg=self.model_cfg.cfg,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device, # TODO Hard Code
            requires_grad=self.env.requires_grad,
            env=self.env,
        )
            
        if checkpoint_path is not None:
            try:
                self.agent.load(checkpoint_path)
                print(f"Successfully loaded weights: {checkpoint_path}")
            except Exception as e:
                raise ValueError(f"Failed to load, please check if the path and model structure match: {e}")

        self.requires_grad = self.env.requires_grad
        self._init_battle_tracking()


    def train_custom(self) -> None:
        self.agent.init(trainer_cfg=self.train_cfg.cfg_trainer) # 嘗試從這個函數找出訓練日志，模型參數保存的路徑，然後把 Train 和 model config 的數據也保存進去
        config_path = os.path.join(self.agent.experiment_dir, "config")
        os.makedirs(config_path, exist_ok=True)
        self.save_run_config(config_path, self.env.game.level.level_configs)

        self.agent.set_mode("train")
        obs, _ = self.env.reset() 

        try:
            print("開始解析策略梯度 (APG) 訓練...")

            for epoch in range(self.train_cfg.total_epochs): # 總訓練輪數
                
                # === 第一步：建立 Tape 記錄整個 Rollout 軌跡 ===
                tape = wp.Tape()
                with tape:
                    for t in range(self.horizon):
                        # 1. PyTorch 計算動作 (PyTorch 端會自動追蹤 Policy 權重到 actions_pt 的梯度)
                        actions_pt = self.agent.act(obs, timestep=self.timestep, timesteps=self.timesteps)[0]
                        actions_pt.retain_grad()

                        # 2. 執行物理步進 (將 actions_pt 轉給 Warp，並記錄在 global tape)
                        next_obs, rewards_pt, terminated, truncated, info = self.env.step_Diff(actions_pt)

                        if self.enable_window:
                            self.env.render()

                        self.agent.record_transition(states=obs, actions=actions_pt, rewards=rewards_pt, next_states=next_obs, terminated=terminated, truncated=truncated, infos=info, timestep=self.timestep, timesteps=self.timesteps)

                        obs = next_obs
                
                self.agent.post_interaction(timestep=self.timestep, timesteps=self.timesteps, tape=tape)
                self.timestep += 1


                if epoch % self.train_cfg.max_episode_epochs == 0:
                    # print(f"Epoch {epoch}, Loss: {loss_val:.4f}, Average Gradient: {self.agent.episode_total_grad:.8f}")
                    print(f"Epoch {epoch}, Average Gradient: {self.agent.episode_total_grad:.8f}")

                    obs, _ = self.env.reset() 
                    self.agent.reset()

            self.agent.write_tracking_data(timestep=self.timestep, timesteps=self.timesteps)

        except Exception as e:
            print(f"Training crashed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.env.close()


    def evaluate_custom(self, test_episodes=10):
        self.agent.set_mode("eval")
        obs, _ = self.env.reset()
        current_episode = 0
        is_testing = True
        
        target_fps = GameConfig.FPS_ACTION
        frame_duration = 1.0 / target_fps
        self.env.game.max_episode_step = self.train_cfg.max_episode_step_evaluate 

        try:
            # 評估模式，與訓練模式類似但不更新權重
            while is_testing:
                start_time = time.perf_counter()

                with torch.no_grad():
                    actions_pt = self.agent.act(obs, timestep=self.timestep, timesteps=self.timesteps)[0]

                # print(f"Episode {current_episode}, Action sample: {actions_pt[0].cpu().numpy()}")

                next_obs, rewards_pt, terminated, truncated, info = self.env.step_Diff(actions_pt)

                if self.enable_window:
                    self.env.render()

                    elapsed_time = time.perf_counter() - start_time

                    # If executing too fast, sleep for the remaining time
                    if elapsed_time < frame_duration:
                        time.sleep(frame_duration - elapsed_time)

                obs = next_obs

                if terminated.any() or truncated.any():
                    current_episode += terminated.sum()

                    if current_episode >= test_episodes:
                        is_testing = False

                    print("==============================================================================================================================")
                    print(f"Current episode {current_episode} ended, environment reset.")
                    self.update_winner()

                    obs, _ = self.env.reset() 
                    self.agent.reset()

        except Exception as e:
            print(f"Training crashed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.env.close()



