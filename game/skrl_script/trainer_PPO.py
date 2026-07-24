# This file contains code adapted from:
# https://github.com/Toni-SM/skrl
#
# Modified for Action_Game_RL.
#
# The original project is licensed under the MIT License.
# See the LICENSE file in the project root for details.

import torch
import sys
import tqdm
import time
import os
import numpy as np

from script.game_config import GameConfig

from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

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

        self.device = device
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

            preset_key = resolve_preset_id("PPO", level, sub_level, obs_type)
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
        self.Value = loaded_config.Value

        if num_envs is None:
            num_envs = getattr(self.train_cfg, "num_envs_default", 4096)

        self.seed = self.train_cfg.seed
        self.enable_window = enable_window
        self.window_num_envs = window_num_envs if window_num_envs is not None else 1
        self.env = WarpEnv(
            num_envs=num_envs,
            device=device, 
            model_cfg=self.model_cfg,
            train_cfg=self.train_cfg,
            level_config_path=self.level_config_path,
            is_training=is_training,
            step_mode="CUDA_Graph",
            enable_window=self.enable_window,
            window_num_envs=self.window_num_envs,
        )

        cfg = self.model_cfg.cfg
        if is_training and not checkpoint_path:
            meta = loaded_config.meta
            cfg["experiment"]["experiment_name"] = make_experiment_name(
                meta.level, meta.sub_level, meta.algorithm
            )

        memory = RandomMemory(
            memory_size=cfg["memory_size"],
            num_envs=self.env.num_envs,
            device=self.device,
        )

        models = {
            "policy": self.Policy(self.env.observation_space, self.env.action_space, device=self.device),
            "value": self.Value(self.env.observation_space, self.env.action_space, device=self.device),
        }

        self.agent = PPO(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device,
        )

        self._eval_deterministic_policy = False
        if checkpoint_path is not None:
            try:
                if self._load_mjlab_actor(checkpoint_path, models["policy"]):
                    # mjlab 的 actor 自帶 EmpiricalNormalization，觀測正規化在策略內部完成。
                    # 因此要 (1) 繞過 skrl 的 state 預處理器 (RunningStandardScaler 會把觀測
                    # 夾在 ±5 並再正規化一次)，(2) 評估時用確定性 (mean) 動作，才能完全重現
                    # mjlab 的部署行為。
                    self.agent._state_preprocessor = self.agent._empty_preprocessor
                    self._eval_deterministic_policy = True
                    normalizer = getattr(models["policy"], "obs_normalizer", None)
                    if normalizer is not None and hasattr(normalizer, "_mean"):
                        print(
                            "[mjlab] obs_normalizer mean/std sample: "
                            f"{normalizer._mean.flatten()[:4].tolist()} "
                            f"{normalizer._std.flatten()[:4].tolist()}"
                        )
                    print(f"Loaded mjlab checkpoint (actor + obs normalizer): {checkpoint_path}")
                else:
                    self.agent.load(checkpoint_path)
                    print(f"Successfully loaded weights: {checkpoint_path}")
            except Exception as e:
                raise ValueError(f"Failed to load, please check if the path and model structure match: {e}")

        self._init_battle_tracking()

    def _load_mjlab_actor(self, checkpoint_path, policy_model) -> bool:
        """把 mjlab rsl-rl checkpoint 的 actor (MLP + EmpiricalNormalization) 載入 skrl 策略。

        mjlab 存的是 ``{"actor_state_dict": ..., "critic_state_dict": ...}`` (或舊版
        ``model_state_dict``)。skrl 的 ``agent.load`` 無法對應這些 key，會直接跳過並
        留下隨機初始化的網路 (這正是「載入後仍秒倒」的根因)。偵測到此格式時改用
        ``load_mjlab_checkpoint`` 正確搬入權重與觀測正規化統計。

        回傳 True 表示已當作 mjlab checkpoint 處理；否則回傳 False 讓呼叫端退回
        skrl 原生的 ``agent.load``。
        """
        try:
            probe = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception:
            return False
        if not isinstance(probe, dict) or (
            "actor_state_dict" not in probe and "model_state_dict" not in probe
        ):
            return False

        from skrl_script.policy_PPO_g1_velocity import load_mjlab_checkpoint

        # 推論只需要 actor。mjlab 的 critic 觀測維度 (含足部狀態) 與本專案不同，
        # 故不載入 value 網路。strict=False 容忍分佈 std 的 key 名稱差異
        # (scalar 的 std_param vs log 的 log_std_param)，對確定性評估無影響。
        load_mjlab_checkpoint(
            checkpoint_path,
            policy=policy_model,
            value=None,
            map_location=self.device,
            strict=False,
        )
        return True

    def train(self):
        cfg = self.train_cfg.cfg_trainer
        cfg["headless"] = not self.enable_window
        trainer = SequentialTrainer(env=self.env, agents=self.agent, cfg=cfg)

        config_path = os.path.join(self.agent.experiment_dir, "config")
        self.save_run_config(config_path, self.env.game.level.level_configs)

        try:
            print("Starting training with optimized config...")
            trainer.train()
        except Exception as e:
            print(f"Training crashed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.env.close()

    def train_custom(self) -> None:
        cfg = self.train_cfg.cfg_trainer
        cfg["headless"] = not self.enable_window
        trainer = SequentialTrainer(env=self.env, agents=self.agent, cfg=cfg)

        config_path = os.path.join(self.agent.experiment_dir, "config")
        self.save_run_config(config_path, self.env.game.level.level_configs)

        if trainer.num_simultaneous_agents > 1:
            for agent in trainer.agents:
                agent.set_running_mode("train")
        else:
            trainer.agents.set_running_mode("train")

        states, infos = trainer.env.reset()

        for timestep in tqdm.tqdm(
            range(trainer.initial_timestep, trainer.timesteps),
            disable=trainer.disable_progressbar,
            file=sys.stdout,
        ):
            trainer.agents.pre_interaction(timestep=timestep, timesteps=trainer.timesteps)

            with torch.no_grad():
                actions = trainer.agents.act(states, timestep=timestep, timesteps=trainer.timesteps)[0]
                next_states, rewards, terminated, truncated, infos = trainer.env.step(actions)

                if not trainer.headless:
                    trainer.env.render()

                trainer.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=trainer.timesteps,
                )

                if trainer.environment_info in infos:
                    for k, v in infos[trainer.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            trainer.agents.track_data(f"Info / {k}", v.item())

            trainer.agents.post_interaction(timestep=timestep, timesteps=trainer.timesteps)

            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = trainer.env.reset()
            else:
                states = next_states

    def evaluate_custom(self, test_episodes=10):
        try:
            self.agent.set_mode("eval")
            obs, _ = self.env.reset()
            current_episode = 0
            timestep = self.agent._random_timesteps + 1

            target_fps = GameConfig.FPS_ACTION
            frame_duration = 1.0 / target_fps

            is_testing = True
            while is_testing:
                start_time = time.perf_counter()

                with torch.no_grad():
                    if getattr(self, "_eval_deterministic_policy", False):
                        # mjlab 部署用確定性 mean 動作；predict_actions 內部會套用
                        # 策略自帶的 EmpiricalNormalization，觀測為原始物理量。
                        actions_pt = self.agent.models["policy"].predict_actions(
                            obs, deterministic=True
                        )
                    else:
                        actions_pt = self.agent.act(obs, timestep=timestep, timesteps=None)[0]

                next_obs, rewards_pt, terminated, truncated, info = self.env.step_CUDA_Graph(actions_pt)

                if self.enable_window:
                    self.env.render()
                    elapsed_time = time.perf_counter() - start_time
                    if elapsed_time < frame_duration:
                        time.sleep(frame_duration - elapsed_time)

                obs = next_obs

                if terminated.any() or truncated.any():
                    current_episode += terminated.sum()

                    if current_episode >= test_episodes:
                        is_testing = False

                    print(f"Current episode {current_episode} ended, environment reset.")
                    self.update_winner()
                    obs, _ = self.env.reset()

        except Exception as e:
            print(f"Evaluation crashed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.env.close()
