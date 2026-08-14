"""RSL-rl trainer compatible with the project's launcher contract.

This trainer is selected through the preset field ``meta.trainer_module``
(e.g. ``rsl_rl_script.trainer``) and driven by ``training/launcher.py``:

  python -m training.launcher train --preset level5_0_rsl_rl_ppo_state_based

It builds the project's ``WarpEnv`` (CUDA-graph stepping), wraps it with
``RslRlVecEnvWrapper`` and runs ``rsl_rl.runners.OnPolicyRunner`` with the
mjlab-aligned PPO configuration from :mod:`rsl_rl_script.ppo_config`.
"""

from __future__ import annotations

import os
from datetime import datetime

from skrl_script.trainer_base import Trainer_base
from skrl_script.wrapperSKRL import WarpEnv
from training.runtime_env import ensure_runtime_env, get_project_root


class Trainer(Trainer_base):
    """RSL-RL PPO trainer (peer framework to SKRL; PPO is currently the only algorithm)."""

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
        dump_rollouts=False,
        dump_actions_steps=0,
        dump_obs_steps=0,
    ):
        ensure_runtime_env()

        self.device = device
        self.model_cfg = None
        self.train_cfg = None
        self.level_config_path = None
        self._resume_from = checkpoint_path
        self.dump_rollouts = bool(dump_rollouts)
        self.dump_actions_steps = int(dump_actions_steps or 0)
        self.dump_obs_steps = int(dump_obs_steps or 0)

        if loaded_config is None and preset_id is not None:
            from training.loader import TrainingPresetLoader

            loaded_config = TrainingPresetLoader.load(preset_id)

        if loaded_config is None and level is not None and sub_level is not None and obs_type is not None:
            from training.level_defaults import resolve_preset_id
            from training.loader import TrainingPresetLoader

            preset_key = resolve_preset_id("PPO", level, sub_level, obs_type, framework="RSL_RL")
            loaded_config = TrainingPresetLoader.load(preset_key)

        if checkpoint_path is not None:
            self.model_cfg, self.train_cfg, self.level_config_path, loaded_from_ckpt = self.load_config_from_checkpoint(
                checkpoint_path
            )
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
        self.preset_path = preset_path or (
            str(loaded_config.preset_path) if loaded_config and loaded_config.preset_path else None
        )

        if num_envs is None:
            num_envs = getattr(self.train_cfg, "num_envs_default", 4096)

        self.seed = self.train_cfg.seed
        self.enable_window = enable_window
        self.window_num_envs = window_num_envs if window_num_envs is not None else 1

        # Build the project environment (identical to the skrl trainer).
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

        # Build the mjlab-aligned PPO runner configuration from the preset.
        from rsl_rl_script.ppo_config import RslRlPpoRunnerCfg
        from rsl_rl_script.runner import RslRlOnPolicyRunner
        from rsl_rl_script.vec_env_wrapper import RslRlVecEnvWrapper

        self.runner_cfg = RslRlPpoRunnerCfg()
        self._apply_preset_to_runner_cfg(self.runner_cfg)

        self.wrapped_env = RslRlVecEnvWrapper(
            env=self.env,
            max_episode_length=self.train_cfg.max_episode_step,
            device=device,
            clip_actions=self.runner_cfg.clip_actions,
        )

        # Create the run directory under logs/rsl_rl/ (mjlab convention),
        # nested as <experiment>/<timestamp>/ like the reference implementation.
        from training.runtime_env import make_experiment_name

        project_root = get_project_root()
        run_root = project_root / "logs" / "rsl_rl" / make_experiment_name(
            loaded_config.meta.level,
            loaded_config.meta.sub_level,
            loaded_config.meta.algorithm,
            framework=loaded_config.meta.framework,
        )
        run_dir = run_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = str(run_dir)
        print(f"[RSL-rl] Run directory: {self.log_dir}")

        # Same dump layout as mjlab StepDataRecorder: actions.npy + obs_<group>.npy
        if self.dump_rollouts and (self.dump_actions_steps > 0 or self.dump_obs_steps > 0):
            from training.rollout_dump import StepDataRecorder

            dump_dir = os.path.join(self.log_dir, "dumps")
            self.wrapped_env = StepDataRecorder(
                env=self.wrapped_env,
                output_dir=dump_dir,
                save_action_steps=self.dump_actions_steps,
                save_obs_steps=self.dump_obs_steps,
            )
            print(
                f"[RSL-rl] StepDataRecorder enabled -> {dump_dir} "
                f"(actions={self.dump_actions_steps}, obs={self.dump_obs_steps})"
            )

        self.runner = RslRlOnPolicyRunner(
            env=self.wrapped_env,
            train_cfg=self.runner_cfg.to_on_policy_runner_cfg(),
            log_dir=self.log_dir,
            device=device,
        )

        if checkpoint_path is not None:
            print(f"[RSL-rl] Resuming from checkpoint: {checkpoint_path}")
            self.runner.load(checkpoint_path, map_location=device)

        self._init_battle_tracking()

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def _apply_preset_to_runner_cfg(self, cfg) -> None:
        """Overlay the preset PPO hyperparameters on top of the mjlab defaults.

        The preset file (``model.ppo`` / ``train.timesteps``) is the single
        source of truth for the run; fields omitted there keep the mjlab
        reference values from ``RslRlPpoRunnerCfg``.
        """
        preset = self.loaded_config.preset
        ppo = preset.model.ppo

        cfg.algorithm.learning_rate = ppo.learning_rate
        cfg.algorithm.entropy_coef = ppo.entropy_loss_scale
        cfg.algorithm.value_loss_coef = ppo.value_loss_scale
        cfg.algorithm.clip_param = ppo.ratio_clip
        cfg.algorithm.gamma = ppo.discount_factor
        cfg.algorithm.lam = ppo.lambda_
        cfg.algorithm.num_learning_epochs = ppo.learning_epochs
        cfg.algorithm.num_mini_batches = ppo.mini_batches
        cfg.algorithm.desired_kl = ppo.kl_threshold
        cfg.algorithm.max_grad_norm = ppo.grad_norm_clip
        cfg.num_steps_per_env = ppo.rollouts
        cfg.save_interval = ppo.checkpoint_interval
        # For RSL-rl, ``train.timesteps`` is interpreted as the number of
        # learning iterations (each collects num_steps_per_env steps per env).
        cfg.max_iterations = preset.train.timesteps

    # ------------------------------------------------------------------
    # Launcher entry points
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run the RSL-rl PPO training loop."""
        config_path = os.path.join(self.log_dir, "config")
        self.save_run_config(config_path, self.env.game.level.level_configs)

        try:
            self.runner.learn(
                num_learning_iterations=self.runner_cfg.max_iterations,
                init_at_random_ep_len=True,
            )
        except Exception as e:
            print(f"[RSL-rl] Training crashed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.wrapped_env.close()

    def train_custom(self) -> None:
        """Alias of :meth:`train` for launcher compatibility."""
        self.train()

    def evaluate_custom(self, test_episodes=10) -> None:
        """Evaluate a checkpoint using the deterministic policy (mjlab deploy behaviour)."""
        import torch

        policy = self.runner.get_inference_policy(self.device)
        obs = self.wrapped_env.get_observations().to(self.device)

        current_episode = 0
        try:
            while current_episode < test_episodes:
                with torch.no_grad():
                    # Deterministic (mean) action, matching mjlab deployment.
                    actions = policy(obs, stochastic_output=False)

                obs, rewards, dones, extras = self.wrapped_env.step(actions)

                # The wrapper auto-resets terminated environments inside step(),
                # so `obs` is already the post-reset observation for done envs.
                if dones.any():
                    current_episode += int(dones.sum().item())
                    print(f"[RSL-rl] Episode {current_episode}/{test_episodes} ended.")
        except Exception as e:
            print(f"[RSL-rl] Evaluation crashed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.wrapped_env.close()
