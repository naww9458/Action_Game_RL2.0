"""Standalone RSL-rl training entry point for G1 velocity tracking.

This script mirrors ``tests/mjlab/.../train.py`` so the project can train the
G1 flat-ground walking policy with ``rsl_rl`` without depending on the external
mjlab package.

Usage examples
──────────────
# Train with the default preset (4096 envs, 30 000 iterations):
  python train.py

# Smaller smoke-test run:
  python train.py --num-envs 64 --max-iterations 100

# Resume from an existing RSL-rl checkpoint:
  python train.py --resume logs/rsl_rl/g1_velocity/<run>/model_500.pt

Output
──────
Checkpoints, TensorBoard logs and the preset config snapshot are written to:
  logs/rsl_rl/g1_velocity/<YYYY-MM-DD_HH-MM-SS>/

Monitor training with:
  tensorboard --logdir logs/rsl_rl/g1_velocity
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime

import torch

from rsl_rl_script.ppo_config import RslRlPpoRunnerCfg
from rsl_rl_script.runner import RslRlOnPolicyRunner
from rsl_rl_script.vec_env_wrapper import RslRlVecEnvWrapper
from skrl_script.wrapperSKRL import WarpEnv
from training.loader import TrainingPresetLoader
from training.runtime_env import ensure_runtime_env, get_project_root


@dataclass
class TrainArgs:
    """CLI configuration for the RSL-rl training run."""

    preset: str = "level5_0_rsl_rl_ppo_state_based"
    num_envs: int = 4096
    device: str = "cuda:0"
    max_iterations: int = 0  # 0 = use the value from the preset
    resume: str | None = None
    enable_window: bool = False
    window_envs: int = 1


def _parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(
        description="Train G1 Velocity-Flat policy with RSL-RL PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", type=str, default=TrainArgs.preset)
    parser.add_argument("--num-envs", type=int, default=TrainArgs.num_envs)
    parser.add_argument("--device", type=str, default=TrainArgs.device)
    parser.add_argument("--max-iterations", type=int, default=TrainArgs.max_iterations)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--enable-window", action="store_true")
    parser.add_argument("--window-envs", type=int, default=1)
    raw = parser.parse_args()
    return TrainArgs(
        preset=raw.preset,
        num_envs=raw.num_envs,
        device=raw.device,
        max_iterations=raw.max_iterations,
        resume=raw.resume,
        enable_window=raw.enable_window,
        window_envs=raw.window_envs,
    )


def run_train(args: TrainArgs) -> None:
    """Build the environment + PPO config and run the RSL-rl training loop."""
    ensure_runtime_env()

    # Load the training preset (reward components, terminator, PPO hyperparams).
    loaded = TrainingPresetLoader.load(args.preset)
    model_cfg = loaded.model_cfg
    train_cfg = loaded.train_cfg
    print(
        f"[INFO] Preset '{args.preset}' | level={loaded.meta.level}_{loaded.meta.sub_level} "
        f"| obs={model_cfg.state_obs_size} | num_envs={args.num_envs} | device={args.device}"
    )

    # Build the project environment.
    env = WarpEnv(
        num_envs=args.num_envs,
        device=args.device,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        level_config_path=None,
        is_training=True,
        step_mode="CUDA_Graph",
        enable_window=args.enable_window,
        window_num_envs=args.window_envs,
    )

    # Build the mjlab-aligned PPO runner configuration.
    runner_cfg = RslRlPpoRunnerCfg()
    _apply_preset_to_runner_cfg(runner_cfg, loaded.preset)
    if args.max_iterations > 0:
        runner_cfg.max_iterations = args.max_iterations

    # Wrap the environment for RSL-rl.
    wrapped_env = RslRlVecEnvWrapper(
        env=env,
        max_episode_length=train_cfg.max_episode_step,
        device=args.device,
        clip_actions=runner_cfg.clip_actions,
    )

    # Create the log directory.
    project_root = get_project_root()
    log_root = project_root / "logs" / "rsl_rl" / runner_cfg.experiment_name
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logs -> {log_dir}")

    # Save a preset config snapshot next to the run for reproducibility/resume.
    import yaml

    config_dir = log_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "preset.yaml", "w", encoding="utf-8") as f:
        yaml.dump(loaded.preset.model_dump(by_alias=True), f, default_flow_style=False, sort_keys=False)

    # Build the runner and optionally resume.
    runner = RslRlOnPolicyRunner(
        env=wrapped_env,
        train_cfg=runner_cfg.to_on_policy_runner_cfg(),
        log_dir=str(log_dir),
        device=args.device,
    )

    if args.resume:
        print(f"[INFO] Resuming from: {args.resume}")
        runner.load(args.resume, map_location=args.device)

    runner.learn(
        num_learning_iterations=runner_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    wrapped_env.close()


def _apply_preset_to_runner_cfg(cfg: RslRlPpoRunnerCfg, preset) -> None:
    """Overlay the preset PPO hyperparameters on top of the mjlab defaults."""
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
    # ``train.timesteps`` is interpreted as the number of RSL-rl iterations.
    cfg.max_iterations = preset.train.timesteps


if __name__ == "__main__":
    run_train(_parse_args())
