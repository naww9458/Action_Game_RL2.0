"""RSL-rl training framework integration for the project's Warp-based environments.

RSL-RL currently exposes PPO

This package provides:

  - ``ppo_config``     : mjlab-aligned PPO runner configuration (``RslRlPpoRunnerCfg``).
  - ``vec_env_wrapper``: adapts the project's Gymnasium-style ``WarpEnv`` to the
                         ``rsl_rl.env.VecEnv`` interface (with an asymmetric critic).
  - ``trainer``        : launcher-compatible trainer used by ``training.launcher``.
  - ``train``          : standalone training entry point mirroring mjlab's ``train.py``.
"""

from rsl_rl_script.ppo_config import RslRlPpoRunnerCfg, build_rsl_rl_cfg
from rsl_rl_script.vec_env_wrapper import RslRlVecEnvWrapper

__all__ = [
    "RslRlPpoRunnerCfg",
    "build_rsl_rl_cfg",
    "RslRlVecEnvWrapper",
]
