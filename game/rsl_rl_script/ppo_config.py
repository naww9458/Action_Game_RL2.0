"""PPO algorithm and network configuration for RSL-rl (mjlab-aligned).

This module mirrors ``tests/mjlab/Mjlab-Velocity-Flat-Unitree-G1_simplified/
configs/rsl_rl_ppo_cfg.py`` so the training behaviour is reproducible inside the
project without depending on the external mjlab package.

Network architecture (identical for actor and critic)
----------------------------------------------------
  - MLP hidden dims ``(512, 256, 128)`` with ELU activations
  - ``EmpiricalNormalization`` (running mean/std) on the inputs
  - Actor outputs a ``GaussianDistribution`` with a single shared scalar std
    initialised to 1.0 (``std_type="scalar"``)

PPO algorithm
-------------
  value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
  entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4,
  learning_rate=1e-3 with adaptive KL schedule (desired_kl=0.01),
  gamma=0.99, lam=0.95, max_grad_norm=1.0

Runner
------
  24 steps per environment per iteration, checkpoint every 50 iterations,
  30 000 iterations by default, TensorBoard logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RslRlModelCfg:
    """Configuration for one MLP model (actor or critic)."""

    class_name: str = "MLPModel"
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    obs_normalization: bool = True
    distribution_cfg: dict[str, Any] | None = None


@dataclass
class RslRlPpoAlgorithmCfg:
    """PPO algorithm hyperparameters (mjlab ``RslRlPpoAlgorithmCfg``)."""

    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    optimizer: str = "adam"
    rnd_cfg: dict[str, Any] | None = None
    symmetry_cfg: dict[str, Any] | None = None


@dataclass
class RslRlPpoRunnerCfg:
    """Full RSL-rl PPO runner configuration (network + algorithm + runner)."""

    experiment_name: str = "g1_velocity"
    logger: str = "tensorboard"
    save_interval: int = 50
    num_steps_per_env: int = 24
    max_iterations: int = 30_000
    clip_actions: float | None = None
    check_for_nan: bool = True
    # Map rsl_rl observation sets to the TensorDict groups exposed by
    # ``RslRlVecEnvWrapper``: the actor sees the proprioceptive policy group,
    # the critic additionally sees the foot-state group (asymmetric critic).
    obs_groups: dict[str, list[str]] = field(
        default_factory=lambda: {"actor": ["policy"], "critic": ["critic"]}
    )

    actor: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        )
    )
    critic: RslRlModelCfg = field(default_factory=RslRlModelCfg)
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)

    def to_on_policy_runner_cfg(self) -> dict[str, Any]:
        """Convert to the flat dictionary expected by ``rsl_rl.runners.OnPolicyRunner``.

        ``class_name`` entries are resolved by ``rsl_rl.utils.resolve_callable``:
        ``PPO``, ``MLPModel`` and ``GaussianDistribution`` are all discoverable
        inside the installed ``rsl_rl`` package.
        """
        return {
            "experiment_name": self.experiment_name,
            "logger": self.logger,
            "save_interval": self.save_interval,
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "clip_actions": self.clip_actions,
            "check_for_nan": self.check_for_nan,
            "obs_groups": dict(self.obs_groups),
            "algorithm": {
                "class_name": self.algorithm.class_name,
                "value_loss_coef": self.algorithm.value_loss_coef,
                "use_clipped_value_loss": self.algorithm.use_clipped_value_loss,
                "clip_param": self.algorithm.clip_param,
                "entropy_coef": self.algorithm.entropy_coef,
                "num_learning_epochs": self.algorithm.num_learning_epochs,
                "num_mini_batches": self.algorithm.num_mini_batches,
                "learning_rate": self.algorithm.learning_rate,
                "schedule": self.algorithm.schedule,
                "gamma": self.algorithm.gamma,
                "lam": self.algorithm.lam,
                "desired_kl": self.algorithm.desired_kl,
                "max_grad_norm": self.algorithm.max_grad_norm,
                "optimizer": self.algorithm.optimizer,
                "rnd_cfg": self.algorithm.rnd_cfg,
                "symmetry_cfg": self.algorithm.symmetry_cfg,
            },
            "actor": {
                "class_name": self.actor.class_name,
                "hidden_dims": list(self.actor.hidden_dims),
                "activation": self.actor.activation,
                "obs_normalization": self.actor.obs_normalization,
                "distribution_cfg": dict(self.actor.distribution_cfg or {}),
            },
            "critic": {
                "class_name": self.critic.class_name,
                "hidden_dims": list(self.critic.hidden_dims),
                "activation": self.critic.activation,
                "obs_normalization": self.critic.obs_normalization,
            },
        }


def build_rsl_rl_cfg() -> RslRlPpoRunnerCfg:
    """Return the default PPO runner configuration for G1 velocity tracking."""
    return RslRlPpoRunnerCfg()
