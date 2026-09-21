"""Vectorized environment adapter between the project's ``WarpEnv`` and ``rsl_rl``.

``rsl_rl.runners.OnPolicyRunner`` expects a ``rsl_rl.env.VecEnv`` whose
:meth:`get_observations` / :meth:`step` return ``TensorDict`` observation groups
and 1D reward / done vectors. This adapter bridges the Gymnasium-style
``rl_framework.skrl_script.wrapperSKRL.WarpEnv`` used by the project to that interface.

Observation groups
------------------
  - ``policy`` : observation consumed by the actor (the G1 proprioceptive state).
  - ``critic`` : observation consumed by the critic. When the wrapped level
    exposes an asymmetric critic observation (attribute ``critic_obs`` on
    ``WarpEnv``, i.e. policy obs + foot-state extras), it is used verbatim;
    otherwise the critic falls back to the policy observation (symmetric critic).

Both groups are cloned before they are handed to rsl_rl. The project's Warp
observation buffers are zero-copy ``wp.to_torch`` views overwritten in place
on every ``step()``; rsl_rl's ``act()`` stores the TensorDict by reference and
only ``copy_``s it *after* ``env.step()``. Without a snapshot, storage would
pair ``a_t`` with ``s_{t+1}``, which makes adaptive-KL look huge even with a
frozen policy and floors the learning rate at ``1e-5``.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv


class RslRlVecEnvWrapper(VecEnv):
    """Wrap a project ``WarpEnv`` instance into an ``rsl_rl.env.VecEnv``."""

    def __init__(
        self,
        env: Any,
        max_episode_length: int,
        device: str,
        clip_actions: float | None = None,
        cfg: dict | object | None = None,
        nan_guard: bool = False,
    ) -> None:
        """Initialize the wrapper.

        Args:
            env: The project Gymnasium-style vectorized environment (``WarpEnv``).
            max_episode_length: Maximum episode length in environment steps.
            device: Torch device string used for buffers.
            clip_actions: If not None, clip actions to ``[-clip_actions, clip_actions]``.
            cfg: Optional configuration object exposed as ``env.cfg`` (defaults to ``env`` itself).
            nan_guard: When True, non-finite obs/reward rows are zeroed, marked
                terminated, and auto-reset (``--nan-check`` guard).
        """
        self.env = env
        self.device = device
        self.cfg = env if cfg is None else cfg
        self.nan_guard = bool(nan_guard)

        self.num_envs = env.num_envs
        action_space = env.action_space
        if action_space is None:
            raise ValueError("The wrapped environment must expose a Gymnasium action space.")
        self.num_actions = int(action_space.shape[0])

        self.max_episode_length = max_episode_length
        self.clip_actions = clip_actions

        # Episode length buffer (incremented every step, reset to 0 on done).
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Observation buffer backing the TensorDict returned by get_observations().
        self.obs_policy = torch.zeros((self.num_envs, 0), device=self.device, dtype=torch.float32)

        # RSL-rl does not call reset() itself; initialize the simulation and
        # refresh the observation cache once on construction (mirrors the
        # mjlab reference wrapper behaviour).
        self.reset()

    # ------------------------------------------------------------------
    # rsl_rl VecEnv interface
    # ------------------------------------------------------------------
    def get_observations(self) -> TensorDict:
        """Return a snapshot TensorDict with ``policy`` / ``critic`` groups.

        The returned tensors must remain valid across a subsequent ``step()``:
        rsl_rl stores this TensorDict by reference until ``process_env_step``
        copies it into rollout storage *after* the environment has already
        advanced.
        """
        return TensorDict(
            {
                "policy": self.obs_policy,
                "critic": self._get_critic_obs(),
            },
            batch_size=[self.num_envs],
        )

    def reset(self) -> TensorDict:
        """Reset all environments, refresh the cached observations, and return them."""
        obs, _ = self.env.reset()
        self.episode_length_buf.zero_()
        self.obs_policy = self._extract_obs(obs)
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step the environment and return (observations, rewards, dones, extras).

        Termination semantics (from the game's episode-end detector):
          - ``terminated`` (fell over / unstable) is a *true terminal state*.
          - ``truncated`` (reached ``max_episode_step``) is a *timeout*.
        Both end the episode, so ``dones`` combines them for RSL-rl bookkeeping
        (episode stats, environment auto-reset). Timeouts are additionally
        exposed as ``extras["time_outs"]``: RSL-rl then bootstraps the value at
        those transitions, i.e. the timeout is NOT treated as a terminal state
        when computing GAE returns.

        RSL-rl's ``OnPolicyRunner`` never calls ``env.reset()`` itself — it
        expects terminated environments to be reset automatically inside
        ``step()`` (the mjlab / IsaacLab convention). The project's
        ``Game.reset()`` only re-initializes the environments flagged as
        terminated and recomputes observations for all of them, so calling it
        whenever any environment terminates leaves the surviving environments
        untouched.
        """
        # Clip actions before applying them to the environment.
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs, rewards, terminated, truncated, info = self.env.step(actions)

        # The project environment returns rewards/dones shaped (num_envs, 1)
        # and terminated/truncated as tensors. Convert to bool so the bitwise
        # OR (and rsl_rl's done handling) works correctly.
        rewards = rewards.view(-1).float()
        terminated = terminated.view(-1).bool()
        truncated = truncated.view(-1).bool()

        # Optional NaN guard (--nan-check): one world can return NaN q/qd from
        # the solver. Zero that step's reward and mark the env terminated so
        # the auto-reset below restores finite physics. Report BEFORE mutating
        # so the layout still shows the original non-finite columns / terms.
        if self.nan_guard:
            bad = self._nonfinite_env_mask(obs, rewards)
            if bad.any():
                hits = int(getattr(self, "_nan_guard_hits", 0)) + 1
                self._nan_guard_hits = hits
                if hits <= 3 or hits % 64 == 0:
                    from rl_framework.nan_report import print_nan_guard_report

                    print_nan_guard_report(
                        obs,
                        rewards,
                        env=self,
                        critic_obs=getattr(self.env, "critic_obs", None),
                        count=hits,
                        n_bad=int(bad.sum().item()),
                        action="zeroing reward and resetting",
                    )
            terminated = terminated | bad
            mark = getattr(self.env, "mark_envs_terminated", None)
            if callable(mark):
                mark(bad)
            rewards = torch.where(torch.isfinite(rewards), rewards, torch.zeros_like(rewards))

        dones = terminated | truncated

        # Update episode length buffer.
        self.episode_length_buf += 1
        reset_env_ids = dones.nonzero(as_tuple=False).flatten()
        self.episode_length_buf[reset_env_ids] = 0

        # Auto-reset terminated environments (RSL-rl contract) and refresh the
        # observations with the post-reset state.
        if dones.any():
            obs = self.env.reset()[0]

        # Cache the latest policy observation for get_observations().
        self.obs_policy = self._extract_obs(obs)

        extras = {
            "time_outs": truncated,
            "log": self.env.get_training_log(),
        }
        return self.get_observations(), rewards, dones, extras

    def get_training_log(self) -> dict:
        """Forward the unified training log to the underlying environment."""
        return self.env.get_training_log()

    def get_training_diagnostics(self) -> list:
        """Forward reward diagnostics to the underlying environment."""
        return self.env.get_training_diagnostics()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _row_nonfinite_mask(self, tensor: Any) -> torch.Tensor | None:
        if tensor is None or not torch.is_tensor(tensor):
            return None
        if not tensor.is_floating_point() and not tensor.is_complex():
            return None
        flat = tensor.reshape(self.num_envs, -1)
        return ~torch.isfinite(flat).all(dim=-1)

    def _nonfinite_env_mask(self, obs: Any, rewards: torch.Tensor) -> torch.Tensor:
        """True for env rows whose reward, policy obs, or critic obs is non-finite."""
        bad = ~torch.isfinite(rewards.reshape(self.num_envs))
        policy = obs["state"] if isinstance(obs, dict) and "state" in obs else obs
        policy_mask = self._row_nonfinite_mask(torch.as_tensor(policy, device=self.device))
        if policy_mask is not None:
            bad = bad | policy_mask
        critic_mask = self._row_nonfinite_mask(getattr(self.env, "critic_obs", None))
        if critic_mask is not None:
            bad = bad | critic_mask
        return bad

    def _snapshot_obs(self, obs: Any) -> torch.Tensor:
        """Flatten ``obs`` to ``(num_envs, obs_dim)`` and detach it from env storage.

        Warp observations are ``wp.to_torch`` views of arrays that kernels
        overwrite in place. ``torch.as_tensor`` on an existing tensor does not
        copy, so a clone is required for rsl_rl rollout storage to keep ``s_t``.
        """
        if isinstance(obs, dict):
            obs = obs["state"] if "state" in obs else obs
        tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        return tensor.detach().contiguous().clone()

    def _extract_obs(self, obs: Any) -> torch.Tensor:
        """Snapshot the policy observation returned by the environment."""
        return self._snapshot_obs(obs)

    def _get_critic_obs(self) -> torch.Tensor:
        """Return a snapshot of the critic observation group.

        Falls back to the (already snapshotted) policy observation when the
        level does not expose an asymmetric critic observation.
        """
        critic_obs = getattr(self.env, "critic_obs", None)
        if critic_obs is None:
            return self.obs_policy
        return self._snapshot_obs(critic_obs)

    def close(self) -> None:
        """Close the underlying environment."""
        self.env.close()
