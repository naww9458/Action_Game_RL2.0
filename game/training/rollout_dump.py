"""Record the first N env steps of policy actions and observations to disk.

Mirrors ``mjlab`` ``StepDataRecorder`` so dumped artifacts are interchangeable:

* **actions** — model outputs passed into ``step()`` (policy-facing)
* **obs** — observations that produced those actions (cached from the
  previous ``get_observations()`` / ``step()`` return)

Data is written under ``<output_dir>/`` once each buffer is full (or on
``close()`` / ``flush()``):

* ``actions.npy`` — shape ``[T, num_envs, action_dim]``
* ``obs_<group>.npy`` — shape ``[T, num_envs, obs_dim]`` per obs group
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict


def _to_cpu(data: Any) -> Any:
    """Recursively move tensors to CPU (detached)."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    if isinstance(data, TensorDict):
        return {key: _to_cpu(value) for key, value in data.items()}
    if isinstance(data, dict):
        return {key: _to_cpu(value) for key, value in data.items()}
    return data


def _normalize_obs_frame(obs: Any) -> dict[str, torch.Tensor]:
    """Convert a TensorDict / dict / tensor observation into a keyed frame dict."""
    frame = _to_cpu(obs)
    if isinstance(frame, torch.Tensor):
        # Plain tensor observations (e.g. skrl) map to the policy group name
        # used by RSL-RL / mjlab dumps.
        return {"policy": frame}
    if isinstance(frame, dict):
        # Prefer TensorDict / dict groups as-is; map skrl "state" → "policy"
        # only when no "policy" key exists, so file names stay comparable.
        if "policy" not in frame and "state" in frame:
            remapped = dict(frame)
            remapped["policy"] = remapped.pop("state")
            return remapped
        return frame
    raise TypeError(f"Unsupported observation type for dump: {type(obs)!r}")


def _stack_obs(frames: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of obs-dicts along a new leading time dimension."""
    if not frames:
        return {}
    keys = frames[0].keys()
    return {key: torch.stack([frame[key] for frame in frames], dim=0) for key in keys}


def save_actions_npy(output_dir: str | Path, action_buf: list[torch.Tensor]) -> np.ndarray | None:
    """Write ``actions.npy`` with shape ``[T, num_envs, action_dim]``."""
    if not action_buf:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "actions.npy"
    stacked = torch.stack(action_buf, dim=0).numpy()
    np.save(path, stacked)
    print(
        f"[INFO] Saved {stacked.shape[0]} action steps -> {path}  "
        f"shape={tuple(stacked.shape)}"
    )
    return stacked


def save_obs_npy(output_dir: str | Path, obs_buf: list[dict[str, torch.Tensor]]) -> dict[str, tuple[int, ...]]:
    """Write ``obs_<group>.npy`` files with shape ``[T, num_envs, obs_dim]``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stacked = _stack_obs(obs_buf)
    shapes: dict[str, tuple[int, ...]] = {}
    for key, value in stacked.items():
        path = output_dir / f"obs_{key}.npy"
        arr = value.numpy()
        np.save(path, arr)
        shapes[key] = tuple(arr.shape)
        print(
            f"[INFO] Saved {arr.shape[0]} obs steps -> {path}  "
            f"shape={tuple(arr.shape)}"
        )
    if not shapes:
        print(f"[INFO] No obs steps to save under {output_dir}")
    return shapes


class StepDataRecorder:
    """VecEnv wrapper that dumps the first N actions / observations to ``.npy`` files.

    Args:
        env: Underlying RSL-RL vec env (typically ``RslRlVecEnvWrapper``).
        output_dir: Directory that will receive ``actions.npy`` / ``obs_*.npy``.
        save_action_steps: Number of ``step()`` calls whose actions to keep.
        save_obs_steps: Number of policy-facing observations to keep.
    """

    _OWN_ATTRS = frozenset(
        {
            "env",
            "output_dir",
            "save_action_steps",
            "save_obs_steps",
            "_action_buf",
            "_obs_buf",
            "_last_obs",
            "_actions_saved",
            "_obs_saved",
        }
    )

    def __init__(
        self,
        env: Any,
        output_dir: str | Path,
        save_action_steps: int,
        save_obs_steps: int,
    ) -> None:
        if save_action_steps < 0 or save_obs_steps < 0:
            raise ValueError("save_action_steps and save_obs_steps must be >= 0")

        object.__setattr__(self, "env", env)
        object.__setattr__(self, "output_dir", Path(output_dir))
        object.__setattr__(self, "save_action_steps", save_action_steps)
        object.__setattr__(self, "save_obs_steps", save_obs_steps)
        object.__setattr__(self, "_action_buf", [])
        object.__setattr__(self, "_obs_buf", [])
        object.__setattr__(self, "_last_obs", None)
        object.__setattr__(self, "_actions_saved", False)
        object.__setattr__(self, "_obs_saved", False)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Attribute passthrough ────────────────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in StepDataRecorder._OWN_ATTRS:
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    # ── VecEnv interface ─────────────────────────────────────────────────────

    def get_observations(self) -> TensorDict:
        obs = self.env.get_observations()
        self._last_obs = obs
        return obs

    def reset(self) -> TensorDict:
        """Reset and cache observations.

        Project ``RslRlVecEnvWrapper.reset`` returns only a TensorDict (not the
        mjlab ``(obs, extras)`` pair), so this wrapper preserves that contract.
        """
        result = self.env.reset()
        if isinstance(result, tuple):
            obs, extras = result
            self._last_obs = obs
            return obs, extras
        self._last_obs = result
        return result

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if (
            not self._obs_saved
            and len(self._obs_buf) < self.save_obs_steps
            and self._last_obs is not None
        ):
            self._obs_buf.append(_normalize_obs_frame(self._last_obs))

        if not self._actions_saved and len(self._action_buf) < self.save_action_steps:
            self._action_buf.append(_to_cpu(actions))

        obs, rewards, dones, extras = self.env.step(actions)
        self._last_obs = obs

        self._maybe_flush()
        return obs, rewards, dones, extras

    def close(self) -> None:
        self.flush()
        return self.env.close()

    # ── Persistence ──────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Write any remaining buffered data to disk."""
        self._ensure_output_dir()
        if not self._actions_saved and self._action_buf:
            self._save_actions()
        if not self._obs_saved and self._obs_buf:
            self._save_obs()

    def _maybe_flush(self) -> None:
        if (
            not self._actions_saved
            and self.save_action_steps > 0
            and len(self._action_buf) >= self.save_action_steps
        ):
            self._save_actions()
        if (
            not self._obs_saved
            and self.save_obs_steps > 0
            and len(self._obs_buf) >= self.save_obs_steps
        ):
            self._save_obs()

    def _ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_actions(self) -> None:
        save_actions_npy(self.output_dir, self._action_buf)
        self._actions_saved = True
        self._action_buf.clear()

    def _save_obs(self) -> None:
        save_obs_npy(self.output_dir, self._obs_buf)
        self._obs_saved = True
        self._obs_buf.clear()


class RolloutDumper:
    """Record API used by skrl trainers; writes the same ``.npy`` layout as mjlab."""

    def __init__(
        self,
        output_dir: str,
        *,
        enabled: bool = False,
        actions_steps: int = 0,
        obs_steps: int = 0,
    ) -> None:
        self.enabled = bool(enabled) and (int(actions_steps) > 0 or int(obs_steps) > 0)
        self.actions_steps = max(0, int(actions_steps))
        self.obs_steps = max(0, int(obs_steps))
        self.output_dir = str(Path(output_dir) / "dumps") if output_dir else "dumps"

        self._action_buf: list[torch.Tensor] = []
        self._obs_buf: list[dict[str, torch.Tensor]] = []
        self._actions_saved = not self.enabled or self.actions_steps <= 0
        self._obs_saved = not self.enabled or self.obs_steps <= 0
        self._finalized = False

        if self.enabled:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            print(
                f"[RolloutDump] Enabled — saving up to {self.actions_steps} action steps "
                f"and {self.obs_steps} obs steps under {self.output_dir}"
            )

    @property
    def active(self) -> bool:
        return self.enabled and not (self._actions_saved and self._obs_saved)

    def record(self, obs: Any, actions: Any) -> None:
        """Record one environment step (obs that produced ``actions``, then actions)."""
        if not self.active:
            return

        if not self._obs_saved and len(self._obs_buf) < self.obs_steps:
            self._obs_buf.append(_normalize_obs_frame(obs))
            if len(self._obs_buf) >= self.obs_steps:
                self._save_obs()

        if not self._actions_saved and len(self._action_buf) < self.actions_steps:
            action_t = actions if isinstance(actions, torch.Tensor) else torch.as_tensor(actions)
            self._action_buf.append(_to_cpu(action_t))
            if len(self._action_buf) >= self.actions_steps:
                self._save_actions()

        if self._actions_saved and self._obs_saved and not self._finalized:
            self._finalized = True
            print(f"[RolloutDump] Capture complete under {self.output_dir}")

    def finalize(self) -> None:
        """Flush any remaining buffers (e.g. training stopped early)."""
        if not self.enabled or self._finalized:
            return
        if not self._obs_saved and self._obs_buf:
            self._save_obs()
        if not self._actions_saved and self._action_buf:
            self._save_actions()
        self._finalized = True
        print(f"[RolloutDump] Finalized under {self.output_dir}")

    def _save_actions(self) -> None:
        save_actions_npy(self.output_dir, self._action_buf)
        self._actions_saved = True
        self._action_buf.clear()

    def _save_obs(self) -> None:
        save_obs_npy(self.output_dir, self._obs_buf)
        self._obs_saved = True
        self._obs_buf.clear()
