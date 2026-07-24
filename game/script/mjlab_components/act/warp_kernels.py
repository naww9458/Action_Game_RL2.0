# This file contains code adapted from:
# https://github.com/mujocolab/mjlab
#
# Modified for Action_Game_RL.
#
# The original project is licensed under the Apache License 2.0.

"""GPU kernels for batched mjlab joint-position action processing (MuJoCo Warp)."""

from __future__ import annotations

import numpy as np

try:
    import warp as wp
except ImportError:  # pragma: no cover - optional dep
    wp = None  # type: ignore[assignment]

if wp is not None:

    @wp.kernel
    def process_actions_to_ctrl_kernel(
        raw_actions: wp.array2d(dtype=float),
        scale: wp.array(dtype=float),
        offset: wp.array(dtype=float),
        encoder_bias: wp.array2d(dtype=float),
        ctrl_indices: wp.array(dtype=int),
        ctrl: wp.array2d(dtype=float),
        action_dim: int,
        clip_actions: float,
        use_clip: int,
    ):
        """One thread per env: affine map + scatter into ``ctrl``."""
        env_id = wp.tid()
        for action_id in range(action_dim):
            raw = raw_actions[env_id, action_id]
            if use_clip == 1:
                raw = wp.clamp(raw, -clip_actions, clip_actions)
            target = raw * scale[action_id] + offset[action_id]
            target = target - encoder_bias[env_id, action_id]
            ctrl_id = ctrl_indices[action_id]
            ctrl[env_id, ctrl_id] = target

    @wp.kernel
    def process_actions_to_targets_kernel(
        raw_actions: wp.array2d(dtype=float),
        scale: wp.array(dtype=float),
        offset: wp.array(dtype=float),
        encoder_bias: wp.array2d(dtype=float),
        targets: wp.array2d(dtype=float),
        action_dim: int,
        clip_actions: float,
        use_clip: int,
    ):
        """One thread per env: write processed joint targets (no ctrl scatter)."""
        env_id = wp.tid()
        for action_id in range(action_dim):
            raw = raw_actions[env_id, action_id]
            if use_clip == 1:
                raw = wp.clamp(raw, -clip_actions, clip_actions)
            targets[env_id, action_id] = (
                raw * scale[action_id]
                + offset[action_id]
                - encoder_bias[env_id, action_id]
            )

    class MjlabWarpActionApplier:
        """Pre-uploaded per-joint scales/offsets + launch helpers for MuJoCo Warp simulators."""

        def __init__(
            self,
            device: str,
            *,
            scale: np.ndarray,
            default_joint_pos: np.ndarray,
            action_dim: int,
            ctrl_indices: np.ndarray | None = None,
            clip_actions: float | None = None,
        ) -> None:
            scale_np = np.asarray(scale, dtype=np.float32)
            offset_np = np.asarray(default_joint_pos, dtype=np.float32)
            if scale_np.shape[0] != action_dim or offset_np.shape[0] != action_dim:
                raise ValueError("scale and default_joint_pos length must match action_dim")

            self.device = device
            self.action_dim = action_dim
            self.clip_actions = clip_actions
            self.use_clip = 1 if clip_actions is not None else 0
            self.clip_value = float(clip_actions if clip_actions is not None else 0.0)
            self._ctrl_indices_wp = None
            if ctrl_indices is not None:
                ctrl_np = np.asarray(ctrl_indices, dtype=np.int32)
                if ctrl_np.shape[0] != action_dim:
                    raise ValueError("ctrl_indices length must match action_dim")
                self._ctrl_indices_wp = wp.array(ctrl_np, dtype=wp.int32, device=device)

            self._scale_wp = wp.array(scale_np, dtype=wp.float32, device=device)
            self._offset_wp = wp.array(offset_np, dtype=wp.float32, device=device)

        def launch_to_ctrl(
            self,
            raw_actions_wp: wp.array,
            encoder_bias_wp: wp.array,
            ctrl_wp: wp.array,
        ) -> None:
            """Launch kernel writing processed targets into ``ctrl_wp``."""
            if self._ctrl_indices_wp is None:
                raise RuntimeError("ctrl_indices were not provided at construction time")
            num_envs = raw_actions_wp.shape[0]
            wp.launch(
                kernel=process_actions_to_ctrl_kernel,
                dim=num_envs,
                inputs=[
                    raw_actions_wp,
                    self._scale_wp,
                    self._offset_wp,
                    encoder_bias_wp,
                    self._ctrl_indices_wp,
                    ctrl_wp,
                    self.action_dim,
                    self.clip_value,
                    self.use_clip,
                ],
                device=self.device,
            )

        def launch_to_targets(
            self,
            raw_actions_wp: wp.array,
            encoder_bias_wp: wp.array,
            targets_wp: wp.array,
        ) -> None:
            """Launch kernel writing processed joint targets only."""
            num_envs = raw_actions_wp.shape[0]
            wp.launch(
                kernel=process_actions_to_targets_kernel,
                dim=num_envs,
                inputs=[
                    raw_actions_wp,
                    self._scale_wp,
                    self._offset_wp,
                    encoder_bias_wp,
                    targets_wp,
                    self.action_dim,
                    self.clip_value,
                    self.use_clip,
                ],
                device=self.device,
            )

else:  # pragma: no cover - optional dep

    class MjlabWarpActionApplier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("warp is required for MjlabWarpActionApplier")
