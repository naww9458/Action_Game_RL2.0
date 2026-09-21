"""Torch/Warp boxing rewards for Unitree G1. Constants come from boxing YAML.

  r_total  = r_walk + r_boxing
  r_boxing = r_distance + r_ready_pose + r_punch + r_retract + r_hit + r_pose

Formulas run through ``wp_kernel.boxing_reward_ops`` so this file has no
``@wp.kernel`` and does not import Newton / mujoco_warp.

Hard-state gating (punch_phase FSM): period rewards gate on ``hard_state``
(0 = retract, 1 = punch) and blend with the continuous ``signal`` (0..1):

  r_ready_pose : retract period (hard_state==0), scaled by (1 - signal)
  r_retract    : retract period (hard_state==0), scaled by (1 - signal)
  r_punch      : punch period (hard_state==1), scaled by signal
  r_hit        : punch period (hard_state==1), scaled by signal
  r_distance   : always on
  r_pose       : always on (torque / tilt regularizer)

All boxing terms short-circuit to 0 when an env has no target bound
(pure-walk envs). Missing ``punch_phase`` degrades to all-zero state (TryGet).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import warp as wp

from script.environments.rewards.reward_calculator import RewardComponent
from script.environments.rewards.wp_kernel.boxing_reward_ops import (
    compute_distance_reward,
    compute_hit_reward,
    compute_pose_reward,
    compute_punch_reward,
    compute_ready_pose_reward,
    compute_retract_reward,
    emit_env_reward,
    fill_hit_metrics,
    reset_punch_state_buffers,
)
from script.game_config import GameConfig
from script.role.objects.object_template.mjlab_unitree_g1.models.boxing_v1.contact_query import (
    ContactQuery,
)
from script.role.objects.object_template.mjlab_unitree_g1.models.boxing_v1.state_query import (
    ArticulationState,
    resolve_newton_body_ids,
)


def _component_params(params: dict, class_name: str) -> dict:
    nested = params.get(class_name)
    if isinstance(nested, dict):
        return nested
    return params


def _init_mjlab_scaling(reward: RewardComponent, class_name: str) -> None:
    cfg = _component_params(reward.params, class_name)
    if "weight" not in cfg:
        raise KeyError(
            f"Missing reward_parameters.{class_name}.weight. "
            "Reward term weights must be declared in the training preset YAML."
        )
    weight = float(cfg["weight"])
    if "step_dt" in reward.params:
        step_dt = float(reward.params["step_dt"])
    elif "step_dt" in cfg:
        step_dt = float(cfg["step_dt"])
    else:
        step_dt = 1.0 / float(GameConfig.FPS_ACTION)
    reward.weight = weight
    reward.step_dt = step_dt
    reward.reward_scale = weight * step_dt


def _try_boxing_cfg(environment) -> Optional[dict]:
    obs_actor = getattr(environment, "g1_obs_actor", None)
    getter = getattr(obs_actor, "get_boxing_obs_cfg", None)
    if not callable(getter):
        return None
    cfg = getter()
    return dict(cfg) if isinstance(cfg, dict) else None


def _get_term_buf(kwargs: dict, log_name: str | None):
    if not log_name:
        return None
    return kwargs.get("reward_term_bufs", {}).get(log_name)


def _resolve_ready_pose_tensors(art, cfg: dict) -> Optional[tuple]:
    """Map ``ready_pose.joints`` (dof name -> pose/weight) to torch tensors.

    Dof names resolve with the established suffix-match convention; every
    configured joint must match exactly one entry of ``joint_dof_names``.
    Returns ``(indices_i32, poses_f32, weights_f32)`` on CPU, or None.
    """
    joints = (cfg.get("ready_pose") or {}).get("joints") or {}
    if not joints:
        return None
    names = list(getattr(art.view, "joint_dof_names", ()) or ())
    if not names:
        return None
    indices: list[int] = []
    poses: list[float] = []
    weights: list[float] = []
    for joint_name, entry in joints.items():
        matches = [i for i, n in enumerate(names) if str(n).endswith(str(joint_name))]
        if len(matches) != 1:
            raise RuntimeError(
                f"boxing ready_pose joint {joint_name!r} matched {len(matches)} dof names "
                f"(need exactly 1): {[names[i] for i in matches] or names}"
            )
        indices.append(matches[0])
        poses.append(float(entry["pose"]))
        weights.append(float(entry["weight"]))
    return (
        torch.tensor(indices, dtype=torch.int32),
        torch.tensor(poses, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


class _BoxingRewardBase(RewardComponent):
    log_name: str | None = None

    def __init__(self, device, pattern: str, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.pattern = pattern
        self.num_env = 0
        self.num_players = 0
        self.cfg = None
        self.art = None
        self.obs_actor = None
        self.environment = None
        self.punch_phase = None
        self._raw = None
        self._hard_state_zero = None
        self._signal_zero = None
        self._ready = False

    def bind_environment(self, environment) -> None:
        self.environment = environment
        self.punch_phase = getattr(environment, "punch_phase", None)
        self.num_env = int(getattr(environment, "num_env", 0) or 0)
        self.num_players = int(getattr(environment.players, "num_total_object_role", 0) or 0)
        self.cfg = _try_boxing_cfg(environment)
        self.art = ArticulationState.try_from_environment(environment, self.pattern)
        self.obs_actor = getattr(environment, "g1_obs_actor", None)
        if self.num_env > 0:
            self._raw = wp.zeros(self.num_env, dtype=float, device=self.device)
            self._hard_state_zero = wp.zeros(self.num_env, dtype=wp.int32, device=self.device)
            self._signal_zero = wp.zeros(self.num_env, dtype=float, device=self.device)
        self._ready = self.cfg is not None and self.art is not None and self._raw is not None

    def _hard_state_wp(self):
        """Per-env hard state (0=retract, 1=punch); zeros when punch_phase absent."""
        pp = self.punch_phase
        if pp is not None:
            hs = getattr(pp, "hard_state_wp", None)
            if hs is not None:
                return hs
        return self._hard_state_zero

    def _signal_wp(self):
        """Per-env action signal (0..1); zeros when punch_phase absent."""
        pp = self.punch_phase
        if pp is not None:
            s = getattr(pp, "signal_wp", None)
            if s is not None:
                return s
        return self._signal_zero

    def _feasibility_mode(self) -> int:
        return 1 if str(self.cfg.get("punch_ready_mode", "linear")).lower() == "gaussian" else 0

    def _bind_ready_pose(self) -> None:
        """Resolve ready-pose joints to wp arrays shared by gated kernels."""
        self._ready_idx_torch = None
        self._ready_pose_torch = None
        self._ready_weight_torch = None
        self._ready_indices_wp = None
        self._ready_poses_wp = None
        self._ready_weights_wp = None
        if not self._ready:
            return
        resolved = _resolve_ready_pose_tensors(self.art, self.cfg)
        if resolved is None:
            return
        dev = torch.device(str(self.device))
        self._ready_idx_torch = resolved[0].to(device=dev).contiguous()
        self._ready_pose_torch = resolved[1].to(device=dev).contiguous()
        self._ready_weight_torch = resolved[2].to(device=dev).contiguous()
        self._ready_indices_wp = wp.from_torch(self._ready_idx_torch, dtype=wp.int32)
        self._ready_poses_wp = wp.from_torch(self._ready_pose_torch, dtype=wp.float32)
        self._ready_weights_wp = wp.from_torch(self._ready_weight_torch, dtype=wp.float32)

    def _alloc_joint_q(self) -> None:
        self._joint_q = None
        if not self._ready:
            return
        dof = int(getattr(self.art.view, "joint_dof_count", 0) or 0)
        n = int(getattr(self.art, "num_instances", self.num_env) or self.num_env)
        if n > 0 and dof > 0:
            self._joint_q = wp.zeros((n, dof), dtype=float, device=self.device)

    def _sync_selected_target(self, physics_manager) -> None:
        sync = getattr(self.obs_actor, "try_sync_selected_target_pose", None)
        if callable(sync) and physics_manager is not None:
            sync(physics_manager)

    def _has_target_wp(self):
        return getattr(self.obs_actor, "has_target_wp", None)

    def _target_pose_wp(self):
        pos = getattr(self.obs_actor, "target_pos_wp", None)
        vel = getattr(self.obs_actor, "target_vel_wp", None)
        if pos is None or vel is None:
            return None, None
        return pos, vel

    def _emit(self, step_total_rewards, kwargs) -> None:
        emit_env_reward(
            self._raw,
            scale=self.reward_scale,
            step_total_rewards=step_total_rewards,
            term_buf=_get_term_buf(kwargs, self.log_name),
            player_shape_ids=kwargs["player_shape_ids_gpu"],
            env_map=kwargs["index_player_obj_to_env_mapping_gpu"],
            num_players=self.num_players,
            num_env=self.num_env,
            device=self.device,
        )

    def reset(self, **kwargs):
        pass


class BoxingDistanceReward(_BoxingRewardBase):
    """``feasibility * base_weight`` in-band, ``-abs(dist-standoff) * penalty_weight`` outside."""

    log_name = "boxing_distance"

    def __init__(self, device, pattern: str, **kwargs):
        super().__init__(device, pattern, **kwargs)
        _init_mjlab_scaling(self, "BoxingDistanceReward")
        self._pelvis_pos = None
        self._pelvis_vel = None

    def bind_environment(self, environment) -> None:
        super().bind_environment(environment)
        if self.num_env > 0:
            self._pelvis_pos = wp.zeros((self.num_env, 3), dtype=float, device=self.device)
            self._pelvis_vel = wp.zeros((self.num_env, 3), dtype=float, device=self.device)

    def calculate(self, physics_manager, step_total_rewards: wp.array, **kwargs):
        if not self._ready:
            return
        self._sync_selected_target(physics_manager)
        target_pos, _ = self._target_pose_wp()
        has_target = self._has_target_wp()
        if target_pos is None or has_target is None:
            return
        self.art.copy_link_position_velocity(
            self.cfg["pelvis_body"], self._pelvis_pos, self._pelvis_vel
        )
        dist_cfg = self.cfg["distance"]
        compute_distance_reward(
            pelvis_pos=self._pelvis_pos,
            target_pos=target_pos,
            has_target=has_target,
            standoff=self.cfg["standoff_distance"],
            band=self.cfg["punch_ready_band"],
            punch_ready_mode=self._feasibility_mode(),
            base_weight=dist_cfg["base_weight"],
            penalty_weight=dist_cfg["penalty_weight"],
            raw=self._raw,
            num_env=self.num_env,
            device=self.device,
        )
        self._emit(step_total_rewards, kwargs)


class BoxingReadyPoseReward(_BoxingRewardBase):
    """Guard-pose convergence; retract period, blended by (1 - signal)."""

    log_name = "boxing_ready_pose"

    def __init__(self, device, pattern: str, **kwargs):
        super().__init__(device, pattern, **kwargs)
        _init_mjlab_scaling(self, "BoxingReadyPoseReward")

    def bind_environment(self, environment) -> None:
        super().bind_environment(environment)
        self._bind_ready_pose()
        self._alloc_joint_q()
        if self._ready_indices_wp is None or self._joint_q is None:
            self._ready = False

    def calculate(self, physics_manager, step_total_rewards: wp.array, **kwargs):
        if not self._ready:
            return
        self._sync_selected_target(physics_manager)
        has_target = self._has_target_wp()
        if has_target is None:
            return
        self.art.copy_joint_positions(self._joint_q)
        ready_cfg = self.cfg["ready_pose"]
        compute_ready_pose_reward(
            has_target=has_target,
            joint_q=self._joint_q,
            joint_indices=self._ready_indices_wp,
            joint_poses=self._ready_poses_wp,
            joint_weights=self._ready_weights_wp,
            hard_state=self._hard_state_wp(),
            signal=self._signal_wp(),
            ready_threshold=ready_cfg["ready_threshold"],
            error_scale=ready_cfg["error_scale"],
            weight=ready_cfg["weight"],
            ready_bonus=ready_cfg["ready_bonus"],
            raw=self._raw,
            num_env=self.num_env,
            device=self.device,
        )
        self._emit(step_total_rewards, kwargs)


class BoxingPunchReward(_BoxingRewardBase):
    """C1 planar hand speed + C2 hand acceleration toward target (full-swing boosted).

    Unlocks at standoff from the ready pose, and stays open while swinging so
    leaving the guard angles mid-extension does not cancel the punch signal.
    """

    log_name = "boxing_punch"

    def __init__(self, device, pattern: str, **kwargs):
        super().__init__(device, pattern, **kwargs)
        _init_mjlab_scaling(self, "BoxingPunchReward")
        self._pelvis_pos = None
        self._pelvis_vel = None
        self._pelvis_yaw = None
        self._hand_pos = None
        self._hand_vel = None
        self._hand_vel_prev = None
        self._hand_fwd_hist = None
        self._hist_len = 0
        self._hist_pos = 0

    def bind_environment(self, environment) -> None:
        super().bind_environment(environment)
        if self.num_env > 0:
            self._pelvis_pos = wp.zeros((self.num_env, 3), dtype=float, device=self.device)
            self._pelvis_vel = wp.zeros((self.num_env, 3), dtype=float, device=self.device)
            self._pelvis_yaw = wp.zeros(self.num_env, dtype=float, device=self.device)
            self._hand_pos = wp.zeros((self.num_env, 3), dtype=float, device=self.device)
            self._hand_vel = wp.zeros((self.num_env, 3), dtype=float, device=self.device)
            self._hand_vel_prev = wp.zeros((self.num_env, 3), dtype=float, device=self.device)
        self._bind_ready_pose()
        self._alloc_joint_q()
        punch_cfg = (self.cfg or {}).get("punch") or {}
        window_s = float(punch_cfg.get("full_swing_window_s", 0.0) or 0.0)
        self._hist_len = max(2, int(round(window_s / max(self.step_dt, 1.0e-6))))
        if self.num_env > 0:
            self._hand_fwd_hist = wp.zeros(
                (self.num_env, self._hist_len), dtype=float, device=self.device
            )
        if (
            self._ready_indices_wp is None
            or self._joint_q is None
            or self._hand_fwd_hist is None
        ):
            self._ready = False

    def calculate(self, physics_manager, step_total_rewards: wp.array, **kwargs):
        if not self._ready:
            return
        self._sync_selected_target(physics_manager)
        target_pos, _ = self._target_pose_wp()
        has_target = self._has_target_wp()
        if target_pos is None or has_target is None:
            return
        self.art.copy_link_position_velocity(
            self.cfg["pelvis_body"], self._pelvis_pos, self._pelvis_vel
        )
        self.art.copy_link_yaw(self.cfg["pelvis_body"], self._pelvis_yaw)
        self.art.copy_link_position_velocity(
            self.cfg["hand_body"], self._hand_pos, self._hand_vel
        )
        self.art.copy_joint_positions(self._joint_q)
        punch_cfg = self.cfg["punch"]
        ready_cfg = self.cfg["ready_pose"]
        compute_punch_reward(
            pelvis_pos=self._pelvis_pos,
            pelvis_vel=self._pelvis_vel,
            pelvis_yaw=self._pelvis_yaw,
            target_pos=target_pos,
            hand_pos=self._hand_pos,
            hand_vel=self._hand_vel,
            hand_vel_prev=self._hand_vel_prev,
            hand_fwd_hist=self._hand_fwd_hist,
            hist_pos=self._hist_pos,
            has_target=has_target,
            hard_state=self._hard_state_wp(),
            signal=self._signal_wp(),
            joint_q=self._joint_q,
            joint_indices=self._ready_indices_wp,
            joint_poses=self._ready_poses_wp,
            joint_weights=self._ready_weights_wp,
            dt=self.step_dt,
            standoff=self.cfg["standoff_distance"],
            band=self.cfg["punch_ready_band"],
            punch_ready_mode=self._feasibility_mode(),
            gate_feasibility=punch_cfg["gate_feasibility"],
            ready_threshold=ready_cfg["ready_threshold"],
            planar_min_speed=punch_cfg["planar_min_speed"],
            planar_dominance_ratio=punch_cfg["planar_dominance_ratio"],
            xy_weight=punch_cfg["xy_weight"],
            acc_weight=punch_cfg["acc_weight"],
            full_swing_min_advance=punch_cfg["full_swing_min_advance"],
            full_swing_acc_scale=punch_cfg["full_swing_acc_scale"],
            unprepared_scale=punch_cfg["unprepared_scale"],
            raw=self._raw,
            num_env=self.num_env,
            device=self.device,
        )
        self._hist_pos = (self._hist_pos + 1) % self._hist_len
        self._emit(step_total_rewards, kwargs)

    def reset(self, terminated, **kwargs):
        if self._hand_vel_prev is None or self._hand_fwd_hist is None or self.num_env <= 0:
            return
        reset_punch_state_buffers(
            terminated,
            self._hand_vel_prev,
            self._hand_fwd_hist,
            num_env=self.num_env,
            device=self.device,
        )


class BoxingRetractReward(_BoxingRewardBase):
    """Slow-away Gaussian + over-speed penalty + fake-punch penalty; retract period."""

    log_name = "boxing_retract"

    def __init__(self, device, pattern: str, **kwargs):
        super().__init__(device, pattern, **kwargs)
        _init_mjlab_scaling(self, "BoxingRetractReward")
        self._pelvis_pos = None
        self._pelvis_vel = None
        self._hand_pos = None
        self._hand_vel = None

    def bind_environment(self, environment) -> None:
        super().bind_environment(environment)
        if self.num_env > 0:
            self._pelvis_pos = wp.zeros((self.num_env, 3), dtype=float, device=self.device)
            self._pelvis_vel = wp.zeros((self.num_env, 3), dtype=float, device=self.device)
            self._hand_pos = wp.zeros((self.num_env, 3), dtype=float, device=self.device)
            self._hand_vel = wp.zeros((self.num_env, 3), dtype=float, device=self.device)

    def calculate(self, physics_manager, step_total_rewards: wp.array, **kwargs):
        if not self._ready:
            return
        self._sync_selected_target(physics_manager)
        target_pos, _ = self._target_pose_wp()
        has_target = self._has_target_wp()
        if target_pos is None or has_target is None:
            return
        self.art.copy_link_position_velocity(
            self.cfg["pelvis_body"], self._pelvis_pos, self._pelvis_vel
        )
        self.art.copy_link_position_velocity(
            self.cfg["hand_body"], self._hand_pos, self._hand_vel
        )
        retract_cfg = self.cfg["retract"]
        punch_cfg = self.cfg["punch"]
        compute_retract_reward(
            pelvis_pos=self._pelvis_pos,
            target_pos=target_pos,
            hand_pos=self._hand_pos,
            hand_vel=self._hand_vel,
            has_target=has_target,
            hard_state=self._hard_state_wp(),
            signal=self._signal_wp(),
            standoff=self.cfg["standoff_distance"],
            band=self.cfg["punch_ready_band"],
            punch_ready_mode=self._feasibility_mode(),
            gate_feasibility=punch_cfg["gate_feasibility"],
            vel_weight=retract_cfg["vel_weight"],
            fake_punch_vel_threshold=retract_cfg["fake_punch_vel_threshold"],
            fake_punch_penalty=retract_cfg["fake_punch_penalty"],
            ideal_away_speed=retract_cfg["ideal_away_speed"],
            away_sigma=retract_cfg["away_sigma"],
            over_speed_threshold=retract_cfg["over_speed_threshold"],
            over_speed_penalty=retract_cfg["over_speed_penalty"],
            raw=self._raw,
            num_env=self.num_env,
            device=self.device,
        )
        self._emit(step_total_rewards, kwargs)


class BoxingHitReward(_BoxingRewardBase):
    """Contact between configured hand bodies and the target at standoff.

    No ready-pose joint gate: impact happens at arm extension.
    """

    log_name = "boxing_hit"
    metric_log_names = ("boxing_hit_rate",)

    def __init__(self, device, pattern: str, **kwargs):
        super().__init__(device, pattern, **kwargs)
        _init_mjlab_scaling(self, "BoxingHitReward")
        self.contact_query = None
        self._hand_ids = None
        self._pelvis_pos = None
        self._pelvis_vel = None
        self._hand_pos = None
        self._hand_vel = None
        self._found_zeros = None
        self._force_zeros = None
        self._hit_flag_buf = None
        self._hit_force_buf = None
        self._hit_force_mean = None

    def bind_environment(self, environment) -> None:
        super().bind_environment(environment)
        pm = getattr(environment, "physics_manager", None)
        self.contact_query = ContactQuery.try_from_physics(pm)
        if pm is not None and self.cfg is not None:
            hand_ids = resolve_newton_body_ids(
                pm, self.cfg["hit_hand_bodies"], self.device
            )
            if hand_ids is not None:
                if hand_ids.ndim == 1:
                    hand_ids = hand_ids.unsqueeze(-1)
                self._hand_ids = hand_ids.to(dtype=torch.int32).contiguous()
        if self.num_env > 0:
            self._pelvis_pos = wp.zeros((self.num_env, 3), dtype=float, device=self.device)
            self._pelvis_vel = wp.zeros((self.num_env, 3), dtype=float, device=self.device)
            self._hand_pos = wp.zeros((self.num_env, 3), dtype=float, device=self.device)
            self._hand_vel = wp.zeros((self.num_env, 3), dtype=float, device=self.device)
            self._found_zeros = wp.zeros(self.num_env, dtype=wp.int32, device=self.device)
            self._force_zeros = wp.zeros(self.num_env, dtype=float, device=self.device)
            self._hit_flag_buf = wp.zeros(self.num_env, dtype=wp.float32, device=self.device)
            self._hit_force_buf = wp.zeros(self.num_env, dtype=wp.float32, device=self.device)
            self._hit_force_mean = torch.zeros(
                (), device=torch.device(str(self.device)), dtype=torch.float32
            )

    def try_get_hit_force_mean(self):
        """Mean contact force among envs that hit this step, or 0 if none hit."""
        if self._hit_flag_buf is None or self._hit_force_buf is None:
            return None
        if self._hit_force_mean is None:
            return None
        flag = wp.to_torch(self._hit_flag_buf)
        force = wp.to_torch(self._hit_force_buf)
        hits = torch.clamp(torch.sum(flag), min=1.0)
        torch.div(torch.sum(force), hits, out=self._hit_force_mean)
        return self._hit_force_mean

    def calculate(self, physics_manager, step_total_rewards: wp.array, **kwargs):
        if self._hit_flag_buf is not None:
            self._hit_flag_buf.zero_()
        if self._hit_force_buf is not None:
            self._hit_force_buf.zero_()
        if not self._ready:
            return
        self._sync_selected_target(physics_manager)
        target_pos, target_vel = self._target_pose_wp()
        has_target = self._has_target_wp()
        if target_pos is None or has_target is None:
            return
        found = self._found_zeros
        force_wp = self._force_zeros
        ids_fn = getattr(self.obs_actor, "try_selected_target_newton_body_ids", None)
        if (
            self.contact_query is not None
            and self._hand_ids is not None
            and callable(ids_fn)
        ):
            body_ids = ids_fn(physics_manager)
            if body_ids is not None:
                self.contact_query.max_force(self._hand_ids, body_ids)
                found = self.contact_query.last_found
                force_wp = self.contact_query.last_max_force
                if found is None:
                    found = self._found_zeros
                if force_wp is None:
                    force_wp = self._force_zeros
        self.art.copy_link_position_velocity(
            self.cfg["pelvis_body"], self._pelvis_pos, self._pelvis_vel
        )
        self.art.copy_link_position_velocity(
            self.cfg["hand_body"], self._hand_pos, self._hand_vel
        )
        hit_cfg = self.cfg["hit"]
        compute_hit_reward(
            pelvis_pos=self._pelvis_pos,
            target_pos=target_pos,
            hand_vel=self._hand_vel,
            target_vel=target_vel,
            has_target=has_target,
            contact_found=found,
            hard_state=self._hard_state_wp(),
            signal=self._signal_wp(),
            standoff=self.cfg["standoff_distance"],
            band=self.cfg["punch_ready_band"],
            punch_ready_mode=self._feasibility_mode(),
            gate_feasibility=self.cfg["punch"]["gate_feasibility"],
            velocity_threshold=hit_cfg["velocity_threshold"],
            base=hit_cfg["base"],
            velocity_bonus=hit_cfg["velocity_bonus"],
            max_hand_vel=self.cfg["max_hand_vel"],
            raw=self._raw,
            num_env=self.num_env,
            device=self.device,
        )
        self._emit(step_total_rewards, kwargs)
        if self._hit_flag_buf is not None and self._hit_force_buf is not None:
            fill_hit_metrics(
                contact_found=found,
                contact_force=force_wp,
                hit_flag=self._hit_flag_buf,
                hit_force=self._hit_force_buf,
                num_env=self.num_env,
                device=self.device,
            )
            rate_buf = (kwargs.get("metric_bufs") or {}).get("boxing_hit_rate")
            if rate_buf is not None:
                wp.copy(rate_buf, self._hit_flag_buf)


class BoxingPoseReward(_BoxingRewardBase):
    """``-(||q-q_nom|| w_n + ||τ|| w_τ + tilt w_tilt)`` on RL-masked joints."""

    log_name = "boxing_pose"

    def __init__(self, device, pattern: str, **kwargs):
        super().__init__(device, pattern, **kwargs)
        _init_mjlab_scaling(self, "BoxingPoseReward")
        self._joint_q = None
        self._joint_tau = None
        self._tilt = None
        self._joint_nom = None
        self._joint_mask = None
        self._dof = 0

    def bind_environment(self, environment) -> None:
        super().bind_environment(environment)
        if not self._ready:
            return
        dof = int(getattr(self.art.view, "joint_dof_count", 0) or 0)
        self._dof = dof
        n = int(getattr(self.art, "num_instances", self.num_env) or self.num_env)
        if n > 0 and dof > 0:
            self._joint_q = wp.zeros((n, dof), dtype=float, device=self.device)
            self._joint_tau = wp.zeros((n, dof), dtype=float, device=self.device)
            self._tilt = wp.zeros(n, dtype=float, device=self.device)
        nom = self.art.joint_nominal_positions()
        mask = self.art.joint_rl_mask()
        if nom is not None:
            self._nom_torch = nom.to(dtype=torch.float32).contiguous()
            self._joint_nom = wp.from_torch(self._nom_torch, dtype=wp.float32)
        if mask is not None:
            self._mask_torch = mask.to(dtype=torch.int32).contiguous()
            self._joint_mask = wp.from_torch(self._mask_torch, dtype=wp.int32)

    def calculate(self, physics_manager, step_total_rewards: wp.array, **kwargs):
        del physics_manager
        if (
            not self._ready
            or self._joint_q is None
            or self._joint_nom is None
            or self._joint_mask is None
        ):
            return
        self.art.copy_joint_positions(self._joint_q)
        if not self.art.copy_joint_torques(self._joint_tau):
            self._joint_tau.zero_()
        self.art.copy_projected_gravity_xy(self.cfg["torso_body"], self._tilt)
        pose_cfg = self.cfg["pose"]
        compute_pose_reward(
            joint_q=self._joint_q,
            joint_nom=self._joint_nom,
            joint_rl_mask=self._joint_mask,
            joint_tau=self._joint_tau,
            tilt=self._tilt,
            dof_count=self._dof,
            w_nominal=pose_cfg["nominal_weight"],
            w_torque=pose_cfg["torque_weight"],
            w_tilt=pose_cfg["tilt_weight"],
            raw=self._raw,
            num_env=self.num_env,
            device=self.device,
        )
        self._emit(step_total_rewards, kwargs)
