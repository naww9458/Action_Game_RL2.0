# This file contains code adapted from:
# https://github.com/mujocolab/mjlab
#
# Modified for Action_Game_RL.
#
# The original project is licensed under the Apache License 2.0.

"""Standalone Warp reward components for G1 velocity-flat locomotion.

Each class mirrors the mathematical definition of the corresponding mjlab
reward term from ``env_cfg.py``.  Data arrays are supplied at runtime via
``calculate(..., **kwargs)`` so this module has no dependency on mjlab.
"""

from __future__ import annotations

import re

import warp as wp

from script.levels.rewards.reward_calculator import RewardComponent
from script.game_config import GameConfig


def _component_params(params: dict, class_name: str) -> dict:
    nested = params.get(class_name)
    if isinstance(nested, dict):
        return nested
    return params


def _init_mjlab_scaling(reward: RewardComponent, class_name: str) -> None:
    """Match mjlab RewardManager: total += raw_value * weight * step_dt.

    ``weight`` must be declared in the training preset under
    ``reward_parameters.<ClassName>.weight``.
    """
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


def _setup_g1_view_reward(reward, device, pattern: str) -> None:
    reward.device = device
    reward.pattern = pattern
    view_idx = next((i for i, p in enumerate(reward.articulation_body.patterns) if p == pattern), -1)
    if view_idx != -1:
        reward.view = reward.articulation_body.views[view_idx]


def _resolve_view_link_index(view, body_name: str) -> int:
    """Resolve the per-view link index whose name ends with ``body_name``.

    Newton ``ArticulationView.link_names`` holds the leaf component of the
    hierarchical body label (e.g. ``torso_link``). Matching uses ``endswith`` so
    the resolved index stays stable regardless of the USD/XPath prefix.
    """
    names = list(getattr(view, "link_names", ()) or ())
    for i, name in enumerate(names):
        if name.endswith(body_name):
            return i
    raise RuntimeError(
        f"Body '{body_name}' not found in view.link_names "
        f"(available: {names[:16]}{'...' if len(names) > 16 else ''})."
    )


def _bind_g1_level_resources(reward, level) -> None:
    reward.foot_sensor = getattr(level, "foot_sensor", None)
    # ``policy_actions`` holds the action of the previous step (a_{t-1}) at reward
    # computation time, matching mjlab's ``action_manager.prev_action``.
    reward.policy_actions = getattr(level, "policy_actions", None)
    reward.prev_actions = getattr(level, "prev_actions", None)
    reward.rl_action_dim = getattr(level, "rl_action_dim", GameConfig.ACTION_SHAPE_OFFSET)
    reward.num_env = getattr(level, "num_env", GameConfig.NUM_PLAYERS)


def _get_log_buffers(kwargs: dict, log_name: str | None, metric_names: tuple[str, ...] = ()):
    """Extract the per-term reward buffer and metric buffers from calculator kwargs.

    ``RewardCalculator`` passes ``reward_term_bufs`` / ``metric_bufs`` dicts keyed
    by the component's ``log_name`` / ``metric_log_names``. Returns the term
    buffer (or ``None``) plus a list of metric buffers in declaration order.
    """
    term_buf = kwargs.get("reward_term_bufs", {}).get(log_name) if log_name else None
    metric_bufs = kwargs.get("metric_bufs", {})
    return term_buf, [metric_bufs.get(n) for n in metric_names]

def _normalize_pattern(pattern: str) -> str:
    if len(pattern) >= 2 and pattern[0] == "r" and pattern[1] in ("'", '"'):
        return pattern[2:-1]
    return pattern


def resolve_joint_stds(pattern_dict: dict[str, float], joint_names: list[str]) -> list[float]:
    """Map regex-style joint-name patterns to per-joint standard deviations.

    Uses ``re.fullmatch`` (as mjlab's ``resolve_matching_names_values``) so
    regex metacharacters like ``.*`` behave like real regexes. Joints that
    match no pattern keep ``default_std`` so the returned list always has one
    entry per joint name (mjlab instead drops unmatched joints, which would
    desync the array from ``num_joints`` used by the reward kernel).
    """
    default_std = 0.1
    result: list[float] = []
    for name in joint_names:
        matched_std = default_std
        for pattern, std in pattern_dict.items():
            if re.fullmatch(_normalize_pattern(pattern), name):
                matched_std = float(std)
                break
        result.append(matched_std)
    return result


@wp.func
def _find_view_obj_idx(
    local_idx: int,
    view_object_indices: wp.array(dtype=int),
    count_per_world: int,
) -> int:
    for i in range(count_per_world):
        if view_object_indices[i] == local_idx:
            return i
    return -1


@wp.func
def _command_active_func(
    command_vel: wp.array2d(dtype=wp.float32),
    env_id: int,
    command_threshold: float,
) -> float:
    lin_x = command_vel[env_id, 0]
    lin_y = command_vel[env_id, 1]
    ang_z = command_vel[env_id, 2]

    linear_norm = wp.sqrt(lin_x * lin_x + lin_y * lin_y)
    total_command = linear_norm + wp.abs(ang_z)
    if total_command > command_threshold:
        return wp.float32(1.0)
    return wp.float32(0.0)


class TrackLinearVelocityReward(RewardComponent):
    """exp(-||v_cmd_xy - v_actual_xy||² / std² - v_actual_z² / std²)."""

    log_name = "track_linear_velocity"

    def __init__(self, device, pattern, **kwargs):
        super().__init__(**kwargs)
        _setup_g1_view_reward(self, device, pattern)
        cfg = _component_params(self.params, "TrackLinearVelocityReward")
        self.std = float(cfg.get("std", 0.5))
        self.std_sq = self.std * self.std
        _init_mjlab_scaling(self, "TrackLinearVelocityReward")

    def calculate(
        self,
        num_players,
        physics_manager,
        step_total_rewards: wp.array,
        player_shape_ids_gpu: wp.array,
        command_vel: wp.array,
        index_player_obj_to_env_mapping_gpu: wp.array,
        **kwargs,
    ):
        pm = physics_manager
        root_tfs = self.view.get_root_transforms(pm.state_0)
        root_vels = self.view.get_root_velocities(pm.state_0)

        term_buf, _ = _get_log_buffers(kwargs, self.log_name)

        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_players,
            inputs=[
                player_shape_ids_gpu,
                command_vel,
                root_tfs,
                root_vels,
                index_player_obj_to_env_mapping_gpu,
                self.articulation_body.view_object_indices_gpus[self.pattern],
                self.articulation_body.num_objects_env,
                self.view.count_per_world,
                step_total_rewards,
                term_buf,
                self.std_sq,
                self.reward_scale,
            ],
            device=self.device,
        )

    @wp.kernel
    def calculate_gpu(
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        command_vel: wp.array2d(dtype=wp.float32),
        root_tfs: wp.array2d(dtype=wp.transform),
        root_vels: wp.array2d(dtype=wp.spatial_vector),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        view_object_indices: wp.array(dtype=int),
        num_objects_env: int,
        count_per_world: int,
        step_total_rewards: wp.array(dtype=wp.float32),
        term_buf: wp.array(dtype=wp.float32),
        std_sq: wp.float32,
        reward_scale: wp.float32,
    ):
        tid = wp.tid()
        shape_id = player_shape_ids_gpu[tid]
        world = index_player_obj_to_env_mapping_gpu[tid]

        local_idx = shape_id % num_objects_env
        obj_idx = _find_view_obj_idx(local_idx, view_object_indices, count_per_world)
        if obj_idx == -1:
            return

        my_tf = root_tfs[world, obj_idx]
        my_rot = wp.transform_get_rotation(my_tf)
        root_qd = root_vels[world, obj_idx]
        inv_rot = wp.quat_inverse(my_rot)
        world_lin = wp.vec3(root_qd[0], root_qd[1], root_qd[2])
        local_lin = wp.quat_rotate(inv_rot, world_lin)

        cmd_x = command_vel[world, 0]
        cmd_y = command_vel[world, 1]
        act_x = local_lin[0]
        act_y = local_lin[1]
        act_z = local_lin[2]

        xy_error = (cmd_x - act_x) * (cmd_x - act_x) + (cmd_y - act_y) * (cmd_y - act_y)
        z_error = act_z * act_z
        lin_vel_error = xy_error + z_error
        reward_value = wp.exp(-lin_vel_error / std_sq) * reward_scale
        wp.atomic_add(step_total_rewards, shape_id, reward_value)
        wp.atomic_add(term_buf, world, reward_value)

    def reset(self, **kwargs):
        pass


class TrackAngularVelocityReward(RewardComponent):
    """exp(-||ω_cmd_z - ω_actual_z||² / std² - ||ω_actual_xy||² / std²)."""

    log_name = "track_angular_velocity"

    def __init__(self, device, pattern, **kwargs):
        super().__init__(**kwargs)
        _setup_g1_view_reward(self, device, pattern)
        cfg = _component_params(self.params, "TrackAngularVelocityReward")
        self.std = float(cfg.get("std", 0.7071067811865476))
        self.std_sq = self.std * self.std
        _init_mjlab_scaling(self, "TrackAngularVelocityReward")

    def calculate(
        self,
        num_players,
        physics_manager,
        step_total_rewards: wp.array,
        player_shape_ids_gpu: wp.array,
        command_vel: wp.array,
        index_player_obj_to_env_mapping_gpu: wp.array,
        **kwargs,
    ):
        pm = physics_manager
        root_tfs = self.view.get_root_transforms(pm.state_0)
        root_vels = self.view.get_root_velocities(pm.state_0)

        term_buf, _ = _get_log_buffers(kwargs, self.log_name)

        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_players,
            inputs=[
                player_shape_ids_gpu,
                command_vel,
                root_tfs,
                root_vels,
                index_player_obj_to_env_mapping_gpu,
                self.articulation_body.view_object_indices_gpus[self.pattern],
                self.articulation_body.num_objects_env,
                self.view.count_per_world,
                step_total_rewards,
                term_buf,
                self.std_sq,
                self.reward_scale,
            ],
            device=self.device,
        )

    @wp.kernel
    def calculate_gpu(
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        command_vel: wp.array2d(dtype=wp.float32),
        root_tfs: wp.array2d(dtype=wp.transform),
        root_vels: wp.array2d(dtype=wp.spatial_vector),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        view_object_indices: wp.array(dtype=int),
        num_objects_env: int,
        count_per_world: int,
        step_total_rewards: wp.array(dtype=wp.float32),
        term_buf: wp.array(dtype=wp.float32),
        std_sq: wp.float32,
        reward_scale: wp.float32,
    ):
        tid = wp.tid()
        shape_id = player_shape_ids_gpu[tid]
        world = index_player_obj_to_env_mapping_gpu[tid]
        local_idx = shape_id % num_objects_env
        obj_idx = _find_view_obj_idx(local_idx, view_object_indices, count_per_world)
        if obj_idx == -1:
            return

        my_tf = root_tfs[world, obj_idx]
        my_rot = wp.transform_get_rotation(my_tf)
        root_qd = root_vels[world, obj_idx]
        inv_rot = wp.quat_inverse(my_rot)
        world_ang = wp.vec3(root_qd[3], root_qd[4], root_qd[5])
        local_ang = wp.quat_rotate(inv_rot, world_ang)

        cmd_wz = command_vel[world, 2]
        act_wx = local_ang[0]
        act_wy = local_ang[1]
        act_wz = local_ang[2]
        z_error = (cmd_wz - act_wz) * (cmd_wz - act_wz)
        xy_error = act_wx * act_wx + act_wy * act_wy
        ang_vel_error = z_error + xy_error
        reward_value = wp.exp(-ang_vel_error / std_sq) * reward_scale
        wp.atomic_add(step_total_rewards, shape_id, reward_value)
        wp.atomic_add(term_buf, world, reward_value)

    def reset(self, **kwargs):
        pass


class ActionRateL2Reward(RewardComponent):
    """||a_t - a_{t-1}||² summed over action dimensions."""

    log_name = "action_rate_l2"

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.policy_actions = None
        self.rl_action_dim = GameConfig.ACTION_SHAPE_OFFSET
        _init_mjlab_scaling(self, "ActionRateL2Reward")

    def bind_level(self, level):
        _bind_g1_level_resources(self, level)

    def calculate(
        self,
        num_players,
        physics_manager,
        step_total_rewards: wp.array,
        player_shape_ids_gpu: wp.array,
        actions: wp.array,
        is_rl_player_mask_gpu: wp.array,
        index_player_obj_to_env_mapping_gpu: wp.array,
        **kwargs,
    ):
        if self.policy_actions is None:
            return

        term_buf, _ = _get_log_buffers(kwargs, self.log_name)

        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_players,
            inputs=[
                player_shape_ids_gpu,
                actions,
                self.policy_actions,
                is_rl_player_mask_gpu,
                index_player_obj_to_env_mapping_gpu,
                step_total_rewards,
                term_buf,
                self.rl_action_dim,
                self.reward_scale,
            ],
            device=self.device,
        )

    @wp.kernel
    def calculate_gpu(
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        actions: wp.array2d(dtype=wp.float32),
        policy_actions: wp.array2d(dtype=wp.float32),
        is_rl_player_mask_gpu: wp.array(dtype=wp.int32),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        step_total_rewards: wp.array(dtype=wp.float32),
        term_buf: wp.array(dtype=wp.float32),
        rl_action_dim: wp.int32,
        reward_scale: wp.float32,
    ):
        tid = wp.tid()
        shape_id = player_shape_ids_gpu[tid]
        rl_row = is_rl_player_mask_gpu[tid]
        if rl_row < 0:
            return
        world = index_player_obj_to_env_mapping_gpu[tid]
        penalty = wp.float32(0.0)
        for i in range(rl_action_dim):
            # policy_actions == a_{t-1} at reward time, so diff == a_t - a_{t-1}.
            diff = actions[rl_row, i] - policy_actions[world, i]
            penalty += diff * diff
        wp.atomic_add(step_total_rewards, shape_id, penalty * reward_scale)
        wp.atomic_add(term_buf, world, penalty * reward_scale)

    def reset(self, **kwargs):
        pass


class AngularMomentumPenaltyReward(RewardComponent):
    """||L||² for the whole-robot angular momentum (mjlab subtreeangmom on a root body)."""

    log_name = "angular_momentum"
    metric_log_names = ("angular_momentum_mean",)

    def __init__(self, device, pattern, **kwargs):
        super().__init__(**kwargs)
        _setup_g1_view_reward(self, device, pattern)
        cfg = _component_params(self.params, "AngularMomentumPenaltyReward")
        if "body_name" not in cfg:
            raise KeyError(
                "Missing reward_parameters.AngularMomentumPenaltyReward.body_name "
                "(mjlab uses pelvis)."
            )
        self.root_body_name = str(cfg["body_name"])
        self.root_mj_body_id = None
        _init_mjlab_scaling(self, "AngularMomentumPenaltyReward")

    def bind_level(self, level):
        from script.role.objects.object_template.mjlab_unitree_g1.g1_foot_sensor_cfg import (
            resolve_g1_body_mj_id,
        )

        pm = getattr(level, "physics_manager", None)
        if pm is None:
            return
        self.root_mj_body_id = resolve_g1_body_mj_id(pm, self.root_body_name)

    def calculate(
        self,
        num_players,
        physics_manager,
        step_total_rewards: wp.array,
        player_shape_ids_gpu: wp.array,
        index_player_obj_to_env_mapping_gpu: wp.array,
        **kwargs,
    ):
        if self.root_mj_body_id is None:
            return
        pm = physics_manager
        solver = getattr(getattr(pm, "solver_handler", None), "solver", None)
        if solver is None or not hasattr(solver, "mjw_data"):
            return
        subtree_angmom = solver.mjw_data.subtree_angmom

        term_buf, metric_bufs = _get_log_buffers(kwargs, self.log_name, self.metric_log_names)

        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_players,
            inputs=[
                player_shape_ids_gpu,
                subtree_angmom,
                index_player_obj_to_env_mapping_gpu,
                step_total_rewards,
                term_buf,
                metric_bufs[0],
                self.root_mj_body_id,
                self.reward_scale,
            ],
            device=self.device,
        )

    @wp.kernel
    def calculate_gpu(
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        subtree_angmom: wp.array2d(dtype=wp.vec3),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        step_total_rewards: wp.array(dtype=wp.float32),
        term_buf: wp.array(dtype=wp.float32),
        ang_mom_metric_buf: wp.array(dtype=wp.float32),
        root_mj_body_id: int,
        reward_scale: wp.float32,
    ):
        tid = wp.tid()
        shape_id = player_shape_ids_gpu[tid]
        world = index_player_obj_to_env_mapping_gpu[tid]

        angmom = subtree_angmom[world, root_mj_body_id]
        penalty = angmom[0] * angmom[0] + angmom[1] * angmom[1] + angmom[2] * angmom[2]
        wp.atomic_add(step_total_rewards, shape_id, penalty * reward_scale)
        wp.atomic_add(term_buf, world, penalty * reward_scale)
        ang_mom_metric_buf[world] = wp.sqrt(penalty)

    def reset(self, **kwargs):
        pass


class BodyAngularVelocityPenaltyReward(RewardComponent):
    """||ω_xy||² for the torso_link body (world-frame angular velocity, mjlab)."""

    log_name = "body_ang_vel"

    def __init__(self, device, pattern, **kwargs):
        super().__init__(**kwargs)
        _setup_g1_view_reward(self, device, pattern)
        cfg = _component_params(self.params, "BodyAngularVelocityPenaltyReward")
        body_name = str(cfg.get("body_name", "torso_link"))
        self.body_link_idx = _resolve_view_link_index(self.view, body_name)
        _init_mjlab_scaling(self, "BodyAngularVelocityPenaltyReward")

    def calculate(
        self,
        num_players,
        physics_manager,
        step_total_rewards: wp.array,
        player_shape_ids_gpu: wp.array,
        index_player_obj_to_env_mapping_gpu: wp.array,
        **kwargs,
    ):
        pm = physics_manager
        link_vels = self.view.get_link_velocities(pm.state_0)

        term_buf, _ = _get_log_buffers(kwargs, self.log_name)

        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_players,
            inputs=[
                player_shape_ids_gpu,
                link_vels,
                index_player_obj_to_env_mapping_gpu,
                self.articulation_body.view_object_indices_gpus[self.pattern],
                self.articulation_body.num_objects_env,
                self.view.count_per_world,
                step_total_rewards,
                term_buf,
                self.body_link_idx,
                self.reward_scale,
            ],
            device=self.device,
        )

    @wp.kernel
    def calculate_gpu(
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        link_vels: wp.array(dtype=wp.spatial_vector, ndim=3),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        view_object_indices: wp.array(dtype=int),
        num_objects_env: int,
        count_per_world: int,
        step_total_rewards: wp.array(dtype=wp.float32),
        term_buf: wp.array(dtype=wp.float32),
        body_link_idx: int,
        reward_scale: wp.float32,
    ):
        tid = wp.tid()
        shape_id = player_shape_ids_gpu[tid]
        world = index_player_obj_to_env_mapping_gpu[tid]
        local_idx = shape_id % num_objects_env
        obj_idx = _find_view_obj_idx(local_idx, view_object_indices, count_per_world)
        if obj_idx == -1:
            return

        link_qd = link_vels[world, obj_idx, body_link_idx]
        # Warp spatial_vector layout is [lin.x, lin.y, lin.z, ang.x, ang.y, ang.z].
        # mjlab penalizes world-frame angular velocity xy (body_link_ang_vel_w).
        world_ang = wp.vec3(link_qd[3], link_qd[4], link_qd[5])
        penalty = world_ang[0] * world_ang[0] + world_ang[1] * world_ang[1]
        wp.atomic_add(step_total_rewards, shape_id, penalty * reward_scale)
        wp.atomic_add(term_buf, world, penalty * reward_scale)

    def reset(self, **kwargs):
        pass


class FeetAirTimeReward(RewardComponent):
    """Count feet whose air time lies in [threshold_min, threshold_max]."""

    log_name = "air_time"

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.foot_sensor = None
        self.num_feet = 2
        cfg = _component_params(self.params, "FeetAirTimeReward")
        self.threshold_min = float(cfg.get("threshold_min", 0.05))
        self.threshold_max = float(cfg.get("threshold_max", 0.5))
        self.command_threshold = float(cfg.get("command_threshold", 0.5))
        _init_mjlab_scaling(self, "FeetAirTimeReward")

    def bind_level(self, level):
        _bind_g1_level_resources(self, level)

    def calculate(
        self,
        num_players,
        physics_manager,
        step_total_rewards: wp.array,
        player_shape_ids_gpu: wp.array,
        command_vel: wp.array,
        index_player_obj_to_env_mapping_gpu: wp.array,
        **kwargs,
    ):
        if self.foot_sensor is None:
            return

        term_buf, _ = _get_log_buffers(kwargs, self.log_name)

        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_players,
            inputs=[
                player_shape_ids_gpu,
                self.foot_sensor.foot_air_time,
                command_vel,
                index_player_obj_to_env_mapping_gpu,
                step_total_rewards,
                term_buf,
                self.num_feet,
                self.threshold_min,
                self.threshold_max,
                self.command_threshold,
                self.reward_scale,
            ],
            device=self.device,
        )

    @wp.kernel
    def calculate_gpu(
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        current_air_time: wp.array2d(dtype=wp.float32),
        command_vel: wp.array2d(dtype=wp.float32),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        step_total_rewards: wp.array(dtype=wp.float32),
        term_buf: wp.array(dtype=wp.float32),
        num_feet: wp.int32,
        threshold_min: wp.float32,
        threshold_max: wp.float32,
        command_threshold: wp.float32,
        reward_scale: wp.float32,
    ):
        tid = wp.tid()
        shape_id = player_shape_ids_gpu[tid]
        world = index_player_obj_to_env_mapping_gpu[tid]
        reward = wp.float32(0.0)
        for foot in range(num_feet):
            air_t = current_air_time[world, foot]
            if air_t > threshold_min and air_t < threshold_max:
                reward += wp.float32(1.0)
        active = _command_active_func(command_vel, world, command_threshold)
        wp.atomic_add(step_total_rewards, shape_id, reward * active * reward_scale)
        wp.atomic_add(term_buf, world, reward * active * reward_scale)

    def reset(self, **kwargs):
        pass


class FeetClearanceReward(RewardComponent):
    """|h - h_target| weighted by foot XY velocity magnitude."""

    log_name = "foot_clearance"

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.foot_sensor = None
        self.num_feet = 2
        cfg = _component_params(self.params, "FeetClearanceReward")
        self.target_height = float(cfg.get("target_height", 0.1))
        self.command_threshold = float(cfg.get("command_threshold", 0.05))
        _init_mjlab_scaling(self, "FeetClearanceReward")

    def bind_level(self, level):
        _bind_g1_level_resources(self, level)

    def calculate(
        self,
        num_players,
        physics_manager,
        step_total_rewards: wp.array,
        player_shape_ids_gpu: wp.array,
        command_vel: wp.array,
        index_player_obj_to_env_mapping_gpu: wp.array,
        **kwargs,
    ):
        if self.foot_sensor is None:
            return

        term_buf, _ = _get_log_buffers(kwargs, self.log_name)

        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_players,
            inputs=[
                player_shape_ids_gpu,
                self.foot_sensor.foot_height,
                self.foot_sensor.foot_lin_vel_xy_sq,
                command_vel,
                index_player_obj_to_env_mapping_gpu,
                step_total_rewards,
                term_buf,
                self.num_feet,
                self.target_height,
                self.command_threshold,
                self.reward_scale,
            ],
            device=self.device,
        )

    @wp.kernel
    def calculate_gpu(
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        foot_heights: wp.array2d(dtype=wp.float32),
        foot_vel_xy_sq: wp.array2d(dtype=wp.float32),
        command_vel: wp.array2d(dtype=wp.float32),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        step_total_rewards: wp.array(dtype=wp.float32),
        term_buf: wp.array(dtype=wp.float32),
        num_feet: wp.int32,
        target_height: wp.float32,
        command_threshold: wp.float32,
        reward_scale: wp.float32,
    ):
        tid = wp.tid()
        shape_id = player_shape_ids_gpu[tid]
        world = index_player_obj_to_env_mapping_gpu[tid]
        cost = wp.float32(0.0)
        for foot in range(num_feet):
            height = foot_heights[world, foot]
            vel_norm = wp.sqrt(foot_vel_xy_sq[world, foot])
            delta = wp.abs(height - target_height)
            cost += delta * vel_norm
        active = _command_active_func(command_vel, world, command_threshold)
        wp.atomic_add(step_total_rewards, shape_id, cost * active * reward_scale)
        wp.atomic_add(term_buf, world, cost * active * reward_scale)

    def reset(self, **kwargs):
        pass


class FeetSlipReward(RewardComponent):
    """||v_foot_xy||² while foot is in contact."""

    log_name = "foot_slip"
    metric_log_names = ("slip_velocity_mean",)

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.foot_sensor = None
        self.num_feet = 2
        cfg = _component_params(self.params, "FeetSlipReward")
        self.command_threshold = float(cfg.get("command_threshold", 0.05))
        _init_mjlab_scaling(self, "FeetSlipReward")

    def bind_level(self, level):
        _bind_g1_level_resources(self, level)

    def calculate(
        self,
        num_players,
        physics_manager,
        step_total_rewards: wp.array,
        player_shape_ids_gpu: wp.array,
        command_vel: wp.array,
        index_player_obj_to_env_mapping_gpu: wp.array,
        **kwargs,
    ):
        if self.foot_sensor is None:
            return

        term_buf, metric_bufs = _get_log_buffers(kwargs, self.log_name, self.metric_log_names)

        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_players,
            inputs=[
                player_shape_ids_gpu,
                self.foot_sensor.foot_found,
                self.foot_sensor.foot_lin_vel_xy_sq,
                command_vel,
                index_player_obj_to_env_mapping_gpu,
                step_total_rewards,
                term_buf,
                metric_bufs[0],
                self.num_feet,
                self.command_threshold,
                self.reward_scale,
            ],
            device=self.device,
        )

    @wp.kernel
    def calculate_gpu(
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        in_contact: wp.array2d(dtype=wp.int32),
        foot_vel_xy_sq: wp.array2d(dtype=wp.float32),
        command_vel: wp.array2d(dtype=wp.float32),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        step_total_rewards: wp.array(dtype=wp.float32),
        term_buf: wp.array(dtype=wp.float32),
        slip_metric_buf: wp.array(dtype=wp.float32),
        num_feet: wp.int32,
        command_threshold: wp.float32,
        reward_scale: wp.float32,
    ):
        tid = wp.tid()
        shape_id = player_shape_ids_gpu[tid]
        world = index_player_obj_to_env_mapping_gpu[tid]
        cost = wp.float32(0.0)
        slip_sum = wp.float32(0.0)
        slip_count = wp.float32(0.0)
        for foot in range(num_feet):
            contact = wp.float32(in_contact[world, foot])
            if contact > 0.0:
                vel_norm = wp.sqrt(foot_vel_xy_sq[world, foot])
                cost += foot_vel_xy_sq[world, foot] * contact
                slip_sum += vel_norm
                slip_count += wp.float32(1.0)
        active = _command_active_func(command_vel, world, command_threshold)
        wp.atomic_add(step_total_rewards, shape_id, cost * active * reward_scale)
        wp.atomic_add(term_buf, world, cost * active * reward_scale)
        slip_metric_buf[world] = slip_sum / wp.max(slip_count, wp.float32(1.0))

    def reset(self, **kwargs):
        pass


class FeetSwingHeightReward(RewardComponent):
    """Penalize deviation from target swing height, evaluated at landing."""

    log_name = "foot_swing_height"
    metric_log_names = ("peak_height_mean",)

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.foot_sensor = None
        self.num_env = GameConfig.NUM_PLAYERS
        self.num_feet = 2
        cfg = _component_params(self.params, "FeetSwingHeightReward")
        self.target_height = float(cfg.get("target_height", 0.1))
        self.command_threshold = float(cfg.get("command_threshold", 0.05))
        self.peak_heights = None
        _init_mjlab_scaling(self, "FeetSwingHeightReward")

    def bind_level(self, level):
        _bind_g1_level_resources(self, level)
        if self.peak_heights is None:
            self.peak_heights = wp.zeros(
                (self.num_env, self.num_feet), dtype=wp.float32, device=self.device
            )

    def calculate(
        self,
        num_players,
        physics_manager,
        step_total_rewards: wp.array,
        player_shape_ids_gpu: wp.array,
        command_vel: wp.array,
        index_player_obj_to_env_mapping_gpu: wp.array,
        **kwargs,
    ):
        if self.foot_sensor is None or self.peak_heights is None:
            return

        term_buf, metric_bufs = _get_log_buffers(kwargs, self.log_name, self.metric_log_names)

        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_players,
            inputs=[
                player_shape_ids_gpu,
                self.foot_sensor.foot_height,
                self.foot_sensor.foot_found,
                self.foot_sensor.foot_first_contact,
                command_vel,
                self.peak_heights,
                index_player_obj_to_env_mapping_gpu,
                step_total_rewards,
                term_buf,
                metric_bufs[0],
                self.num_feet,
                self.target_height,
                self.command_threshold,
                self.reward_scale,
            ],
            device=self.device,
        )

    @wp.kernel
    def calculate_gpu(
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        foot_heights: wp.array2d(dtype=wp.float32),
        foot_found: wp.array2d(dtype=wp.int32),
        first_contact: wp.array2d(dtype=wp.int32),
        command_vel: wp.array2d(dtype=wp.float32),
        peak_heights: wp.array2d(dtype=wp.float32),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        step_total_rewards: wp.array(dtype=wp.float32),
        term_buf: wp.array(dtype=wp.float32),
        peak_metric_buf: wp.array(dtype=wp.float32),
        num_feet: wp.int32,
        target_height: wp.float32,
        command_threshold: wp.float32,
        reward_scale: wp.float32,
    ):
        tid = wp.tid()
        shape_id = player_shape_ids_gpu[tid]
        world = index_player_obj_to_env_mapping_gpu[tid]
        cost = wp.float32(0.0)
        peak_sum = wp.float32(0.0)
        peak_count = wp.float32(0.0)
        for foot in range(num_feet):
            height = foot_heights[world, foot]
            in_air = foot_found[world, foot] == 0
            if in_air:
                peak_heights[world, foot] = wp.max(peak_heights[world, foot], height)
            landed = wp.float32(first_contact[world, foot])
            if landed > 0.0:
                error = peak_heights[world, foot] / target_height - wp.float32(1.0)
                cost += error * error * landed
                peak_sum += peak_heights[world, foot]
                peak_count += wp.float32(1.0)
                peak_heights[world, foot] = wp.float32(0.0)
        active = _command_active_func(command_vel, world, command_threshold)
        wp.atomic_add(step_total_rewards, shape_id, cost * active * reward_scale)
        wp.atomic_add(term_buf, world, cost * active * reward_scale)
        peak_metric_buf[world] = peak_sum / wp.max(peak_count, wp.float32(1.0))

    def reset(
        self,
        num_players,
        terminated,
        index_player_obj_to_env_mapping_gpu,
        **kwargs,
    ):
        if self.peak_heights is None:
            return
        wp.launch(
            kernel=self.reset_gpu,
            dim=num_players,
            inputs=[
                terminated,
                index_player_obj_to_env_mapping_gpu,
                self.peak_heights,
                self.num_feet,
            ],
            device=self.device,
        )

    @wp.kernel
    def reset_gpu(
        terminated: wp.array(dtype=wp.bool),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        peak_heights: wp.array2d(dtype=wp.float32),
        num_feet: wp.int32,
    ):
        tid = wp.tid()
        index_env = index_player_obj_to_env_mapping_gpu[tid]
        if terminated[index_env] == False:
            return
        for foot in range(num_feet):
            peak_heights[index_env, foot] = wp.float32(0.0)


class JointPosLimitsReward(RewardComponent):
    """Soft joint-limit violation penalty."""

    log_name = "dof_pos_limits"

    def __init__(self, device, pattern, **kwargs):
        super().__init__(**kwargs)
        _setup_g1_view_reward(self, device, pattern)
        self.num_joints = self.view.joint_dof_count
        self.joint_rl_mask = self.articulation_body.control_joint_rl_mask_gpus[self.pattern]
        _init_mjlab_scaling(self, "JointPosLimitsReward")

    def calculate(
        self,
        num_players,
        physics_manager,
        step_total_rewards: wp.array,
        player_shape_ids_gpu: wp.array,
        index_player_obj_to_env_mapping_gpu: wp.array,
        **kwargs,
    ):
        pm = physics_manager
        joint_qs = self.view.get_dof_positions(pm.state_0)

        term_buf, _ = _get_log_buffers(kwargs, self.log_name)

        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_players,
            inputs=[
                player_shape_ids_gpu,
                joint_qs,
                self.articulation_body.control_joint_limits_min_gpus[self.pattern],
                self.articulation_body.control_joint_limits_max_gpus[self.pattern],
                index_player_obj_to_env_mapping_gpu,
                self.articulation_body.view_object_indices_gpus[self.pattern],
                self.articulation_body.num_objects_env,
                self.view.count_per_world,
                step_total_rewards,
                term_buf,
                self.joint_rl_mask,
                self.num_joints,
                self.reward_scale,
            ],
            device=self.device,
        )

    @wp.kernel
    def calculate_gpu(
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        joint_qs: wp.array3d(dtype=wp.float32),
        joint_limits_min: wp.array(dtype=wp.float32),
        joint_limits_max: wp.array(dtype=wp.float32),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        view_object_indices: wp.array(dtype=int),
        num_objects_env: int,
        count_per_world: int,
        step_total_rewards: wp.array(dtype=wp.float32),
        term_buf: wp.array(dtype=wp.float32),
        joint_rl_mask: wp.array(dtype=wp.int32),
        num_joints: wp.int32,
        reward_scale: wp.float32,
    ):
        tid = wp.tid()
        shape_id = player_shape_ids_gpu[tid]
        world = index_player_obj_to_env_mapping_gpu[tid]
        local_idx = shape_id % num_objects_env
        obj_idx = _find_view_obj_idx(local_idx, view_object_indices, count_per_world)
        if obj_idx == -1:
            return

        penalty = wp.float32(0.0)
        for joint in range(num_joints):
            if joint_rl_mask[joint] == 0:
                continue
            pos = joint_qs[world, obj_idx, joint]
            lower = joint_limits_min[joint]
            upper = joint_limits_max[joint]
            below = pos - lower
            if below < 0.0:
                penalty += -below
            above = pos - upper
            if above > 0.0:
                penalty += above
        wp.atomic_add(step_total_rewards, shape_id, penalty * reward_scale)
        wp.atomic_add(term_buf, world, penalty * reward_scale)

    def reset(self, **kwargs):
        pass


class SelfCollisionCostReward(RewardComponent):
    """Count self-collision substeps where force exceeds threshold."""

    log_name = "self_collisions"

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.self_collision_sensor = None
        cfg = _component_params(self.params, "SelfCollisionCostReward")
        self.num_history = int(cfg.get("num_history", 4))
        self.force_threshold = float(cfg.get("force_threshold", 10.0))
        self.use_force_history = self.num_history > 0
        _init_mjlab_scaling(self, "SelfCollisionCostReward")

    def bind_level(self, level):
        """Bind the level's per-substep self-collision sensor (mjlab semantics).

        The sensor keeps a rolling ``force_history`` of the max robot-on-robot
        contact force over the last ``history_length`` substeps (index 0 = most
        recent), matching mjlab's ``self_collision`` ContactSensor with
        ``history_length`` set to the decimation value.
        """
        self.self_collision_sensor = getattr(level, "self_collision_sensor", None)

    def calculate(
        self,
        num_players,
        physics_manager,
        step_total_rewards: wp.array,
        player_shape_ids_gpu: wp.array,
        index_player_obj_to_env_mapping_gpu: wp.array,
        **kwargs,
    ):
        if self.self_collision_sensor is None:
            return

        term_buf, _ = _get_log_buffers(kwargs, self.log_name)
        if self.use_force_history:
            history = self.self_collision_sensor.force_history
            # Clamp to the sensor's actual width so a stale preset cannot index
            # out of bounds.
            num_history = min(self.num_history, self.self_collision_sensor.history_length)
            wp.launch(
                kernel=self.calculate_from_force_history_gpu,
                dim=num_players,
                inputs=[
                    player_shape_ids_gpu,
                    index_player_obj_to_env_mapping_gpu,
                    history,
                    step_total_rewards,
                    term_buf,
                    num_history,
                    self.force_threshold,
                    self.reward_scale,
                ],
                device=self.device,
            )
        else:
            found = self.self_collision_sensor.found
            wp.launch(
                kernel=self.calculate_from_found_gpu,
                dim=num_players,
                inputs=[
                    player_shape_ids_gpu,
                    index_player_obj_to_env_mapping_gpu,
                    found,
                    step_total_rewards,
                    term_buf,
                    self.reward_scale,
                ],
                device=self.device,
            )

    @wp.kernel
    def calculate_from_force_history_gpu(
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        force_history: wp.array2d(dtype=wp.float32),
        step_total_rewards: wp.array(dtype=wp.float32),
        term_buf: wp.array(dtype=wp.float32),
        num_history: wp.int32,
        force_threshold: wp.float32,
        reward_scale: wp.float32,
    ):
        tid = wp.tid()
        shape_id = player_shape_ids_gpu[tid]
        world = index_player_obj_to_env_mapping_gpu[tid]
        cost = wp.float32(0.0)
        for h in range(num_history):
            if force_history[world, h] > force_threshold:
                cost += wp.float32(1.0)
        wp.atomic_add(step_total_rewards, shape_id, cost * reward_scale)
        wp.atomic_add(term_buf, world, cost * reward_scale)

    @wp.kernel
    def calculate_from_found_gpu(
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        self_collision_found: wp.array(dtype=wp.int32),
        step_total_rewards: wp.array(dtype=wp.float32),
        term_buf: wp.array(dtype=wp.float32),
        reward_scale: wp.float32,
    ):
        tid = wp.tid()
        shape_id = player_shape_ids_gpu[tid]
        world = index_player_obj_to_env_mapping_gpu[tid]
        wp.atomic_add(step_total_rewards, shape_id, wp.float32(self_collision_found[world]) * reward_scale)
        wp.atomic_add(term_buf, world, wp.float32(self_collision_found[world]) * reward_scale)

    def reset(self, **kwargs):
        pass


class SoftLandingReward(RewardComponent):
    """Penalize high impact forces at first foot contact."""

    log_name = "soft_landing"
    metric_log_names = ("landing_force_mean",)

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.foot_sensor = None
        self.num_feet = 2
        cfg = _component_params(self.params, "SoftLandingReward")
        self.command_threshold = float(cfg.get("command_threshold", 0.05))
        _init_mjlab_scaling(self, "SoftLandingReward")

    def bind_level(self, level):
        _bind_g1_level_resources(self, level)

    def calculate(
        self,
        num_players,
        physics_manager,
        step_total_rewards: wp.array,
        player_shape_ids_gpu: wp.array,
        command_vel: wp.array,
        index_player_obj_to_env_mapping_gpu: wp.array,
        **kwargs,
    ):
        if self.foot_sensor is None:
            return

        term_buf, metric_bufs = _get_log_buffers(kwargs, self.log_name, self.metric_log_names)

        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_players,
            inputs=[
                player_shape_ids_gpu,
                self.foot_sensor.foot_force_z,
                self.foot_sensor.foot_first_contact,
                command_vel,
                index_player_obj_to_env_mapping_gpu,
                step_total_rewards,
                term_buf,
                metric_bufs[0],
                self.num_feet,
                self.command_threshold,
                self.reward_scale,

            ],
            device=self.device,
        )

    @wp.kernel
    def calculate_gpu(
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        contact_forces: wp.array2d(dtype=wp.float32),
        first_contact: wp.array2d(dtype=wp.int32),
        command_vel: wp.array2d(dtype=wp.float32),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        step_total_rewards: wp.array(dtype=wp.float32),
        term_buf: wp.array(dtype=wp.float32),
        landing_metric_buf: wp.array(dtype=wp.float32),
        num_feet: wp.int32,
        command_threshold: wp.float32,
        reward_scale: wp.float32,

    ):
        tid = wp.tid()
        shape_id = player_shape_ids_gpu[tid]
        world = index_player_obj_to_env_mapping_gpu[tid]
        cost = wp.float32(0.0)
        force_sum = wp.float32(0.0)
        force_count = wp.float32(0.0)
        for foot in range(num_feet):
            force_mag = contact_forces[world, foot]
            landed = wp.float32(first_contact[world, foot])
            cost += force_mag * landed
            # if landed > 0.0:
            force_sum += force_mag
            force_count += wp.float32(1.0)
        active = _command_active_func(command_vel, world, command_threshold)
        wp.atomic_add(step_total_rewards, shape_id, cost * active * reward_scale)
        wp.atomic_add(term_buf, world, cost * active * reward_scale)

        landing_metric_buf[world] = force_sum / wp.max(force_count, wp.float32(1.0))

    def reset(self, **kwargs):
        pass


class UprightReward(RewardComponent):
    """exp(-||projected_gravity_xy||² / std²) for keeping the body upright."""

    log_name = "upright"

    def __init__(self, device, pattern, **kwargs):
        super().__init__(**kwargs)
        _setup_g1_view_reward(self, device, pattern)
        cfg = _component_params(self.params, "UprightReward")
        self.std = float(cfg.get("std", 0.4472135954999579))
        self.std_sq = self.std * self.std
        gravity = cfg.get("gravity_vec", (0.0, 0.0, -1.0))
        self.gravity_vec = wp.vec3(float(gravity[0]), float(gravity[1]), float(gravity[2]))
        body_name = str(cfg.get("body_name", "torso_link"))
        self.body_link_idx = _resolve_view_link_index(self.view, body_name)
        _init_mjlab_scaling(self, "UprightReward")

    def calculate(
        self,
        num_players,
        physics_manager,
        step_total_rewards: wp.array,
        player_shape_ids_gpu: wp.array,
        index_player_obj_to_env_mapping_gpu: wp.array,
        **kwargs,
    ):
        pm = physics_manager
        link_tfs = self.view.get_link_transforms(pm.state_0)

        term_buf, _ = _get_log_buffers(kwargs, self.log_name)

        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_players,
            inputs=[
                player_shape_ids_gpu,
                link_tfs,
                index_player_obj_to_env_mapping_gpu,
                self.articulation_body.view_object_indices_gpus[self.pattern],
                self.articulation_body.num_objects_env,
                self.view.count_per_world,
                step_total_rewards,
                term_buf,
                self.body_link_idx,
                self.gravity_vec,
                self.std_sq,
                self.reward_scale,
            ],
            device=self.device,
        )

    @wp.kernel
    def calculate_gpu(
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        link_tfs: wp.array(dtype=wp.transform, ndim=3),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        view_object_indices: wp.array(dtype=int),
        num_objects_env: int,
        count_per_world: int,
        step_total_rewards: wp.array(dtype=wp.float32),
        term_buf: wp.array(dtype=wp.float32),
        body_link_idx: int,
        gravity_vec: wp.vec3,
        std_sq: wp.float32,
        reward_scale: wp.float32,
    ):
        tid = wp.tid()
        shape_id = player_shape_ids_gpu[tid]
        world = index_player_obj_to_env_mapping_gpu[tid]
        local_idx = shape_id % num_objects_env
        obj_idx = _find_view_obj_idx(local_idx, view_object_indices, count_per_world)
        if obj_idx == -1:
            return

        quat = wp.transform_get_rotation(link_tfs[world, obj_idx, body_link_idx])
        projected_gravity_b = wp.quat_rotate_inv(quat, gravity_vec)
        xy_squared = projected_gravity_b[0] * projected_gravity_b[0] + projected_gravity_b[1] * projected_gravity_b[1]
        reward_value = wp.exp(-xy_squared / std_sq) * reward_scale
        wp.atomic_add(step_total_rewards, shape_id, reward_value)
        wp.atomic_add(term_buf, world, reward_value)

    def reset(self, **kwargs):
        pass


class VariablePostureReward(RewardComponent):
    """exp(-mean((q - q_default)² / std(speed)²)) with speed-dependent tolerance."""

    log_name = "pose"

    def __init__(self, device, pattern: str, **kwargs):
        super().__init__(**kwargs)
        _setup_g1_view_reward(self, device, pattern)
        # Restrict pose / limit terms to RL-controlled DoFs (mjlab JOINT_NAMES,
        # 29 on G1). Hand joints stay in the view arrays but must not enter the
        # mean / penalty — they have no corresponding mjlab reward term.
        joint_names = self.view.joint_dof_names
        self.num_joints = len(joint_names)
        self.joint_rl_mask = self.articulation_body.control_joint_rl_mask_gpus[self.pattern]
        posture_cfg = _component_params(self.params, "VariablePostureReward")
        self.walking_threshold = float(posture_cfg.get("walking_threshold", 0.05))
        self.running_threshold = float(posture_cfg.get("running_threshold", 1.5))
        self.std_standing = wp.array(
            resolve_joint_stds(posture_cfg["std_standing"], joint_names),
            dtype=wp.float32,
            device=device,
        )
        self.std_walking = wp.array(
            resolve_joint_stds(posture_cfg["std_walking"], joint_names),
            dtype=wp.float32,
            device=device,
        )
        self.std_running = wp.array(
            resolve_joint_stds(posture_cfg["std_running"], joint_names),
            dtype=wp.float32,
            device=device,
        )
        _init_mjlab_scaling(self, "VariablePostureReward")

    def calculate(
        self,
        num_players,
        physics_manager,
        step_total_rewards: wp.array,
        player_shape_ids_gpu: wp.array,
        command_vel: wp.array,
        index_player_obj_to_env_mapping_gpu: wp.array,
        **kwargs,
    ):
        pm = physics_manager
        joint_qs = self.view.get_dof_positions(pm.state_0)

        term_buf, _ = _get_log_buffers(kwargs, self.log_name)

        wp.launch(
            kernel=self.calculate_gpu,
            dim=num_players,
            inputs=[
                player_shape_ids_gpu,
                joint_qs,
                self.articulation_body.control_joint_nominal_qs_gpus[self.pattern],
                command_vel,
                index_player_obj_to_env_mapping_gpu,
                self.articulation_body.view_object_indices_gpus[self.pattern],
                self.articulation_body.num_objects_env,
                self.view.count_per_world,
                self.std_standing,
                self.std_walking,
                self.std_running,
                step_total_rewards,
                term_buf,
                self.joint_rl_mask,
                self.num_joints,
                self.walking_threshold,
                self.running_threshold,
                self.reward_scale,
            ],
            device=self.device,
        )

    @wp.kernel
    def calculate_gpu(
        player_shape_ids_gpu: wp.array(dtype=wp.int32),
        joint_qs: wp.array3d(dtype=wp.float32),
        default_joint_pos: wp.array(dtype=wp.float32),
        command_vel: wp.array2d(dtype=wp.float32),
        index_player_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32),
        view_object_indices: wp.array(dtype=int),
        num_objects_env: int,
        count_per_world: int,
        std_standing: wp.array(dtype=wp.float32),
        std_walking: wp.array(dtype=wp.float32),
        std_running: wp.array(dtype=wp.float32),
        step_total_rewards: wp.array(dtype=wp.float32),
        term_buf: wp.array(dtype=wp.float32),
        joint_rl_mask: wp.array(dtype=wp.int32),
        num_joints: wp.int32,
        walking_threshold: wp.float32,
        running_threshold: wp.float32,
        reward_scale: wp.float32,
    ):
        tid = wp.tid()
        shape_id = player_shape_ids_gpu[tid]
        world = index_player_obj_to_env_mapping_gpu[tid]
        local_idx = shape_id % num_objects_env
        obj_idx = _find_view_obj_idx(local_idx, view_object_indices, count_per_world)
        if obj_idx == -1:
            return

        lin_x = command_vel[world, 0]
        lin_y = command_vel[world, 1]
        ang_z = command_vel[world, 2]
        linear_speed = wp.sqrt(lin_x * lin_x + lin_y * lin_y)
        total_speed = linear_speed + wp.abs(ang_z)
        mean_term = wp.float32(0.0)
        n_rl = wp.float32(0.0)
        for joint in range(num_joints):
            if joint_rl_mask[joint] == 0:
                continue
            error = joint_qs[world, obj_idx, joint] - default_joint_pos[joint]
            error_sq = error * error
            if total_speed < walking_threshold:
                std_val = std_standing[joint]
            elif total_speed < running_threshold:
                std_val = std_walking[joint]
            else:
                std_val = std_running[joint]
            std_sq = std_val * std_val
            mean_term += error_sq / std_sq
            n_rl += wp.float32(1.0)
        if n_rl > 0.0:
            mean_term /= n_rl
        reward_value = wp.exp(-mean_term) * reward_scale
        wp.atomic_add(step_total_rewards, shape_id, reward_value)
        wp.atomic_add(term_buf, world, reward_value)

    def reset(self, **kwargs):
        pass
