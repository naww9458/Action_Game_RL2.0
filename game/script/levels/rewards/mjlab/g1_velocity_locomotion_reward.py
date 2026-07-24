# This file contains code adapted from:
# https://github.com/mujocolab/mjlab
#
# Modified for Action_Game_RL.
#
# The original project is licensed under the Apache License 2.0.

import math
import os
import warp as wp
import numpy as np

from ..reward_calculator import RewardComponent

# Per-term debug indices (used when G1_DEBUG_NAN=1)
DEBUG_TERM_TRACK_LIN = 0
DEBUG_TERM_TRACK_ANG = 1
DEBUG_TERM_UPRIGHT = 2
DEBUG_TERM_POSTURE = 3
DEBUG_TERM_BODY_ANG = 4
DEBUG_TERM_ANG_MOM = 5
DEBUG_TERM_LIMIT_PEN = 6
DEBUG_TERM_ACTION_RATE = 7
DEBUG_TERM_FOOT_COSTS = 8
DEBUG_TERM_TOTAL = 9
DEBUG_TERM_COUNT = 10

# Sanity bounds for unbounded penalty terms
MAX_ANG_VEL_SQ = 2500.0       # ~50 rad/s
MAX_LIN_VEL_XY_SQ = 100.0     # ~10 m/s
MAX_LIMIT_PEN = 100.0
MAX_FOOT_COST = 50.0
MAX_ACTION_RATE = 10.0        # cap action-rate penalty so high-std exploration can't dominate
MAX_TOTAL_REWARD = 50.0


@wp.func
def exp_reward(err_sq: float, std: float) -> float:
    return wp.exp(-err_sq / (std * std))


@wp.func
def clampf(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@wp.kernel
def calculate_g1_velocity_locomotion(
    root_tfs: wp.array2d(dtype=wp.transform),
    root_vels: wp.array2d(dtype=wp.spatial_vector),
    joint_qs: wp.array3d(dtype=float),
    joint_qds: wp.array3d(dtype=float),
    joint_nominal_qs: wp.array(dtype=float),
    joint_limits_max: wp.array(dtype=float),
    joint_limits_min: wp.array(dtype=float),
    joint_rl_mask: wp.array(dtype=wp.int32),
    joint_rl_action_indices: wp.array(dtype=wp.int32),
    commands: wp.array2d(dtype=float),
    actions: wp.array2d(dtype=float),
    prev_actions: wp.array2d(dtype=float),
    action_offset: int,
    rl_action_dim: int,
    foot_found: wp.array2d(dtype=wp.int32),
    foot_height: wp.array2d(dtype=float),
    foot_lin_vel_xy_sq: wp.array2d(dtype=float),
    foot_peak_height: wp.array2d(dtype=float),
    foot_first_contact: wp.array2d(dtype=wp.int32),
    foot_force_z: wp.array2d(dtype=float),
    env_players_index_offset: wp.array(dtype=wp.int32),
    player_shape_ids_gpu: wp.array(dtype=wp.int32),
    player_health: wp.array(dtype=wp.float32),
    step_total_rewards: wp.array(dtype=wp.float32),
    debug_terms: wp.array2d(dtype=float),
    joint_dof_count: int,
    std_track_lin: float,
    std_track_ang: float,
    std_upright: float,
    std_standing: float,
    std_walking: float,
    w_track_lin: float,
    w_track_ang: float,
    w_upright: float,
    w_posture: float,
    w_body_ang_vel: float,
    w_ang_momentum: float,
    w_dof_limits: float,
    w_action_rate: float,
    w_foot_clearance: float,
    w_foot_swing: float,
    w_foot_slip: float,
    w_soft_landing: float,
    w_alive: float,
    cmd_threshold: float,
    target_swing_height: float,
    enable_debug: int,
):
    env = wp.tid()

    player_idx = env_players_index_offset[env]
    shape_id = player_shape_ids_gpu[player_idx]
    if player_health[shape_id] <= 0.0:
        return

    my_tf = root_tfs[env, 0]
    my_rot = wp.transform_get_rotation(my_tf)
    root_qd = root_vels[env, 0]

    inv_rot = wp.quat_inverse(my_rot)
    world_lin = wp.vec3(root_qd[0], root_qd[1], root_qd[2])
    world_ang = wp.vec3(root_qd[3], root_qd[4], root_qd[5])
    local_lin = wp.quat_rotate(inv_rot, world_lin)
    local_ang = wp.quat_rotate(inv_rot, world_ang)

    cmd_vx = commands[env, 0]
    cmd_vy = commands[env, 1]
    cmd_wz = commands[env, 2]

    lin_xy_err = (cmd_vx - local_lin[0]) * (cmd_vx - local_lin[0]) + (cmd_vy - local_lin[1]) * (cmd_vy - local_lin[1])
    lin_z_err = local_lin[2] * local_lin[2]
    r_track_lin = exp_reward(lin_xy_err + lin_z_err, std_track_lin)

    ang_xy_err = local_ang[0] * local_ang[0] + local_ang[1] * local_ang[1]
    ang_z_err = (cmd_wz - local_ang[2]) * (cmd_wz - local_ang[2])
    r_track_ang = exp_reward(ang_z_err + ang_xy_err, std_track_ang)

    grav = wp.quat_rotate(inv_rot, wp.vec3(0.0, 0.0, -1.0))
    tilt_sq = grav[0] * grav[0] + grav[1] * grav[1]
    r_upright = exp_reward(tilt_sq, std_upright)

    cmd_speed = wp.abs(cmd_vx) + wp.abs(cmd_vy) + wp.abs(cmd_wz)
    posture_std = std_standing
    if cmd_speed >= 1.5:
        posture_std = std_walking
    elif cmd_speed >= 0.05:
        posture_std = std_walking

    posture_err = wp.float32(0.0)
    posture_count = wp.float32(0.0)
    for d in range(joint_dof_count):
        if joint_rl_mask[d] == 0:
            continue
        q_err = joint_qs[env, 0, d] - joint_nominal_qs[d]
        posture_err = posture_err + q_err * q_err
        posture_count = posture_count + 1.0
    if posture_count > 0.0:
        posture_err = posture_err / posture_count
    r_posture = exp_reward(posture_err, posture_std)

    ang_xy_sq = clampf(
        local_ang[0] * local_ang[0] + local_ang[1] * local_ang[1],
        0.0,
        MAX_ANG_VEL_SQ,
    )
    ang_mom_sq = clampf(
        local_ang[0] * local_ang[0] + local_ang[1] * local_ang[1] + local_ang[2] * local_ang[2],
        0.0,
        MAX_ANG_VEL_SQ,
    )
    r_body_ang = -ang_xy_sq
    r_ang_mom = -ang_mom_sq

    limit_pen = wp.float32(0.0)
    for d in range(joint_dof_count):
        q = joint_qs[env, 0, d]
        if q > joint_limits_max[d]:
            limit_pen = limit_pen + (q - joint_limits_max[d]) * (q - joint_limits_max[d])
        if q < joint_limits_min[d]:
            limit_pen = limit_pen + (q - joint_limits_min[d]) * (q - joint_limits_min[d])
    limit_pen = clampf(limit_pen, 0.0, MAX_LIMIT_PEN)

    action_rate = wp.float32(0.0)
    for a in range(rl_action_dim):
        diff = actions[env, action_offset + a] - prev_actions[env, a]
        action_rate = action_rate + diff * diff
    action_rate = clampf(action_rate, 0.0, MAX_ACTION_RATE)

    cmd_active = 0.0
    if cmd_speed > cmd_threshold:
        cmd_active = 1.0

    foot_clearance_cost = 0.0
    foot_slip_cost = 0.0
    foot_swing_cost = 0.0
    soft_landing_cost = 0.0
    for f in range(2):
        vel_xy_sq = clampf(foot_lin_vel_xy_sq[env, f], 0.0, MAX_LIN_VEL_XY_SQ)
        h_err = wp.abs(foot_height[env, f] - target_swing_height)
        foot_clearance_cost = foot_clearance_cost + h_err * vel_xy_sq
        if foot_found[env, f] == 1:
            foot_slip_cost = foot_slip_cost + vel_xy_sq
        if foot_first_contact[env, f] == 1:
            ph = foot_peak_height[env, f]
            if target_swing_height > 0.0:
                swing_err = ph / target_swing_height - 1.0
                foot_swing_cost = foot_swing_cost + swing_err * swing_err
            soft_landing_cost = soft_landing_cost + foot_force_z[env, f]
    foot_clearance_cost = clampf(foot_clearance_cost, 0.0, MAX_FOOT_COST)
    foot_slip_cost = clampf(foot_slip_cost, 0.0, MAX_FOOT_COST)
    foot_swing_cost = clampf(foot_swing_cost, 0.0, MAX_FOOT_COST)
    soft_landing_cost = clampf(soft_landing_cost, 0.0, MAX_FOOT_COST)

    term_track_lin = w_track_lin * r_track_lin
    term_track_ang = w_track_ang * r_track_ang
    term_upright = w_upright * r_upright
    term_posture = w_posture * r_posture
    term_body_ang = w_body_ang_vel * r_body_ang
    term_ang_mom = w_ang_momentum * r_ang_mom
    term_limit_pen = w_dof_limits * limit_pen
    term_action_rate = w_action_rate * action_rate
    term_foot_costs = cmd_active * (
        w_foot_clearance * foot_clearance_cost
        + w_foot_swing * foot_swing_cost
        + w_foot_slip * foot_slip_cost
        + w_soft_landing * soft_landing_cost
    )

    # Constant per-step alive bonus: guarantees that staying upright is net-positive
    # so PPO cannot exploit immediate termination (the "learned to give up" failure).
    term_alive = w_alive

    total = (
        term_track_lin
        + term_track_ang
        + term_upright
        + term_posture
        + term_body_ang
        + term_ang_mom
        + term_limit_pen
        + term_action_rate
        + term_foot_costs
        + term_alive
    )
    total = clampf(total, -MAX_TOTAL_REWARD, MAX_TOTAL_REWARD)
    step_total_rewards[shape_id] += total

    if enable_debug == 1:
        debug_terms[env, DEBUG_TERM_TRACK_LIN] = term_track_lin
        debug_terms[env, DEBUG_TERM_TRACK_ANG] = term_track_ang
        debug_terms[env, DEBUG_TERM_UPRIGHT] = term_upright
        debug_terms[env, DEBUG_TERM_POSTURE] = term_posture
        debug_terms[env, DEBUG_TERM_BODY_ANG] = term_body_ang
        debug_terms[env, DEBUG_TERM_ANG_MOM] = term_ang_mom
        debug_terms[env, DEBUG_TERM_LIMIT_PEN] = term_limit_pen
        debug_terms[env, DEBUG_TERM_ACTION_RATE] = term_action_rate
        debug_terms[env, DEBUG_TERM_FOOT_COSTS] = term_foot_costs
        debug_terms[env, DEBUG_TERM_TOTAL] = total


class G1VelocityLocomotionReward(RewardComponent):
    DEBUG_TERM_NAMES = [
        "track_lin", "track_ang", "upright", "posture",
        "body_ang", "ang_mom", "limit_pen", "action_rate",
        "foot_costs", "total",
    ]

    def __init__(self, device, num_max_players, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        cfg = self.params.get("G1VelocityLocomotionReward", self.params)

        self.w_track_lin = cfg.get("w_track_lin", 2.0)
        self.w_track_ang = cfg.get("w_track_ang", 2.0)
        self.w_upright = cfg.get("w_upright", 1.0)
        self.w_posture = cfg.get("w_posture", 1.0)
        self.w_body_ang_vel = cfg.get("w_body_ang_vel", -0.05)
        self.w_ang_momentum = cfg.get("w_ang_momentum", -0.02)
        self.w_dof_limits = cfg.get("w_dof_limits", -1.0)
        self.w_action_rate = cfg.get("w_action_rate", -0.1)
        self.w_foot_clearance = cfg.get("w_foot_clearance", -2.0)
        self.w_foot_swing = cfg.get("w_foot_swing", -0.25)
        self.w_foot_slip = cfg.get("w_foot_slip", -0.1)
        self.w_soft_landing = cfg.get("w_soft_landing", -1.0e-5)
        self.w_alive = cfg.get("w_alive", 0.5)
        self.std_track_lin = cfg.get("std_track_lin", math.sqrt(0.25))
        self.std_track_ang = cfg.get("std_track_ang", math.sqrt(0.5))
        self.std_upright = cfg.get("std_upright", math.sqrt(0.2))
        self.std_standing = cfg.get("std_standing", 0.05)
        self.std_walking = cfg.get("std_walking", 0.15)
        self.cmd_threshold = cfg.get("cmd_threshold", 0.05)
        self.target_swing_height = cfg.get("target_swing_height", 0.1)

        pattern = kwargs.get("pattern")
        if not pattern:
            raise ValueError("G1VelocityLocomotionReward requires pattern from level config.")
        self.pattern = pattern

        self.view_idx = next(
            (i for i, p in enumerate(self.articulation_body.patterns) if p == self.pattern), -1
        )
        self.view = self.articulation_body.views[self.view_idx]
        self.joint_dof_count = self.view.joint_dof_count
        self.rl_action_dim = self.articulation_body.control_rl_action_dim.get(self.pattern, 29)
        self.action_offset = 0

        self.commands_wp = None
        self.prev_actions_wp = None
        self.foot_sensor = None
        self.num_env = 0
        self.debug_enabled = os.environ.get("G1_DEBUG_NAN", "0") == "1"
        self.debug_terms = None

    def bind_level(self, level):
        self.commands_wp = level.commands
        self.prev_actions_wp = level.prev_actions
        self.foot_sensor = level.foot_sensor
        self.num_env = level.num_env
        self.action_offset = 0
        if self.debug_enabled and self.debug_terms is None:
            self.debug_terms = wp.zeros(
                (self.num_env, DEBUG_TERM_COUNT), dtype=wp.float32, device=self.device
            )

    def calculate(self, num_players, physics_manager, player_shape_ids_gpu, actions,
                  env_players_index_offset, player_health, step_total_rewards, **kwargs):
        if self.commands_wp is None or self.foot_sensor is None:
            return

        pm = physics_manager
        root_tfs = self.view.get_root_transforms(pm.state_0)
        root_vels = self.view.get_root_velocities(pm.state_0)
        joint_qs = self.view.get_dof_positions(pm.state_0)
        joint_qds = self.view.get_dof_velocities(pm.state_0)

        prev = self.prev_actions_wp
        if prev is None:
            prev = wp.zeros((self.num_env, self.rl_action_dim), dtype=wp.float32, device=self.device)

        debug_terms = self.debug_terms
        if debug_terms is None:
            debug_terms = wp.zeros((self.num_env, DEBUG_TERM_COUNT), dtype=wp.float32, device=self.device)

        wp.launch(
            kernel=calculate_g1_velocity_locomotion,
            dim=self.num_env,
            inputs=[
                root_tfs,
                root_vels,
                joint_qs,
                joint_qds,
                self.articulation_body.control_joint_nominal_qs_gpus[self.pattern],
                self.articulation_body.control_joint_limits_max_gpus[self.pattern],
                self.articulation_body.control_joint_limits_min_gpus[self.pattern],
                self.articulation_body.control_joint_rl_mask_gpus[self.pattern],
                self.articulation_body.control_joint_rl_action_indices_gpus[self.pattern],
                self.commands_wp,
                actions,
                prev,
                self.action_offset,
                self.rl_action_dim,
                self.foot_sensor.foot_found,
                self.foot_sensor.foot_height,
                self.foot_sensor.foot_lin_vel_xy_sq,
                self.foot_sensor.foot_peak_height,
                self.foot_sensor.foot_first_contact,
                self.foot_sensor.foot_force_z,
                env_players_index_offset,
                player_shape_ids_gpu,
                player_health,
                step_total_rewards,
                debug_terms,
                self.joint_dof_count,
                self.std_track_lin,
                self.std_track_ang,
                self.std_upright,
                self.std_standing,
                self.std_walking,
                self.w_track_lin,
                self.w_track_ang,
                self.w_upright,
                self.w_posture,
                self.w_body_ang_vel,
                self.w_ang_momentum,
                self.w_dof_limits,
                self.w_action_rate,
                self.w_foot_clearance,
                self.w_foot_swing,
                self.w_foot_slip,
                self.w_soft_landing,
                self.w_alive,
                self.cmd_threshold,
                self.target_swing_height,
                1 if self.debug_enabled else 0,
            ],
            device=self.device,
        )

    def reset(self, **kwargs):
        pass
