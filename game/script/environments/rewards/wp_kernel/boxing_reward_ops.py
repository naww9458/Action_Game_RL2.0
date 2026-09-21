"""Warp kernels for G1 boxing reward formulas.

Reward decomposition (weights come from ``models/boxing_v1/control_configs.yaml``):

  r_total  = r_walk + r_boxing
  r_boxing = r_distance + r_ready_pose + r_punch + r_retract + r_hit + r_pose

Every boxing kernel short-circuits to 0 when ``has_target == 0`` (pure-walk
environments).

Hard-state gating (punch_phase FSM): the env keeps a boolean ``hard_state``
(0 = retract, 1 = punch) plus a continuous ``signal`` (0..1). Period rewards
gate on the hard state and blend with the signal:

  r_ready_pose : retract period (hard_state==0), scaled by (1 - signal)
  r_retract    : retract period (hard_state==0), scaled by (1 - signal)
  r_punch      : punch period (hard_state==1), scaled by signal
  r_hit        : punch period (hard_state==1), scaled by signal
  r_distance   : always on
  r_pose       : always on (torque / tilt regularizer)

Task-layer reward classes stay free of ``@wp.kernel`` / Newton imports; they
call these adapters with YAML constants and Torch/Warp tensor views so the
work stays on the physics stream (CUDA-graph replay).
"""

from __future__ import annotations

from typing import Any, Optional

import warp as wp

from script.role.objects.object_template.mjlab_unitree_g1.models.boxing_v1.state_query import (
    scatter_env_reward,
)


@wp.func
def _feasibility_fn(
    dist: float,
    standoff: float,
    band: float,
    mode: int,
    has_target: int,
) -> float:
    """distance_feasibility: 1 at standoff, decays to 0 across the band."""
    if has_target == 0:
        return 0.0
    ratio = wp.abs(dist - standoff) / band
    if mode == 1:
        return wp.exp(-ratio * ratio)
    return 1.0 - wp.clamp(ratio, 0.0, 1.0)


@wp.func
def _xy_dist(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return wp.sqrt(dx * dx + dy * dy)


@wp.func
def _ready_pose_error(
    joint_q: wp.array2d(dtype=float),
    joint_indices: wp.array(dtype=wp.int32),
    joint_poses: wp.array(dtype=float),
    joint_weights: wp.array(dtype=float),
    n_joints: int,
    tid: int,
) -> float:
    """Weighted absolute joint error against the configured ready pose."""
    err = float(0.0)
    for j in range(n_joints):
        idx = joint_indices[j]
        dq = joint_q[tid, idx] - joint_poses[j]
        err += wp.abs(dq) * joint_weights[j]
    return err


@wp.func
def _vel_toward_target(
    hx: float,
    hy: float,
    hz: float,
    hvx: float,
    hvy: float,
    hvz: float,
    tx: float,
    ty: float,
    tz: float,
) -> float:
    """Hand linear speed along the vector from hand to target (m/s)."""
    dx = tx - hx
    dy = ty - hy
    dz = tz - hz
    dir_len = wp.sqrt(dx * dx + dy * dy + dz * dz)
    if dir_len <= 1.0e-8:
        return 0.0
    return (hvx * dx + hvy * dy + hvz * dz) / dir_len


@wp.kernel
def _distance_reward_kernel(
    pelvis_pos: wp.array2d(dtype=float),
    target_pos: wp.array2d(dtype=float),
    has_target: wp.array(dtype=wp.int32),
    standoff: float,
    band: float,
    mode: int,
    base_weight: float,
    penalty_weight: float,
    raw: wp.array(dtype=float),
):
    tid = wp.tid()
    if has_target[tid] == 0:
        raw[tid] = 0.0
        return
    px = pelvis_pos[tid, 0]
    py = pelvis_pos[tid, 1]
    tx = target_pos[tid, 0]
    ty = target_pos[tid, 1]
    if (
        (not wp.isfinite(px))
        or (not wp.isfinite(py))
        or (not wp.isfinite(tx))
        or (not wp.isfinite(ty))
    ):
        raw[tid] = 0.0
        return
    dist = _xy_dist(px, py, tx, ty)
    feasibility = _feasibility_fn(dist, standoff, band, mode, 1)
    if feasibility > 0.0:
        raw[tid] = feasibility * base_weight
    else:
        raw[tid] = -wp.abs(dist - standoff) * penalty_weight


@wp.kernel
def _ready_pose_kernel(
    has_target: wp.array(dtype=wp.int32),
    joint_q: wp.array2d(dtype=float),
    joint_indices: wp.array(dtype=wp.int32),
    joint_poses: wp.array(dtype=float),
    joint_weights: wp.array(dtype=float),
    n_joints: int,
    hard_state: wp.array(dtype=wp.int32),
    signal: wp.array(dtype=float),
    ready_threshold: float,
    error_scale: float,
    weight: float,
    ready_bonus: float,
    raw: wp.array(dtype=float),
):
    """Guard-pose convergence; retract period, blended by (1 - signal)."""
    tid = wp.tid()
    if has_target[tid] == 0:
        raw[tid] = 0.0
        return
    if hard_state[tid] != 0:
        raw[tid] = 0.0
        return
    err = _ready_pose_error(joint_q, joint_indices, joint_poses, joint_weights, n_joints, tid)
    if not wp.isfinite(err):
        raw[tid] = 0.0
        return
    base = 1.0 - err / error_scale
    if base < 0.0:
        base = 0.0
    r = base * weight
    if err < ready_threshold:
        r = r + ready_bonus
    r = r * (1.0 - signal[tid])
    raw[tid] = r


@wp.kernel
def _punch_reward_kernel(
    pelvis_pos: wp.array2d(dtype=float),
    pelvis_vel: wp.array2d(dtype=float),
    pelvis_yaw: wp.array(dtype=float),
    target_pos: wp.array2d(dtype=float),
    hand_pos: wp.array2d(dtype=float),
    hand_vel: wp.array2d(dtype=float),
    hand_vel_prev: wp.array2d(dtype=float),
    hand_fwd_hist: wp.array2d(dtype=float),
    hist_pos: int,
    has_target: wp.array(dtype=wp.int32),
    hard_state: wp.array(dtype=wp.int32),
    signal: wp.array(dtype=float),
    joint_q: wp.array2d(dtype=float),
    joint_indices: wp.array(dtype=wp.int32),
    joint_poses: wp.array(dtype=float),
    joint_weights: wp.array(dtype=float),
    n_joints: int,
    dt: float,
    standoff: float,
    band: float,
    mode: int,
    gate_feasibility: float,
    ready_threshold: float,
    planar_min_speed: float,
    planar_dominance_ratio: float,
    xy_weight: float,
    acc_weight: float,
    full_swing_min_advance: float,
    full_swing_acc_scale: float,
    unprepared_scale: float,
    raw: wp.array(dtype=float),
):
    """C1 planar-velocity term + C2 hand-acceleration-toward-target term.

    Punch period (hard_state==1), blended by signal. C2 gets a
    ``full_swing_acc_scale`` multiplier when the hand advanced at least
    ``full_swing_min_advance`` meters (pelvis yaw frame) within the ring-buffer
    window, and is reduced by ``unprepared_scale`` when the hand left the ready
    pose before the punch fired.
    """
    tid = wp.tid()
    hist_len = hand_fwd_hist.shape[1]

    px = pelvis_pos[tid, 0]
    py = pelvis_pos[tid, 1]
    hx = hand_pos[tid, 0]
    hy = hand_pos[tid, 1]
    hz = hand_pos[tid, 2]
    hvx = hand_vel[tid, 0]
    hvy = hand_vel[tid, 1]
    hvz = hand_vel[tid, 2]
    tx = target_pos[tid, 0]
    ty = target_pos[tid, 1]
    tz = target_pos[tid, 2]

    # Hand forward offset in the pelvis yaw frame, written into the ring buffer.
    yaw = pelvis_yaw[tid]
    cy = wp.cos(yaw)
    sy = wp.sin(yaw)
    fwd_x = cy * (hx - px) + sy * (hy - py)
    hand_fwd_hist[tid, hist_pos] = fwd_x
    oldest = hand_fwd_hist[tid, (hist_pos + 1) % hist_len]

    # Hand acceleration toward the target (3D, hand-based punch direction).
    dtx = tx - hx
    dty = ty - hy
    dtz = tz - hz
    dir_len = wp.sqrt(dtx * dtx + dty * dty + dtz * dtz)
    inv_len = 0.0
    if dir_len > 1.0e-8:
        inv_len = 1.0 / dir_len
    ax = (hvx - hand_vel_prev[tid, 0]) / dt
    ay = (hvy - hand_vel_prev[tid, 1]) / dt
    az = (hvz - hand_vel_prev[tid, 2]) / dt
    acc_proj = wp.max(0.0, (ax * dtx + ay * dty + az * dtz) * inv_len)
    hand_vel_prev[tid, 0] = hvx
    hand_vel_prev[tid, 1] = hvy
    hand_vel_prev[tid, 2] = hvz

    if (
        (not wp.isfinite(px))
        or (not wp.isfinite(py))
        or (not wp.isfinite(hx))
        or (not wp.isfinite(hy))
        or (not wp.isfinite(hz))
        or (not wp.isfinite(hvx))
        or (not wp.isfinite(hvy))
        or (not wp.isfinite(hvz))
        or (not wp.isfinite(tx))
        or (not wp.isfinite(ty))
        or (not wp.isfinite(tz))
    ):
        raw[tid] = 0.0
        return
    if has_target[tid] == 0:
        raw[tid] = 0.0
        return
    if hard_state[tid] != 1:
        raw[tid] = 0.0
        return

    # Relative hand velocity in the pelvis yaw frame (matches obs extras).
    rvx = hvx - pelvis_vel[tid, 0]
    rvy = hvy - pelvis_vel[tid, 1]
    rvz = hvz - pelvis_vel[tid, 2]
    v_bx = cy * rvx + sy * rvy
    v_by = -sy * rvx + cy * rvy
    v_xy = wp.sqrt(v_bx * v_bx + v_by * v_by)
    v_z = wp.abs(rvz)

    dist = _xy_dist(px, py, tx, ty)
    feasibility = _feasibility_fn(dist, standoff, band, mode, 1)
    if feasibility <= gate_feasibility:
        raw[tid] = 0.0
        return

    err = _ready_pose_error(joint_q, joint_indices, joint_poses, joint_weights, n_joints, tid)
    if not wp.isfinite(err):
        raw[tid] = 0.0
        return

    if oldest < fwd_x - full_swing_min_advance:
        acc_proj = acc_proj * full_swing_acc_scale

    # Unprepared reduction: punch fired without the ready pose -> cut the acc term.
    if err >= ready_threshold:
        acc_proj = acc_proj * unprepared_scale

    r = 0.0
    if v_xy > planar_dominance_ratio * v_z and v_xy > planar_min_speed:
        r = r + v_xy * xy_weight
    r = r + acc_proj * acc_weight
    r = r * signal[tid]
    raw[tid] = r


@wp.kernel
def _retract_reward_kernel(
    pelvis_pos: wp.array2d(dtype=float),
    target_pos: wp.array2d(dtype=float),
    hand_pos: wp.array2d(dtype=float),
    hand_vel: wp.array2d(dtype=float),
    has_target: wp.array(dtype=wp.int32),
    hard_state: wp.array(dtype=wp.int32),
    signal: wp.array(dtype=float),
    standoff: float,
    band: float,
    mode: int,
    gate_feasibility: float,
    vel_weight: float,
    fake_punch_vel_threshold: float,
    fake_punch_penalty: float,
    ideal_away_speed: float,
    away_sigma: float,
    over_speed_threshold: float,
    over_speed_penalty: float,
    raw: wp.array(dtype=float),
):
    """Slow-away Gaussian + over-speed penalty + fake-punch penalty; retract period."""
    tid = wp.tid()
    if has_target[tid] == 0:
        raw[tid] = 0.0
        return
    if hard_state[tid] != 0:
        raw[tid] = 0.0
        return
    px = pelvis_pos[tid, 0]
    py = pelvis_pos[tid, 1]
    tx = target_pos[tid, 0]
    ty = target_pos[tid, 1]
    tz = target_pos[tid, 2]
    hx = hand_pos[tid, 0]
    hy = hand_pos[tid, 1]
    hz = hand_pos[tid, 2]
    hvx = hand_vel[tid, 0]
    hvy = hand_vel[tid, 1]
    hvz = hand_vel[tid, 2]
    if (
        (not wp.isfinite(px))
        or (not wp.isfinite(py))
        or (not wp.isfinite(tx))
        or (not wp.isfinite(ty))
        or (not wp.isfinite(tz))
        or (not wp.isfinite(hx))
        or (not wp.isfinite(hy))
        or (not wp.isfinite(hz))
        or (not wp.isfinite(hvx))
        or (not wp.isfinite(hvy))
        or (not wp.isfinite(hvz))
    ):
        raw[tid] = 0.0
        return
    dist = _xy_dist(px, py, tx, ty)
    feasibility = _feasibility_fn(dist, standoff, band, mode, 1)
    vel_toward = _vel_toward_target(hx, hy, hz, hvx, hvy, hvz, tx, ty, tz)

    ax = hx - tx
    ay = hy - ty
    az = hz - tz
    away_len = wp.sqrt(ax * ax + ay * ay + az * az)
    vel_away = 0.0
    if away_len > 1.0e-8:
        vel_away = (hvx * ax + hvy * ay + hvz * az) / away_len

    # Slow-away bell curve: peak at ideal_away_speed, decays too fast or too slow.
    diff = vel_away - ideal_away_speed
    r = vel_weight * wp.exp(-(diff * diff) / (away_sigma * away_sigma))

    # Over-speed penalty: proportional to how far vel_away exceeds the cap.
    overspeed = vel_away - over_speed_threshold
    if overspeed > 0.0:
        r = r - overspeed * over_speed_penalty

    # Fake-punch only outside the punch distance band (too far to score a punch).
    if feasibility <= gate_feasibility and vel_toward > fake_punch_vel_threshold:
        r = r - fake_punch_penalty * vel_toward

    r = r * (1.0 - signal[tid])
    raw[tid] = r


@wp.kernel
def _hit_reward_kernel(
    pelvis_pos: wp.array2d(dtype=float),
    target_pos: wp.array2d(dtype=float),
    hand_vel: wp.array2d(dtype=float),
    target_vel: wp.array2d(dtype=float),
    has_target: wp.array(dtype=wp.int32),
    contact_found: wp.array(dtype=wp.int32),
    hard_state: wp.array(dtype=wp.int32),
    signal: wp.array(dtype=float),
    standoff: float,
    band: float,
    mode: int,
    gate_feasibility: float,
    velocity_threshold: float,
    base: float,
    velocity_bonus: float,
    max_hand_vel: float,
    raw: wp.array(dtype=float),
):
    """Contact hit at standoff; punch period (hard_state==1), blended by signal."""
    tid = wp.tid()
    if has_target[tid] == 0 or contact_found[tid] == 0:
        raw[tid] = 0.0
        return
    if hard_state[tid] != 1:
        raw[tid] = 0.0
        return
    px = pelvis_pos[tid, 0]
    py = pelvis_pos[tid, 1]
    tx = target_pos[tid, 0]
    ty = target_pos[tid, 1]
    if (
        (not wp.isfinite(px))
        or (not wp.isfinite(py))
        or (not wp.isfinite(tx))
        or (not wp.isfinite(ty))
    ):
        raw[tid] = 0.0
        return
    dist = _xy_dist(px, py, tx, ty)
    feasibility = _feasibility_fn(dist, standoff, band, mode, 1)
    if feasibility <= gate_feasibility:
        raw[tid] = 0.0
        return

    rvx = hand_vel[tid, 0] - target_vel[tid, 0]
    rvy = hand_vel[tid, 1] - target_vel[tid, 1]
    rvz = hand_vel[tid, 2] - target_vel[tid, 2]
    if (not wp.isfinite(rvx)) or (not wp.isfinite(rvy)) or (not wp.isfinite(rvz)):
        raw[tid] = 0.0
        return
    rel_speed = wp.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
    r = base
    if rel_speed > velocity_threshold:
        r = r + velocity_bonus * rel_speed / max_hand_vel
    r = r * signal[tid]
    raw[tid] = r


@wp.kernel
def _pose_reward_kernel(
    joint_q: wp.array2d(dtype=float),
    joint_nom: wp.array(dtype=float),
    joint_rl_mask: wp.array(dtype=wp.int32),
    joint_tau: wp.array2d(dtype=float),
    tilt: wp.array(dtype=float),
    dof_count: int,
    has_tau: int,
    w_nominal: float,
    w_torque: float,
    w_tilt: float,
    raw: wp.array(dtype=float),
):
    tid = wp.tid()
    q_term = float(0.0)
    tau_term = float(0.0)
    for d in range(dof_count):
        if joint_rl_mask[d] == 0:
            continue
        dq = joint_q[tid, d] - joint_nom[d]
        q_term += dq * dq
        if has_tau == 1:
            t = joint_tau[tid, d]
            tau_term += t * t
    q_term = wp.sqrt(q_term)
    tau_term = wp.sqrt(tau_term)
    tilt_v = tilt[tid]
    if (not wp.isfinite(q_term)) or (not wp.isfinite(tau_term)) or (not wp.isfinite(tilt_v)):
        raw[tid] = 0.0
        return
    raw[tid] = -(w_nominal * q_term + w_torque * tau_term + w_tilt * tilt_v)


@wp.kernel
def _reset_punch_buffers_kernel(
    terminated: wp.array(dtype=wp.bool),
    hand_vel_prev: wp.array2d(dtype=float),
    hand_fwd_hist: wp.array2d(dtype=float),
):
    tid = wp.tid()
    if terminated[tid] == False:
        return
    hand_vel_prev[tid, 0] = 0.0
    hand_vel_prev[tid, 1] = 0.0
    hand_vel_prev[tid, 2] = 0.0
    hist_len = hand_fwd_hist.shape[1]
    for i in range(hist_len):
        hand_fwd_hist[tid, i] = 0.0


def _as_wp_2d(tensor, dtype=wp.float32) -> wp.array:
    if isinstance(tensor, wp.array):
        return tensor
    return wp.from_torch(tensor.contiguous(), dtype=dtype)


def _as_wp_1d(tensor, dtype=wp.int32) -> wp.array:
    if isinstance(tensor, wp.array):
        return tensor
    return wp.from_torch(tensor.contiguous(), dtype=dtype)


def compute_distance_reward(
    *,
    pelvis_pos,
    target_pos,
    has_target,
    standoff: float,
    band: float,
    punch_ready_mode: int,
    base_weight: float,
    penalty_weight: float,
    raw: wp.array,
    num_env: int,
    device: Any,
) -> None:
    wp.launch(
        _distance_reward_kernel,
        dim=num_env,
        inputs=[
            _as_wp_2d(pelvis_pos),
            _as_wp_2d(target_pos),
            _as_wp_1d(has_target, dtype=wp.int32),
            float(standoff),
            float(band),
            int(punch_ready_mode),
            float(base_weight),
            float(penalty_weight),
            raw,
        ],
        device=device,
    )


def compute_ready_pose_reward(
    *,
    has_target,
    joint_q,
    joint_indices: wp.array,
    joint_poses: wp.array,
    joint_weights: wp.array,
    hard_state,
    signal,
    ready_threshold: float,
    error_scale: float,
    weight: float,
    ready_bonus: float,
    raw: wp.array,
    num_env: int,
    device: Any,
) -> None:
    wp.launch(
        _ready_pose_kernel,
        dim=num_env,
        inputs=[
            _as_wp_1d(has_target, dtype=wp.int32),
            _as_wp_2d(joint_q),
            joint_indices,
            joint_poses,
            joint_weights,
            int(joint_indices.shape[0]),
            _as_wp_1d(hard_state, dtype=wp.int32),
            _as_wp_1d(signal, dtype=wp.float32),
            float(ready_threshold),
            float(error_scale),
            float(weight),
            float(ready_bonus),
            raw,
        ],
        device=device,
    )


def compute_punch_reward(
    *,
    pelvis_pos,
    pelvis_vel,
    pelvis_yaw,
    target_pos,
    hand_pos,
    hand_vel,
    hand_vel_prev: wp.array,
    hand_fwd_hist: wp.array,
    hist_pos: int,
    has_target,
    hard_state,
    signal,
    joint_q,
    joint_indices: wp.array,
    joint_poses: wp.array,
    joint_weights: wp.array,
    dt: float,
    standoff: float,
    band: float,
    punch_ready_mode: int,
    gate_feasibility: float,
    ready_threshold: float,
    planar_min_speed: float,
    planar_dominance_ratio: float,
    xy_weight: float,
    acc_weight: float,
    full_swing_min_advance: float,
    full_swing_acc_scale: float,
    unprepared_scale: float,
    raw: wp.array,
    num_env: int,
    device: Any,
) -> None:
    wp.launch(
        _punch_reward_kernel,
        dim=num_env,
        inputs=[
            _as_wp_2d(pelvis_pos),
            _as_wp_2d(pelvis_vel),
            _as_wp_1d(pelvis_yaw, dtype=wp.float32),
            _as_wp_2d(target_pos),
            _as_wp_2d(hand_pos),
            _as_wp_2d(hand_vel),
            hand_vel_prev,
            hand_fwd_hist,
            int(hist_pos),
            _as_wp_1d(has_target, dtype=wp.int32),
            _as_wp_1d(hard_state, dtype=wp.int32),
            _as_wp_1d(signal, dtype=wp.float32),
            _as_wp_2d(joint_q),
            joint_indices,
            joint_poses,
            joint_weights,
            int(joint_indices.shape[0]),
            float(dt),
            float(standoff),
            float(band),
            int(punch_ready_mode),
            float(gate_feasibility),
            float(ready_threshold),
            float(planar_min_speed),
            float(planar_dominance_ratio),
            float(xy_weight),
            float(acc_weight),
            float(full_swing_min_advance),
            float(full_swing_acc_scale),
            float(unprepared_scale),
            raw,
        ],
        device=device,
    )


def compute_retract_reward(
    *,
    pelvis_pos,
    target_pos,
    hand_pos,
    hand_vel,
    has_target,
    hard_state,
    signal,
    standoff: float,
    band: float,
    punch_ready_mode: int,
    gate_feasibility: float,
    vel_weight: float,
    fake_punch_vel_threshold: float,
    fake_punch_penalty: float,
    ideal_away_speed: float,
    away_sigma: float,
    over_speed_threshold: float,
    over_speed_penalty: float,
    raw: wp.array,
    num_env: int,
    device: Any,
) -> None:
    wp.launch(
        _retract_reward_kernel,
        dim=num_env,
        inputs=[
            _as_wp_2d(pelvis_pos),
            _as_wp_2d(target_pos),
            _as_wp_2d(hand_pos),
            _as_wp_2d(hand_vel),
            _as_wp_1d(has_target, dtype=wp.int32),
            _as_wp_1d(hard_state, dtype=wp.int32),
            _as_wp_1d(signal, dtype=wp.float32),
            float(standoff),
            float(band),
            int(punch_ready_mode),
            float(gate_feasibility),
            float(vel_weight),
            float(fake_punch_vel_threshold),
            float(fake_punch_penalty),
            float(ideal_away_speed),
            float(away_sigma),
            float(over_speed_threshold),
            float(over_speed_penalty),
            raw,
        ],
        device=device,
    )


def compute_hit_reward(
    *,
    pelvis_pos,
    target_pos,
    hand_vel,
    target_vel,
    has_target,
    contact_found,
    hard_state,
    signal,
    standoff: float,
    band: float,
    punch_ready_mode: int,
    gate_feasibility: float,
    velocity_threshold: float,
    base: float,
    velocity_bonus: float,
    max_hand_vel: float,
    raw: wp.array,
    num_env: int,
    device: Any,
) -> None:
    wp.launch(
        _hit_reward_kernel,
        dim=num_env,
        inputs=[
            _as_wp_2d(pelvis_pos),
            _as_wp_2d(target_pos),
            _as_wp_2d(hand_vel),
            _as_wp_2d(target_vel),
            _as_wp_1d(has_target, dtype=wp.int32),
            _as_wp_1d(contact_found, dtype=wp.int32),
            _as_wp_1d(hard_state, dtype=wp.int32),
            _as_wp_1d(signal, dtype=wp.float32),
            float(standoff),
            float(band),
            int(punch_ready_mode),
            float(gate_feasibility),
            float(velocity_threshold),
            float(base),
            float(velocity_bonus),
            float(max_hand_vel),
            raw,
        ],
        device=device,
    )


def compute_pose_reward(
    *,
    joint_q,
    joint_nom,
    joint_rl_mask,
    joint_tau,
    tilt,
    dof_count: int,
    w_nominal: float,
    w_torque: float,
    w_tilt: float,
    raw: wp.array,
    num_env: int,
    device: Any,
) -> None:
    has_tau = 1
    wp.launch(
        _pose_reward_kernel,
        dim=num_env,
        inputs=[
            _as_wp_2d(joint_q),
            _as_wp_1d(joint_nom, dtype=wp.float32),
            _as_wp_1d(joint_rl_mask, dtype=wp.int32),
            _as_wp_2d(joint_tau),
            _as_wp_1d(tilt, dtype=wp.float32),
            int(dof_count),
            has_tau,
            float(w_nominal),
            float(w_torque),
            float(w_tilt),
            raw,
        ],
        device=device,
    )


def reset_punch_state_buffers(
    terminated,
    hand_vel_prev: wp.array,
    hand_fwd_hist: wp.array,
    *,
    num_env: int,
    device: Any,
) -> None:
    """Clear per-env hand velocity / forward-history buffers on termination."""
    term = terminated if isinstance(terminated, wp.array) else wp.from_torch(
        terminated.contiguous(), dtype=wp.bool
    )
    wp.launch(
        _reset_punch_buffers_kernel,
        dim=num_env,
        inputs=[term, hand_vel_prev, hand_fwd_hist],
        device=device,
    )


@wp.kernel
def _fill_hit_metrics_kernel(
    found: wp.array(dtype=wp.int32),
    force: wp.array(dtype=float),
    hit_flag: wp.array(dtype=float),
    hit_force: wp.array(dtype=float),
):
    tid = wp.tid()
    flag = float(found[tid])
    hit_flag[tid] = flag
    hit_force[tid] = force[tid] * flag


def fill_hit_metrics(
    *,
    contact_found,
    contact_force,
    hit_flag: wp.array,
    hit_force: wp.array,
    num_env: int,
    device: Any,
) -> None:
    """Write per-env hit flag (0/1) and force (0 on miss) for logging."""
    wp.launch(
        _fill_hit_metrics_kernel,
        dim=num_env,
        inputs=[
            _as_wp_1d(contact_found, dtype=wp.int32),
            _as_wp_1d(contact_force, dtype=wp.float32),
            hit_flag,
            hit_force,
        ],
        device=device,
    )


def emit_env_reward(
    raw: wp.array,
    *,
    scale: float,
    step_total_rewards: wp.array,
    term_buf: Optional[Any],
    player_shape_ids: wp.array,
    env_map: wp.array,
    num_players: int,
    num_env: int,
    device: Any,
) -> None:
    scatter_env_reward(
        raw,
        scale=scale,
        step_total_rewards=step_total_rewards,
        term_buf=term_buf,
        player_shape_ids=player_shape_ids,
        env_map=env_map,
        num_players=num_players,
        num_env=num_env,
        device=device,
    )
