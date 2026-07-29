"""Camera-aligned turret aim helpers (signed error + PD torque)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import warp as wp


@dataclass
class ToolAimControlConfig:
    yaw_torque_gain: float = 200.0
    yaw_damping: float = 20.0
    max_yaw_torque: float = 500.0
    pitch_torque_gain: float = 150.0
    pitch_damping: float = 15.0
    max_pitch_torque: float = 300.0
    angle_dead_zone_deg: float = 0.5
    weld_yaw_drive_gain: float = 8.0
    aim_forward_local: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    world_up: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    @classmethod
    def from_mapping(cls, raw: dict | None, defaults: "ToolAimControlConfig | None" = None) -> "ToolAimControlConfig":
        base = defaults or cls()
        if not raw:
            return base
        forward = raw.get("aim_forward_local", base.aim_forward_local)
        up = raw.get("world_up", base.world_up)
        return cls(
            yaw_torque_gain=float(raw.get("yaw_torque_gain", base.yaw_torque_gain)),
            yaw_damping=float(raw.get("yaw_damping", base.yaw_damping)),
            max_yaw_torque=float(raw.get("max_yaw_torque", base.max_yaw_torque)),
            pitch_torque_gain=float(raw.get("pitch_torque_gain", base.pitch_torque_gain)),
            pitch_damping=float(raw.get("pitch_damping", base.pitch_damping)),
            max_pitch_torque=float(raw.get("max_pitch_torque", base.max_pitch_torque)),
            angle_dead_zone_deg=float(raw.get("angle_dead_zone_deg", base.angle_dead_zone_deg)),
            weld_yaw_drive_gain=float(raw.get("weld_yaw_drive_gain", base.weld_yaw_drive_gain)),
            aim_forward_local=tuple(float(v) for v in forward),
            world_up=tuple(float(v) for v in up),
        )


def camera_forward_z_up(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Match newton Camera.get_front() for Z-up worlds."""
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    cos_pitch = math.cos(pitch)
    return np.array(
        [
            math.cos(yaw) * cos_pitch,
            math.sin(yaw) * cos_pitch,
            math.sin(pitch),
        ],
        dtype=np.float64,
    )


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec
    return vec / norm


def _wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def clamp_angle_to_limits(angle: float, lo: float, hi: float, reference: float) -> float:
    """Clamp *angle* to [lo, hi], preserving continuity near *reference*."""
    span = float(hi) - float(lo)
    if span >= 2.0 * math.pi - 1e-4:
        return _wrap_pi(angle)

    wrapped = _wrap_pi(angle)
    candidates = [wrapped, wrapped + 2.0 * math.pi, wrapped - 2.0 * math.pi]
    valid = [value for value in candidates if float(lo) - 1e-6 <= value <= float(hi) + 1e-6]
    if not valid:
        if wrapped < lo:
            return float(lo)
        if wrapped > hi:
            return float(hi)
        return wrapped
    return min(valid, key=lambda value: abs(value - float(reference)))


def measure_mount_yaw_in_host_frame(
    host_body_q,
    aim_body_q,
    forward_local: Sequence[float],
) -> float:
    host_q = _quat_to_np(host_body_q[3:])
    host_inv = _quat_inverse(host_q)
    current_world = body_forward_world(aim_body_q, forward_local)
    current_local = _rotate_vec_by_quat(host_inv, current_world)
    return math.atan2(current_local[1], current_local[0])


def _quat_to_np(q) -> np.ndarray:
    return np.array([float(q[0]), float(q[1]), float(q[2]), float(q[3])], dtype=np.float64)


def _rotate_vec_by_quat(q: np.ndarray, vec: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    vx, vy, vz = vec
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return np.array(
        [
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx),
        ],
        dtype=np.float64,
    )


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    norm_sq = qx * qx + qy * qy + qz * qz + qw * qw
    if norm_sq < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    inv = 1.0 / norm_sq
    return np.array([-qx * inv, -qy * inv, -qz * inv, qw * inv], dtype=np.float64)


def body_forward_world(body_q_entry, forward_local: Sequence[float]) -> np.ndarray:
    q = _quat_to_np(body_q_entry[3:])
    local = np.array([float(forward_local[0]), float(forward_local[1]), float(forward_local[2])])
    return _normalize(_rotate_vec_by_quat(q, local))


def compute_host_local_aim_errors(
    host_body_q,
    aim_body_q,
    desired_world: np.ndarray,
    forward_local: Sequence[float],
) -> Tuple[float, float]:
    """
    Returns (yaw_error_rad, pitch_error_rad) in host-local frame.

    Positive yaw_error means the camera look direction is to the left of the
    current barrel forward (counter-clockwise about +Z when viewed from above).
    Positive pitch_error means the camera is above the current barrel aim.
    """
    host_q = _quat_to_np(host_body_q[3:])
    host_inv = _quat_inverse(host_q)
    desired_local = _rotate_vec_by_quat(host_inv, _normalize(desired_world))
    current_world = body_forward_world(aim_body_q, forward_local)
    current_local = _rotate_vec_by_quat(host_inv, current_world)

    desired_yaw = math.atan2(desired_local[1], desired_local[0])
    current_yaw = math.atan2(current_local[1], current_local[0])
    yaw_error = _wrap_pi(desired_yaw - current_yaw)

    desired_pitch = math.atan2(
        desired_local[2],
        max(1e-8, math.hypot(desired_local[0], desired_local[1])),
    )
    current_pitch = math.atan2(
        current_local[2],
        max(1e-8, math.hypot(current_local[0], current_local[1])),
    )
    pitch_error = desired_pitch - current_pitch
    return yaw_error, pitch_error


def pd_torque(error: float, rate: float, gain: float, damping: float, max_torque: float) -> float:
    if abs(error) < 1e-8:
        torque = -damping * rate
    else:
        torque = gain * error - damping * rate
    if torque > max_torque:
        return max_torque
    if torque < -max_torque:
        return -max_torque
    return torque


def soft_limit_torque(torque: float, max_torque: float) -> float:
    limit = max(abs(max_torque), 1e-6)
    return limit * math.tanh(torque / limit)


def world_torque_about_axis(axis_world: np.ndarray, torque_scalar: float) -> wp.vec3:
    axis = _normalize(axis_world)
    return wp.vec3(
        float(axis[0] * torque_scalar),
        float(axis[1] * torque_scalar),
        float(axis[2] * torque_scalar),
    )
