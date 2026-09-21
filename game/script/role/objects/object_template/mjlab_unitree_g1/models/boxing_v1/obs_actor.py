# This file contains code adapted from:
# https://github.com/mujocolab/mjlab
#
# Modified for Action_Game_RL.
#
# The original project is licensed under the Apache License 2.0.

"""G1 velocity locomotion observation and command (mjlab-aligned)."""

from __future__ import annotations

from typing import Optional, Sequence, TYPE_CHECKING

import numpy as np
import torch
import warp as wp
import yaml

from pathlib import Path

from script.game_config import GameConfig
from script.role.objects.object_template.mjlab_unitree_g1.models.boxing_v1.state_query import (
    fill_selected_newton_body_ids,
    gather_override_body_poses,
)

if TYPE_CHECKING:
    from script.role.bodies.articulation_body import ArticulationBody
    from script.simulate.physics_manager import PhysicsManager

_VERSION_DIR = Path(__file__).resolve().parent


def _load_version_yaml(name: str) -> dict:
    path = _VERSION_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return data


def _task_mapping() -> dict:
    raw = _load_version_yaml("control_configs.yaml")
    robot_cfg = raw.get("unitree_g1") or {}
    if not isinstance(robot_cfg, dict):
        robot_cfg = {}
    task_cfg = robot_cfg.get("velocity_locomotion") or {}
    return task_cfg if isinstance(task_cfg, dict) else {}


def _action_dim() -> int:
    action = _load_version_yaml("control_policy.yaml").get("action") or {}
    if "low_level_dim" not in action:
        raise KeyError(
            f"{_VERSION_DIR / 'control_policy.yaml'}: action.low_level_dim is required"
        )
    return int(action["low_level_dim"])


def _default_command_velocity_ranges() -> dict[str, tuple[float, float]]:
    raw = _task_mapping().get("command_sample_ranges") or {}
    if not isinstance(raw, dict):
        raw = {}
    required = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(
            f"{_VERSION_DIR / 'control_configs.yaml'}: unitree_g1.velocity_locomotion."
            f"command_sample_ranges missing {missing}"
        )
    return {key: (float(raw[key][0]), float(raw[key][1])) for key in required}


def _command_follow_metric_stds() -> dict[str, float]:
    """Per-axis Gaussian std for command-following scores (object-template config)."""
    raw = _task_mapping().get("command_follow_metrics") or {}
    if not isinstance(raw, dict):
        raw = {}
    stds = raw.get("std") or {}
    if not isinstance(stds, dict):
        stds = {}
    required = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
    missing = [key for key in required if key not in stds]
    if missing:
        raise ValueError(
            f"{_VERSION_DIR / 'control_configs.yaml'}: unitree_g1.velocity_locomotion."
            f"command_follow_metrics.std missing {missing}"
        )
    out: dict[str, float] = {}
    for key in required:
        value = float(stds[key])
        if value <= 0.0:
            raise ValueError(
                f"{_VERSION_DIR / 'control_configs.yaml'}: unitree_g1.velocity_locomotion."
                f"command_follow_metrics.std.{key} must be > 0"
            )
        out[key] = value
    return out


def _require_unit_interval(value: float, *, context: str) -> float:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{context} must be in [0, 1], got {value}")
    return value


def _command_resample_cfg() -> dict[str, float]:
    """Load command-resample mixture / heading PD from this version's YAML."""
    path = _VERSION_DIR / "control_configs.yaml"
    raw = _task_mapping().get("command_resample") or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: unitree_g1.velocity_locomotion.command_resample must be a mapping"
        )
    required_floats = (
        "rel_standing",
        "rel_forward",
        "rel_lateral",
        "rel_yaw",
        "rel_heading",
        "heading_stiffness",
        "heading_wz_clip",
        "forward_min_speed",
        "lateral_min_speed",
        "yaw_min_speed",
    )
    missing = [key for key in required_floats if key not in raw]
    if missing:
        raise ValueError(
            f"{path}: unitree_g1.velocity_locomotion.command_resample missing {missing}"
        )
    interval = raw.get("interval_s")
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError(
            f"{path}: unitree_g1.velocity_locomotion.command_resample.interval_s "
            "must be a [min, max] pair"
        )
    interval_min = float(interval[0])
    interval_max = float(interval[1])
    if interval_min <= 0.0 or interval_max < interval_min:
        raise ValueError(
            f"{path}: unitree_g1.velocity_locomotion.command_resample.interval_s "
            "must satisfy 0 < min <= max"
        )
    # mjlab uses independent Bernoulli probs (each in [0,1]); lateral/yaw are
    # retained for schema compatibility but ignored by the resample kernels.
    rel_standing = _require_unit_interval(
        float(raw["rel_standing"]), context=f"{path}: command_resample.rel_standing"
    )
    rel_forward = _require_unit_interval(
        float(raw["rel_forward"]), context=f"{path}: command_resample.rel_forward"
    )
    rel_lateral = _require_unit_interval(
        float(raw["rel_lateral"]), context=f"{path}: command_resample.rel_lateral"
    )
    rel_yaw = _require_unit_interval(
        float(raw["rel_yaw"]), context=f"{path}: command_resample.rel_yaw"
    )
    rel_heading = _require_unit_interval(
        float(raw["rel_heading"]), context=f"{path}: command_resample.rel_heading"
    )
    heading_wz_clip = float(raw["heading_wz_clip"])
    forward_min_speed = float(raw["forward_min_speed"])
    lateral_min_speed = float(raw["lateral_min_speed"])
    yaw_min_speed = float(raw["yaw_min_speed"])
    if heading_wz_clip <= 0.0:
        raise ValueError(f"{path}: command_resample.heading_wz_clip must be > 0")
    if forward_min_speed < 0.0:
        raise ValueError(f"{path}: command_resample.forward_min_speed must be >= 0")
    if lateral_min_speed < 0.0:
        raise ValueError(f"{path}: command_resample.lateral_min_speed must be >= 0")
    if yaw_min_speed < 0.0:
        raise ValueError(f"{path}: command_resample.yaw_min_speed must be >= 0")
    return {
        "interval_min": interval_min,
        "interval_max": interval_max,
        "rel_standing": rel_standing,
        "rel_forward": rel_forward,
        "rel_lateral": rel_lateral,
        "rel_yaw": rel_yaw,
        "rel_heading": rel_heading,
        "heading_stiffness": float(raw["heading_stiffness"]),
        "heading_wz_clip": heading_wz_clip,
        "forward_min_speed": forward_min_speed,
        "lateral_min_speed": lateral_min_speed,
        "yaw_min_speed": yaw_min_speed,
    }


def _tracking_reward_cfg() -> dict[str, dict]:
    """Load tracking-reward std / axis coupling from this version's YAML."""
    path = _VERSION_DIR / "control_configs.yaml"
    raw = _task_mapping().get("tracking_rewards") or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: unitree_g1.velocity_locomotion.tracking_rewards must be a mapping"
        )
    ang = raw.get("TrackAngularVelocityReward")
    if not isinstance(ang, dict):
        raise ValueError(
            f"{path}: unitree_g1.velocity_locomotion.tracking_rewards."
            "TrackAngularVelocityReward must be a mapping"
        )
    missing = [key for key in ("std", "include_roll_pitch") if key not in ang]
    if missing:
        raise ValueError(
            f"{path}: tracking_rewards.TrackAngularVelocityReward missing {missing}"
        )
    std = float(ang["std"])
    if std <= 0.0:
        raise ValueError(
            f"{path}: tracking_rewards.TrackAngularVelocityReward.std must be > 0"
        )
    return {
        "TrackAngularVelocityReward": {
            "std": std,
            "include_roll_pitch": bool(ang["include_roll_pitch"]),
        }
    }


_DEFAULT_COMMAND_VELOCITY_RANGES = _default_command_velocity_ranges()
_COMMAND_FOLLOW_METRIC_STDS = _command_follow_metric_stds()
_COMMAND_RESAMPLE_CFG = _command_resample_cfg()
_TRACKING_REWARD_CFG = _tracking_reward_cfg()
ACTION_DIM = _action_dim()


def _boxing_mapping() -> dict:
    raw = _load_version_yaml("control_configs.yaml")
    robot_cfg = raw.get("unitree_g1") or {}
    if not isinstance(robot_cfg, dict):
        robot_cfg = {}
    boxing = robot_cfg.get("boxing") or {}
    if not isinstance(boxing, dict):
        raise ValueError(
            f"{_VERSION_DIR / 'control_configs.yaml'}: unitree_g1.boxing must be a mapping"
        )
    return boxing


def _require_non_negative(value: float, *, context: str) -> float:
    if value < 0.0:
        raise ValueError(f"{context} must be >= 0, got {value}")
    return value


def _require_body_list(value, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{context} must be a non-empty list of body names")
    names = tuple(str(item).strip() for item in value)
    if any(not name for name in names):
        raise ValueError(f"{context} entries must be non-empty strings")
    return names


def _require_positive(value: float, *, context: str) -> float:
    if value <= 0.0:
        raise ValueError(f"{context} must be > 0, got {value}")
    return value


def _policy_obs_dim() -> int:
    observation = _load_version_yaml("control_policy.yaml").get("observation") or {}
    if "obs_dim" not in observation:
        raise KeyError(
            f"{_VERSION_DIR / 'control_policy.yaml'}: observation.obs_dim is required"
        )
    return int(observation["obs_dim"])


def _command_policy_cfg() -> dict:
    """Load boxing_v1 command layout from this version's control_policy.yaml."""
    path = _VERSION_DIR / "control_policy.yaml"
    raw = _load_version_yaml("control_policy.yaml").get("commands") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: commands must be a mapping")
    if raw.get("dim") is None:
        raise ValueError(f"{path}: commands.dim is required")
    dim = int(raw["dim"])
    labels = tuple(str(item) for item in (raw.get("labels") or ()))
    ranges_raw = raw.get("ranges") or ()
    ranges = tuple((float(item[0]), float(item[1])) for item in ranges_raw)
    if len(labels) != dim or len(ranges) != dim:
        raise ValueError(
            f"{path}: commands.dim={dim} but labels={len(labels)} ranges={len(ranges)}"
        )
    return {
        "dim": dim,
        "labels": labels,
        "ranges": ranges,
    }


def _v1_segment_slices(rl_action_dim: int) -> dict[str, tuple[int, int]]:
    """Column slices of the 99-dim V1 locomotion frame."""
    d = int(rl_action_dim)
    return {
        "base_lin_vel": (0, 3),
        "base_ang_vel": (3, 6),
        "projected_gravity": (6, 9),
        "joint_pos": (9, 9 + d),
        "joint_vel": (9 + d, 9 + 2 * d),
        "actions": (9 + 2 * d, 9 + 3 * d),
        "command": (9 + 3 * d, 9 + 3 * d + 3),
    }


def _history_source_indices(rl_action_dim: int, segments: Sequence[str]) -> list[int]:
    slices = _v1_segment_slices(rl_action_dim)
    indices: list[int] = []
    for name in segments:
        if name not in slices:
            raise ValueError(
                f"{_VERSION_DIR / 'control_configs.yaml'}: boxing.obs_history.segments "
                f"unknown segment {name!r}; expected one of {sorted(slices)}"
            )
        start, end = slices[name]
        indices.extend(range(start, end))
    if not indices:
        raise ValueError(
            f"{_VERSION_DIR / 'control_configs.yaml'}: boxing.obs_history.segments "
            "must select at least one column"
        )
    return indices


def _require_mapping(value, *, context: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _section_floats(raw: dict, keys, *, context: str) -> dict[str, float]:
    missing = [key for key in keys if key not in raw]
    if missing:
        raise ValueError(f"{context} missing {missing}")
    return {key: float(raw[key]) for key in keys}


def _ready_pose_cfg(raw: dict, *, context: str) -> dict:
    section = _require_mapping(raw, context=context)
    cfg = _section_floats(
        section,
        ("ready_threshold", "error_scale", "weight", "ready_bonus"),
        context=context,
    )
    for key in ("ready_threshold", "error_scale"):
        cfg[key] = _require_positive(cfg[key], context=f"{context}.{key}")
    for key in ("weight", "ready_bonus"):
        cfg[key] = _require_non_negative(cfg[key], context=f"{context}.{key}")
    joints_raw = section.get("joints")
    if not isinstance(joints_raw, dict) or not joints_raw:
        raise ValueError(f"{context}.joints must be a non-empty mapping of dof name to pose/weight")
    joints: dict[str, dict[str, float]] = {}
    for name, entry in joints_raw.items():
        entry = _require_mapping(entry, context=f"{context}.joints.{name}")
        for key in ("pose", "weight"):
            if key not in entry:
                raise ValueError(f"{context}.joints.{name} missing {key!r}")
        joints[str(name)] = {
            "pose": float(entry["pose"]),
            "weight": _require_non_negative(
                float(entry["weight"]), context=f"{context}.joints.{name}.weight"
            ),
        }
    cfg["joints"] = joints
    return cfg


def _punch_phase_cfg(raw: dict, *, context: str) -> dict:
    section = _require_mapping(raw, context=context)
    cfg = _section_floats(
        section,
        (
            "cooldown_lo_s",
            "cooldown_hi_s",
            "transition_duration_s",
            "contact_hold_min_s",
            "contact_hold_max_s",
            "direction_deviation_cos",
            "joint_limit_proximity",
            "punch_timeout_s",
            "arm_extension_threshold",
        ),
        context=context,
    )
    for key in (
        "cooldown_lo_s",
        "cooldown_hi_s",
        "transition_duration_s",
        "contact_hold_min_s",
        "contact_hold_max_s",
        "joint_limit_proximity",
        "punch_timeout_s",
        "arm_extension_threshold",
    ):
        cfg[key] = _require_positive(cfg[key], context=f"{context}.{key}")
    if cfg["cooldown_hi_s"] < cfg["cooldown_lo_s"]:
        raise ValueError(f"{context}: cooldown_hi_s must be >= cooldown_lo_s")
    if not 0.0 < cfg["direction_deviation_cos"] <= 1.0:
        raise ValueError(f"{context}.direction_deviation_cos must be in (0, 1]")
    return cfg


def _boxing_obs_cfg() -> dict:
    """Load boxing extras / history / overlay / reward constants from this version's YAML."""
    path = _VERSION_DIR / "control_configs.yaml"
    raw = _boxing_mapping()
    required = (
        "max_dist",
        "max_hand_vel",
        "standoff_distance",
        "punch_ready_band",
        "punch_ready_mode",
        "pelvis_body",
        "hand_body",
        "extras_dim",
        "signal_dim",
        "hit_hand_bodies",
        "target_body",
        "torso_body",
        "command_clip",
        "overlay",
        "obs_history",
        "checkpoint_expand",
        "distance",
        "ready_pose",
        "punch_phase",
        "punch",
        "retract",
        "hit",
        "pose",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"{path}: unitree_g1.boxing missing {missing}")

    clip = _require_mapping(raw["command_clip"], context=f"{path}: boxing.command_clip")
    if "cmd_x_max" not in clip or "cmd_yaw_max" not in clip:
        raise ValueError(f"{path}: boxing.command_clip must declare cmd_x_max and cmd_yaw_max")
    overlay = _require_mapping(raw["overlay"], context=f"{path}: boxing.overlay")
    overlay_required = (
        "lin_vel_deadband",
        "yaw_deadband",
    )
    overlay_missing = [key for key in overlay_required if key not in overlay]
    if overlay_missing:
        raise ValueError(f"{path}: boxing.overlay missing {overlay_missing}")

    hist = _require_mapping(raw["obs_history"], context=f"{path}: boxing.obs_history")
    for key in ("samples", "stride", "segments"):
        if key not in hist:
            raise ValueError(f"{path}: boxing.obs_history missing {key!r}")
    segments = hist["segments"]
    if not isinstance(segments, (list, tuple)) or not segments:
        raise ValueError(f"{path}: boxing.obs_history.segments must be a non-empty list")

    mode = str(raw["punch_ready_mode"]).strip().lower()
    if mode not in ("linear", "gaussian"):
        raise ValueError(
            f"{path}: boxing.punch_ready_mode must be 'linear' or 'gaussian', got {mode!r}"
        )

    samples = int(hist["samples"])
    stride = int(hist["stride"])
    if samples <= 0 or stride <= 0:
        raise ValueError(f"{path}: boxing.obs_history samples/stride must be > 0")

    extras_dim = int(raw["extras_dim"])
    signal_dim = int(raw["signal_dim"])
    if signal_dim != 1:
        raise ValueError(f"{path}: boxing.signal_dim must be 1, got {signal_dim}")

    distance_cfg = _section_floats(
        _require_mapping(raw["distance"], context=f"{path}: boxing.distance"),
        ("base_weight", "penalty_weight"),
        context=f"{path}: boxing.distance",
    )
    ready_cfg = _ready_pose_cfg(raw["ready_pose"], context=f"{path}: boxing.ready_pose")
    punch_phase_cfg = _punch_phase_cfg(raw["punch_phase"], context=f"{path}: boxing.punch_phase")
    punch_cfg = _section_floats(
        _require_mapping(raw["punch"], context=f"{path}: boxing.punch"),
        (
            "gate_feasibility",
            "planar_min_speed",
            "planar_dominance_ratio",
            "xy_weight",
            "acc_weight",
            "full_swing_window_s",
            "full_swing_min_advance",
            "full_swing_acc_scale",
            "unprepared_scale",
        ),
        context=f"{path}: boxing.punch",
    )
    punch_cfg["gate_feasibility"] = _require_positive(
        punch_cfg["gate_feasibility"], context=f"{path}: boxing.punch.gate_feasibility"
    )
    retract_cfg = _section_floats(
        _require_mapping(raw["retract"], context=f"{path}: boxing.retract"),
        (
            "vel_weight",
            "fake_punch_vel_threshold",
            "fake_punch_penalty",
            "ideal_away_speed",
            "away_sigma",
            "over_speed_threshold",
            "over_speed_penalty",
        ),
        context=f"{path}: boxing.retract",
    )
    if retract_cfg["fake_punch_penalty"] <= punch_cfg["acc_weight"]:
        raise ValueError(
            f"{path}: boxing.retract.fake_punch_penalty "
            f"({retract_cfg['fake_punch_penalty']}) must exceed boxing.punch.acc_weight "
            f"({punch_cfg['acc_weight']}) so fake punches are never profitable"
        )
    retract_cfg["ideal_away_speed"] = _require_positive(
        retract_cfg["ideal_away_speed"], context=f"{path}: boxing.retract.ideal_away_speed"
    )
    retract_cfg["away_sigma"] = _require_positive(
        retract_cfg["away_sigma"], context=f"{path}: boxing.retract.away_sigma"
    )
    retract_cfg["over_speed_threshold"] = _require_non_negative(
        retract_cfg["over_speed_threshold"], context=f"{path}: boxing.retract.over_speed_threshold"
    )
    retract_cfg["over_speed_penalty"] = _require_non_negative(
        retract_cfg["over_speed_penalty"], context=f"{path}: boxing.retract.over_speed_penalty"
    )
    if not 0.0 <= punch_cfg["unprepared_scale"] <= 1.0:
        raise ValueError(
            f"{path}: boxing.punch.unprepared_scale must be in [0, 1]"
        )
    hit_cfg = _section_floats(
        _require_mapping(raw["hit"], context=f"{path}: boxing.hit"),
        ("base", "velocity_bonus", "velocity_threshold"),
        context=f"{path}: boxing.hit",
    )
    pose_cfg = _section_floats(
        _require_mapping(raw["pose"], context=f"{path}: boxing.pose"),
        ("nominal_weight", "torque_weight", "tilt_weight"),
        context=f"{path}: boxing.pose",
    )

    return {
        "max_dist": _require_positive(float(raw["max_dist"]), context=f"{path}: boxing.max_dist"),
        "max_hand_vel": _require_positive(
            float(raw["max_hand_vel"]), context=f"{path}: boxing.max_hand_vel"
        ),
        "standoff_distance": _require_positive(
            float(raw["standoff_distance"]), context=f"{path}: boxing.standoff_distance"
        ),
        "punch_ready_band": _require_positive(
            float(raw["punch_ready_band"]), context=f"{path}: boxing.punch_ready_band"
        ),
        "punch_ready_mode": mode,
        "pelvis_body": str(raw["pelvis_body"]),
        "hand_body": str(raw["hand_body"]),
        "hit_hand_bodies": _require_body_list(
            raw["hit_hand_bodies"], context=f"{path}: boxing.hit_hand_bodies"
        ),
        "target_body": str(raw["target_body"]).strip(),
        "torso_body": str(raw["torso_body"]).strip(),
        "extras_dim": extras_dim,
        "signal_dim": signal_dim,
        "cmd_x_max": _require_positive(
            float(clip["cmd_x_max"]), context=f"{path}: boxing.command_clip.cmd_x_max"
        ),
        "cmd_yaw_max": _require_positive(
            float(clip["cmd_yaw_max"]), context=f"{path}: boxing.command_clip.cmd_yaw_max"
        ),
        "lin_vel_deadband": float(overlay["lin_vel_deadband"]),
        "yaw_deadband": float(overlay["yaw_deadband"]),
        "hist_samples": samples,
        "hist_stride": stride,
        "hist_segments": tuple(str(s) for s in segments),
        "hist_span": samples * stride + 1,
        "packed_obs_dim": _policy_obs_dim(),
        "distance": distance_cfg,
        "ready_pose": ready_cfg,
        "punch_phase": punch_phase_cfg,
        "punch": punch_cfg,
        "retract": retract_cfg,
        "hit": hit_cfg,
        "pose": pose_cfg,
    }


# Matches ``overlay_and_extras_kernel`` column order (object-template layout).
_BOXING_EXTRAS_NAMES: tuple[str, ...] = (
    "tgt_pos_x",
    "tgt_pos_y",
    "tgt_pos_z",
    "tgt_vel_x",
    "tgt_vel_y",
    "tgt_vel_z",
    "hand_pos_x",
    "hand_pos_y",
    "hand_pos_z",
    "hand_vel_x",
    "hand_vel_y",
    "hand_vel_z",
    "dist",
)

_COMMAND_POLICY_CFG = _command_policy_cfg()
_BOXING_OBS_CFG = _boxing_obs_cfg()
if not _BOXING_OBS_CFG["target_body"]:
    raise ValueError(
        f"{_VERSION_DIR / 'control_configs.yaml'}: boxing.target_body must be a non-empty body name"
    )
if not _BOXING_OBS_CFG["torso_body"]:
    raise ValueError(
        f"{_VERSION_DIR / 'control_configs.yaml'}: boxing.torso_body must be a non-empty body name"
    )
if len(_BOXING_EXTRAS_NAMES) != int(_BOXING_OBS_CFG["extras_dim"]):
    raise ValueError(
        "boxing extras name list length "
        f"{len(_BOXING_EXTRAS_NAMES)} != extras_dim {_BOXING_OBS_CFG['extras_dim']}"
    )


def _load_imu_site_offset_b() -> wp.vec3:
    """Read ``imu_site_offset`` from this version's ``control_configs.yaml``."""
    path = _VERSION_DIR / "control_configs.yaml"
    offset = _task_mapping().get("imu_site_offset")
    if not isinstance(offset, (list, tuple)) or len(offset) != 3:
        raise ValueError(
            f"{path}: unitree_g1.velocity_locomotion.imu_site_offset must be a "
            "3-element [x, y, z] list (mjlab imu_in_pelvis)."
        )
    return wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))


@wp.kernel
def init_encoder_bias_kernel(
    encoder_bias: wp.array2d(dtype=float),
    seeds: wp.array(dtype=wp.int32),
    seed_offsets: wp.array(dtype=wp.int32),
    bias_min: float,
    bias_max: float,
    rl_action_dim: int,
):
    """Sample per-(env, joint) encoder bias once at startup (mjlab startup event)."""
    tid = wp.tid()
    rng = wp.rand_init(seeds[tid], seed_offsets[tid])
    seed_offsets[tid] = seed_offsets[tid] + 1
    for i in range(rl_action_dim):
        encoder_bias[tid, i] = wp.randf(rng, bias_min, bias_max)


@wp.kernel
def init_base_com_offset_kernel(
    base_com_offset: wp.array2d(dtype=float),
    seeds: wp.array(dtype=wp.int32),
    seed_offsets: wp.array(dtype=wp.int32),
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    z_min: float, z_max: float,
):
    """Sample per-env torso COM offset once at startup (mjlab startup event)."""
    tid = wp.tid()
    rng = wp.rand_init(seeds[tid], seed_offsets[tid])
    seed_offsets[tid] = seed_offsets[tid] + 1
    base_com_offset[tid, 0] = wp.randf(rng, x_min, x_max)
    base_com_offset[tid, 1] = wp.randf(rng, y_min, y_max)
    base_com_offset[tid, 2] = wp.randf(rng, z_min, z_max)


@wp.kernel
def apply_encoder_bias_kernel(
    noisy_obs: wp.array2d(dtype=float),
    encoder_bias: wp.array2d(dtype=float),
    rl_action_dim: int,
):
    """Corrupt the actor joint-pos segment with the per-env encoder bias.

    Only the actor frame (``single_obs_wp``) gets the bias; the asymmetric
    critic's noiseless frame stays clean (mjlab ``enable_corruption`` applies to
    the actor observation group only).
    """
    tid = wp.tid()
    for i in range(rl_action_dim):
        noisy_obs[tid, 9 + i] = noisy_obs[tid, 9 + i] - encoder_bias[tid, i]


@wp.kernel
def shift_and_append_history_kernel(
    obs_history: wp.array2d(dtype=float),
    new_obs: wp.array2d(dtype=float),
    obs_dim: int,
    history_len: int,
):
    tid = wp.tid()
    shift_range = (history_len - 1) * obs_dim
    for i in range(shift_range):
        obs_history[tid, i] = obs_history[tid, i + obs_dim]
    for i in range(obs_dim):
        obs_history[tid, shift_range + i] = new_obs[tid, i]


@wp.kernel
def apply_obs_noise_kernel(
    clean_obs: wp.array2d(dtype=float),
    noisy_obs: wp.array2d(dtype=float),
    seeds: wp.array(dtype=wp.int32),
    seed_offsets: wp.array(dtype=wp.int32),
    lin_vel_noise: float,
    ang_vel_noise: float,
    gravity_noise: float,
    joint_pos_noise: float,
    joint_vel_noise: float,
    rl_action_dim: int,
):
    """Add per-segment uniform noise (mjlab ``enable_corruption`` semantics).

    Layout matches ``compute_obs_locomotion_kernel``:
      [0:3] base_lin_vel, [3:6] base_ang_vel, [6:9] projected_gravity,
      [9:9+D] joint_pos, [9+D:9+2D] joint_vel, [9+2D:9+3D] actions,
      [9+3D:9+3D+3] command.
    Only proprioceptive segments get noise (actions/command stay clean),
    mirroring mjlab's actor obs corruption config.
    """
    tid = wp.tid()
    rng = wp.rand_init(seeds[tid], seed_offsets[tid])
    seed_offsets[tid] = seed_offsets[tid] + 1

    for i in range(3):
        noisy_obs[tid, i] = clean_obs[tid, i] + wp.randf(rng, -lin_vel_noise, lin_vel_noise)
    for i in range(3):
        noisy_obs[tid, 3 + i] = clean_obs[tid, 3 + i] + wp.randf(rng, -ang_vel_noise, ang_vel_noise)
    for i in range(3):
        noisy_obs[tid, 6 + i] = clean_obs[tid, 6 + i] + wp.randf(rng, -gravity_noise, gravity_noise)
    for i in range(rl_action_dim):
        noisy_obs[tid, 9 + i] = clean_obs[tid, 9 + i] + wp.randf(rng, -joint_pos_noise, joint_pos_noise)
    for i in range(rl_action_dim):
        noisy_obs[tid, 9 + rl_action_dim + i] = clean_obs[tid, 9 + rl_action_dim + i] + wp.randf(rng, -joint_vel_noise, joint_vel_noise)
    # actions and command segments carry no noise.
    for i in range(9 + 2 * rl_action_dim, 9 + 3 * rl_action_dim):
        noisy_obs[tid, i] = clean_obs[tid, i]
    for i in range(9 + 3 * rl_action_dim, 9 + 3 * rl_action_dim + 3):
        noisy_obs[tid, i] = clean_obs[tid, i]


@wp.kernel
def reset_history_kernel(
    obs_history: wp.array2d(dtype=float),
    new_obs: wp.array2d(dtype=float),
    reset_mask: wp.array(dtype=int),
    instance_world_indices: wp.array(dtype=wp.int32),
    obs_dim: int,
    history_len: int,
):
    tid = wp.tid()
    if reset_mask[instance_world_indices[tid]] == 1:
        for step in range(history_len):
            offset = step * obs_dim
            for i in range(obs_dim):
                obs_history[tid, offset + i] = new_obs[tid, i]


@wp.kernel
def compute_obs_locomotion_kernel(
    obs: wp.array2d(dtype=float),
    view_link_q: wp.array(dtype=wp.transform, ndim=3),
    view_link_qd: wp.array(dtype=wp.spatial_vector, ndim=3),
    view_joint_q: wp.array(dtype=float, ndim=3),
    view_joint_qd: wp.array(dtype=float, ndim=3),
    joint_nominal_qs: wp.array(dtype=float),
    joint_rl_mask: wp.array(dtype=wp.int32),
    joint_rl_action_indices: wp.array(dtype=wp.int32),
    commands: wp.array2d(dtype=float),
    policy_actions: wp.array2d(dtype=float),
    instance_world_indices: wp.array(dtype=wp.int32),
    instance_view_indices: wp.array(dtype=wp.int32),
    rl_action_dim: int,
    joint_dof_count: int,
    gravity_vector: wp.vec3,
    imu_site_offset_b: wp.vec3,
    base_com_offsets: wp.array2d(dtype=float),
):
    tid = wp.tid()
    world_idx = instance_world_indices[tid]
    view_idx = instance_view_indices[tid]
    root_tf = view_link_q[world_idx, view_idx, 0]
    root_qd_spatial = view_link_qd[world_idx, view_idx, 0]
    base_rot = root_tf.q

    world_lin_vel = wp.vec3(root_qd_spatial[0], root_qd_spatial[1], root_qd_spatial[2])
    world_ang_vel = wp.vec3(root_qd_spatial[3], root_qd_spatial[4], root_qd_spatial[5])
    inv_base_rot = wp.quat_inverse(base_rot)
    local_lin_vel = wp.quat_rotate(inv_base_rot, world_lin_vel)
    local_ang_vel = wp.quat_rotate(inv_base_rot, world_ang_vel)
    local_gravity = wp.quat_rotate(inv_base_rot, gravity_vector)

    # Obs-domain approximation of mjlab dr.body_com_offset: the torso COM is
    # shifted per env, changing the IMU lever arm. Zero offsets when disabled.
    com_offset = wp.vec3(base_com_offsets[tid, 0], base_com_offsets[tid, 1], base_com_offsets[tid, 2])
    local_com_offset = imu_site_offset_b + com_offset
    imu_lin_vel = local_lin_vel + wp.cross(local_ang_vel, local_com_offset)

    idx = 0
    obs[tid, idx] = imu_lin_vel[0]; idx += 1
    obs[tid, idx] = imu_lin_vel[1]; idx += 1
    obs[tid, idx] = imu_lin_vel[2]; idx += 1
    obs[tid, idx] = local_ang_vel[0]; idx += 1
    obs[tid, idx] = local_ang_vel[1]; idx += 1
    obs[tid, idx] = local_ang_vel[2]; idx += 1
    obs[tid, idx] = local_gravity[0]; idx += 1
    obs[tid, idx] = local_gravity[1]; idx += 1
    obs[tid, idx] = local_gravity[2]; idx += 1

    for a in range(rl_action_dim):
        dof = wp.int32(0)
        for d in range(joint_dof_count):
            if joint_rl_action_indices[d] == a:
                dof = d
                break
        nom_q = joint_nominal_qs[dof]
        pos_err = view_joint_q[world_idx, view_idx, dof] - nom_q
        obs[tid, idx] = pos_err
        idx += 1

    for a in range(rl_action_dim):
        dof = wp.int32(0)
        for d in range(joint_dof_count):
            if joint_rl_action_indices[d] == a:
                dof = d
                break
        obs[tid, idx] = view_joint_qd[world_idx, view_idx, dof]
        idx += 1

    for a in range(rl_action_dim):
        obs[tid, idx] = policy_actions[tid, a]
        idx += 1

    obs[tid, idx] = commands[tid, 0]; idx += 1
    obs[tid, idx] = commands[tid, 1]; idx += 1
    obs[tid, idx] = commands[tid, 2]


@wp.func
def _apply_command_mode(
    commands: wp.array2d(dtype=float),
    is_heading_env: wp.array(dtype=wp.int32),
    is_standing_env: wp.array(dtype=wp.int32),
    tid: int,
    u_standing: float,
    u_heading: float,
    u_forward: float,
    rel_standing: float,
    rel_forward: float,
    rel_heading: float,
    forward_min_speed: float,
):
    """Apply mjlab UniformVelocityCommand flags (independent Bernoullis).

    standing / heading / forward may overlap. Forward clamps |vx| and zeros
    vy/wz at resample; heading PD overwrites wz each step; standing zeros all
    each step (wins over heading). Unused lateral/yaw modes are not applied.
    """
    is_standing = 0
    is_heading = 0
    if u_standing < rel_standing:
        is_standing = 1
    if u_heading < rel_heading:
        is_heading = 1
    if u_forward < rel_forward:
        # mjlab: vx = abs(vx).clamp(min=forward_min_speed); vy = wz = 0
        vx = wp.abs(commands[tid, 0])
        if vx < forward_min_speed:
            vx = forward_min_speed
        commands[tid, 0] = vx
        commands[tid, 1] = 0.0
        commands[tid, 2] = 0.0
    is_heading_env[tid] = is_heading
    is_standing_env[tid] = is_standing


@wp.kernel
def resample_velocity_commands_kernel(
    commands: wp.array2d(dtype=float),
    heading_target: wp.array(dtype=float),
    is_heading_env: wp.array(dtype=wp.int32),
    is_standing_env: wp.array(dtype=wp.int32),
    resample_timer: wp.array(dtype=float),
    seeds: wp.array(dtype=wp.int32),
    seed_offsets: wp.array(dtype=wp.int32),
    dt: float,
    interval_min: float,
    interval_max: float,
    rel_standing: float,
    rel_forward: float,
    rel_heading: float,
    forward_min_speed: float,
    command_ranges: wp.array(dtype=wp.float32),
    skip_command_update: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if skip_command_update[tid] == 1:
        return
    resample_timer[tid] = resample_timer[tid] - dt
    if resample_timer[tid] > 0.0:
        return

    rng = wp.rand_init(seeds[tid], seed_offsets[tid])
    seed_offsets[tid] = seed_offsets[tid] + 1

    resample_timer[tid] = wp.randf(rng, interval_min, interval_max)
    commands[tid, 0] = wp.randf(rng, command_ranges[0], command_ranges[1])
    commands[tid, 1] = wp.randf(rng, command_ranges[2], command_ranges[3])
    commands[tid, 2] = wp.randf(rng, command_ranges[4], command_ranges[5])

    u_standing = wp.randf(rng, 0.0, 1.0)
    u_heading = wp.randf(rng, 0.0, 1.0)
    u_forward = wp.randf(rng, 0.0, 1.0)
    _apply_command_mode(
        commands,
        is_heading_env,
        is_standing_env,
        tid,
        u_standing,
        u_heading,
        u_forward,
        rel_standing,
        rel_forward,
        rel_heading,
        forward_min_speed,
    )

    heading_target[tid] = wp.randf(rng, -3.1415926, 3.1415926)


@wp.kernel
def update_heading_command_kernel(
    commands: wp.array2d(dtype=float),
    heading_target: wp.array(dtype=float),
    is_heading_env: wp.array(dtype=wp.int32),
    is_standing_env: wp.array(dtype=wp.int32),
    root_tfs: wp.array2d(dtype=wp.transform),
    instance_world_indices: wp.array(dtype=wp.int32),
    instance_view_indices: wp.array(dtype=wp.int32),
    heading_stiffness: float,
    heading_wz_clip: float,
    skip_command_update: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if skip_command_update[tid] == 1:
        return
    # mjlab UniformVelocityCommand._update_command order:
    # heading PD first, then standing zeros (vx, vy, wz).
    if is_heading_env[tid] == 1:
        my_tf = root_tfs[instance_world_indices[tid], instance_view_indices[tid]]
        rot = my_tf.q
        siny_cosp = 2.0 * (rot[3] * rot[2] + rot[0] * rot[1])
        cosy_cosp = 1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2])
        heading = wp.atan2(siny_cosp, cosy_cosp)

        err = heading_target[tid] - heading
        while err > 3.1415926:
            err = err - 6.2831852
        while err < -3.1415926:
            err = err + 6.2831852

        wz = heading_stiffness * err
        if wz > heading_wz_clip:
            wz = heading_wz_clip
        if wz < -heading_wz_clip:
            wz = -heading_wz_clip
        commands[tid, 2] = wz

    if is_standing_env[tid] == 1:
        commands[tid, 0] = 0.0
        commands[tid, 1] = 0.0
        commands[tid, 2] = 0.0


@wp.kernel
def or_skip_from_world_mask_kernel(
    skip_command_update: wp.array(dtype=wp.int32),
    instance_world_indices: wp.array(dtype=wp.int32),
    world_hold: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if world_hold[instance_world_indices[tid]] == 1:
        skip_command_update[tid] = 1


@wp.kernel
def clear_skip_command_update_kernel(skip_command_update: wp.array(dtype=wp.int32)):
    skip_command_update[wp.tid()] = 0


@wp.kernel
def reset_velocity_command_on_env_kernel(
    commands: wp.array2d(dtype=float),
    heading_target: wp.array(dtype=float),
    is_heading_env: wp.array(dtype=wp.int32),
    is_standing_env: wp.array(dtype=wp.int32),
    resample_timer: wp.array(dtype=float),
    reset_mask: wp.array(dtype=wp.int32),
    instance_world_indices: wp.array(dtype=wp.int32),
    seeds: wp.array(dtype=wp.int32),
    seed_offsets: wp.array(dtype=wp.int32),
    rel_standing: float,
    rel_forward: float,
    rel_heading: float,
    forward_min_speed: float,
    command_ranges: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if reset_mask[instance_world_indices[tid]] != 1:
        return
    rng = wp.rand_init(seeds[tid], seed_offsets[tid])
    seed_offsets[tid] = seed_offsets[tid] + 1
    resample_timer[tid] = 0.0
    commands[tid, 0] = wp.randf(rng, command_ranges[0], command_ranges[1])
    commands[tid, 1] = wp.randf(rng, command_ranges[2], command_ranges[3])
    commands[tid, 2] = wp.randf(rng, command_ranges[4], command_ranges[5])

    u_standing = wp.randf(rng, 0.0, 1.0)
    u_heading = wp.randf(rng, 0.0, 1.0)
    u_forward = wp.randf(rng, 0.0, 1.0)
    _apply_command_mode(
        commands,
        is_heading_env,
        is_standing_env,
        tid,
        u_standing,
        u_heading,
        u_forward,
        rel_standing,
        rel_forward,
        rel_heading,
        forward_min_speed,
    )
    heading_target[tid] = wp.randf(rng, -3.1415926, 3.1415926)


@wp.kernel
def write_commands_from_rl_actions_kernel(
    commands: wp.array2d(dtype=float),
    actions: wp.array2d(dtype=float),
    action_shape_offset: int,
    command_dim: int,
    player_action_rows: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    action_row = player_action_rows[tid]
    if action_row < 0:
        return
    for i in range(command_dim):
        commands[tid, i] = actions[action_row][action_shape_offset + i]


@wp.func
def _yaw_from_quat(rot: wp.quat) -> float:
    """Extract heading (yaw about +Z) from a Warp quaternion (x, y, z, w)."""
    siny_cosp = 2.0 * (rot[3] * rot[2] + rot[0] * rot[1])
    cosy_cosp = 1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2])
    return wp.atan2(siny_cosp, cosy_cosp)


@wp.func
def _world_to_yaw_frame(delta: wp.vec3, yaw: float) -> wp.vec3:
    """Rotate a world-frame offset into the pelvis yaw frame (Z up)."""
    cy = wp.cos(yaw)
    sy = wp.sin(yaw)
    return wp.vec3(
        cy * delta[0] + sy * delta[1],
        -sy * delta[0] + cy * delta[1],
        delta[2],
    )


@wp.func
def _bang_bang_command(error: float, deadband: float, max_val: float) -> float:
    """Return ±max_val when |error| exceeds deadband, else 0."""
    if wp.abs(error) < deadband:
        return 0.0
    if error > 0.0:
        return max_val
    return -max_val


@wp.kernel
def gather_columns_kernel(
    src: wp.array2d(dtype=float),
    dst: wp.array2d(dtype=float),
    columns: wp.array(dtype=wp.int32),
    n_cols: int,
):
    tid = wp.tid()
    for i in range(n_cols):
        dst[tid, i] = src[tid, columns[i]]


@wp.kernel
def overlay_and_extras_kernel(
    extras: wp.array2d(dtype=float),
    commands: wp.array2d(dtype=float),
    v1_frame: wp.array2d(dtype=float),
    v1_frame_clean: wp.array2d(dtype=float),
    view_link_q: wp.array(dtype=wp.transform, ndim=3),
    view_link_qd: wp.array(dtype=wp.spatial_vector, ndim=3),
    has_target: wp.array(dtype=wp.int32),
    target_pos: wp.array2d(dtype=float),
    target_vel: wp.array2d(dtype=float),
    instance_world_indices: wp.array(dtype=wp.int32),
    instance_view_indices: wp.array(dtype=wp.int32),
    v1_cmd_offset: int,
    pelvis_link_idx: int,
    hand_link_idx: int,
    max_dist: float,
    max_hand_vel: float,
    standoff_distance: float,
    lin_vel_deadband: float,
    yaw_deadband: float,
    cmd_x_max: float,
    cmd_yaw_max: float,
    skip_command_update: wp.array(dtype=wp.int32),
):
    """Fill extras from the selected target and overlay approach commands.

    Target columns use post-normalization sentinels when no target is bound
    (``tgt_pos_b = [-1, -1, -1]``, ``dist = -1``) so the policy can tell
    "no target" apart from a target sitting at the value-0 pose. The hand
    columns always carry the real pelvis-yaw-frame state. The bang-bang
    approach overlay (vx -> standoff, wz -> heading at max command) is active
    whenever a valid target is bound and the command is not externally held.
    """
    tid = wp.tid()
    world_idx = instance_world_indices[tid]
    view_idx = instance_view_indices[tid]

    pelvis_tf = view_link_q[world_idx, view_idx, pelvis_link_idx]
    hand_tf = view_link_q[world_idx, view_idx, hand_link_idx]
    pelvis_qd = view_link_qd[world_idx, view_idx, pelvis_link_idx]
    hand_qd = view_link_qd[world_idx, view_idx, hand_link_idx]

    yaw = _yaw_from_quat(pelvis_tf.q)
    pelvis_pos = pelvis_tf.p
    hand_delta = hand_tf.p - pelvis_pos
    hand_pos_b = _world_to_yaw_frame(hand_delta, yaw)

    hand_lin_w = wp.vec3(hand_qd[0], hand_qd[1], hand_qd[2])
    pelvis_lin_w = wp.vec3(pelvis_qd[0], pelvis_qd[1], pelvis_qd[2])
    hand_vel_b = _world_to_yaw_frame(hand_lin_w - pelvis_lin_w, yaw)

    tgt_pos_b = wp.vec3(-1.0, -1.0, -1.0)
    tgt_vel_b = wp.vec3(0.0, 0.0, 0.0)
    dist_out = -1.0
    if has_target[tid] == 1:
        tx = target_pos[tid, 0]
        ty = target_pos[tid, 1]
        tz = target_pos[tid, 2]
        vx = target_vel[tid, 0]
        vy = target_vel[tid, 1]
        vz = target_vel[tid, 2]
        valid = 1
        if (
            wp.isnan(tx)
            or wp.isnan(ty)
            or wp.isnan(tz)
            or wp.isnan(vx)
            or wp.isnan(vy)
            or wp.isnan(vz)
        ):
            valid = 0
        if valid == 1:
            tgt_delta = wp.vec3(
                tx - pelvis_pos[0],
                ty - pelvis_pos[1],
                tz - pelvis_pos[2],
            )
            tgt_pos_raw = _world_to_yaw_frame(tgt_delta, yaw)
            tgt_vel_w = wp.vec3(vx, vy, vz)
            tgt_vel_raw = _world_to_yaw_frame(tgt_vel_w - pelvis_lin_w, yaw)
            dist = wp.sqrt(tgt_pos_raw[0] * tgt_pos_raw[0] + tgt_pos_raw[1] * tgt_pos_raw[1])
            heading_err = wp.atan2(tgt_pos_raw[1], tgt_pos_raw[0])
            tgt_pos_b = wp.vec3(
                tgt_pos_raw[0] / max_dist,
                tgt_pos_raw[1] / max_dist,
                tgt_pos_raw[2] / max_dist,
            )
            tgt_vel_b = wp.vec3(
                tgt_vel_raw[0] / max_hand_vel,
                tgt_vel_raw[1] / max_hand_vel,
                tgt_vel_raw[2] / max_hand_vel,
            )
            dist_out = dist / max_dist
            if skip_command_update[tid] == 0:
                vx = _bang_bang_command(
                    dist - standoff_distance, lin_vel_deadband, cmd_x_max
                )
                wz = _bang_bang_command(heading_err, yaw_deadband, cmd_yaw_max)
                if wp.isfinite(vx) and wp.isfinite(wz):
                    commands[tid, 0] = vx
                    commands[tid, 2] = wz

    commands[tid, 0] = wp.clamp(commands[tid, 0], -cmd_x_max, cmd_x_max)
    commands[tid, 2] = wp.clamp(commands[tid, 2], -cmd_yaw_max, cmd_yaw_max)
    v1_frame[tid, v1_cmd_offset] = commands[tid, 0]
    v1_frame[tid, v1_cmd_offset + 1] = commands[tid, 1]
    v1_frame[tid, v1_cmd_offset + 2] = commands[tid, 2]
    v1_frame_clean[tid, v1_cmd_offset] = commands[tid, 0]
    v1_frame_clean[tid, v1_cmd_offset + 1] = commands[tid, 1]
    v1_frame_clean[tid, v1_cmd_offset + 2] = commands[tid, 2]

    extras[tid, 0] = tgt_pos_b[0]
    extras[tid, 1] = tgt_pos_b[1]
    extras[tid, 2] = tgt_pos_b[2]
    extras[tid, 3] = tgt_vel_b[0]
    extras[tid, 4] = tgt_vel_b[1]
    extras[tid, 5] = tgt_vel_b[2]
    extras[tid, 6] = hand_pos_b[0] / max_dist
    extras[tid, 7] = hand_pos_b[1] / max_dist
    extras[tid, 8] = hand_pos_b[2] / max_dist
    extras[tid, 9] = hand_vel_b[0] / max_hand_vel
    extras[tid, 10] = hand_vel_b[1] / max_hand_vel
    extras[tid, 11] = hand_vel_b[2] / max_hand_vel
    extras[tid, 12] = dist_out


@wp.kernel
def pack_actor_obs_kernel(
    obs: wp.array2d(dtype=float),
    v1_frame: wp.array2d(dtype=float),
    extras: wp.array2d(dtype=float),
    signal: wp.array(dtype=float),
    raw_hist: wp.array2d(dtype=float),
    instance_world_indices: wp.array(dtype=wp.int32),
    reset_mask: wp.array(dtype=wp.int32),
    apply_mask: int,
    v1_dim: int,
    extras_dim: int,
    signal_dim: int,
    hist_dim: int,
    hist_span: int,
    hist_samples: int,
    hist_stride: int,
):
    tid = wp.tid()
    if apply_mask == 1:
        if reset_mask[instance_world_indices[tid]] != 1:
            return
    for i in range(v1_dim):
        obs[tid, i] = v1_frame[tid, i]
    for i in range(extras_dim):
        obs[tid, v1_dim + i] = extras[tid, i]
    for i in range(signal_dim):
        obs[tid, v1_dim + extras_dim + i] = signal[tid]
    last = hist_span - 1
    hist_offset = v1_dim + extras_dim + signal_dim
    for k in range(hist_samples):
        src_frame = last - (hist_samples - k) * hist_stride
        src = src_frame * hist_dim
        dst = hist_offset + k * hist_dim
        for i in range(hist_dim):
            obs[tid, dst + i] = raw_hist[tid, src + i]


class G1VelocityLocomotionObsActor:
    """Observation/command contract for boxing_v1 (V1 locomotion + extras)."""

    command_labels = list(_COMMAND_POLICY_CFG["labels"])

    def __init__(
        self,
        *,
        num_env: int,
        device: str,
        articulation_body: "ArticulationBody",
        pattern: str,
        history_len: int,
        instance_world_indices: Optional[list[int]] = None,
        instance_view_indices: Optional[list[int]] = None,
        enable_obs_noise: bool = False,
        obs_noise_cfg: Optional[dict] = None,
        encoder_bias_range: Optional[tuple[float, float]] = None,
        base_com_offset_range: Optional[dict[str, tuple[float, float]]] = None,
    ) -> None:
        self.num_env = num_env
        self.device = device
        self.articulation_body = articulation_body
        self.pattern = pattern
        self.history_len = history_len
        self.instance_world_indices = instance_world_indices or list(range(num_env))
        self.instance_view_indices = instance_view_indices or [0] * num_env
        if len(self.instance_world_indices) != len(self.instance_view_indices):
            raise ValueError("G1 obs actor instance world and view index counts must match.")
        self.num_instances = len(self.instance_world_indices)

        self.enable_obs_noise = bool(enable_obs_noise)
        noise = obs_noise_cfg or {}
        required_noise = (
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
        )
        if self.enable_obs_noise:
            missing = [key for key in required_noise if key not in noise]
            if missing:
                raise KeyError(
                    f"environment_configs.observation_noise missing {missing}"
                )
            self.lin_vel_noise = float(noise["base_lin_vel"])
            self.ang_vel_noise = float(noise["base_ang_vel"])
            self.gravity_noise = float(noise["projected_gravity"])
            self.joint_pos_noise = float(noise["joint_pos"])
            self.joint_vel_noise = float(noise["joint_vel"])
        else:
            self.lin_vel_noise = 0.0
            self.ang_vel_noise = 0.0
            self.gravity_noise = 0.0
            self.joint_pos_noise = 0.0
            self.joint_vel_noise = 0.0

        # C4 DR (mjlab startup events): encoder_bias corrupts only the actor
        # obs; base_com_offset approximates the torso COM shift in obs space.
        self.encoder_bias_range: Optional[tuple[float, float]] = encoder_bias_range
        self.base_com_offset_range: Optional[dict[str, tuple[float, float]]] = base_com_offset_range
        self.enable_encoder_bias = encoder_bias_range is not None
        self.enable_base_com = base_com_offset_range is not None
        self.encoder_bias_wp: Optional[wp.array2d] = None
        self.base_com_offset_wp: Optional[wp.array2d] = None

        # C3 curriculum: host-side current ranges + device buffer read by the
        # command-resample kernels (updated by ``set_command_velocity_ranges``).
        self._command_velocity_ranges: dict[str, tuple[float, float]] = {
            k: (float(v[0]), float(v[1]))
            for k, v in _DEFAULT_COMMAND_VELOCITY_RANGES.items()
        }
        self._command_follow_metric_stds: dict[str, float] = {
            k: float(v) for k, v in _COMMAND_FOLLOW_METRIC_STDS.items()
        }
        self._command_resample_cfg: dict[str, float] = {
            k: float(v) for k, v in _COMMAND_RESAMPLE_CFG.items()
        }
        self._tracking_reward_cfg: dict[str, dict] = {
            name: dict(values) for name, values in _TRACKING_REWARD_CFG.items()
        }
        self._skip_command_update_wp: Optional[wp.array] = None
        self._world_hold_wp: Optional[wp.array] = None
        self.cmd_ranges_wp: Optional[wp.array] = None
        self.imu_site_offset_b = _load_imu_site_offset_b()

        self.view = None
        self.rl_action_dim = ACTION_DIM
        self.obs_dim = 0
        self.flat_obs_dim = 0

        self.commands: Optional[wp.array2d] = None
        self.policy_actions: Optional[wp.array2d] = None
        self.prev_actions: Optional[wp.array2d] = None
        self.obs_wp: Optional[wp.array2d] = None
        self.obs_torch: Optional[torch.Tensor] = None
        self.single_obs_wp: Optional[wp.array2d] = None
        self.single_obs_clean_wp: Optional[wp.array2d] = None
        self.noiseless_obs_wp: Optional[wp.array2d] = None
        self.obs_noise_seeds: Optional[wp.array] = None
        self.obs_noise_seed_offsets: Optional[wp.array] = None

        self.heading_target = None
        self.is_heading_env = None
        self.is_standing_env = None
        self.resample_timer = None
        self.cmd_seeds = None
        self.cmd_seed_offsets = None
        self.torch_device = None
        self._boxing_cfg = _BOXING_OBS_CFG
        self.v1_frame_dim = 0
        self.hist_dim = 0
        self.hist_samples = 0
        self.hist_stride = 0
        self.hist_span = 0
        self.extras_dim = 0
        self.signal_dim = 0
        self.signal_wp = None
        self._hist_source_indices: list[int] = []
        self.raw_hist_wp = None
        self.raw_hist_clean_wp = None
        self.hist_frame_wp = None
        self.hist_frame_clean_wp = None
        self.extras_wp = None
        self.has_target_wp = None
        self.target_pos_wp = None
        self.target_vel_wp = None
        self.command_dim = int(_COMMAND_POLICY_CFG["dim"])
        self.command_ranges = list(_COMMAND_POLICY_CFG["ranges"])
        self._target_local_body_wp = None
        self._target_role_local: list[int] = []
        self._target_role_labels: list[str] = []
        self._selected_newton_ids_wp = None
        self._selected_newton_ids = None
        self.defer_command_hold_clear = True

    def setup(self) -> None:
        view_idx = next(
            (i for i, p in enumerate(self.articulation_body.patterns) if p == self.pattern),
            -1,
        )
        if view_idx == -1:
            raise RuntimeError(f"Articulation pattern '{self.pattern}' not found for G1 obs actor.")
        self.view = self.articulation_body.views[view_idx]
        self.rl_action_dim = self.articulation_body.control_rl_action_dim.get(self.pattern, ACTION_DIM)
        cfg = _BOXING_OBS_CFG
        self._boxing_cfg = cfg
        self.v1_frame_dim = 12 + 3 * self.rl_action_dim
        hist_indices = _history_source_indices(self.rl_action_dim, cfg["hist_segments"])
        self._hist_source_indices = list(hist_indices)
        self.hist_dim = len(hist_indices)
        self.hist_samples = int(cfg["hist_samples"])
        self.hist_stride = int(cfg["hist_stride"])
        self.hist_span = int(cfg["hist_span"])
        self.extras_dim = int(cfg["extras_dim"])
        self.signal_dim = int(cfg["signal_dim"])
        packed = (
            self.v1_frame_dim
            + self.extras_dim
            + self.signal_dim
            + self.hist_samples * self.hist_dim
        )
        if packed != int(cfg["packed_obs_dim"]):
            raise ValueError(
                f"Packed boxing obs dim {packed} != control_policy.yaml observation.obs_dim "
                f"{cfg['packed_obs_dim']} (v1={self.v1_frame_dim}, extras={self.extras_dim}, "
                f"signal={self.signal_dim}, hist={self.hist_samples}x{self.hist_dim})"
            )
        self.obs_dim = packed
        self.flat_obs_dim = packed
        self.hist_columns_wp = wp.array(hist_indices, dtype=wp.int32, device=self.device)
        self.pelvis_link_idx = self._resolve_view_link_index(cfg["pelvis_body"])
        self.hand_link_idx = self._resolve_view_link_index(cfg["hand_body"])

        self.instance_world_indices_wp = wp.array(
            self.instance_world_indices, dtype=wp.int32, device=self.device
        )
        self.instance_view_indices_wp = wp.array(
            self.instance_view_indices, dtype=wp.int32, device=self.device
        )
        self.commands = wp.zeros(
            (self.num_instances, self.command_dim), dtype=wp.float32, device=self.device
        )
        self.policy_actions = wp.zeros(
            (self.num_instances, self.rl_action_dim), dtype=wp.float32, device=self.device
        )
        self.prev_actions = wp.zeros(
            (self.num_instances, self.rl_action_dim), dtype=wp.float32, device=self.device
        )
        self.heading_target = wp.zeros(self.num_instances, dtype=wp.float32, device=self.device)
        self.is_heading_env = wp.zeros(self.num_instances, dtype=wp.int32, device=self.device)
        self.is_standing_env = wp.zeros(self.num_instances, dtype=wp.int32, device=self.device)
        self.resample_timer = wp.zeros(self.num_instances, dtype=wp.float32, device=self.device)
        self._skip_command_update_wp = wp.zeros(
            self.num_instances, dtype=wp.int32, device=self.device
        )
        self._world_hold_wp = wp.zeros(self.num_env, dtype=wp.int32, device=self.device)

        seed_base = getattr(GameConfig, "SEED", 31415926)
        self.cmd_seeds = wp.array(
            np.arange(seed_base, seed_base + self.num_instances, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.cmd_seed_offsets = wp.zeros(self.num_instances, dtype=wp.int32, device=self.device)

        self.obs_wp = wp.zeros(
            shape=(self.num_instances, self.flat_obs_dim), dtype=float, device=self.device
        )
        self.noiseless_obs_wp = wp.zeros(
            shape=(self.num_instances, self.flat_obs_dim), dtype=float, device=self.device
        )
        self.single_obs_wp = wp.zeros(
            (self.num_instances, self.v1_frame_dim), dtype=float, device=self.device
        )
        self.single_obs_clean_wp = wp.zeros(
            (self.num_instances, self.v1_frame_dim), dtype=float, device=self.device
        )
        hist_cols = self.hist_span * self.hist_dim
        self.raw_hist_wp = wp.zeros((self.num_instances, hist_cols), dtype=float, device=self.device)
        self.raw_hist_clean_wp = wp.zeros(
            (self.num_instances, hist_cols), dtype=float, device=self.device
        )
        self.hist_frame_wp = wp.zeros(
            (self.num_instances, self.hist_dim), dtype=float, device=self.device
        )
        self.hist_frame_clean_wp = wp.zeros(
            (self.num_instances, self.hist_dim), dtype=float, device=self.device
        )
        self.extras_wp = wp.zeros(
            (self.num_instances, self.extras_dim), dtype=float, device=self.device
        )
        # Action-signal column (retract->punch transition). Starts as a zero
        # array for pure-walk / no-punch-phase environments; punch_phase binds
        # its own signal_wp here when the boxing runtime attaches.
        self.signal_wp = wp.zeros(self.num_instances, dtype=float, device=self.device)
        # Post-normalization no-target sentinels so a skipped overlay cannot
        # leak "target at origin" (legal 0) into the packed actor obs.
        extras_t = wp.to_torch(self.extras_wp)
        extras_t[:, 0:3] = -1.0
        extras_t[:, 12] = -1.0
        self.has_target_wp = wp.zeros(self.num_instances, dtype=wp.int32, device=self.device)
        self.target_pos_wp = wp.zeros((self.num_instances, 3), dtype=float, device=self.device)
        self.target_vel_wp = wp.zeros((self.num_instances, 3), dtype=float, device=self.device)
        self._target_local_body_wp = wp.zeros(
            self.num_instances, dtype=wp.int32, device=self.device
        )
        wp.to_torch(self._target_local_body_wp).fill_(-1)
        self._target_role_local = [-1] * self.num_instances
        self._target_role_labels = [""] * self.num_instances
        self._pack_reset_mask_wp = wp.zeros(self.num_env, dtype=wp.int32, device=self.device)

        noise_seed_base = getattr(GameConfig, "SEED", 31415926) + 70000
        self.obs_noise_seeds = wp.array(
            np.arange(noise_seed_base, noise_seed_base + self.num_instances, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.obs_noise_seed_offsets = wp.zeros(self.num_instances, dtype=wp.int32, device=self.device)

        self.torch_device = wp.device_to_torch(self.device)
        self._selected_newton_ids_wp = wp.zeros(
            (self.num_env, 1), dtype=wp.int32, device=self.device
        )
        wp.to_torch(self._selected_newton_ids_wp).fill_(-1)
        self._selected_newton_ids = wp.to_torch(self._selected_newton_ids_wp)

        # C3: command-range device buffer, seeded from the default ranges.
        self.cmd_ranges_wp = wp.zeros(6, dtype=wp.float32, device=self.device)
        self.set_command_velocity_ranges(self._command_velocity_ranges)

        # C4: encoder bias (per-env, per-RL-joint) and base-COM offset buffers.
        # Zero buffers keep kernel signatures stable when a term is disabled.
        self.encoder_bias_wp = wp.zeros(
            (self.num_instances, self.rl_action_dim), dtype=float, device=self.device
        )
        self.base_com_offset_wp = wp.zeros(
            (self.num_instances, 3), dtype=float, device=self.device
        )
        dr_seed_base = getattr(GameConfig, "SEED", 31415926) + 90000
        dr_seeds = wp.array(
            np.arange(dr_seed_base, dr_seed_base + self.num_instances, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        dr_seed_offsets = wp.zeros(self.num_instances, dtype=wp.int32, device=self.device)
        if self.enable_encoder_bias:
            bmin, bmax = self.encoder_bias_range
            wp.launch(
                init_encoder_bias_kernel,
                dim=self.num_instances,
                inputs=[
                    self.encoder_bias_wp, dr_seeds, dr_seed_offsets,
                    float(bmin), float(bmax), self.rl_action_dim,
                ],
                device=self.device,
            )
        if self.enable_base_com:
            ox = self.base_com_offset_range["x"]
            oy = self.base_com_offset_range["y"]
            oz = self.base_com_offset_range["z"]
            wp.launch(
                init_base_com_offset_kernel,
                dim=self.num_instances,
                inputs=[
                    self.base_com_offset_wp, dr_seeds, dr_seed_offsets,
                    float(ox[0]), float(ox[1]),
                    float(oy[0]), float(oy[1]),
                    float(oz[0]), float(oz[1]),
                ],
                device=self.device,
            )

        self.obs_torch = wp.to_torch(self.obs_wp)

    def validate_dims(self, *, expected_low_level_action_dim: Optional[int] = None) -> None:
        if expected_low_level_action_dim is not None and self.rl_action_dim != expected_low_level_action_dim:
            raise ValueError(
                f"obs_actor rl_action_dim={self.rl_action_dim} != expected {expected_low_level_action_dim}"
            )

    def write_commands_from_rl_actions(
        self,
        actions: wp.array2d,
        action_shape_offset: int,
        player_action_rows: wp.array,
    ) -> None:
        wp.launch(
            write_commands_from_rl_actions_kernel,
            dim=player_action_rows.shape[0],
            inputs=[
                self.commands,
                actions,
                action_shape_offset,
                int(self.command_dim),
                player_action_rows,
            ],
            device=self.device,
        )

    def get_command_velocity_ranges(self) -> dict[str, tuple[float, float]]:
        """Return the current velocity-command sampling ranges.

        Mirrors mjlab's ``Curriculum/command_vel`` so iteration logs can emit
        ``Curriculum/command_vel/<axis>_{min,max}`` matching the reference. A
        copy is returned so callers cannot mutate the shared state.
        """
        return {k: (float(v[0]), float(v[1])) for k, v in self._command_velocity_ranges.items()}

    def get_command_follow_metric_stds(self) -> dict[str, float]:
        """Return per-axis stds for command-following scores.

        Used by ``Metrics/twist/follow_vel_*``: each axis is scored independently
        as ``exp(-(v - cmd)^2 / std^2)``. A copy is returned so callers cannot
        mutate the shared state.
        """
        return {k: float(v) for k, v in self._command_follow_metric_stds.items()}

    def get_tracking_reward_cfg(self) -> dict[str, dict]:
        """Return tracking-reward std / axis-coupling from this version's YAML."""
        return {name: dict(values) for name, values in self._tracking_reward_cfg.items()}

    def get_boxing_obs_cfg(self) -> dict:
        """Return boxing extras / reward constants from this version's YAML."""
        return dict(self._boxing_cfg)

    def try_preferred_target_body(self) -> Optional[str]:
        """Inspector TryGet: body name suffix used when picking a role target."""
        body = str(self._boxing_cfg.get("target_body") or "").strip()
        return body or None

    def try_set_command_role_target(
        self,
        instance_idx: int,
        *,
        local_role_idx: int,
        local_body_idx: int,
        label: str = "",
    ) -> bool:
        """Inspector TryGet: bind one instance's extras target to a role body."""
        return self.try_select_target(
            local_body_idx=int(local_body_idx),
            local_role_idx=int(local_role_idx),
            label=str(label or ""),
            instance_indices=(int(instance_idx),),
        )

    def try_clear_command_role_target(self, instance_idx: int) -> bool:
        idx = int(instance_idx)
        if self._target_local_body_wp is None or idx < 0 or idx >= self.num_instances:
            return False
        wp.to_torch(self._target_local_body_wp)[idx] = -1
        self._target_role_local[idx] = -1
        self._target_role_labels[idx] = ""
        if self.has_target_wp is not None:
            wp.to_torch(self.has_target_wp)[idx] = 0
        if self.target_pos_wp is not None:
            wp.to_torch(self.target_pos_wp)[idx] = 0.0
            wp.to_torch(self.target_vel_wp)[idx] = 0.0
        return True

    def try_clear_all_command_role_targets(self) -> bool:
        """Play/default TryGet: drop every selected target."""
        if self._target_local_body_wp is None:
            return False
        wp.to_torch(self._target_local_body_wp).fill_(-1)
        self._target_role_local = [-1] * self.num_instances
        self._target_role_labels = [""] * self.num_instances
        if self.has_target_wp is not None:
            wp.to_torch(self.has_target_wp).zero_()
        if self.target_pos_wp is not None:
            wp.to_torch(self.target_pos_wp).zero_()
            wp.to_torch(self.target_vel_wp).zero_()
        return True

    def try_get_command_role_target(self, instance_idx: int) -> Optional[dict]:
        idx = int(instance_idx)
        if idx < 0 or idx >= len(self._target_role_local):
            return None
        role = int(self._target_role_local[idx])
        if role < 0:
            return None
        return {
            "local_role_idx": role,
            "label": str(self._target_role_labels[idx] or ""),
        }

    def try_select_target(
        self,
        *,
        local_body_idx: int,
        local_role_idx: int = -1,
        label: str = "",
        instance_indices: Optional[Sequence[int]] = None,
        world_mask: Optional[object] = None,
        physics_manager: Optional["PhysicsManager"] = None,
    ) -> bool:
        """Bind extras to an env-local body for the given instances or worlds."""
        if self._target_local_body_wp is None:
            return False
        selected = list(self._iter_select_instances(instance_indices, world_mask))
        if not selected:
            return False
        buf = wp.to_torch(self._target_local_body_wp)
        body = int(local_body_idx)
        role = int(local_role_idx)
        text = str(label or "")
        for idx in selected:
            buf[idx] = body
            self._target_role_local[idx] = role
            self._target_role_labels[idx] = text
        if physics_manager is not None:
            self.try_sync_selected_target_pose(physics_manager)
        return True

    def _iter_select_instances(
        self,
        instance_indices: Optional[Sequence[int]],
        world_mask: Optional[object],
    ):
        if instance_indices is not None:
            for raw in instance_indices:
                idx = int(raw)
                if 0 <= idx < self.num_instances:
                    yield idx
            return
        if world_mask is None:
            yield from range(self.num_instances)
            return
        mask = world_mask
        if isinstance(mask, wp.array):
            mask = wp.to_torch(mask)
        mask_host = torch.as_tensor(mask).reshape(-1).detach().cpu().tolist()
        for idx, world in enumerate(self.instance_world_indices):
            world_idx = int(world)
            if 0 <= world_idx < len(mask_host) and bool(mask_host[world_idx]):
                yield idx

    def try_sync_selected_target_pose(self, physics_manager: "PhysicsManager") -> None:
        """Copy the selected body pose into extras buffers (graph-safe)."""
        if self._target_local_body_wp is None or self.target_pos_wp is None:
            return
        gather_override_body_poses(
            physics_manager,
            self.instance_world_indices_wp,
            self._target_local_body_wp,
            self.target_pos_wp,
            self.target_vel_wp,
            self.has_target_wp,
            count=int(self.num_instances),
            device=self.device,
        )
        if self._selected_newton_ids_wp is not None:
            fill_selected_newton_body_ids(
                physics_manager,
                self.instance_world_indices_wp,
                self._target_local_body_wp,
                self._selected_newton_ids_wp,
                count=int(self.num_instances),
                num_env=int(self.num_env),
                device=self.device,
            )

    def try_selected_target_newton_body_ids(
        self, physics_manager: "PhysicsManager" = None
    ) -> Optional[torch.Tensor]:
        """Per-env Newton ids of the selected body, shape ``(num_env, 1)``. Unselected is -1."""
        del physics_manager
        if self._selected_newton_ids is None:
            return None
        return self._selected_newton_ids

    def _apply_role_target_overrides(self, physics_manager: "PhysicsManager") -> None:
        self.try_sync_selected_target_pose(physics_manager)

    def try_describe_obs_index(self, index: int) -> Optional[str]:
        """Layout label for a packed actor-obs column (TryGet for NaN reports)."""
        i = int(index)
        v1 = int(self.v1_frame_dim)
        extras = int(self.extras_dim)
        signal_dim = int(self.signal_dim)
        hist_dim = int(self.hist_dim)
        packed = int(self.obs_dim)
        if i < 0 or packed <= 0 or i >= packed:
            return None
        if i < v1:
            return self._v1_column_name(i)
        extra_i = i - v1
        if extra_i < extras:
            names = _BOXING_EXTRAS_NAMES
            if extra_i < len(names):
                return f"extras.{names[extra_i]}"
            return f"extras[{extra_i}]"
        signal_i = extra_i - extras
        if signal_i < signal_dim:
            return "action_signal"
        hist_i = signal_i - signal_dim
        if hist_dim <= 0:
            return f"hist[{hist_i}]"
        sample = hist_i // hist_dim
        col = hist_i % hist_dim
        sources = self._hist_source_indices
        if col < len(sources):
            return f"hist[{sample}].{self._v1_column_name(int(sources[col]))}"
        return f"hist[{sample}][{col}]"

    def _v1_column_name(self, col: int) -> str:
        i = int(col)
        for name, (start, end) in _v1_segment_slices(self.rl_action_dim).items():
            if start <= i < end:
                return f"v1.{name}[{i - start}]"
        return f"v1[{i}]"

    def set_command_velocity_ranges(self, ranges: dict[str, tuple[float, float]]) -> None:
        """Update velocity-command sampling ranges on host and device.

        Writes the 6-float device buffer in-place; the captured CUDA graph's
        command-resample kernels read it by address, so the next graph replay
        picks up the new ranges immediately.
        """
        self._command_velocity_ranges = {
            k: (float(v[0]), float(v[1])) for k, v in ranges.items()
        }
        if self.cmd_ranges_wp is None:
            return
        vals = torch.tensor(
            [
                self._command_velocity_ranges["lin_vel_x"][0],
                self._command_velocity_ranges["lin_vel_x"][1],
                self._command_velocity_ranges["lin_vel_y"][0],
                self._command_velocity_ranges["lin_vel_y"][1],
                self._command_velocity_ranges["ang_vel_z"][0],
                self._command_velocity_ranges["ang_vel_z"][1],
            ],
            dtype=torch.float32,
            device=self.torch_device,
        )
        wp.to_torch(self.cmd_ranges_wp).copy_(vals)

    def hold_external_commands(
        self,
        world_indices: Optional[Sequence[int]] = None,
        instance_indices: Optional[Sequence[int]] = None,
    ) -> None:
        """Skip resample / heading PD only on the worlds or instances being driven.

        A global skip would freeze every training env when ObjectInspector pins
        a single command. Flags live on device so CUDA-graph replay still honors them.
        """
        if self._skip_command_update_wp is None:
            return
        if instance_indices:
            skip = wp.to_torch(self._skip_command_update_wp)
            n = int(skip.shape[0])
            for raw_idx in instance_indices:
                idx = int(raw_idx)
                if 0 <= idx < n:
                    skip[idx] = 1
        if world_indices:
            if self._world_hold_wp is None:
                return
            hold = wp.to_torch(self._world_hold_wp)
            hold.zero_()
            n_world = int(hold.shape[0])
            any_hold = False
            for raw_world in world_indices:
                world = int(raw_world)
                if 0 <= world < n_world:
                    hold[world] = 1
                    any_hold = True
            if any_hold:
                wp.launch(
                    or_skip_from_world_mask_kernel,
                    dim=self.num_instances,
                    inputs=[
                        self._skip_command_update_wp,
                        self.instance_world_indices_wp,
                        self._world_hold_wp,
                    ],
                    device=self.device,
                )

    def update_velocity_commands(self, physics_manager: "PhysicsManager", dt: float) -> None:
        cfg = self._command_resample_cfg
        root_tfs = self.view.get_root_transforms(physics_manager.state_0)
        wp.launch(
            resample_velocity_commands_kernel,
            dim=self.num_instances,
            inputs=[
                self.commands,
                self.heading_target,
                self.is_heading_env,
                self.is_standing_env,
                self.resample_timer,
                self.cmd_seeds,
                self.cmd_seed_offsets,
                dt,
                cfg["interval_min"],
                cfg["interval_max"],
                cfg["rel_standing"],
                cfg["rel_forward"],
                cfg["rel_heading"],
                cfg["forward_min_speed"],
                self.cmd_ranges_wp,
                self._skip_command_update_wp,
            ],
            device=self.device,
        )
        wp.launch(
            update_heading_command_kernel,
            dim=self.num_instances,
            inputs=[
                self.commands,
                self.heading_target,
                self.is_heading_env,
                self.is_standing_env,
                root_tfs,
                self.instance_world_indices_wp,
                self.instance_view_indices_wp,
                cfg["heading_stiffness"],
                cfg["heading_wz_clip"],
                self._skip_command_update_wp,
            ],
            device=self.device,
        )

    def clear_external_command_hold(self) -> None:
        """Consume skip flags after every command-update site in the step has run."""
        if self._skip_command_update_wp is None:
            return
        wp.launch(
            clear_skip_command_update_kernel,
            dim=self.num_instances,
            inputs=[self._skip_command_update_wp],
            device=self.device,
        )

    def reset_commands(self, reset_mask: wp.array) -> None:
        cfg = self._command_resample_cfg
        wp.launch(
            reset_velocity_command_on_env_kernel,
            dim=self.num_instances,
            inputs=[
                self.commands,
                self.heading_target,
                self.is_heading_env,
                self.is_standing_env,
                self.resample_timer,
                reset_mask,
                self.instance_world_indices_wp,
                self.cmd_seeds,
                self.cmd_seed_offsets,
                cfg["rel_standing"],
                cfg["rel_forward"],
                cfg["rel_heading"],
                cfg["forward_min_speed"],
                self.cmd_ranges_wp,
            ],
            device=self.device,
        )

    def reset_policy_actions(self, reset_mask_torch: torch.Tensor) -> None:
        if self.policy_actions is None:
            return
        policy_torch = wp.to_torch(self.policy_actions)
        prev_torch = wp.to_torch(self.prev_actions)
        mask = reset_mask_torch.bool()
        policy_torch[mask] = 0.0
        prev_torch[mask] = 0.0

    def compute_single_frame_obs(self, physics_manager: "PhysicsManager") -> None:
        pm = physics_manager
        view_link_q = self.view.get_link_transforms(pm.state_0)
        view_link_qd = self.view.get_link_velocities(pm.state_0)
        view_joint_q = self.view.get_dof_positions(pm.state_0)
        view_joint_qd = self.view.get_dof_velocities(pm.state_0)

        wp.launch(
            compute_obs_locomotion_kernel,
            dim=self.num_instances,
            inputs=[
                self.single_obs_clean_wp,
                view_link_q,
                view_link_qd,
                view_joint_q,
                view_joint_qd,
                self.articulation_body.control_joint_nominal_qs_gpus[self.pattern],
                self.articulation_body.control_joint_rl_mask_gpus[self.pattern],
                self.articulation_body.control_joint_rl_action_indices_gpus[self.pattern],
                self.commands,
                self.policy_actions,
                self.instance_world_indices_wp,
                self.instance_view_indices_wp,
                self.rl_action_dim,
                self.view.joint_dof_count,
                wp.vec3(0.0, 0.0, -1.0),
                self.imu_site_offset_b,
                self.base_com_offset_wp,
            ],
            device=self.device,
        )

    def apply_obs_noise(self) -> None:
        """Copy the clean frame into the noisy single-frame buffer with per-segment noise."""
        wp.launch(
            apply_obs_noise_kernel,
            dim=self.num_instances,
            inputs=[
                self.single_obs_clean_wp,
                self.single_obs_wp,
                self.obs_noise_seeds,
                self.obs_noise_seed_offsets,
                self.lin_vel_noise,
                self.ang_vel_noise,
                self.gravity_noise,
                self.joint_pos_noise,
                self.joint_vel_noise,
                self.rl_action_dim,
            ],
            device=self.device,
        )

    def _sync_single_frame(self) -> None:
        """Fill ``single_obs_wp`` from the clean frame, adding noise when enabled."""
        if self.enable_obs_noise:
            self.apply_obs_noise()
        else:
            wp.copy(self.single_obs_wp, self.single_obs_clean_wp)
        # C4 encoder bias: corrupt only the actor frame (critic stays clean).
        if self.enable_encoder_bias and self.encoder_bias_wp is not None:
            wp.launch(
                apply_encoder_bias_kernel,
                dim=self.num_instances,
                inputs=[
                    self.single_obs_wp,
                    self.encoder_bias_wp,
                    self.rl_action_dim,
                ],
                device=self.device,
            )

    def _resolve_view_link_index(self, body_name: str) -> int:
        names = list(getattr(self.view, "link_names", ()) or ())
        for i, name in enumerate(names):
            if str(name).endswith(body_name):
                return i
        raise RuntimeError(
            f"Body '{body_name}' not found in view.link_names "
            f"(available: {names[:16]}{'...' if len(names) > 16 else ''})."
        )

    def _extract_history_frames(self) -> None:
        wp.launch(
            gather_columns_kernel,
            dim=self.num_instances,
            inputs=[
                self.single_obs_wp,
                self.hist_frame_wp,
                self.hist_columns_wp,
                self.hist_dim,
            ],
            device=self.device,
        )
        wp.launch(
            gather_columns_kernel,
            dim=self.num_instances,
            inputs=[
                self.single_obs_clean_wp,
                self.hist_frame_clean_wp,
                self.hist_columns_wp,
                self.hist_dim,
            ],
            device=self.device,
        )

    def _append_sampled_history(self) -> None:
        wp.launch(
            shift_and_append_history_kernel,
            dim=self.num_instances,
            inputs=[self.raw_hist_wp, self.hist_frame_wp, self.hist_dim, self.hist_span],
            device=self.device,
        )
        wp.launch(
            shift_and_append_history_kernel,
            dim=self.num_instances,
            inputs=[
                self.raw_hist_clean_wp,
                self.hist_frame_clean_wp,
                self.hist_dim,
                self.hist_span,
            ],
            device=self.device,
        )

    def _compute_extras(self, physics_manager: "PhysicsManager") -> None:
        self._apply_role_target_overrides(physics_manager)
        cfg = self._boxing_cfg
        view_link_q = self.view.get_link_transforms(physics_manager.state_0)
        view_link_qd = self.view.get_link_velocities(physics_manager.state_0)
        wp.launch(
            overlay_and_extras_kernel,
            dim=self.num_instances,
            inputs=[
                self.extras_wp,
                self.commands,
                self.single_obs_wp,
                self.single_obs_clean_wp,
                view_link_q,
                view_link_qd,
                self.has_target_wp,
                self.target_pos_wp,
                self.target_vel_wp,
                self.instance_world_indices_wp,
                self.instance_view_indices_wp,
                self.v1_frame_dim - 3,
                self.pelvis_link_idx,
                self.hand_link_idx,
                float(cfg["max_dist"]),
                float(cfg["max_hand_vel"]),
                float(cfg["standoff_distance"]),
                float(cfg["lin_vel_deadband"]),
                float(cfg["yaw_deadband"]),
                float(cfg["cmd_x_max"]),
                float(cfg["cmd_yaw_max"]),
                self._skip_command_update_wp,
            ],
            device=self.device,
        )

    def _pack_obs(self, reset_mask: Optional[wp.array] = None) -> None:
        apply_mask = 0
        mask = self._pack_reset_mask_wp
        if reset_mask is not None:
            apply_mask = 1
            mask = reset_mask
        pack_inputs = [
            self.obs_wp,
            self.single_obs_wp,
            self.extras_wp,
            self.signal_wp,
            self.raw_hist_wp,
            self.instance_world_indices_wp,
            mask,
            apply_mask,
            self.v1_frame_dim,
            self.extras_dim,
            self.signal_dim,
            self.hist_dim,
            self.hist_span,
            self.hist_samples,
            self.hist_stride,
        ]
        wp.launch(
            pack_actor_obs_kernel,
            dim=self.num_instances,
            inputs=pack_inputs,
            device=self.device,
        )
        pack_clean = list(pack_inputs)
        pack_clean[0] = self.noiseless_obs_wp
        pack_clean[1] = self.single_obs_clean_wp
        # Signal (index 3) stays shared with punch_phase; raw history moved to
        # index 4 after the action-signal column was inserted at index 3.
        pack_clean[4] = self.raw_hist_clean_wp
        wp.launch(
            pack_actor_obs_kernel,
            dim=self.num_instances,
            inputs=pack_clean,
            device=self.device,
        )

    def append_history(self) -> None:
        self._extract_history_frames()
        self._append_sampled_history()

    def reset_history(self, reset_mask: wp.array, physics_manager: "PhysicsManager") -> None:
        if self.obs_wp is None:
            return
        self.compute_single_frame_obs(physics_manager)
        self._sync_single_frame()
        self._extract_history_frames()
        self._compute_extras(physics_manager)
        wp.launch(
            reset_history_kernel,
            dim=self.num_instances,
            inputs=[
                self.raw_hist_wp,
                self.hist_frame_wp,
                reset_mask,
                self.instance_world_indices_wp,
                self.hist_dim,
                self.hist_span,
            ],
            device=self.device,
        )
        wp.launch(
            reset_history_kernel,
            dim=self.num_instances,
            inputs=[
                self.raw_hist_clean_wp,
                self.hist_frame_clean_wp,
                reset_mask,
                self.instance_world_indices_wp,
                self.hist_dim,
                self.hist_span,
            ],
            device=self.device,
        )
        self._pack_obs(reset_mask=reset_mask)

    def get_observation(self, physics_manager: "PhysicsManager") -> torch.Tensor:
        self.compute_single_frame_obs(physics_manager)
        self._sync_single_frame()
        self.append_history()
        self._compute_extras(physics_manager)
        self._pack_obs()
        clearer = getattr(self, "clear_external_command_hold", None)
        if callable(clearer):
            clearer()
        obs = wp.to_torch(self.obs_wp)
        if obs.device != self.torch_device:
            obs = obs.to(self.torch_device, non_blocking=True)
        self.obs_torch = obs
        return obs

    def store_low_level_actions(self, low_level_actions: wp.array2d) -> None:
        expected_shape = (self.num_instances, self.rl_action_dim)
        if low_level_actions.shape != expected_shape:
            raise ValueError(
                "Low-level action shape does not match G1 obs actor instances: "
                f"expected {expected_shape}, got {low_level_actions.shape}."
            )
        wp.copy(self.prev_actions, self.policy_actions)
        wp.copy(self.policy_actions, low_level_actions)


def create_g1_velocity_locomotion_obs_actor(
    *,
    num_env: int,
    device: str,
    articulation_body: "ArticulationBody",
    pattern: str,
    history_len: int,
    instance_world_indices: Optional[list[int]] = None,
    instance_view_indices: Optional[list[int]] = None,
    enable_obs_noise: bool = False,
    obs_noise_cfg: Optional[dict] = None,
    encoder_bias_range: Optional[tuple[float, float]] = None,
    base_com_offset_range: Optional[dict[str, tuple[float, float]]] = None,
) -> G1VelocityLocomotionObsActor:
    obs_actor = G1VelocityLocomotionObsActor(
        num_env=num_env,
        device=device,
        articulation_body=articulation_body,
        pattern=pattern,
        history_len=history_len,
        instance_world_indices=instance_world_indices,
        instance_view_indices=instance_view_indices,
        enable_obs_noise=enable_obs_noise,
        obs_noise_cfg=obs_noise_cfg,
        encoder_bias_range=encoder_bias_range,
        base_com_offset_range=base_com_offset_range,
    )
    obs_actor.setup()
    return obs_actor


create_obs_actor = create_g1_velocity_locomotion_obs_actor
