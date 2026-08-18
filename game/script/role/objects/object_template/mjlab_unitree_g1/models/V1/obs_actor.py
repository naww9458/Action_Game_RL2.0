# This file contains code adapted from:
# https://github.com/mujocolab/mjlab
#
# Modified for Action_Game_RL.
#
# The original project is licensed under the Apache License 2.0.

"""G1 velocity locomotion observation/command provider (mjlab-aligned)."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
import warp as wp
import yaml

from pathlib import Path

from script.game_config import GameConfig

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


_DEFAULT_COMMAND_VELOCITY_RANGES = _default_command_velocity_ranges()
ACTION_DIM = _action_dim()


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


@wp.kernel
def resample_velocity_commands_kernel(
    commands: wp.array2d(dtype=float),
    heading_target: wp.array(dtype=float),
    is_heading_env: wp.array(dtype=wp.int32),
    is_standing_env: wp.array(dtype=wp.int32),
    is_forward_env: wp.array(dtype=wp.int32),
    resample_timer: wp.array(dtype=float),
    resample_interval: wp.array(dtype=float),
    seeds: wp.array(dtype=wp.int32),
    seed_offsets: wp.array(dtype=wp.int32),
    dt: float,
    rel_standing: float,
    rel_heading: float,
    rel_forward: float,
    command_ranges: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    resample_timer[tid] = resample_timer[tid] - dt
    if resample_timer[tid] > 0.0:
        return

    rng = wp.rand_init(seeds[tid], seed_offsets[tid])
    seed_offsets[tid] = seed_offsets[tid] + 1

    lin_vel_x_min = command_ranges[0]
    lin_vel_x_max = command_ranges[1]
    lin_vel_y_min = command_ranges[2]
    lin_vel_y_max = command_ranges[3]
    ang_vel_z_min = command_ranges[4]
    ang_vel_z_max = command_ranges[5]

    resample_timer[tid] = wp.randf(rng, 3.0, 8.0)
    commands[tid, 0] = wp.randf(rng, lin_vel_x_min, lin_vel_x_max)
    commands[tid, 1] = wp.randf(rng, lin_vel_y_min, lin_vel_y_max)
    commands[tid, 2] = wp.randf(rng, ang_vel_z_min, ang_vel_z_max)

    is_heading_env[tid] = 1 if wp.randf(rng, 0.0, 1.0) <= rel_heading else 0
    is_standing_env[tid] = 1 if wp.randf(rng, 0.0, 1.0) <= rel_standing else 0
    is_forward_env[tid] = 1 if wp.randf(rng, 0.0, 1.0) <= rel_forward else 0

    if is_forward_env[tid] == 1:
        vx = wp.abs(commands[tid, 0])
        if vx < 0.3:
            vx = 0.3
        commands[tid, 0] = vx
        commands[tid, 1] = 0.0
        commands[tid, 2] = 0.0

    if is_standing_env[tid] == 1:
        commands[tid, 0] = 0.0
        commands[tid, 1] = 0.0
        commands[tid, 2] = 0.0

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
):
    tid = wp.tid()
    if is_standing_env[tid] == 1 or is_heading_env[tid] == 0:
        return

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
    if wz > 0.5:
        wz = 0.5
    if wz < -0.5:
        wz = -0.5
    commands[tid, 2] = wz


@wp.kernel
def reset_velocity_command_on_env_kernel(
    commands: wp.array2d(dtype=float),
    heading_target: wp.array(dtype=float),
    is_heading_env: wp.array(dtype=wp.int32),
    is_standing_env: wp.array(dtype=wp.int32),
    is_forward_env: wp.array(dtype=wp.int32),
    resample_timer: wp.array(dtype=float),
    reset_mask: wp.array(dtype=wp.int32),
    instance_world_indices: wp.array(dtype=wp.int32),
    seeds: wp.array(dtype=wp.int32),
    seed_offsets: wp.array(dtype=wp.int32),
    rel_standing: float,
    rel_heading: float,
    rel_forward: float,
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
    is_heading_env[tid] = 1 if wp.randf(rng, 0.0, 1.0) <= rel_heading else 0
    is_standing_env[tid] = 1 if wp.randf(rng, 0.0, 1.0) <= rel_standing else 0
    is_forward_env[tid] = 1 if wp.randf(rng, 0.0, 1.0) <= rel_forward else 0
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


class G1VelocityLocomotionProvider:
    """Observation/command contract for Mjlab-Velocity-Flat-Unitree-G1."""

    command_labels = ["vx (m/s)", "vy (m/s)", "wz (rad/s)"]

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
            raise ValueError("G1 provider instance world and view index counts must match.")
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
        self.is_forward_env = None
        self.resample_timer = None
        self.cmd_seeds = None
        self.cmd_seed_offsets = None
        self.torch_device = None

    def setup(self) -> None:
        view_idx = next(
            (i for i, p in enumerate(self.articulation_body.patterns) if p == self.pattern),
            -1,
        )
        if view_idx == -1:
            raise RuntimeError(f"Articulation pattern '{self.pattern}' not found for G1 provider.")
        self.view = self.articulation_body.views[view_idx]
        self.rl_action_dim = self.articulation_body.control_rl_action_dim.get(self.pattern, ACTION_DIM)
        self.obs_dim = 12 + 3 * self.rl_action_dim
        self.flat_obs_dim = self.obs_dim * self.history_len

        self.instance_world_indices_wp = wp.array(
            self.instance_world_indices, dtype=wp.int32, device=self.device
        )
        self.instance_view_indices_wp = wp.array(
            self.instance_view_indices, dtype=wp.int32, device=self.device
        )
        self.commands = wp.zeros((self.num_instances, 3), dtype=wp.float32, device=self.device)
        self.policy_actions = wp.zeros(
            (self.num_instances, self.rl_action_dim), dtype=wp.float32, device=self.device
        )
        self.prev_actions = wp.zeros(
            (self.num_instances, self.rl_action_dim), dtype=wp.float32, device=self.device
        )
        self.heading_target = wp.zeros(self.num_instances, dtype=wp.float32, device=self.device)
        self.is_heading_env = wp.zeros(self.num_instances, dtype=wp.int32, device=self.device)
        self.is_standing_env = wp.zeros(self.num_instances, dtype=wp.int32, device=self.device)
        self.is_forward_env = wp.zeros(self.num_instances, dtype=wp.int32, device=self.device)
        self.resample_timer = wp.zeros(self.num_instances, dtype=wp.float32, device=self.device)

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
        self.single_obs_wp = wp.zeros((self.num_instances, self.obs_dim), dtype=float, device=self.device)
        self.single_obs_clean_wp = wp.zeros((self.num_instances, self.obs_dim), dtype=float, device=self.device)

        noise_seed_base = getattr(GameConfig, "SEED", 31415926) + 70000
        self.obs_noise_seeds = wp.array(
            np.arange(noise_seed_base, noise_seed_base + self.num_instances, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.obs_noise_seed_offsets = wp.zeros(self.num_instances, dtype=wp.int32, device=self.device)

        self.torch_device = wp.device_to_torch(self.device)

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
                f"Provider rl_action_dim={self.rl_action_dim} != expected {expected_low_level_action_dim}"
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
                3,
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

    def update_velocity_commands(self, physics_manager: "PhysicsManager", dt: float) -> None:
        root_tfs = self.view.get_root_transforms(physics_manager.state_0)
        wp.launch(
            resample_velocity_commands_kernel,
            dim=self.num_instances,
            inputs=[
                self.commands,
                self.heading_target,
                self.is_heading_env,
                self.is_standing_env,
                self.is_forward_env,
                self.resample_timer,
                self.resample_timer,
                self.cmd_seeds,
                self.cmd_seed_offsets,
                dt,
                0.1,
                0.3,
                0.2,
                self.cmd_ranges_wp,
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
                0.5,
            ],
            device=self.device,
        )

    def reset_commands(self, reset_mask: wp.array) -> None:
        wp.launch(
            reset_velocity_command_on_env_kernel,
            dim=self.num_instances,
            inputs=[
                self.commands,
                self.heading_target,
                self.is_heading_env,
                self.is_standing_env,
                self.is_forward_env,
                self.resample_timer,
                reset_mask,
                self.instance_world_indices_wp,
                self.cmd_seeds,
                self.cmd_seed_offsets,
                0.1,
                0.3,
                0.2,
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

    def append_history(self) -> None:
        if self.history_len <= 1:
            wp.copy(self.obs_wp, self.single_obs_wp)
            wp.copy(self.noiseless_obs_wp, self.single_obs_clean_wp)
            return
        wp.launch(
            shift_and_append_history_kernel,
            dim=self.num_instances,
            inputs=[self.obs_wp, self.single_obs_wp, self.obs_dim, self.history_len],
            device=self.device,
        )
        wp.launch(
            shift_and_append_history_kernel,
            dim=self.num_instances,
            inputs=[self.noiseless_obs_wp, self.single_obs_clean_wp, self.obs_dim, self.history_len],
            device=self.device,
        )

    def reset_history(self, reset_mask: wp.array, physics_manager: "PhysicsManager") -> None:
        if self.obs_wp is None:
            return
        self.compute_single_frame_obs(physics_manager)
        self._sync_single_frame()
        wp.launch(
            reset_history_kernel,
            dim=self.num_instances,
            inputs=[
                self.obs_wp,
                self.single_obs_wp,
                reset_mask,
                self.instance_world_indices_wp,
                self.obs_dim,
                self.history_len,
            ],
            device=self.device,
        )
        wp.launch(
            reset_history_kernel,
            dim=self.num_instances,
            inputs=[
                self.noiseless_obs_wp,
                self.single_obs_clean_wp,
                reset_mask,
                self.instance_world_indices_wp,
                self.obs_dim,
                self.history_len,
            ],
            device=self.device,
        )

    def get_observation(self, physics_manager: "PhysicsManager") -> torch.Tensor:
        self.compute_single_frame_obs(physics_manager)
        self._sync_single_frame()
        self.append_history()
        obs = wp.to_torch(self.obs_wp)
        if obs.device != self.torch_device:
            obs = obs.to(self.torch_device, non_blocking=True)
        self.obs_torch = obs
        return obs

    def store_low_level_actions(self, low_level_actions: wp.array2d) -> None:
        expected_shape = (self.num_instances, self.rl_action_dim)
        if low_level_actions.shape != expected_shape:
            raise ValueError(
                "Low-level action shape does not match G1 provider instances: "
                f"expected {expected_shape}, got {low_level_actions.shape}."
            )
        wp.copy(self.prev_actions, self.policy_actions)
        wp.copy(self.policy_actions, low_level_actions)


def create_g1_velocity_locomotion_provider(
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
) -> G1VelocityLocomotionProvider:
    provider = G1VelocityLocomotionProvider(
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
    provider.setup()
    return provider


create_obs_actor = create_g1_velocity_locomotion_provider
