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

from script.game_config import GameConfig
from script.role.objects.object_template.mjlab_unitree_g1.g1_actuator_model import ACTION_DIM

if TYPE_CHECKING:
    from script.role.bodies.articulation_body import ArticulationBody
    from script.simulate.physics_manager import PhysicsManager

# Pelvis IMU site offset [m] (mjlab unitree_g1 g1.xml ``imu_in_pelvis``).
_IMU_SITE_OFFSET_B = wp.vec3(0.04525, 0.0, -0.08339)


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

    imu_lin_vel = local_lin_vel + wp.cross(local_ang_vel, imu_site_offset_b)

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
):
    tid = wp.tid()
    resample_timer[tid] = resample_timer[tid] - dt
    if resample_timer[tid] > 0.0:
        return

    rng = wp.rand_init(seeds[tid], seed_offsets[tid])
    seed_offsets[tid] = seed_offsets[tid] + 1

    resample_timer[tid] = wp.randf(rng, 3.0, 8.0)
    commands[tid, 0] = wp.randf(rng, -1.0, 1.0)
    commands[tid, 1] = wp.randf(rng, -1.0, 1.0)
    commands[tid, 2] = wp.randf(rng, -0.5, 0.5)

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
):
    tid = wp.tid()
    if reset_mask[instance_world_indices[tid]] != 1:
        return
    rng = wp.rand_init(seeds[tid], seed_offsets[tid])
    seed_offsets[tid] = seed_offsets[tid] + 1
    resample_timer[tid] = 0.0
    commands[tid, 0] = wp.randf(rng, -1.0, 1.0)
    commands[tid, 1] = wp.randf(rng, -1.0, 1.0)
    commands[tid, 2] = wp.randf(rng, -0.5, 0.5)
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
        history_len: int = 1,
        instance_world_indices: Optional[list[int]] = None,
        instance_view_indices: Optional[list[int]] = None,
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
        self.torch_device = wp.device_to_torch(self.device)
        self.obs_torch = wp.to_torch(self.obs_wp)
        self.single_obs_wp = wp.zeros((self.num_instances, self.obs_dim), dtype=float, device=self.device)

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

    def set_commands_torch(self, commands: torch.Tensor) -> None:
        if commands.shape != (self.num_instances, 3):
            raise ValueError(f"Expected commands shape {(self.num_instances, 3)}, got {tuple(commands.shape)}")
        wp.copy(self.commands, wp.from_torch(commands.contiguous()))

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
                self.single_obs_wp if self.history_len > 1 else self.obs_wp,
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
                _IMU_SITE_OFFSET_B,
            ],
            device=self.device,
        )

    def append_history(self) -> None:
        if self.history_len <= 1:
            return
        wp.launch(
            shift_and_append_history_kernel,
            dim=self.num_instances,
            inputs=[self.obs_wp, self.single_obs_wp, self.obs_dim, self.history_len],
            device=self.device,
        )

    def reset_history(self, reset_mask: wp.array, physics_manager: "PhysicsManager") -> None:
        if self.obs_wp is None:
            return
        self.compute_single_frame_obs(physics_manager)
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

    def get_observation(self, physics_manager: "PhysicsManager") -> torch.Tensor:
        self.compute_single_frame_obs(physics_manager)
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
    history_len: int = 1,
    instance_world_indices: Optional[list[int]] = None,
    instance_view_indices: Optional[list[int]] = None,
) -> G1VelocityLocomotionProvider:
    provider = G1VelocityLocomotionProvider(
        num_env=num_env,
        device=device,
        articulation_body=articulation_body,
        pattern=pattern,
        history_len=history_len,
        instance_world_indices=instance_world_indices,
        instance_view_indices=instance_view_indices,
    )
    provider.setup()
    return provider


def register_g1_obs_providers() -> None:
    from script.role.objects.object_template.loader import ensure_object_templates_registered

    ensure_object_templates_registered()
