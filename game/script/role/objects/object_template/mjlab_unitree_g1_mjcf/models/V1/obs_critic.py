"""Unitree G1 foot-state asymmetric critic observation extension.

This module lives inside the ``mjlab_unitree_g1_mjcf`` object template so that the
robot-specific critic observation (policy obs + foot-state extras) is defined
next to the robot itself instead of inside a level. Levels discover it through
``CriticObsRegistry`` and degrade to the symmetric critic when the
robot has no registered ``obs_critic``.
"""

from __future__ import annotations

import torch
import warp as wp

from .foot_sensor_cfg import load_g1_foot_sensor_config, resolve_g1_foot_body_mapping


@wp.kernel
def compute_foot_critic_obs_kernel(
    critic_obs: wp.array2d(dtype=float),
    policy_obs: wp.array2d(dtype=float),
    foot_height: wp.array2d(dtype=float),
    foot_air_time: wp.array2d(dtype=float),
    foot_found: wp.array2d(dtype=wp.int32),
    foot_contact_forces: wp.array2d(dtype=float),
    policy_obs_dim: int,
    critic_extra_dim: int,
    num_feet: int,
):
    tid = wp.tid()
    for i in range(policy_obs_dim):
        critic_obs[tid, i] = policy_obs[tid, i]

    idx = policy_obs_dim
    for f in range(num_feet):
        critic_obs[tid, idx] = foot_height[tid, f]
        idx += 1
    for f in range(num_feet):
        critic_obs[tid, idx] = foot_air_time[tid, f]
        idx += 1
    for f in range(num_feet):
        critic_obs[tid, idx] = wp.float32(foot_found[tid, f])
        idx += 1
    # sign(x) * log1p(|x|) log-compression, mirroring mjlab's foot_contact_forces term.
    for c in range(num_feet * 3):
        v = foot_contact_forces[tid, c]
        critic_obs[tid, idx] = wp.sign(v) * wp.log(1.0 + wp.abs(v))
        idx += 1


class G1FootCriticObs:
    """Asymmetric critic observation for Unitree G1 (policy obs + foot extras).

    The critic sees the same policy observation plus foot-state extras
    (foot_height, foot_air_time, foot_contact, foot_contact_forces per foot).
    Contact buffers are filled each policy step by ``FootContactSensor``;
    this module only packs them into the critic observation.
    """

    def __init__(
        self,
        *,
        num_instances: int,
        policy_obs_dim: int,
        foot_sensor,
        physics_manager,
        device: str,
    ) -> None:
        self.num_instances = num_instances
        self.policy_obs_dim = policy_obs_dim
        self.foot_sensor = foot_sensor
        self.physics_manager = physics_manager
        self.device = device

        foot_cfg = load_g1_foot_sensor_config()
        num_feet = foot_sensor.NUM_FEET if foot_sensor is not None else foot_cfg.num_feet
        # 2 height + 2 air + 2 contact + 2*3 forces (for 2 feet) = 6 * num_feet
        self.critic_obs_dim = num_feet * 6
        self.num_feet = num_feet
        self.critic_obs_wp = None
        self.critic_obs_torch = None
        self._mapping = None

    def setup(self) -> None:
        self.critic_obs_wp = wp.zeros(
            shape=(self.num_instances, self.policy_obs_dim + self.critic_obs_dim),
            dtype=float,
            device=self.device,
        )
        self.critic_obs_torch = wp.to_torch(self.critic_obs_wp)

        # Ensure foot sensor has solver constants / body mapping if the level
        # has not already bound them (e.g. critic constructed before sensor bind).
        if self.foot_sensor is not None and not getattr(self.foot_sensor, "_solver_bound", False):
            mapping = resolve_g1_foot_body_mapping(self.physics_manager, self.device)
            self._mapping = mapping
            self.foot_sensor.bind_solver_constants(
                ngeom=mapping.ngeom,
                njmax=mapping.njmax,
                nbody_mj=mapping.nbody_mj,
                naconmax=mapping.naconmax,
                opt_cone=mapping.opt_cone,
            )

    def get_critic_observation(self, policy_obs_wp) -> torch.Tensor:
        """Return the asymmetric critic observation (policy obs + foot extras).

        Runs outside the CUDA graph. Foot contact buffers must already be
        updated by ``FootContactSensor.update_from_solver`` in the level step.
        """
        if self.critic_obs_wp is None or self.foot_sensor is None:
            return None

        wp.launch(
            kernel=compute_foot_critic_obs_kernel,
            dim=self.num_instances,
            inputs=[
                self.critic_obs_wp,
                policy_obs_wp,
                self.foot_sensor.foot_height,
                self.foot_sensor.foot_air_time,
                self.foot_sensor.foot_found,
                self.foot_sensor.foot_contact_forces,
                self.policy_obs_dim,
                self.critic_obs_dim,
                self.num_feet,
            ],
            device=self.device,
        )
        return self.critic_obs_torch


def create_obs_critic(**kwargs) -> G1FootCriticObs:
    return G1FootCriticObs(**kwargs)
