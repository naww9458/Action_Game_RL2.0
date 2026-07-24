import warp as wp
import numpy as np


@wp.kernel
def update_foot_contact_kernel(
    foot_found: wp.array2d(dtype=wp.int32),
    foot_air_time: wp.array2d(dtype=float),
    foot_contact_time: wp.array2d(dtype=float),
    foot_force_z: wp.array2d(dtype=float),
    foot_height: wp.array2d(dtype=float),
    foot_lin_vel_xy_sq: wp.array2d(dtype=float),
    foot_peak_height: wp.array2d(dtype=float),
    foot_first_contact: wp.array2d(dtype=wp.int32),
    root_tfs: wp.array2d(dtype=wp.transform),
    root_vels: wp.array2d(dtype=wp.spatial_vector),
    dt: float,
    contact_height_threshold: float,
    force_threshold: float,
):
    env = wp.tid()
    root_tf = root_tfs[env, 0]
    root_pos = wp.transform_get_translation(root_tf)
    root_qd = root_vels[env, 0]
    root_vx = root_qd[0]
    root_vy = root_qd[1]

    for f in range(2):
        prev_found = foot_found[env, f]
        height = root_pos[2] - contact_height_threshold
        if f == 1:
            height = height - 0.02
        foot_height[env, f] = height

        vel_xy_sq = root_vx * root_vx + root_vy * root_vy
        foot_lin_vel_xy_sq[env, f] = vel_xy_sq

        found = 0
        if height < 0.03:
            found = 1
            foot_force_z[env, f] = force_threshold + 1.0
        else:
            foot_force_z[env, f] = 0.0

        if found == 1:
            foot_air_time[env, f] = 0.0
            foot_contact_time[env, f] = foot_contact_time[env, f] + dt
            if prev_found == 0:
                foot_first_contact[env, f] = 1
            else:
                foot_first_contact[env, f] = 0
            foot_peak_height[env, f] = 0.0
        else:
            foot_contact_time[env, f] = 0.0
            foot_air_time[env, f] = foot_air_time[env, f] + dt
            foot_first_contact[env, f] = 0
            if height > foot_peak_height[env, f]:
                foot_peak_height[env, f] = height

        foot_found[env, f] = found


@wp.kernel
def reset_foot_sensor_kernel(
    reset_mask: wp.array(dtype=wp.int32),
    foot_found: wp.array2d(dtype=wp.int32),
    foot_air_time: wp.array2d(dtype=float),
    foot_contact_time: wp.array2d(dtype=float),
    foot_peak_height: wp.array2d(dtype=float),
    foot_first_contact: wp.array2d(dtype=wp.int32),
):
    env = wp.tid()
    if reset_mask[env] == 1:
        for f in range(2):
            foot_found[env, f] = 0
            foot_air_time[env, f] = 0.0
            foot_contact_time[env, f] = 0.0
            foot_peak_height[env, f] = 0.0
            foot_first_contact[env, f] = 0


class FootContactSensor:
    """Simplified foot contact / height sensor for flat terrain locomotion."""

    NUM_FEET = 2

    def __init__(self, num_env: int, device: str):
        self.num_env = num_env
        self.device = device
        self.dt = 0.02

        self.foot_found = wp.zeros((num_env, self.NUM_FEET), dtype=wp.int32, device=device)
        self.foot_air_time = wp.zeros((num_env, self.NUM_FEET), dtype=float, device=device)
        self.foot_contact_time = wp.zeros((num_env, self.NUM_FEET), dtype=float, device=device)
        self.foot_force_z = wp.zeros((num_env, self.NUM_FEET), dtype=float, device=device)
        self.foot_height = wp.zeros((num_env, self.NUM_FEET), dtype=float, device=device)
        self.foot_lin_vel_xy_sq = wp.zeros((num_env, self.NUM_FEET), dtype=float, device=device)
        self.foot_peak_height = wp.zeros((num_env, self.NUM_FEET), dtype=float, device=device)
        self.foot_first_contact = wp.zeros((num_env, self.NUM_FEET), dtype=wp.int32, device=device)

    def update(self, root_tfs, root_vels, dt: float):
        self.dt = dt
        wp.launch(
            kernel=update_foot_contact_kernel,
            dim=self.num_env,
            inputs=[
                self.foot_found,
                self.foot_air_time,
                self.foot_contact_time,
                self.foot_force_z,
                self.foot_height,
                self.foot_lin_vel_xy_sq,
                self.foot_peak_height,
                self.foot_first_contact,
                root_tfs,
                root_vels,
                dt,
                0.74,
                10.0,
            ],
            device=self.device,
        )

    def reset_envs(self, terminated_mask):
        if isinstance(terminated_mask, wp.array):
            mask = terminated_mask
        else:
            mask = wp.from_torch(terminated_mask, dtype=wp.int32)

        wp.launch(
            kernel=reset_foot_sensor_kernel,
            dim=self.num_env,
            inputs=[mask, self.foot_found, self.foot_air_time, self.foot_contact_time,
                    self.foot_peak_height, self.foot_first_contact],
            device=self.device,
        )
