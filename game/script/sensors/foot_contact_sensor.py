# This file contains code adapted from:
# https://github.com/mujocolab/mjlab
#
# Modified for Action_Game_RL.
#
# The original project is licensed under the Apache License 2.0.

"""Foot contact / height sensor for flat-terrain locomotion training.

Wraps :class:`ContactSensor` (mjlab found / force / air-time semantics via
mujoco-warp contact/efc decode) and adds per-foot height, XY speed, peak
height, and force magnitude buffers consumed by gait rewards and the
asymmetric critic.
"""

from __future__ import annotations

import warp as wp

from script.sensors.contact_sensor import (
    ContactSensor,
    accumulate_primary_contact_forces_kernel,
)


@wp.kernel
def update_foot_kinematics_kernel(
    foot_height: wp.array2d(dtype=float),
    foot_lin_vel_xy_sq: wp.array2d(dtype=float),
    foot_peak_height: wp.array2d(dtype=float),
    foot_force_z: wp.array2d(dtype=float),
    foot_found: wp.array2d(dtype=wp.int32),
    foot_first_contact: wp.array2d(dtype=wp.int32),
    foot_contact_forces: wp.array2d(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    primary_newton_bodies: wp.array2d(dtype=wp.int32),
    ground_height: float,
    foot_height_site_offset: wp.vec3,
    num_feet: int,
):
    """Fill height / slip velocity / force magnitude; track peak height in air."""
    env = wp.tid()
    for f in range(num_feet):
        body = primary_newton_bodies[env, f]
        tf = body_q[body]
        pos = wp.transform_get_translation(tf)
        site = wp.quat_rotate(wp.transform_get_rotation(tf), foot_height_site_offset)
        height = (pos[2] + site[2]) - ground_height
        foot_height[env, f] = height

        qd = body_qd[body]
        vx = qd[0]
        vy = qd[1]
        foot_lin_vel_xy_sq[env, f] = vx * vx + vy * vy

        base = f * 3
        fx = foot_contact_forces[env, base + 0]
        fy = foot_contact_forces[env, base + 1]
        fz = foot_contact_forces[env, base + 2]
        foot_force_z[env, f] = wp.sqrt(fx * fx + fy * fy + fz * fz)

        if foot_found[env, f] == 1:
            # Preserve peak on the first-contact frame so swing rewards can read it;
            # clear on subsequent contact frames.
            if foot_first_contact[env, f] == 0:
                foot_peak_height[env, f] = 0.0
        else:
            if height > foot_peak_height[env, f]:
                foot_peak_height[env, f] = height


@wp.kernel
def reset_foot_extras_kernel(
    reset_mask: wp.array(dtype=wp.int32),
    foot_peak_height: wp.array2d(dtype=float),
    foot_height: wp.array2d(dtype=float),
    foot_lin_vel_xy_sq: wp.array2d(dtype=float),
    foot_force_z: wp.array2d(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    primary_newton_bodies: wp.array2d(dtype=wp.int32),
    ground_height: float,
    foot_height_site_offset: wp.vec3,
    num_feet: int,
):
    env = wp.tid()
    if reset_mask[env] == 0:
        return
    for f in range(num_feet):
        body = primary_newton_bodies[env, f]
        tf = body_q[body]
        pos = wp.transform_get_translation(tf)
        site = wp.quat_rotate(wp.transform_get_rotation(tf), foot_height_site_offset)
        foot_height[env, f] = (pos[2] + site[2]) - ground_height
        foot_lin_vel_xy_sq[env, f] = 0.0
        foot_peak_height[env, f] = 0.0
        foot_force_z[env, f] = 0.0


class FootContactSensor:
    """Per-foot training sensor: real contact + flat-terrain foot kinematics."""

    def __init__(
        self,
        num_env: int,
        device: str,
        *,
        primary_newton_bodies: wp.array,
        num_feet: int,
        ground_geom_id: int = 0,
        ground_height: float = 0.0,
        foot_height_offset: float = 0.0,
        foot_height_site_offset: tuple[float, float, float] | None = None,
        track_air_time: bool = True,
    ):
        if num_feet <= 0:
            raise ValueError("num_feet must be > 0")
        self.num_env = num_env
        self.device = device
        self.NUM_FEET = int(num_feet)
        self.ground_geom_id = int(ground_geom_id)
        self.ground_height = float(ground_height)
        if foot_height_site_offset is None:
            site = (0.0, 0.0, float(foot_height_offset))
        else:
            site = (
                float(foot_height_site_offset[0]),
                float(foot_height_site_offset[1]),
                float(foot_height_site_offset[2]),
            )
        self.foot_height_site_offset = wp.vec3(site[0], site[1], site[2])
        # z-only alias kept so older diagnostics can still print the site drop.
        self.foot_height_offset = float(site[2])
        self.dt = 0.02

        self._contact = ContactSensor(
            num_env=num_env,
            num_primaries=self.NUM_FEET,
            primary_newton_bodies=primary_newton_bodies,
            device=device,
            ground_geom_id=self.ground_geom_id,
            track_air_time=track_air_time,
        )

        # Aliases matching reward / critic buffer names.
        self.foot_found = self._contact.found
        self.foot_air_time = self._contact.current_air_time
        self.foot_contact_time = self._contact.current_contact_time
        self.foot_first_contact = self._contact.first_contact
        self.foot_contact_forces = self._contact.force

        self.foot_force_z = wp.zeros((num_env, self.NUM_FEET), dtype=float, device=device)
        self.foot_height = wp.zeros((num_env, self.NUM_FEET), dtype=float, device=device)
        self.foot_lin_vel_xy_sq = wp.zeros((num_env, self.NUM_FEET), dtype=float, device=device)
        self.foot_peak_height = wp.zeros((num_env, self.NUM_FEET), dtype=float, device=device)

        # Solver constants filled by bind_solver / update_from_solver.
        self._solver_bound = False
        self._ngeom = 0
        self._njmax = 0
        self._nbody_mj = 0
        self._naconmax = 0
        self._opt_cone = 0

    @property
    def primary_newton_bodies(self) -> wp.array:
        return self._contact.primary_newton_bodies

    def bind_solver_constants(
        self,
        *,
        ngeom: int,
        njmax: int,
        nbody_mj: int,
        naconmax: int,
        opt_cone: int,
    ) -> None:
        self._ngeom = int(ngeom)
        self._njmax = int(njmax)
        self._nbody_mj = int(nbody_mj)
        self._naconmax = int(naconmax)
        self._opt_cone = int(opt_cone)
        self._solver_bound = True

    def update_from_solver(
        self,
        *,
        contact,
        efc_force,
        nacon,
        geom_bodyid,
        mjc_body_to_newton,
        body_q,
        body_qd,
        dt: float,
        ngeom: int | None = None,
        njmax: int | None = None,
        nbody_mj: int | None = None,
        naconmax: int | None = None,
        opt_cone: int | None = None,
    ) -> None:
        """Decode real contacts then refresh kinematics (call before rewards)."""
        self.dt = float(dt)
        if ngeom is not None:
            self.bind_solver_constants(
                ngeom=ngeom,
                njmax=int(njmax),
                nbody_mj=int(nbody_mj),
                naconmax=int(naconmax),
                opt_cone=int(opt_cone),
            )
        if not self._solver_bound:
            raise RuntimeError(
                "FootContactSensor.update_from_solver requires bind_solver_constants "
                "or ngeom/njmax/... kwargs."
            )

        self._contact.update_from_solver(
            contact,
            efc_force,
            nacon,
            geom_bodyid,
            mjc_body_to_newton,
            self._ngeom,
            self._njmax,
            self._nbody_mj,
            self._naconmax,
            self._opt_cone,
            self.dt,
        )

        wp.launch(
            kernel=update_foot_kinematics_kernel,
            dim=self.num_env,
            inputs=[
                self.foot_height,
                self.foot_lin_vel_xy_sq,
                self.foot_peak_height,
                self.foot_force_z,
                self.foot_found,
                self.foot_first_contact,
                self.foot_contact_forces,
                body_q,
                body_qd,
                self.primary_newton_bodies,
                self.ground_height,
                self.foot_height_site_offset,
                self.NUM_FEET,
            ],
            device=self.device,
        )

    def update_substep_from_solver(
        self,
        *,
        contact,
        efc_force,
        nacon,
        geom_bodyid,
        mjc_body_to_newton,
        dt: float,
        ngeom: int | None = None,
        njmax: int | None = None,
        nbody_mj: int | None = None,
        naconmax: int | None = None,
        opt_cone: int | None = None,
    ) -> None:
        """Per-physics-substep contact decode + air/contact-time accumulation.

        Mirrors mjlab's per-substep ``ContactSensor.update`` (air-time state
        advances at physics dt). ``first_contact`` / ``first_air`` are NOT
        recomputed here; call ``refresh_policy_step`` at policy-step time so the
        edge windows use the policy-step dt (mjlab ``compute_first_contact(
        dt=step_dt)``).
        """
        self.dt = float(dt)
        if ngeom is not None:
            self.bind_solver_constants(
                ngeom=ngeom,
                njmax=int(njmax),
                nbody_mj=int(nbody_mj),
                naconmax=int(naconmax),
                opt_cone=int(opt_cone),
            )
        if not self._solver_bound:
            raise RuntimeError(
                "FootContactSensor.update_substep_from_solver requires "
                "bind_solver_constants or ngeom/njmax/... kwargs."
            )
        self._contact.substep_update(
            contact,
            efc_force,
            nacon,
            geom_bodyid,
            mjc_body_to_newton,
            self._ngeom,
            self._njmax,
            self._nbody_mj,
            self._naconmax,
            self._opt_cone,
            self.dt,
        )

    def refresh_policy_step(self, *, body_q, body_qd, dt: float) -> None:
        """Policy-step refresh: first-contact/air windows + foot kinematics.

        Air/contact times already accumulated per substep; this recomputes the
        first-contact / first-air flags with the policy-step dt and refreshes
        kinematics for the rewards / critic. No contact re-decode.
        """
        self.dt = float(dt)
        self._contact.compute_first_contact(self.dt)
        self._contact.compute_first_air(self.dt)
        wp.launch(
            kernel=update_foot_kinematics_kernel,
            dim=self.num_env,
            inputs=[
                self.foot_height,
                self.foot_lin_vel_xy_sq,
                self.foot_peak_height,
                self.foot_force_z,
                self.foot_found,
                self.foot_first_contact,
                self.foot_contact_forces,
                body_q,
                body_qd,
                self.primary_newton_bodies,
                self.ground_height,
                self.foot_height_site_offset,
                self.NUM_FEET,
            ],
            device=self.device,
        )

    def update_foot_contact_forces(
        self,
        contact,
        efc_force,
        nacon,
        geom_bodyid,
        mjc_body_to_newton,
        ankle_newton_bodies,
        ngeom: int,
        njmax: int,
        nbody_mj: int,
        naconmax: int,
        opt_cone: int,
    ) -> None:
        """Backward-compat: re-decode forces only (kinematics unchanged).

        Prefer ``update_from_solver`` once per policy step so rewards and critic
        share the same buffers. Kept for callers that still pass ankle bodies.
        """
        del ankle_newton_bodies  # primary bodies already bound on construction
        self.bind_solver_constants(
            ngeom=ngeom,
            njmax=njmax,
            nbody_mj=nbody_mj,
            naconmax=naconmax,
            opt_cone=opt_cone,
        )
        self._contact.found.zero_()
        self._contact.force.zero_()
        wp.launch(
            kernel=accumulate_primary_contact_forces_kernel,
            dim=naconmax,
            inputs=[
                self._contact.found,
                self._contact.force,
                contact.worldid,
                contact.geom,
                contact.frame,
                contact.friction,
                contact.dim,
                contact.efc_address,
                geom_bodyid,
                mjc_body_to_newton,
                self.primary_newton_bodies,
                efc_force,
                ngeom,
                njmax,
                nbody_mj,
                self.NUM_FEET,
                self.ground_geom_id,
                nacon,
                opt_cone,
            ],
            device=self.device,
        )

    def reset_envs(self, terminated_mask, body_q=None, **_kwargs):
        """Reset contact + kinematic extras for terminated environments."""
        if body_q is None:
            raise ValueError("FootContactSensor.reset_envs requires body_q.")
        if isinstance(terminated_mask, wp.array):
            mask = terminated_mask
        else:
            mask = wp.from_torch(terminated_mask, dtype=wp.int32)

        self._contact.reset(mask)
        wp.launch(
            kernel=reset_foot_extras_kernel,
            dim=self.num_env,
            inputs=[
                mask,
                self.foot_peak_height,
                self.foot_height,
                self.foot_lin_vel_xy_sq,
                self.foot_force_z,
                body_q,
                self.primary_newton_bodies,
                self.ground_height,
                self.foot_height_site_offset,
                self.NUM_FEET,
            ],
            device=self.device,
        )
