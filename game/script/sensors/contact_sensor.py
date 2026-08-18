# This file contains code adapted from:
# https://github.com/mujocolab/mjlab
#
# Modified for Action_Game_RL.
#
# The original project is licensed under the Apache License 2.0.

"""Contact sensors for gameplay role matrix and training gait signals.

``RoleContactSensor`` tracks Newton rigid contacts at the role/object level
(used by ``physics_manager`` for gameplay collision queries).

``ContactSensor`` mirrors mjlab's contact-sensor semantics (found / force /
air-time) but decodes from mujoco-warp ``contact`` + ``efc.force`` instead of
``mjSENS_CONTACT`` / ``sensordata``, which Newton does not wire through.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5

# Contact kind tags for raw buffer extensibility (soft/fluid future use)
CONTACT_KIND_RIGID = 0
CONTACT_KIND_SOFT = 1
CONTACT_KIND_FLUID = 2

INT32_MAX = 2**31 - 1
GROUND_SHAPE_INDEX = 0


@wp.kernel
def record_contacts_kernel(
    contact_count: wp.array(dtype=int),
    shape0_array: wp.array(dtype=int),
    shape1_array: wp.array(dtype=int),
    shape_to_role: wp.array(dtype=int),
    num_roles: int,
    role_contact_matrix: wp.array(dtype=int),
    ground_contact_flags: wp.array(dtype=int),
    raw_shape0: wp.array(dtype=int),
    raw_shape1: wp.array(dtype=int),
    raw_contact_kind: wp.array(dtype=int),
    raw_count: wp.array(dtype=int),
    raw_capacity: int,
    contact_kind: int,
):
    tid = wp.tid()
    limit = contact_count[0]

    if tid >= limit:
        return

    s0 = shape0_array[tid]
    s1 = shape1_array[tid]

    slot = wp.atomic_add(raw_count, 0, 1)
    if slot < raw_capacity:
        raw_shape0[slot] = s0
        raw_shape1[slot] = s1
        raw_contact_kind[slot] = contact_kind

    if s0 == GROUND_SHAPE_INDEX or s1 == GROUND_SHAPE_INDEX:
        other = s1
        if s1 == GROUND_SHAPE_INDEX:
            other = s0
        if other > 0 and other < shape_to_role.shape[0]:
            role = shape_to_role[other]
            if role >= 0 and role < num_roles:
                wp.atomic_max(ground_contact_flags, role, 1)

    if s0 <= 0 or s1 <= 0:
        return
    if s0 >= shape_to_role.shape[0] or s1 >= shape_to_role.shape[0]:
        return

    r0 = shape_to_role[s0]
    r1 = shape_to_role[s1]

    if r0 < 0 or r1 < 0:
        return
    if r0 >= num_roles or r1 >= num_roles:
        return

    idx1 = r0 * num_roles + r1
    idx2 = r1 * num_roles + r0
    wp.atomic_max(role_contact_matrix, idx1, 1)
    wp.atomic_max(role_contact_matrix, idx2, 1)


def build_shape_to_role_map(
    shape_count: int,
    role_shape_ranges: list[tuple[int, int, int]],
    template_shape_count: int,
    num_env: int,
    num_objects_env: int,
) -> np.ndarray:
    """Map Newton shape indices to logical role object indices."""
    shape_to_role = np.full(shape_count, -1, dtype=np.int32)

    for world in range(num_env):
        role_offset = world * num_objects_env
        shape_offset = 1 + world * template_shape_count

        for shape_begin, shape_end, role_index in role_shape_ranges:
            global_role = role_index + role_offset
            for local_s in range(shape_begin, shape_end):
                global_shape = shape_offset + local_s
                if 0 <= global_shape < shape_count:
                    shape_to_role[global_shape] = global_role

    return shape_to_role


class RoleContactSensor:
    """Scalable contact tracking at role/object level with bounded raw contact buffer."""

    def __init__(
        self,
        num_roles: int,
        shape_count: int,
        shape_to_role_np: np.ndarray,
        raw_capacity: int,
        device: str,
    ):
        self.num_roles = num_roles
        self.shape_count = shape_count
        self.device = device
        self.raw_capacity = raw_capacity

        matrix_size = num_roles * num_roles
        if matrix_size > INT32_MAX:
            raise ValueError(
                f"Role contact matrix size {matrix_size} exceeds int32 limit. "
                f"Reduce num_objects_total ({num_roles})."
            )

        self.role_contact_matrix = wp.zeros(matrix_size, dtype=wp.int32, device=device)
        self.ground_contact_flags = wp.zeros(num_roles, dtype=wp.int32, device=device)
        self.shape_to_role_gpu = wp.array(shape_to_role_np, dtype=wp.int32, device=device)

        self.raw_shape0 = wp.zeros(raw_capacity, dtype=wp.int32, device=device)
        self.raw_shape1 = wp.zeros(raw_capacity, dtype=wp.int32, device=device)
        self.raw_contact_kind = wp.zeros(raw_capacity, dtype=wp.int32, device=device)
        self.raw_count = wp.zeros(1, dtype=wp.int32, device=device)

        matrix_bytes = matrix_size * 4
        map_bytes = shape_count * 4
        raw_bytes = raw_capacity * 12
        print(
            f"RoleContactSensor: num_roles={num_roles}, shape_count={shape_count}, "
            f"role_matrix={matrix_bytes / (1024 * 1024):.2f} MB, "
            f"shape_map={map_bytes / (1024 * 1024):.2f} MB, "
            f"raw_buffer_cap={raw_capacity} ({raw_bytes / 1024:.1f} KB)"
        )

    @property
    def collision_matrix(self):
        """Backward-compat alias for role-level contact matrix."""
        return self.role_contact_matrix

    def reset_frame(self):
        self.role_contact_matrix.zero_()
        self.ground_contact_flags.zero_()
        self.raw_count.zero_()

    def record_rigid_contacts(self, contacts):
        wp.launch(
            kernel=record_contacts_kernel,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                self.shape_to_role_gpu,
                self.num_roles,
                self.role_contact_matrix,
                self.ground_contact_flags,
                self.raw_shape0,
                self.raw_shape1,
                self.raw_contact_kind,
                self.raw_count,
                self.raw_capacity,
                CONTACT_KIND_RIGID,
            ],
            device=self.device,
        )

    def check_role_collision(self, role_a: int, role_b: int) -> bool:
        matrix_np = self.role_contact_matrix.numpy()
        idx = role_a * self.num_roles + role_b
        return matrix_np[idx] == 1


# ---------------------------------------------------------------------------
# Training ContactSensor (mjlab semantics via mujoco-warp contact/efc decode)
# ---------------------------------------------------------------------------


@wp.kernel
def accumulate_primary_contact_forces_kernel(
    found: wp.array2d(dtype=wp.int32),
    force_flat: wp.array2d(dtype=float),
    contact_worldid: wp.array(dtype=wp.int32),
    contact_geom: wp.array(dtype=wp.vec2i),
    contact_frame: wp.array(dtype=wp.mat33),
    contact_friction: wp.array(dtype=vec5),
    contact_dim: wp.array(dtype=wp.int32),
    contact_efc_address: wp.array(dtype=wp.int32, ndim=2),
    geom_bodyid: wp.array(dtype=wp.int32),
    mjc_body_to_newton: wp.array(dtype=wp.int32, ndim=2),
    primary_newton_bodies: wp.array2d(dtype=wp.int32),
    efc_force: wp.array2d(dtype=float),
    ngeom: int,
    njmax: int,
    nbody_mj: int,
    num_primaries: int,
    ground_geom_id: int,
    nacon: wp.array(dtype=wp.int32),
    opt_cone: int,
):
    """Accumulate world-frame GRF per primary that contacts the ground geom.

    Thread per contact slot. ``force_flat`` layout is ``[B, P*3]`` (x,y,z per
    primary). ``found`` is set to 1 for any matched ground contact.
    """
    c = wp.tid()
    if c >= nacon[0]:
        return

    w = contact_worldid[c]
    g = contact_geom[c]
    if g[0] != ground_geom_id and g[1] != ground_geom_id:
        return

    primary = int(-1)
    force_sign = float(1.0)
    for k in range(2):
        geom = g[k]
        if geom < 0 or geom >= ngeom:
            continue
        body = geom_bodyid[geom]
        if body < 0 or body >= nbody_mj:
            continue
        nb = mjc_body_to_newton[w, body]
        for p in range(num_primaries):
            if nb == primary_newton_bodies[w, p]:
                primary = p
                force_sign = -1.0 if k == 0 else 1.0
                break
        if primary >= 0:
            break
    if primary < 0:
        return

    force = contact_force_fn(
        opt_cone,
        contact_frame,
        contact_friction,
        contact_dim,
        contact_efc_address,
        efc_force,
        njmax,
        nacon,
        w,
        c,
        True,
    )
    found[w, primary] = 1
    base = primary * 3
    for i in range(3):
        wp.atomic_add(force_flat, w, base + i, force_sign * force[i])


@wp.kernel
def update_air_time_kernel(
    found: wp.array2d(dtype=wp.int32),
    current_air_time: wp.array2d(dtype=float),
    last_air_time: wp.array2d(dtype=float),
    current_contact_time: wp.array2d(dtype=float),
    last_contact_time: wp.array2d(dtype=float),
    dt: float,
    num_primaries: int,
):
    """Port of mjlab ContactSensor._update_air_time_tracking (elapsed = dt)."""
    env = wp.tid()
    for p in range(num_primaries):
        is_contact = found[env, p] > 0
        was_air = current_air_time[env, p] > 0.0
        was_contact = current_contact_time[env, p] > 0.0

        if is_contact and was_air:
            last_air_time[env, p] = current_air_time[env, p] + dt
        if (not is_contact) and was_contact:
            last_contact_time[env, p] = current_contact_time[env, p] + dt

        if is_contact:
            current_air_time[env, p] = 0.0
            current_contact_time[env, p] = current_contact_time[env, p] + dt
        else:
            current_contact_time[env, p] = 0.0
            current_air_time[env, p] = current_air_time[env, p] + dt


@wp.kernel
def compute_first_contact_kernel(
    current_contact_time: wp.array2d(dtype=float),
    first_contact: wp.array2d(dtype=wp.int32),
    dt: float,
    abs_tol: float,
    num_primaries: int,
):
    """mjlab compute_first_contact: landed within the last dt seconds."""
    env = wp.tid()
    for p in range(num_primaries):
        t = current_contact_time[env, p]
        if t > 0.0 and t < (dt + abs_tol):
            first_contact[env, p] = 1
        else:
            first_contact[env, p] = 0


@wp.kernel
def compute_first_air_kernel(
    current_air_time: wp.array2d(dtype=float),
    first_air: wp.array2d(dtype=wp.int32),
    dt: float,
    abs_tol: float,
    num_primaries: int,
):
    """mjlab compute_first_air: took off within the last dt seconds."""
    env = wp.tid()
    for p in range(num_primaries):
        t = current_air_time[env, p]
        if t > 0.0 and t < (dt + abs_tol):
            first_air[env, p] = 1
        else:
            first_air[env, p] = 0


@wp.kernel
def reset_contact_sensor_kernel(
    reset_mask: wp.array(dtype=wp.int32),
    found: wp.array2d(dtype=wp.int32),
    force_flat: wp.array2d(dtype=float),
    current_air_time: wp.array2d(dtype=float),
    last_air_time: wp.array2d(dtype=float),
    current_contact_time: wp.array2d(dtype=float),
    last_contact_time: wp.array2d(dtype=float),
    num_primaries: int,
):
    env = wp.tid()
    if reset_mask[env] == 0:
        return
    for p in range(num_primaries):
        found[env, p] = 0
        current_air_time[env, p] = 0.0
        last_air_time[env, p] = 0.0
        current_contact_time[env, p] = 0.0
        last_contact_time[env, p] = 0.0
        base = p * 3
        force_flat[env, base + 0] = 0.0
        force_flat[env, base + 1] = 0.0
        force_flat[env, base + 2] = 0.0


class ContactSensor:
    """Training contact sensor with mjlab found / force / air-time semantics.

    Decodes ground-reaction forces from mujoco-warp solver contact data for a
    set of primary Newton bodies (one column per primary). Shape conventions
    match mjlab with ``num_slots=1``: ``found`` is ``[B, P]``, force is stored
    flat as ``[B, P*3]`` for warp reward / critic consumers.
    """

    def __init__(
        self,
        num_env: int,
        num_primaries: int,
        primary_newton_bodies: wp.array,
        device: str,
        *,
        ground_geom_id: int = 0,
        track_air_time: bool = True,
    ):
        if num_primaries <= 0:
            raise ValueError("num_primaries must be > 0")
        self.num_env = num_env
        self.num_primaries = num_primaries
        self.device = device
        self.ground_geom_id = int(ground_geom_id)
        self.track_air_time = bool(track_air_time)
        self.primary_newton_bodies = primary_newton_bodies

        self.found = wp.zeros((num_env, num_primaries), dtype=wp.int32, device=device)
        self.force = wp.zeros((num_env, num_primaries * 3), dtype=float, device=device)
        self.current_air_time = wp.zeros((num_env, num_primaries), dtype=float, device=device)
        self.last_air_time = wp.zeros((num_env, num_primaries), dtype=float, device=device)
        self.current_contact_time = wp.zeros((num_env, num_primaries), dtype=float, device=device)
        self.last_contact_time = wp.zeros((num_env, num_primaries), dtype=float, device=device)
        self.first_contact = wp.zeros((num_env, num_primaries), dtype=wp.int32, device=device)
        self.first_air = wp.zeros((num_env, num_primaries), dtype=wp.int32, device=device)

    def _decode_contacts(
        self,
        contact,
        efc_force,
        nacon,
        geom_bodyid,
        mjc_body_to_newton,
        ngeom: int,
        njmax: int,
        nbody_mj: int,
        naconmax: int,
        opt_cone: int,
    ) -> None:
        """Re-decode ``found`` / ``force`` from the solver contact buffer."""
        self.found.zero_()
        self.force.zero_()
        wp.launch(
            kernel=accumulate_primary_contact_forces_kernel,
            dim=naconmax,
            inputs=[
                self.found,
                self.force,
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
                self.num_primaries,
                self.ground_geom_id,
                nacon,
                opt_cone,
            ],
            device=self.device,
        )

    def update_air_time(self, dt: float) -> None:
        """Accumulate air/contact times by ``dt`` for the current ``found`` state."""
        if not self.track_air_time:
            return
        wp.launch(
            kernel=update_air_time_kernel,
            dim=self.num_env,
            inputs=[
                self.found,
                self.current_air_time,
                self.last_air_time,
                self.current_contact_time,
                self.last_contact_time,
                float(dt),
                self.num_primaries,
            ],
            device=self.device,
        )

    def substep_update(
        self,
        contact,
        efc_force,
        nacon,
        geom_bodyid,
        mjc_body_to_newton,
        ngeom: int,
        njmax: int,
        nbody_mj: int,
        naconmax: int,
        opt_cone: int,
        dt: float,
    ) -> None:
        """Per-substep: decode contacts + accumulate air-time (mjlab granularity).

        ``first_contact`` / ``first_air`` are intentionally not recomputed here;
        they are refreshed once per policy step at the step dt so the edge
        windows match mjlab's ``compute_first_contact(dt=step_dt)``.
        """
        self._decode_contacts(
            contact, efc_force, nacon, geom_bodyid, mjc_body_to_newton,
            ngeom, njmax, nbody_mj, naconmax, opt_cone,
        )
        self.update_air_time(dt)

    def update_from_solver(
        self,
        contact,
        efc_force,
        nacon,
        geom_bodyid,
        mjc_body_to_newton,
        ngeom: int,
        njmax: int,
        nbody_mj: int,
        naconmax: int,
        opt_cone: int,
        dt: float,
        abs_tol: float = 1.0e-6,
    ) -> None:
        """Decode contacts, then update air-time / first-contact flags.

        One-shot convenience (single policy-step call). Per-substep callers
        should use ``substep_update`` + ``policy_step_update`` instead.
        """
        self._decode_contacts(
            contact, efc_force, nacon, geom_bodyid, mjc_body_to_newton,
            ngeom, njmax, nbody_mj, naconmax, opt_cone,
        )
        self.update_air_time(dt)
        if self.track_air_time:
            self.compute_first_contact(dt, abs_tol=abs_tol)
            self.compute_first_air(dt, abs_tol=abs_tol)

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-6) -> wp.array:
        if not self.track_air_time:
            raise RuntimeError("track_air_time=True required for compute_first_contact")
        wp.launch(
            kernel=compute_first_contact_kernel,
            dim=self.num_env,
            inputs=[
                self.current_contact_time,
                self.first_contact,
                float(dt),
                float(abs_tol),
                self.num_primaries,
            ],
            device=self.device,
        )
        return self.first_contact

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-6) -> wp.array:
        if not self.track_air_time:
            raise RuntimeError("track_air_time=True required for compute_first_air")
        wp.launch(
            kernel=compute_first_air_kernel,
            dim=self.num_env,
            inputs=[
                self.current_air_time,
                self.first_air,
                float(dt),
                float(abs_tol),
                self.num_primaries,
            ],
            device=self.device,
        )
        return self.first_air

    def reset(self, reset_mask) -> None:
        if isinstance(reset_mask, wp.array):
            mask = reset_mask
        else:
            mask = wp.from_torch(reset_mask, dtype=wp.int32)
        wp.launch(
            kernel=reset_contact_sensor_kernel,
            dim=self.num_env,
            inputs=[
                mask,
                self.found,
                self.force,
                self.current_air_time,
                self.last_air_time,
                self.current_contact_time,
                self.last_contact_time,
                self.num_primaries,
            ],
            device=self.device,
        )
        self.first_contact.zero_()
        self.first_air.zero_()


# ---------------------------------------------------------------------------
# SelfCollisionSensor (mjlab self_collision contact sensor semantics)
# ---------------------------------------------------------------------------


@wp.kernel
def accumulate_self_collision_max_force_kernel(
    found: wp.array(dtype=wp.int32),
    max_force: wp.array(dtype=float),
    contact_worldid: wp.array(dtype=wp.int32),
    contact_geom: wp.array(dtype=wp.vec2i),
    contact_frame: wp.array(dtype=wp.mat33),
    contact_friction: wp.array(dtype=vec5),
    contact_dim: wp.array(dtype=wp.int32),
    contact_efc_address: wp.array(dtype=wp.int32, ndim=2),
    geom_bodyid: wp.array(dtype=wp.int32),
    mjc_body_to_newton: wp.array(dtype=wp.int32, ndim=2),
    efc_force: wp.array2d(dtype=float),
    ngeom: int,
    njmax: int,
    nbody_mj: int,
    ground_geom_id: int,
    nacon: wp.array(dtype=wp.int32),
    opt_cone: int,
):
    """Accumulate max self-contact force magnitude per env (robot-on-robot only).

    Ground contacts (either geom == ``ground_geom_id``, or either body is the
    MuJoCo world body 0) are skipped; both geoms must belong to loaded robot
    bodies (``mjc_body_to_newton`` maps them to a non-negative Newton body).
    Thread per contact slot.
    """
    c = wp.tid()
    if c >= nacon[0]:
        return

    w = contact_worldid[c]
    g = contact_geom[c]
    if g[0] == ground_geom_id or g[1] == ground_geom_id:
        return
    if g[0] < 0 or g[0] >= ngeom or g[1] < 0 or g[1] >= ngeom:
        return

    b0 = geom_bodyid[g[0]]
    b1 = geom_bodyid[g[1]]
    if b0 < 0 or b0 >= nbody_mj or b1 < 0 or b1 >= nbody_mj:
        return
    # World/ground body (mj body 0) is never a robot self-collision, even when
    # its geom id is not ``ground_geom_id``.
    if b0 == 0 or b1 == 0:
        return
    if mjc_body_to_newton[w, b0] < 0 or mjc_body_to_newton[w, b1] < 0:
        return

    force = contact_force_fn(
        opt_cone,
        contact_frame,
        contact_friction,
        contact_dim,
        contact_efc_address,
        efc_force,
        njmax,
        nacon,
        w,
        c,
        True,
    )
    mag = wp.sqrt(force[0] * force[0] + force[1] * force[1] + force[2] * force[2])
    wp.atomic_max(max_force, w, mag)
    wp.atomic_max(found, w, 1)


@wp.kernel
def push_self_collision_history_kernel(
    force_history: wp.array2d(dtype=float),
    current_max_force: wp.array(dtype=float),
    history_length: int,
):
    """Roll history so index 0 = most recent substep (mjlab ``roll(1)`` layout)."""
    env = wp.tid()
    for h in range(history_length - 1, 0, -1):
        force_history[env, h] = force_history[env, h - 1]
    force_history[env, 0] = current_max_force[env]


@wp.kernel
def reset_self_collision_kernel(
    reset_mask: wp.array(dtype=wp.int32),
    force_history: wp.array2d(dtype=float),
    found: wp.array(dtype=wp.int32),
    current_max_force: wp.array(dtype=float),
    history_length: int,
):
    env = wp.tid()
    if reset_mask[env] == 0:
        return
    found[env] = 0
    current_max_force[env] = 0.0
    for h in range(history_length):
        force_history[env, h] = 0.0


class SelfCollisionSensor:
    """Track max robot-on-robot contact force per substep for self-collision rewards.

    Mirrors mjlab's ``self_collision`` ContactSensor with ``history_length``:
    each substep the max self-contact force magnitude is decoded from
    mujoco-warp contact/efc data (ground contacts excluded) and pushed into a
    rolling ``force_history`` buffer (index 0 = most recent substep). Rewards
    count how many of the last ``history_length`` substeps exceeded a force
    threshold.
    """

    def __init__(
        self,
        num_env: int,
        device: str,
        *,
        history_length: int,
        ground_geom_id: int = 0,
    ):
        self.num_env = num_env
        self.device = device
        self.history_length = int(history_length)
        self.ground_geom_id = int(ground_geom_id)

        self.force_history = wp.zeros(
            (num_env, self.history_length), dtype=float, device=device
        )
        self.found = wp.zeros(num_env, dtype=wp.int32, device=device)
        self.current_max_force = wp.zeros(num_env, dtype=float, device=device)

        # Solver constants filled by bind_solver_constants.
        self._solver_bound = False
        self._ngeom = 0
        self._njmax = 0
        self._nbody_mj = 0
        self._naconmax = 0
        self._opt_cone = 0

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

    def begin_step(self) -> None:
        """Clear per-step flags (call once per policy step, before substeps)."""
        self.found.zero_()

    def record_substep(
        self,
        contact,
        efc_force,
        nacon,
        geom_bodyid,
        mjc_body_to_newton,
    ) -> None:
        """Decode this substep's max self-contact force and push it to history."""
        if not self._solver_bound:
            raise RuntimeError(
                "SelfCollisionSensor.record_substep requires bind_solver_constants."
            )
        self.current_max_force.zero_()
        wp.launch(
            kernel=accumulate_self_collision_max_force_kernel,
            dim=self._naconmax,
            inputs=[
                self.found,
                self.current_max_force,
                contact.worldid,
                contact.geom,
                contact.frame,
                contact.friction,
                contact.dim,
                contact.efc_address,
                geom_bodyid,
                mjc_body_to_newton,
                efc_force,
                self._ngeom,
                self._njmax,
                self._nbody_mj,
                self.ground_geom_id,
                nacon,
                self._opt_cone,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=push_self_collision_history_kernel,
            dim=self.num_env,
            inputs=[
                self.force_history,
                self.current_max_force,
                self.history_length,
            ],
            device=self.device,
        )

    def reset_history(self, reset_mask) -> None:
        """Clear history / flags for terminated environments."""
        if isinstance(reset_mask, wp.array):
            mask = reset_mask
        else:
            mask = wp.from_torch(reset_mask, dtype=wp.int32)
        wp.launch(
            kernel=reset_self_collision_kernel,
            dim=self.num_env,
            inputs=[
                mask,
                self.force_history,
                self.found,
                self.current_max_force,
                self.history_length,
            ],
            device=self.device,
        )
