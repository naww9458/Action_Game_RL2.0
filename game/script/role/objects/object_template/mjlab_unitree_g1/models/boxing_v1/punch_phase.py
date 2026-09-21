"""Hard-state punch/retract FSM and smoothed action signal for boxing_v1.

The environment keeps a boolean hard state (retract / punch) plus a continuous
signal (0 = retract, 1 = punch) that eases between states with a quadratic
curve (slow start, fast end). Rewards gate on the hard state and blend their
branches with the signal; the policy reads the signal as a dedicated obs column.

Newton / Warp details stay in this boxing_v1 plugin. Task-layer rewards and
environments read tensors only and must TryGet when this plugin is absent.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import warp as wp

from .state_query import ArticulationState, resolve_newton_body_ids
from .contact_query import ContactQuery
from .obs_actor import _BOXING_OBS_CFG


@wp.func
def _feasibility_fn(dist: float, standoff: float, band: float, mode: int) -> float:
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
    err = float(0.0)
    for j in range(n_joints):
        idx = joint_indices[j]
        dq = joint_q[tid, idx] - joint_poses[j]
        err += wp.abs(dq) * joint_weights[j]
    return err


@wp.kernel
def _punch_phase_step_kernel(
    hard_state: wp.array(dtype=wp.int32),
    signal: wp.array(dtype=float),
    cooldown: wp.array(dtype=float),
    punch_elapsed: wp.array(dtype=float),
    contact_elapsed: wp.array(dtype=float),
    contact_seen: wp.array(dtype=wp.int32),
    trans_t: wp.array(dtype=float),
    seed_offsets: wp.array(dtype=wp.int32),
    seeds: wp.array(dtype=wp.int32),
    pelvis_pos: wp.array2d(dtype=float),
    hand_pos: wp.array2d(dtype=float),
    hand_vel: wp.array2d(dtype=float),
    target_pos: wp.array2d(dtype=float),
    has_target: wp.array(dtype=wp.int32),
    joint_q: wp.array2d(dtype=float),
    joint_indices: wp.array(dtype=wp.int32),
    joint_poses: wp.array(dtype=float),
    joint_weights: wp.array(dtype=float),
    n_joints: int,
    joint_limits_min: wp.array(dtype=float),
    joint_limits_max: wp.array(dtype=float),
    contact_found: wp.array(dtype=wp.int32),
    standoff: float,
    band: float,
    mode: int,
    gate_feasibility: float,
    ready_threshold: float,
    cooldown_lo: float,
    cooldown_hi: float,
    transition_duration: float,
    contact_hold_min: float,
    contact_hold_max: float,
    direction_deviation_cos: float,
    joint_limit_proximity: float,
    punch_timeout: float,
    arm_extension_threshold: float,
    dt: float,
):
    tid = wp.tid()

    if has_target[tid] == 0:
        hard_state[tid] = 0
        signal[tid] = 0.0
        cooldown[tid] = 0.0
        punch_elapsed[tid] = 0.0
        contact_elapsed[tid] = 0.0
        contact_seen[tid] = 0
        trans_t[tid] = 1.0
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

    bad = 0
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
        bad = 1
    if bad == 1:
        hard_state[tid] = 0
        signal[tid] = 0.0
        cooldown[tid] = 0.0
        punch_elapsed[tid] = 0.0
        contact_elapsed[tid] = 0.0
        contact_seen[tid] = 0
        trans_t[tid] = 1.0
        return

    dist = _xy_dist(px, py, tx, ty)
    feas = _feasibility_fn(dist, standoff, band, mode)
    err = _ready_pose_error(joint_q, joint_indices, joint_poses, joint_weights, n_joints, tid)
    hand_ready = 0
    if err < ready_threshold:
        hand_ready = 1

    cooldown[tid] = wp.max(0.0, cooldown[tid] - dt)

    if hard_state[tid] == 0:
        # RETRACT -> PUNCH: distance feasible AND hand ready AND cooldown done.
        if feas > gate_feasibility and hand_ready == 1 and cooldown[tid] <= 0.0:
            hard_state[tid] = 1
            punch_elapsed[tid] = 0.0
            contact_elapsed[tid] = 0.0
            contact_seen[tid] = 0
            trans_t[tid] = 0.0
    else:
        # PUNCH -> RETRACT
        punch_elapsed[tid] = punch_elapsed[tid] + dt
        exit_punch = 0

        # Condition 3: attack timeout.
        if punch_elapsed[tid] > punch_timeout:
            exit_punch = 1

        # Condition 1: post-contact forced hold.
        if exit_punch == 0 and contact_found[tid] == 1 and contact_seen[tid] == 0:
            contact_seen[tid] = 1
            contact_elapsed[tid] = 0.0
        if exit_punch == 0 and contact_seen[tid] == 1:
            contact_elapsed[tid] = contact_elapsed[tid] + dt
            if contact_elapsed[tid] >= contact_hold_min:
                ddx = tx - hx
                ddy = ty - hy
                ddz = tz - hz
                hand_target_dist = wp.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                arm_extended = 0
                if hand_target_dist < arm_extension_threshold:
                    arm_extended = 1
                if contact_found[tid] == 1 and arm_extended == 1:
                    if contact_elapsed[tid] >= contact_hold_max:
                        exit_punch = 1
                else:
                    exit_punch = 1

        # Condition 2: direction deviation AND arm joint near mechanical limit.
        if exit_punch == 0:
            dtx = tx - hx
            dty = ty - hy
            dtz = tz - hz
            dir_len = wp.sqrt(dtx * dtx + dty * dty + dtz * dtz)
            spd = wp.sqrt(hvx * hvx + hvy * hvy + hvz * hvz)
            if dir_len > 1.0e-8 and spd > 1.0e-4:
                cosang = (hvx * dtx + hvy * dty + hvz * dtz) / (dir_len * spd)
                near_limit = int(0)
                for j in range(n_joints):
                    idx = joint_indices[j]
                    q = joint_q[tid, idx]
                    lo = joint_limits_min[idx]
                    hi = joint_limits_max[idx]
                    if (q - lo) < joint_limit_proximity or (hi - q) < joint_limit_proximity:
                        near_limit = 1
                if cosang < direction_deviation_cos and near_limit == 1:
                    exit_punch = 1

        if exit_punch == 1:
            hard_state[tid] = 0
            trans_t[tid] = 0.0
            rng = wp.rand_init(seeds[tid], seed_offsets[tid])
            cooldown[tid] = wp.randf(rng, cooldown_lo, cooldown_hi)
            seed_offsets[tid] = seed_offsets[tid] + 1
            contact_seen[tid] = 0
            contact_elapsed[tid] = 0.0

    # Quadratic-eased transition (slow start, fast end) in both directions.
    if trans_t[tid] < 1.0:
        trans_t[tid] = wp.min(1.0, trans_t[tid] + dt / transition_duration)
    t = trans_t[tid]
    if hard_state[tid] == 1:
        signal[tid] = t * t
    else:
        signal[tid] = 1.0 - t * t


@wp.kernel
def _reset_punch_phase_kernel(
    terminated: wp.array(dtype=wp.bool),
    hard_state: wp.array(dtype=wp.int32),
    signal: wp.array(dtype=float),
    cooldown: wp.array(dtype=float),
    punch_elapsed: wp.array(dtype=float),
    contact_elapsed: wp.array(dtype=float),
    contact_seen: wp.array(dtype=wp.int32),
    trans_t: wp.array(dtype=float),
):
    tid = wp.tid()
    if terminated[tid] == False:
        return
    hard_state[tid] = 0
    signal[tid] = 0.0
    cooldown[tid] = 0.0
    punch_elapsed[tid] = 0.0
    contact_elapsed[tid] = 0.0
    contact_seen[tid] = 0
    trans_t[tid] = 1.0


class PunchPhase:
    """Per-env punch/retract hard state + smoothed action signal (TryGet source)."""

    def __init__(
        self,
        *,
        environment: Any,
        obs_actor: Any,
        num_env: int,
        device: str,
        dt: float,
        seed: int = 0,
    ) -> None:
        self.environment = environment
        self.obs_actor = obs_actor
        self.device = device
        self.num_env = int(num_env)
        self.dt = float(dt)
        self.seed = int(seed)

        self.cfg = dict(obs_actor.get_boxing_obs_cfg()) if obs_actor is not None else None
        self.art = ArticulationState.try_from_environment(environment, None)
        self.pattern = getattr(self.art, "pattern", "") if self.art is not None else ""

        pm = getattr(environment, "physics_manager", None)
        self.contact_query = ContactQuery.try_from_physics(pm)
        self._hand_ids = None

        self._pelvis_pos = None
        self._pelvis_vel = None
        self._hand_pos = None
        self._hand_vel = None
        self._joint_q = None
        self._joint_indices = None
        self._joint_poses = None
        self._joint_weights = None
        self._n_joints = 0
        self._joint_limits_min = None
        self._joint_limits_max = None

        self.hard_state_wp = None
        self.signal_wp = None
        self._cooldown = None
        self._punch_elapsed = None
        self._contact_elapsed = None
        self._contact_seen = None
        self._trans_t = None
        self._seed_offsets = None
        self._seeds = None
        self._contact_found_zero = None

        self._ready = False
        self._setup()

    def _setup(self) -> None:
        if self.cfg is None or self.art is None or self.num_env <= 0:
            return

        # Ready-pose joint resolution (suffix match against joint_dof_names).
        joints = (self.cfg.get("ready_pose") or {}).get("joints") or {}
        names = list(getattr(self.art.view, "joint_dof_names", ()) or ())
        if not joints or not names:
            return
        indices: list[int] = []
        poses: list[float] = []
        weights: list[float] = []
        for joint_name, entry in joints.items():
            matches = [i for i, n in enumerate(names) if str(n).endswith(str(joint_name))]
            if len(matches) != 1:
                raise RuntimeError(
                    f"boxing punch_phase ready_pose joint {joint_name!r} matched "
                    f"{len(matches)} dof names (need exactly 1)"
                )
            indices.append(matches[0])
            poses.append(float(entry["pose"]))
            weights.append(float(entry["weight"]))
        self._joint_indices = wp.array(indices, dtype=wp.int32, device=self.device)
        self._joint_poses = wp.array(poses, dtype=float, device=self.device)
        self._joint_weights = wp.array(weights, dtype=float, device=self.device)
        self._n_joints = len(indices)

        ab = getattr(self.environment, "articulation_body", None)
        limits_min = getattr(ab, "control_joint_limits_min_gpus", {}).get(self.pattern)
        limits_max = getattr(ab, "control_joint_limits_max_gpus", {}).get(self.pattern)
        if limits_min is None or limits_max is None:
            return
        self._joint_limits_min = limits_min
        self._joint_limits_max = limits_max

        # Contact body ids (hand bodies from config).
        pm = getattr(self.environment, "physics_manager", None)
        if pm is not None:
            hand_ids = resolve_newton_body_ids(pm, self.cfg["hit_hand_bodies"], self.device)
            if hand_ids is not None:
                if hand_ids.ndim == 1:
                    hand_ids = hand_ids.unsqueeze(-1)
                self._hand_ids = hand_ids.to(dtype=torch.int32).contiguous()

        n = self.num_env
        dof = int(getattr(self.art.view, "joint_dof_count", 0) or 0)
        self._pelvis_pos = wp.zeros((n, 3), dtype=float, device=self.device)
        self._pelvis_vel = wp.zeros((n, 3), dtype=float, device=self.device)
        self._hand_pos = wp.zeros((n, 3), dtype=float, device=self.device)
        self._hand_vel = wp.zeros((n, 3), dtype=float, device=self.device)
        self._joint_q = wp.zeros((n, dof), dtype=float, device=self.device)

        self.hard_state_wp = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.signal_wp = wp.zeros(n, dtype=float, device=self.device)
        self._cooldown = wp.zeros(n, dtype=float, device=self.device)
        self._punch_elapsed = wp.zeros(n, dtype=float, device=self.device)
        self._contact_elapsed = wp.zeros(n, dtype=float, device=self.device)
        self._contact_seen = wp.zeros(n, dtype=wp.int32, device=self.device)
        self._trans_t = wp.ones(n, dtype=float, device=self.device)
        self._seed_offsets = wp.zeros(n, dtype=wp.int32, device=self.device)
        self._seeds = wp.array(
            np.arange(self.seed, self.seed + n, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self._contact_found_zero = wp.zeros(n, dtype=wp.int32, device=self.device)

        self._ready = True

    def _sync_contact(self, physics_manager: Any) -> Any:
        """Return a per-env contact-found wp array (0/1), or the zero buffer."""
        if self.contact_query is None or self._hand_ids is None:
            return self._contact_found_zero
        ids_fn = getattr(self.obs_actor, "try_selected_target_newton_body_ids", None)
        if not callable(ids_fn):
            return self._contact_found_zero
        body_ids = ids_fn(physics_manager)
        if body_ids is None:
            return self._contact_found_zero
        self.contact_query.max_force(self._hand_ids, body_ids)
        found = self.contact_query.last_found
        if found is None:
            found = self._contact_found_zero
        return found

    def step(self, physics_manager: Any) -> None:
        """Advance the FSM one policy step (physics already simulated)."""
        if not self._ready:
            return
        # Fresh target pose + newton ids.
        sync = getattr(self.obs_actor, "try_sync_selected_target_pose", None)
        if callable(sync):
            sync(physics_manager)

        self.art.copy_link_position_velocity(
            self.cfg["pelvis_body"], self._pelvis_pos, self._pelvis_vel
        )
        self.art.copy_link_position_velocity(
            self.cfg["hand_body"], self._hand_pos, self._hand_vel
        )
        self.art.copy_joint_positions(self._joint_q)

        contact_found = self._sync_contact(physics_manager)

        target_pos = getattr(self.obs_actor, "target_pos_wp", None)
        has_target = getattr(self.obs_actor, "has_target_wp", None)
        if target_pos is None or has_target is None:
            return

        phase_cfg = self.cfg["punch_phase"]
        punch_cfg = self.cfg["punch"]
        ready_cfg = self.cfg["ready_pose"]
        mode = 1 if str(self.cfg.get("punch_ready_mode", "linear")).lower() == "gaussian" else 0

        wp.launch(
            _punch_phase_step_kernel,
            dim=self.num_env,
            inputs=[
                self.hard_state_wp,
                self.signal_wp,
                self._cooldown,
                self._punch_elapsed,
                self._contact_elapsed,
                self._contact_seen,
                self._trans_t,
                self._seed_offsets,
                self._seeds,
                self._pelvis_pos,
                self._hand_pos,
                self._hand_vel,
                target_pos,
                has_target,
                self._joint_q,
                self._joint_indices,
                self._joint_poses,
                self._joint_weights,
                self._n_joints,
                self._joint_limits_min,
                self._joint_limits_max,
                contact_found,
                float(self.cfg["standoff_distance"]),
                float(self.cfg["punch_ready_band"]),
                mode,
                float(punch_cfg["gate_feasibility"]),
                float(ready_cfg["ready_threshold"]),
                float(phase_cfg["cooldown_lo_s"]),
                float(phase_cfg["cooldown_hi_s"]),
                float(phase_cfg["transition_duration_s"]),
                float(phase_cfg["contact_hold_min_s"]),
                float(phase_cfg["contact_hold_max_s"]),
                float(phase_cfg["direction_deviation_cos"]),
                float(phase_cfg["joint_limit_proximity"]),
                float(phase_cfg["punch_timeout_s"]),
                float(phase_cfg["arm_extension_threshold"]),
                self.dt,
            ],
            device=self.device,
        )

    def reset(self, terminated) -> None:
        if not self._ready or self.hard_state_wp is None:
            return
        if isinstance(terminated, wp.array):
            term = terminated
        else:
            term = wp.from_torch(terminated.contiguous(), dtype=wp.bool)
        wp.launch(
            _reset_punch_phase_kernel,
            dim=self.num_env,
            inputs=[
                term,
                self.hard_state_wp,
                self.signal_wp,
                self._cooldown,
                self._punch_elapsed,
                self._contact_elapsed,
                self._contact_seen,
                self._trans_t,
            ],
            device=self.device,
        )


def create_punch_phase(**kwargs) -> PunchPhase:
    return PunchPhase(**kwargs)
