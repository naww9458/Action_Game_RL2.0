"""turret_110mm aim action — the turret's attached-tool "aim" behavior.

Everything turret-aim-specific lives here (and in ``third_person_aim_view.py``):
config parsing, joint/body reference resolution, per-frame camera-aligned
pitch/yaw driving with online gravity compensation, and the third-person aim
overlay. Common mount/simulation modules only see the generic ``ToolAction``
interface and never import this module directly — it is loaded lazily through
``tool_function_registry`` when a level actually uses ``turret_110mm``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from newton import JointTargetMode

from script.role.objects.tool_anchor import compose_mounted_weld_relpose
from script.simulate.mount_joint_builder import resolve_joint_index_by_leaf_name
from script.simulate.tool_action import ToolAction

from ..third_person_aim_view import (
    Turret110mmAimViewStyle,
    build_turret_110mm_third_person_aim_view,
    draw_turret_110mm_aim_view_overlay,
    joint_pitch_limits_to_camera_pitch_deg,
)
from .recoil import apply_turret_recoil

# ---------------------------------------------------------------------------
# Config / state
# ---------------------------------------------------------------------------


@dataclass
class TurretAimControlConfig:
    """Aim control parameters for turret_110mm.

    All tuning values are declared in the object template
    (``object_template/turret_110mm/template.yaml`` -> ``aim.control``) and
    loaded via :meth:`from_mapping`. The class defaults below are *neutral*
    runtime-safety fallbacks (0.0 = no driving force) or structural
    conventions shared with the template — never authoritative tuning values.
    """

    yaw_torque_gain: float = 0.0
    yaw_damping: float = 0.0
    max_yaw_torque: float = 0.0
    pitch_torque_gain: float = 0.0
    pitch_damping: float = 0.0
    max_pitch_torque: float = 0.0
    # Maps host-local pitch_error → joint-q delta. turret_110mm Y-axis: +q depresses.
    pitch_joint_sign: float = -1.0
    angle_dead_zone_deg: float = 0.0
    weld_yaw_drive_gain: float = 0.0
    aim_forward_local: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    world_up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    # Pitch joint limits used only when the physics model reports none (deg).
    pitch_limit_fallback_deg: Tuple[float, float] = (-10.0, 10.0)
    # Online gravity torque τ_g = coeff * basis(q); coeff learned from torque + Δq.
    pitch_gravity_comp_enable: bool = False
    pitch_gravity_learn_rate: float = 0.08
    pitch_gravity_basis: str = "cos"  # "cos" | "sin"
    pitch_gravity_rate_eps: float = 0.2  # rad/s — quasi-static gate
    pitch_gravity_accel_eps: float = 2.0  # rad/s^2
    pitch_gravity_dq_eps: float = 0.002  # rad — min |Δq| for energy estimate

    @classmethod
    def from_mapping(
        cls,
        raw: dict | None,
        defaults: "TurretAimControlConfig | None" = None,
    ) -> "TurretAimControlConfig":
        base = defaults or cls()
        if not raw:
            return base
        forward = raw.get("aim_forward_local", base.aim_forward_local)
        up = raw.get("world_up", base.world_up)
        fallback = raw.get("pitch_limit_fallback_deg", base.pitch_limit_fallback_deg)
        basis = str(raw.get("pitch_gravity_basis", base.pitch_gravity_basis)).lower()
        if basis not in ("cos", "sin"):
            basis = base.pitch_gravity_basis
        return cls(
            yaw_torque_gain=float(raw.get("yaw_torque_gain", base.yaw_torque_gain)),
            yaw_damping=float(raw.get("yaw_damping", base.yaw_damping)),
            max_yaw_torque=float(raw.get("max_yaw_torque", base.max_yaw_torque)),
            pitch_torque_gain=float(raw.get("pitch_torque_gain", base.pitch_torque_gain)),
            pitch_damping=float(raw.get("pitch_damping", base.pitch_damping)),
            max_pitch_torque=float(raw.get("max_pitch_torque", base.max_pitch_torque)),
            pitch_joint_sign=float(raw.get("pitch_joint_sign", base.pitch_joint_sign)),
            angle_dead_zone_deg=float(raw.get("angle_dead_zone_deg", base.angle_dead_zone_deg)),
            weld_yaw_drive_gain=float(raw.get("weld_yaw_drive_gain", base.weld_yaw_drive_gain)),
            aim_forward_local=tuple(float(v) for v in forward),
            world_up=tuple(float(v) for v in up),
            pitch_limit_fallback_deg=(
                float(fallback[0]),
                float(fallback[1]),
            ),
            pitch_gravity_comp_enable=bool(
                raw.get("pitch_gravity_comp_enable", base.pitch_gravity_comp_enable)
            ),
            pitch_gravity_learn_rate=float(
                raw.get("pitch_gravity_learn_rate", base.pitch_gravity_learn_rate)
            ),
            pitch_gravity_basis=basis,
            pitch_gravity_rate_eps=float(
                raw.get("pitch_gravity_rate_eps", base.pitch_gravity_rate_eps)
            ),
            pitch_gravity_accel_eps=float(
                raw.get("pitch_gravity_accel_eps", base.pitch_gravity_accel_eps)
            ),
            pitch_gravity_dq_eps=float(
                raw.get("pitch_gravity_dq_eps", base.pitch_gravity_dq_eps)
            ),
        )


@dataclass
class TurretRlActionConfig:
    """RL / inspector continuous control for turret_110mm (template ``aim.rl_action``)."""

    shape: int = 3
    range: Tuple[float, float] = (-1.0, 1.0)
    yaw_command_scale_rad: float = 0.0
    pitch_command_scale_rad: float = 0.0
    fire_threshold: float = 0.0
    dim_labels: Tuple[str, ...] = ("turret_yaw", "barrel_pitch", "fire")

    @classmethod
    def from_mapping(cls, raw: dict | None) -> "TurretRlActionConfig":
        if not raw:
            return cls()
        rng = raw.get("range", [-1.0, 1.0])
        if not isinstance(rng, list) or len(rng) != 2:
            rng = [-1.0, 1.0]
        labels: list[str] = []
        dims = raw.get("dims")
        if isinstance(dims, list):
            for entry in dims:
                if isinstance(entry, dict) and entry.get("name"):
                    labels.append(str(entry["name"]))
                elif isinstance(entry, str):
                    labels.append(entry)
        try:
            shape = max(0, int(raw.get("shape", len(labels) or 3)))
        except (TypeError, ValueError):
            shape = len(labels) or 3
        if labels and len(labels) < shape:
            for i in range(len(labels), shape):
                labels.append(f"action_{i}")
        return cls(
            shape=shape,
            range=(float(rng[0]), float(rng[1])),
            yaw_command_scale_rad=float(raw.get("yaw_command_scale_rad", 0.0)),
            pitch_command_scale_rad=float(raw.get("pitch_command_scale_rad", 0.0)),
            fire_threshold=float(raw.get("fire_threshold", 0.0)),
            dim_labels=tuple(labels) if labels else cls.dim_labels,
        )


@dataclass
class PitchGravityState:
    """Runtime estimate of gravity torque coefficient for one pitch joint."""

    coeff: float = 0.0
    prev_q: float | None = None
    prev_rate: float | None = None

    def reset(self) -> None:
        self.coeff = 0.0
        self.prev_q = None
        self.prev_rate = None


@dataclass
class AimDofSpec:
    """Resolved pitch DOF the aim action drives via joint_f."""

    global_dof_idx: int
    local_coord_idx: int
    limit_lower: float
    limit_upper: float
    mouse_axis: str  # "pitch"
    current_target: float = 0.0
    sensitivity: float = 1.0
    saved_target_ke: float = 0.0
    saved_target_kd: float = 0.0
    saved_target_mode: int = int(JointTargetMode.NONE)


# ---------------------------------------------------------------------------
# Pure math helpers (camera-relative direction errors, PD torque, gravity comp)
# ---------------------------------------------------------------------------


def pitch_gravity_basis(q: float, mode: str = "cos") -> float:
    if mode == "sin":
        return math.sin(q)
    return math.cos(q)


def pitch_gravity_potential_delta(q: float, prev_q: float, mode: str = "cos") -> float:
    """Δ(V/G) consistent with τ_g = G * basis(q) via τ_g = -∂V/∂q on the plant."""
    if mode == "sin":
        return -(math.cos(q) - math.cos(prev_q))
    return math.sin(q) - math.sin(prev_q)


def pitch_gravity_compensation_torque(coeff: float, q: float, mode: str = "cos") -> float:
    return float(coeff) * pitch_gravity_basis(q, mode)


def update_pitch_gravity_coeff(
    state: PitchGravityState,
    *,
    torque_applied: float,
    q: float,
    rate: float,
    dt: float,
    learn_rate: float,
    rate_eps: float,
    accel_eps: float,
    dq_eps: float,
    max_abs_coeff: float,
    basis_mode: str = "cos",
) -> None:
    """
    Learn G in τ_g = G * basis(q) from applied joint torque and pitch motion.

    Quasi-static: G ≈ τ / basis(q)
    Slow angle change (energy): G ≈ τ * Δq / Δ(V/G)  (KE neglected)
    """
    lr = float(max(0.0, min(1.0, learn_rate)))
    basis_eps = 1e-3
    g_obs: float | None = None

    basis = pitch_gravity_basis(q, basis_mode)
    alpha = 0.0
    if state.prev_rate is not None and dt > 1e-8:
        alpha = (float(rate) - float(state.prev_rate)) / float(dt)

    if (
        abs(float(rate)) <= float(rate_eps)
        and abs(alpha) <= float(accel_eps)
        and abs(basis) >= basis_eps
    ):
        g_obs = float(torque_applied) / basis
    elif (
        state.prev_q is not None
        and abs(float(rate)) <= float(rate_eps) * 5.0
        and abs(alpha) <= float(accel_eps)
    ):
        dq = float(q) - float(state.prev_q)
        d_pot = pitch_gravity_potential_delta(float(q), float(state.prev_q), basis_mode)
        if abs(dq) >= float(dq_eps) and abs(d_pot) >= basis_eps:
            g_obs = float(torque_applied) * dq / d_pot

    if g_obs is not None and lr > 0.0:
        limit = abs(float(max_abs_coeff))
        if limit > 0.0:
            g_obs = max(-limit, min(limit, g_obs))
        state.coeff = (1.0 - lr) * float(state.coeff) + lr * g_obs
        if limit > 0.0:
            state.coeff = max(-limit, min(limit, float(state.coeff)))

    state.prev_q = float(q)
    state.prev_rate = float(rate)


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


# ---------------------------------------------------------------------------
# Aim body resolution (from USD path maps / pitch joint child)
# ---------------------------------------------------------------------------


def _body_idx_from_suffix(path_body_map: dict, prim_suffix: str, default_body_idx: int) -> int:
    if not path_body_map:
        return default_body_idx
    suffix = prim_suffix.strip().lower()
    for path, idx in path_body_map.items():
        leaf = path.rstrip("/").split("/")[-1].lower()
        if leaf == suffix or leaf.endswith(suffix) or suffix in leaf:
            return int(idx)
    return default_body_idx


def _body_idx_from_pitch_joint(
    builder,
    pitch_joint_idx: Optional[int],
    default_body_idx: int,
) -> int:
    """Prefer the pitch joint child body (barrel) for aim direction feedback."""
    if pitch_joint_idx is None:
        return default_body_idx
    joint_child = getattr(builder, "joint_child", None)
    if joint_child is None:
        return default_body_idx
    try:
        child_idx = int(joint_child[int(pitch_joint_idx)])
    except Exception:
        return default_body_idx
    if child_idx < 0:
        return default_body_idx
    return child_idx


# ---------------------------------------------------------------------------
# The aim action
# ---------------------------------------------------------------------------


class Turret110mmAimAction(ToolAction):
    """Aim: keep the barrel pointing at the third-person camera direction."""

    name = "aim"

    def __init__(self) -> None:
        self.pitch_joint_name: Optional[str] = None
        self.aim_body_prim_suffix: Optional[str] = None
        self.pitch_joint_idx: Optional[int] = None
        self.aim_body_idx: int = -1
        self.pitch_dof_spec: Optional[AimDofSpec] = None
        self.cfg: TurretAimControlConfig = TurretAimControlConfig()
        self.rl_cfg: TurretRlActionConfig = TurretRlActionConfig()
        self.pitch_gravity_state: PitchGravityState = PitchGravityState()
        self._view_style: Turret110mmAimViewStyle = Turret110mmAimViewStyle()
        # Shoot trigger edge-detect + Shoot ability lazy cache
        self._prev_mouse_left: bool = False
        self._prev_rl_fire_active: bool = False
        self._rl_active: bool = False
        self._rl_yaw: float = 0.0
        self._rl_pitch: float = 0.0
        self._rl_fire: float = 0.0
        self._shoot_cache: object = None
        self._physics_manager_cache: object = None

    # -- config -------------------------------------------------------------

    def configure(
        self,
        tool_cfg: dict,
        tool_template: Optional[dict],
    ) -> None:
        aim_cfg = (tool_template or {}).get("aim") or {}

        # Level config wins over template for turret-aim fields (same priority as
        # the legacy flat keys in tool_configs).
        self.pitch_joint_name = tool_cfg.get("pitch_joint_name")
        if self.pitch_joint_name is None:
            self.pitch_joint_name = aim_cfg.get("pitch_joint_name")
        self.pitch_joint_name = (
            str(self.pitch_joint_name).strip() if self.pitch_joint_name is not None else None
        )

        self.aim_body_prim_suffix = tool_cfg.get("aim_body_prim_suffix")
        if self.aim_body_prim_suffix is None:
            self.aim_body_prim_suffix = aim_cfg.get("aim_body_prim_suffix")
        if self.aim_body_prim_suffix is not None:
            self.aim_body_prim_suffix = str(self.aim_body_prim_suffix)

        if self.pitch_joint_name is None:
            internal = list(tool_cfg.get("internal_joint_names") or [])
            if not internal and tool_template:
                internal = list((tool_template or {}).get("internal_joint_names") or [])
            if len(internal) == 1:
                self.pitch_joint_name = str(internal[0]).strip()

        base = TurretAimControlConfig.from_mapping(aim_cfg.get("control"))
        # Per-level overrides keep the old flat key `aim_control`.
        self.cfg = TurretAimControlConfig.from_mapping(
            tool_cfg.get("aim_control"),
            defaults=base,
        )
        self._view_style = Turret110mmAimViewStyle.from_mapping(aim_cfg.get("view"))
        self.rl_cfg = TurretRlActionConfig.from_mapping(aim_cfg.get("rl_action"))

    def set_rl_control(self, values: Sequence[float]) -> None:
        count = min(len(values), self.rl_cfg.shape)
        if count <= 0:
            self.clear_rl_control()
            return
        self._rl_active = True
        self._rl_yaw = float(values[0]) if count > 0 else 0.0
        self._rl_pitch = float(values[1]) if count > 1 else 0.0
        self._rl_fire = float(values[2]) if count > 2 else 0.0

    def clear_rl_control(self) -> None:
        self._rl_active = False
        self._rl_yaw = 0.0
        self._rl_pitch = 0.0
        self._rl_fire = 0.0

    def rl_control_active(self) -> bool:
        return bool(self._rl_active)

    def resolve_mount_refs(
        self,
        *,
        builder,
        path_joint_map: dict,
        path_body_map: dict,
        tool_joint_start: int,
        tool_joint_end: int,
        tool_body_idx: int,
    ) -> None:
        """Resolve pitch joint index + aim (barrel) body index at level setup."""
        if self.pitch_joint_name:
            self.pitch_joint_idx = resolve_joint_index_by_leaf_name(
                path_joint_map,
                self.pitch_joint_name,
                tool_joint_start,
                tool_joint_end,
            )
        else:
            self.pitch_joint_idx = None
        self.aim_body_idx = self._resolve_aim_body_idx(
            builder,
            path_body_map,
            self.pitch_joint_idx,
            int(tool_body_idx),
        )

    def _resolve_aim_body_idx(
        self,
        builder,
        path_body_map: dict,
        pitch_joint_idx: Optional[int],
        tool_body_idx: int,
    ) -> int:
        pitch_child_idx = _body_idx_from_pitch_joint(builder, pitch_joint_idx, tool_body_idx)
        aim_suffix = self.aim_body_prim_suffix
        if aim_suffix:
            return _body_idx_from_suffix(
                path_body_map,
                str(aim_suffix),
                default_body_idx=pitch_child_idx,
            )
        # No explicit aim body: use the pitch child (barrel) when available.
        if pitch_joint_idx is not None and pitch_child_idx != tool_body_idx:
            return pitch_child_idx
        return pitch_child_idx

    # -- physics model binding ----------------------------------------------

    def bind_model(self, registry, record) -> None:
        self.pitch_dof_spec = self._build_pitch_dof_spec(registry.model)

    def _build_pitch_dof_spec(self, model) -> Optional[AimDofSpec]:
        if self.pitch_joint_idx is None:
            return None
        joint_idx = int(self.pitch_joint_idx)

        joint_qd_start = model.joint_qd_start.numpy()
        joint_q_start = (
            model.joint_q_start.numpy() if model.joint_q_start is not None else joint_qd_start
        )
        joint_limit_lower = (
            model.joint_limit_lower.numpy() if model.joint_limit_lower is not None else None
        )
        joint_limit_upper = (
            model.joint_limit_upper.numpy() if model.joint_limit_upper is not None else None
        )
        joint_target_ke = (
            model.joint_target_ke.numpy() if model.joint_target_ke is not None else None
        )
        joint_target_kd = (
            model.joint_target_kd.numpy() if model.joint_target_kd is not None else None
        )
        joint_target_mode = (
            model.joint_target_mode.numpy() if model.joint_target_mode is not None else None
        )

        dof_idx = int(joint_qd_start[joint_idx])
        coord_idx = int(joint_q_start[joint_idx])
        fallback_lo, fallback_hi = self.cfg.pitch_limit_fallback_deg
        lower = (
            float(joint_limit_lower[dof_idx])
            if joint_limit_lower is not None
            else math.radians(fallback_lo)
        )
        upper = (
            float(joint_limit_upper[dof_idx])
            if joint_limit_upper is not None
            else math.radians(fallback_hi)
        )
        return AimDofSpec(
            global_dof_idx=dof_idx,
            local_coord_idx=coord_idx,
            limit_lower=lower,
            limit_upper=upper,
            mouse_axis="pitch",
            sensitivity=1.0,
            saved_target_ke=float(joint_target_ke[dof_idx]) if joint_target_ke is not None else 0.0,
            saved_target_kd=float(joint_target_kd[dof_idx]) if joint_target_kd is not None else 0.0,
            saved_target_mode=(
                int(joint_target_mode[dof_idx])
                if joint_target_mode is not None
                else int(JointTargetMode.POSITION)
            ),
        )

    # -- attach / detach ------------------------------------------------------

    def on_attach(self, registry, record, *, world: int) -> None:
        self._set_pitch_position_actuation(registry, world=world, aim_active=True)
        self.pitch_gravity_state.reset()

    def on_detach(self, registry, record, *, world: int) -> None:
        self._set_pitch_position_actuation(registry, world=world, aim_active=False)
        self.pitch_gravity_state.reset()
        if self.pitch_dof_spec is not None:
            self.pitch_dof_spec.current_target = 0.0

    # -- shoot ---------------------------------------------------------------

    def _handle_shoot(
        self,
        registry,
        record,
        *,
        world: int,
        aim_body_q,
        cfg,
    ) -> None:
        """Spawn a projectile (and recoil) when the human left-clicks.

        Parameters come from the generic ``Shoot`` ability's per-tool fire config
        (muzzle offset / recoil force) read from control_configs.yaml. The recoil
        is applied directly here (turret-owned business logic) after a successful
        muzzle fire, keeping the generic ability pure. The recoil impulse is
        staged in the turret articulation's control-force buffer so it survives
        the per-substep ``clear_forces()`` and reaches the solver.
        """
        # ── Barrel forward direction (world-space, unit) ──────────────────
        barrel_forward = body_forward_world(aim_body_q, cfg.aim_forward_local)

        # ── Resolve the Shoot ability + per-tool firing config ────────────
        shoot = self._resolve_shoot()
        if shoot is None:
            return
        owner_idx = int(record.tool_role_object_id)
        fire_cfg = shoot.get_owner_fire_config(owner_idx)
        if fire_cfg is None:
            return

        # ── Spawn position: barrel body centre + muzzle offset (world space) ──
        aim_body_pos = np.array(
            [float(aim_body_q[0]), float(aim_body_q[1]), float(aim_body_q[2])],
            dtype=np.float64,
        )
        spawn_world = aim_body_pos + _rotate_vec_by_quat(
            _quat_to_np(aim_body_q[3:]),
            np.array(fire_cfg.spawn_offset, dtype=np.float64),
        )

        fired = shoot.fire_from_aim_action(
            physics_manager=self._physics_manager_cache,
            owner_obj_idx=owner_idx,
            spawn_pos_world=(
                float(spawn_world[0]),
                float(spawn_world[1]),
                float(spawn_world[2]),
            ),
            barrel_forward_dir=(
                float(barrel_forward[0]),
                float(barrel_forward[1]),
                float(barrel_forward[2]),
            ),
        )
        if fired and fire_cfg.recoil_force > 0.0:
            apply_turret_recoil(
                articulation_body=getattr(shoot, "articulation_body", None),
                tool_pattern=record.tool_pattern or "",
                world=world,
                tool_root_body_idx=record.tool_root_body_idx,
                barrel_forward_dir=(
                    float(barrel_forward[0]),
                    float(barrel_forward[1]),
                    float(barrel_forward[2]),
                ),
                recoil_force=fire_cfg.recoil_force,
            )

    def _resolve_shoot(self):
        """Lazy-resolve the generic ``Shoot`` ability singleton."""
        if self._shoot_cache is not None:
            return self._shoot_cache
        try:
            from script.role.abilities import get_shared_ability
            shoot = get_shared_ability("Shoot")
        except Exception:
            return None

        self._shoot_cache = shoot
        # Also cache physics_manager reference from the shoot ability
        pm = getattr(shoot, "physics_manager", None)
        if pm is not None:
            self._physics_manager_cache = pm
        return shoot

    def _set_pitch_position_actuation(
        self,
        registry,
        *,
        world: int,
        aim_active: bool,
    ) -> None:
        """Disable the built-in position servo while camera aim drives pitch via joint_f."""
        pitch_spec = self.pitch_dof_spec
        model = registry.model
        if pitch_spec is None or model is None:
            return

        global_dof = registry.global_dof_idx(world, pitch_spec.global_dof_idx)
        if aim_active:
            target_ke = 0.0
            target_kd = 0.0
            target_mode = int(JointTargetMode.NONE)
        else:
            target_ke = float(pitch_spec.saved_target_ke)
            target_kd = float(pitch_spec.saved_target_kd)
            target_mode = int(pitch_spec.saved_target_mode)

        if model.joint_target_ke is not None:
            ke_np = model.joint_target_ke.numpy()
            if 0 <= global_dof < ke_np.shape[0]:
                ke_np[global_dof] = target_ke
                model.joint_target_ke.assign(ke_np)

        if model.joint_target_kd is not None:
            kd_np = model.joint_target_kd.numpy()
            if 0 <= global_dof < kd_np.shape[0]:
                kd_np[global_dof] = target_kd
                model.joint_target_kd.assign(kd_np)

        if model.joint_target_mode is not None:
            mode_np = model.joint_target_mode.numpy()
            if 0 <= global_dof < mode_np.shape[0]:
                mode_np[global_dof] = target_mode
                model.joint_target_mode.assign(mode_np)

        registry.notify_joint_dof_properties()

    # -- per-frame driving -----------------------------------------------------

    def step(
        self,
        registry,
        record,
        *,
        world: int,
        dt: float,
        camera_yaw: float,
        camera_pitch: float,
        host_role_object_id: int | None,
        body_q,
        body_qd,
        control,
        body_q_np=None,
        joint_q=None,
        joint_qd=None,
        mouse_buttons=None,
        body_f=None,
    ) -> None:
        model = registry.model
        if model is None:
            return

        needs_joint_torque = (
            (not record.uses_weld_fallback and record.mount_joint_dof_idx is not None)
            or self.pitch_dof_spec is not None
        )
        needs_joint_q = (
            self.pitch_dof_spec is not None or record.mount_joint_coord_idx is not None
        )
        needs_relpose = registry.uses_mujoco_weld and record.uses_weld_fallback

        joint_q_src = joint_q if joint_q is not None else getattr(model, "joint_q", None)
        joint_qd_src = joint_qd if joint_qd is not None else getattr(model, "joint_qd", None)
        joint_qd_np = (
            joint_qd_src.numpy()
            if needs_joint_torque and joint_qd_src is not None
            else None
        )
        joint_f_np = (
            control.joint_f.numpy()
            if needs_joint_torque and control is not None and control.joint_f is not None
            else None
        )
        joint_q_np = (
            joint_q_src.numpy()
            if needs_joint_q and joint_q_src is not None
            else None
        )
        joint_target_pos_np = (
            control.joint_target_pos.numpy()
            if needs_joint_q
            and control is not None
            and getattr(control, "joint_target_pos", None) is not None
            else None
        )
        relpose_np = (
            model.equality_constraint_relpose.numpy()
            if needs_relpose and model.equality_constraint_relpose is not None
            else None
        )

        if body_q_np is None:
            body_q_np = body_q.numpy()

        cfg = self.cfg
        dead_zone = math.radians(cfg.angle_dead_zone_deg)

        global_host = registry.global_body_idx(world, record.host_body_idx)
        aim_local_idx = self.aim_body_idx if self.aim_body_idx >= 0 else record.tool_body_idx
        global_aim = registry.global_body_idx(world, aim_local_idx)

        host_body_q = body_q_np[global_host]
        aim_body_q = body_q_np[global_aim]

        if self._rl_active:
            yaw_error = float(self._rl_yaw) * float(self.rl_cfg.yaw_command_scale_rad)
            pitch_error = float(self._rl_pitch) * float(self.rl_cfg.pitch_command_scale_rad)
            if abs(yaw_error) < dead_zone:
                yaw_error = 0.0
            if abs(pitch_error) < dead_zone:
                pitch_error = 0.0
            fire_active = float(self._rl_fire) > float(self.rl_cfg.fire_threshold)
            if fire_active and not self._prev_rl_fire_active:
                self._handle_shoot(
                    registry,
                    record,
                    world=world,
                    aim_body_q=aim_body_q,
                    cfg=cfg,
                )
            self._prev_rl_fire_active = fire_active
        else:
            desired_world = camera_forward_z_up(float(camera_yaw), float(camera_pitch))
            yaw_error, pitch_error = compute_host_local_aim_errors(
                host_body_q,
                aim_body_q,
                desired_world,
                cfg.aim_forward_local,
            )

            if abs(yaw_error) < dead_zone:
                yaw_error = 0.0
            if abs(pitch_error) < dead_zone:
                pitch_error = 0.0

            mouse_left = (
                mouse_buttons is not None
                and hasattr(mouse_buttons, "__getitem__")
                and len(mouse_buttons) > 0
                and bool(mouse_buttons[0])
            )
            if mouse_left and not self._prev_mouse_left:
                self._handle_shoot(
                    registry,
                    record,
                    world=world,
                    aim_body_q=aim_body_q,
                    cfg=cfg,
                )
            self._prev_mouse_left = mouse_left
            self._prev_rl_fire_active = False

        relpose_dirty = False
        joint_f_dirty = False
        joint_target_pos_dirty = False

        lo, hi = record.mount_yaw_limits
        previous_yaw = float(record.mount_yaw)
        if record.mount_joint_coord_idx is not None and joint_q_np is not None:
            global_coord = registry.global_coord_idx(world, record.mount_joint_coord_idx)
            if 0 <= global_coord < joint_q_np.shape[0]:
                previous_yaw = float(joint_q_np[global_coord])
        else:
            previous_yaw = measure_mount_yaw_in_host_frame(
                host_body_q,
                aim_body_q,
                cfg.aim_forward_local,
            )
            record.mount_yaw = previous_yaw

        if record.uses_weld_fallback:
            if abs(yaw_error) >= dead_zone:
                proposed_yaw = _wrap_pi(
                    previous_yaw + cfg.weld_yaw_drive_gain * yaw_error * float(dt)
                )
                record.mount_yaw = clamp_angle_to_limits(proposed_yaw, lo, hi, previous_yaw)
            if relpose_np is not None and record.mount_eq_idx is not None:
                global_eq = registry.global_eq_idx(world, record.mount_eq_idx)
                relpose_np[global_eq] = compose_mounted_weld_relpose(
                    record.host_anchor_local,
                    record.tool_anchor_local,
                    yaw_rad=record.mount_yaw,
                    mount_axis=record.mount_axis,
                )
                relpose_dirty = True
        elif record.mount_joint_dof_idx is not None and joint_f_np is not None:
            global_mount_dof = registry.global_dof_idx(world, record.mount_joint_dof_idx)
            mount_rate = 0.0
            if joint_qd_np is not None and 0 <= global_mount_dof < joint_qd_np.shape[0]:
                mount_rate = float(joint_qd_np[global_mount_dof])
            yaw_torque = soft_limit_torque(
                pd_torque(
                    yaw_error,
                    mount_rate,
                    cfg.yaw_torque_gain,
                    cfg.yaw_damping,
                    cfg.max_yaw_torque,
                ),
                cfg.max_yaw_torque,
            )
            joint_f_np[global_mount_dof] = float(joint_f_np[global_mount_dof]) + yaw_torque
            joint_f_dirty = True

        pitch_spec = self.pitch_dof_spec
        current_q = 0.0
        target_q = 0.0
        pitch_torque = 0.0
        if pitch_spec is not None and joint_f_np is not None and joint_q_np is not None:
            global_pitch_dof = registry.global_dof_idx(world, pitch_spec.global_dof_idx)
            global_coord = registry.global_coord_idx(world, pitch_spec.local_coord_idx)
            if (
                0 <= global_coord < joint_q_np.shape[0]
                and 0 <= global_pitch_dof < joint_f_np.shape[0]
            ):
                current_q = float(joint_q_np[global_coord])
                # Desired joint angle from signed pitch error, clamped to USD limits.
                target_q = clamp_angle_to_limits(
                    current_q + float(cfg.pitch_joint_sign) * pitch_error,
                    pitch_spec.limit_lower,
                    pitch_spec.limit_upper,
                    current_q,
                )
                pitch_spec.current_target = target_q

                # Neutralize any leftover POSITION actuator (target==current → zero P term).
                if (
                    joint_target_pos_np is not None
                    and 0 <= global_pitch_dof < joint_target_pos_np.shape[0]
                ):
                    joint_target_pos_np[global_pitch_dof] = current_q
                    joint_target_pos_dirty = True

                pitch_rate = 0.0
                if joint_qd_np is not None and 0 <= global_pitch_dof < joint_qd_np.shape[0]:
                    pitch_rate = float(joint_qd_np[global_pitch_dof])

                # PD on joint-space tracking error (already includes pitch_joint_sign).
                joint_error = target_q - current_q
                pd_term = pd_torque(
                    joint_error,
                    pitch_rate,
                    cfg.pitch_torque_gain,
                    cfg.pitch_damping,
                    cfg.max_pitch_torque,
                )
                gravity_term = 0.0
                if cfg.pitch_gravity_comp_enable:
                    gravity_term = pitch_gravity_compensation_torque(
                        self.pitch_gravity_state.coeff,
                        current_q,
                        cfg.pitch_gravity_basis,
                    )
                pitch_torque = soft_limit_torque(
                    pd_term + gravity_term,
                    cfg.max_pitch_torque,
                )
                if cfg.pitch_gravity_comp_enable:
                    update_pitch_gravity_coeff(
                        self.pitch_gravity_state,
                        torque_applied=pitch_torque,
                        q=current_q,
                        rate=pitch_rate,
                        dt=float(dt),
                        learn_rate=cfg.pitch_gravity_learn_rate,
                        rate_eps=cfg.pitch_gravity_rate_eps,
                        accel_eps=cfg.pitch_gravity_accel_eps,
                        dq_eps=cfg.pitch_gravity_dq_eps,
                        max_abs_coeff=cfg.max_pitch_torque,
                        basis_mode=cfg.pitch_gravity_basis,
                    )
                # Overwrite (do not accumulate leftover joint_f across frames).
                joint_f_np[global_pitch_dof] = pitch_torque
                joint_f_dirty = True

        if (
            relpose_dirty
            and model.equality_constraint_relpose is not None
            and relpose_np is not None
        ):
            model.equality_constraint_relpose.assign(relpose_np)
            registry.notify_solver()
        if joint_f_dirty and control is not None and joint_f_np is not None:
            control.joint_f.assign(joint_f_np)
        if joint_target_pos_dirty and control is not None and joint_target_pos_np is not None:
            control.joint_target_pos.assign(joint_target_pos_np)

    # -- viewer overlay --------------------------------------------------------

    def view_overlay(
        self,
        registry,
        record,
        *,
        host_role_object_id: int | None,
        window_w: float,
        window_h: float,
        camera_pitch_deg: float,
        camera_fov_deg: float,
    ):
        """Enable only when this turret is attached to the given human-controlled host."""
        if not record.attached:
            return None
        if host_role_object_id is not None and record.host_role_object_id != int(
            host_role_object_id
        ):
            return None
        limits = self._camera_pitch_limits_deg()
        if limits is None:
            return None
        return build_turret_110mm_third_person_aim_view(
            window_w=window_w,
            window_h=window_h,
            camera_pitch_deg=camera_pitch_deg,
            camera_fov_deg=camera_fov_deg,
            pitch_limit_cam_deg=limits,
            style=self._view_style,
        )

    def _camera_pitch_limits_deg(self) -> Optional[Tuple[float, float]]:
        if self.pitch_dof_spec is None:
            return None
        return joint_pitch_limits_to_camera_pitch_deg(
            self.pitch_dof_spec.limit_lower,
            self.pitch_dof_spec.limit_upper,
            float(self.cfg.pitch_joint_sign),
        )

    def draw_view_overlay(self, overlay, shapes_module, batch=None):
        return draw_turret_110mm_aim_view_overlay(
            overlay,
            shapes_module=shapes_module,
            batch=batch,
        )

    def forward_local(self) -> Tuple[float, float, float]:
        return tuple(float(v) for v in self.cfg.aim_forward_local)
