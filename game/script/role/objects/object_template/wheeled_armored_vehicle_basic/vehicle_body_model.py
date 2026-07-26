"""Rigid-body mass specs and suspension gain helpers for wheeled_armored_vehicle_basic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np
import newton
import warp as wp
from newton import GeoType, JointTargetMode, ShapeFlags

from .vehicle_joint_model import DofPhysicsSpec


@dataclass(frozen=True)
class BodyMassSpec:
    chassis_mass_kg: float = 14000.0
    suspension_mass_kg: float = 200.0
    wheel_mass_kg: float = 300.0


@dataclass(frozen=True)
class SuspensionGainSpec:
    num_corners: int = 6
    lever_arm_m: float = 0.65
    # Hard pose target: swing-arm vs chassis (diagram left 3°).
    target_angle_rad: float = math.radians(3.0)
    damping_ratio: float = 0.55
    effective_inertia_kgm2: float = 540.0
    armature: float = 0.05
    include_wheel_load: bool = False
    # Residual sag past target under gravity ≈ target_angle / stiffness_scale.
    # Use >= 4 so settle stays close to the hardcoded target pose.
    stiffness_scale: float = 5.0
    servo_stiffness_nm_rad: Optional[float] = None
    servo_damping_nm_s_rad: Optional[float] = None
    servo_max_torque_nm: float = 50000.0
    stiffness: Optional[float] = None
    damping: Optional[float] = None

    @property
    def nominal_sag_rad(self) -> float:
        return self.target_angle_rad


@dataclass(frozen=True)
class WheelSpinGainSpec:
    """Wheel spin properties used by the direct-torque drive."""

    armature: float = 0.01


def classify_body_mass(label: str, spec: BodyMassSpec) -> Optional[float]:
    lower = str(label).lower().rstrip("/")
    parts = lower.split("/")
    name = parts[-1] if parts else lower

    if name == "vehicle_body":
        return spec.chassis_mass_kg
    if name.startswith("wheels_") and len(parts) >= 2 and parts[-2] == name:
        return spec.wheel_mass_kg
    if name.startswith("susp_") and len(parts) >= 2 and parts[-2] == name:
        return spec.suspension_mass_kg
    return None


def apply_body_masses(builder_env, spec: BodyMassSpec) -> int:
    """Override body masses from spec; scale inertia proportionally when mass changes."""
    applied = 0
    for body_idx in range(builder_env.body_count):
        label = str(builder_env.body_label[body_idx])
        target_mass = classify_body_mass(label, spec)
        if target_mass is None:
            continue

        old_mass = float(builder_env.body_mass[body_idx])
        if old_mass <= 0.0:
            builder_env.body_mass[body_idx] = target_mass
            builder_env.body_inv_mass[body_idx] = 1.0 / target_mass
            applied += 1
            continue

        if abs(old_mass - target_mass) <= 1e-3:
            applied += 1
            continue

        scale = target_mass / old_mass
        builder_env.body_mass[body_idx] = target_mass
        builder_env.body_inv_mass[body_idx] = 1.0 / target_mass
        builder_env.body_inertia[body_idx] = builder_env.body_inertia[body_idx] * scale
        inertia = builder_env.body_inertia[body_idx]
        builder_env.body_inv_inertia[body_idx] = wp.inverse(inertia)
        applied += 1

    return applied


def compute_suspension_spec(
    body_spec: BodyMassSpec,
    gain_spec: SuspensionGainSpec,
    gravity: float = 9.81,
) -> DofPhysicsSpec:
    corner_load_n = body_spec.chassis_mass_kg * gravity / float(gain_spec.num_corners)
    carried_load_n = corner_load_n + 0.5 * body_spec.suspension_mass_kg * gravity
    if gain_spec.include_wheel_load:
        carried_load_n += body_spec.wheel_mass_kg * gravity
    ke = gain_spec.stiffness
    if ke is None:
        ke = (
            carried_load_n
            * gain_spec.lever_arm_m
            / max(gain_spec.nominal_sag_rad, 1e-4)
            * gain_spec.stiffness_scale
        )

    kd = gain_spec.damping
    if kd is None:
        kd = (
            2.0
            * gain_spec.damping_ratio
            * math.sqrt(max(ke, 1.0) * max(gain_spec.effective_inertia_kgm2, 1e-3))
        )

    return DofPhysicsSpec(
        stiffness=float(ke),
        damping=float(kd),
        armature=gain_spec.armature,
        target_mode=JointTargetMode.POSITION,
        nominal=0.0,
    )


def parse_body_mass_spec(raw: Mapping[str, float] | None) -> BodyMassSpec:
    raw = raw or {}
    return BodyMassSpec(
        chassis_mass_kg=float(raw.get("chassis_mass_kg", 14000.0)),
        suspension_mass_kg=float(raw.get("suspension_mass_kg", 200.0)),
        wheel_mass_kg=float(raw.get("wheel_mass_kg", 300.0)),
    )


@dataclass(frozen=True)
class WheelMaterialSpec:
    friction: float = 0.4
    torsional_friction: float = 0.001
    rolling_friction: float = 0.0005
    friction_stiffness: float = 500.0


@dataclass(frozen=True)
class VehicleContactSpec:
    """Contact response for chassis / wheel / suspension collision proxies."""

    ke: float = 80.0
    kd: float = 20.0
    restitution: float = 0.0
    margin: float = 0.03
    chassis_box_shrink: float = 0.98


@dataclass(frozen=True)
class RolloverSpec:
    """Mitigate rollover bounce by releasing suspension pose servos when tipped."""

    passive_suspension_upright_dot: float = 0.5


def is_chassis_shape_label(label: str) -> bool:
    lower = str(label).lower().rstrip("/")
    parts = lower.split("/")
    name = parts[-1] if parts else lower
    return name == "vehicle_body"


def is_suspension_shape_label(label: str) -> bool:
    lower = str(label).lower().rstrip("/")
    parts = lower.split("/")
    name = parts[-1] if parts else lower
    return name.startswith("susp_")


def _add_visual_mesh_copy(builder_env, shape_idx: int) -> bool:
    flags = int(builder_env.shape_flags[shape_idx])
    if not (flags & int(ShapeFlags.VISIBLE)):
        return False

    mesh = builder_env.shape_source[shape_idx]
    if mesh is None:
        return False

    cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        margin=builder_env.shape_margin[shape_idx],
        is_solid=builder_env.shape_is_solid[shape_idx],
        has_shape_collision=False,
        has_particle_collision=False,
        is_visible=True,
    )
    builder_env.add_shape_mesh(
        body=builder_env.shape_body[shape_idx],
        xform=builder_env.shape_transform[shape_idx],
        cfg=cfg,
        mesh=mesh,
        color=builder_env.shape_color[shape_idx],
        label=f"{builder_env.shape_label[shape_idx]}_visual",
        scale=builder_env.shape_scale[shape_idx],
    )
    builder_env.shape_flags[shape_idx] &= ~int(ShapeFlags.VISIBLE)
    return True


def _replace_shape_with_aabb_box(
    builder_env,
    shape_idx: int,
    shrink: float = 1.0,
) -> None:
    mesh = builder_env.shape_source[shape_idx]
    if mesh is None:
        return

    scale = builder_env.shape_scale[shape_idx]
    scale_xyz = np.asarray([scale[0], scale[1], scale[2]], dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale_xyz
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) * 0.5
    half = np.maximum((vmax - vmin) * 0.5, 1e-4) * max(float(shrink), 1e-3)

    shape_tf = builder_env.shape_transform[shape_idx]
    builder_env.shape_type[shape_idx] = int(GeoType.BOX)
    builder_env.shape_source[shape_idx] = None
    builder_env.shape_scale[shape_idx] = wp.vec3(float(half[0]), float(half[1]), float(half[2]))
    builder_env.shape_transform[shape_idx] = shape_tf * wp.transform(
        wp.vec3(float(center[0]), float(center[1]), float(center[2])),
        wp.quat_identity(),
    )


def _replace_shape_with_bounding_sphere(builder_env, shape_idx: int) -> None:
    mesh = builder_env.shape_source[shape_idx]
    if mesh is None:
        return

    scale = builder_env.shape_scale[shape_idx]
    scale_xyz = np.asarray([scale[0], scale[1], scale[2]], dtype=np.float32)
    vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale_xyz
    center = np.mean(vertices, axis=0)
    radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
    radius = max(radius, 1e-4)

    shape_tf = builder_env.shape_transform[shape_idx]
    builder_env.shape_type[shape_idx] = int(GeoType.SPHERE)
    builder_env.shape_source[shape_idx] = None
    builder_env.shape_scale[shape_idx] = wp.vec3(radius, 0.0, 0.0)
    builder_env.shape_transform[shape_idx] = shape_tf * wp.transform(
        wp.vec3(float(center[0]), float(center[1]), float(center[2])),
        wp.quat_identity(),
    )


def apply_vehicle_collision_proxies(
    builder_env,
    contact_spec: VehicleContactSpec | None = None,
) -> tuple[int, int, int, int]:
    """Split authored visuals from stable convex collision proxies."""
    contact_spec = contact_spec or VehicleContactSpec()
    visual_copies = 0
    chassis_proxies = 0
    wheel_proxies = 0
    suspension_proxies = 0

    # Snapshot indices because add_shape_mesh appends new shapes.
    shape_indices = list(range(builder_env.shape_count))
    for shape_idx in shape_indices:
        label = str(builder_env.shape_label[shape_idx])
        if label.endswith("_visual"):
            continue

        flags = int(builder_env.shape_flags[shape_idx])
        if not (flags & int(ShapeFlags.COLLIDE_SHAPES)):
            continue
        if int(builder_env.shape_type[shape_idx]) != int(GeoType.MESH):
            continue

        if _add_visual_mesh_copy(builder_env, shape_idx):
            visual_copies += 1

        if is_chassis_shape_label(label):
            _replace_shape_with_aabb_box(
                builder_env,
                shape_idx,
                shrink=contact_spec.chassis_box_shrink,
            )
            chassis_proxies += 1
        elif is_wheel_shape_label(label):
            _replace_shape_with_bounding_sphere(builder_env, shape_idx)
            wheel_proxies += 1
        elif is_suspension_shape_label(label):
            _replace_shape_with_aabb_box(builder_env, shape_idx)
            suspension_proxies += 1

    return visual_copies, chassis_proxies, wheel_proxies, suspension_proxies


def is_vehicle_collision_shape_label(label: str) -> bool:
    return (
        is_chassis_shape_label(label)
        or is_wheel_shape_label(label)
        or is_suspension_shape_label(label)
    )


def apply_vehicle_contact_properties(
    builder_env,
    spec: VehicleContactSpec,
) -> int:
    """Apply soft, non-bouncy contact to all vehicle collision proxies."""
    applied = 0
    for shape_idx in range(builder_env.shape_count):
        label = str(builder_env.shape_label[shape_idx])
        if label.endswith("_visual"):
            continue
        if not is_vehicle_collision_shape_label(label):
            continue
        flags = int(builder_env.shape_flags[shape_idx])
        if not (flags & int(ShapeFlags.COLLIDE_SHAPES)):
            continue

        builder_env.shape_material_ke[shape_idx] = float(spec.ke)
        builder_env.shape_material_kd[shape_idx] = float(spec.kd)
        builder_env.shape_material_restitution[shape_idx] = float(spec.restitution)
        builder_env.shape_margin[shape_idx] = float(spec.margin)
        applied += 1
    return applied


def is_wheel_shape_label(label: str) -> bool:
    lower = str(label).lower()
    if lower.endswith("_visual"):
        return False
    parts = lower.rstrip("/").split("/")
    name = parts[-1] if parts else lower
    return name.startswith("wheels_")


def apply_wheel_shape_materials(builder_env, spec: WheelMaterialSpec) -> int:
    """Lower wheel contact friction so suspension can settle without tipping."""
    applied = 0
    for shape_idx in range(builder_env.shape_count):
        label = str(builder_env.shape_label[shape_idx])
        if not is_wheel_shape_label(label):
            continue
        builder_env.shape_material_mu[shape_idx] = spec.friction
        builder_env.shape_material_mu_torsional[shape_idx] = spec.torsional_friction
        builder_env.shape_material_mu_rolling[shape_idx] = spec.rolling_friction
        builder_env.shape_material_kf[shape_idx] = spec.friction_stiffness
        applied += 1
    return applied


def parse_wheel_material_spec(raw: Mapping[str, float] | None) -> WheelMaterialSpec:
    raw = raw or {}
    return WheelMaterialSpec(
        friction=float(raw.get("friction", 0.4)),
        torsional_friction=float(raw.get("torsional_friction", 0.001)),
        rolling_friction=float(raw.get("rolling_friction", 0.0005)),
        friction_stiffness=float(raw.get("friction_stiffness", 500.0)),
    )


def parse_vehicle_contact_spec(raw: Mapping[str, float] | None) -> VehicleContactSpec:
    raw = raw or {}
    return VehicleContactSpec(
        ke=float(raw.get("ke", 80.0)),
        kd=float(raw.get("kd", 20.0)),
        restitution=float(raw.get("restitution", 0.0)),
        margin=float(raw.get("margin", 0.03)),
        chassis_box_shrink=float(raw.get("chassis_box_shrink", 0.98)),
    )


def parse_rollover_spec(raw: Mapping[str, float] | None) -> RolloverSpec:
    raw = raw or {}
    return RolloverSpec(
        passive_suspension_upright_dot=float(
            raw.get("passive_suspension_upright_dot", 0.5)
        ),
    )


def is_suspension_joint_label(label: str) -> bool:
    lower = str(label).lower()
    return "vb_susp" in lower and "revolute" in lower


def parse_possess_offset(raw) -> Tuple[float, float, float]:
    if raw is None:
        return (0.0, 0.0, 0.0)
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    raise ValueError(f"possess_offset must be [x, y, z], got {raw!r}")


def parse_suspension_gain_spec(raw: Mapping[str, float] | None) -> SuspensionGainSpec:
    raw = raw or {}
    stiffness = raw.get("stiffness")
    damping = raw.get("damping")
    if raw.get("target_angle_deg") is not None:
        target_angle_rad = math.radians(float(raw["target_angle_deg"]))
    elif raw.get("nominal_sag_deg") is not None:
        target_angle_rad = math.radians(float(raw["nominal_sag_deg"]))
    elif raw.get("target_angle_rad") is not None:
        target_angle_rad = float(raw["target_angle_rad"])
    elif raw.get("nominal_sag_rad") is not None:
        target_angle_rad = float(raw["nominal_sag_rad"])
    else:
        target_angle_rad = math.radians(3.0)
    return SuspensionGainSpec(
        num_corners=int(raw.get("num_corners", 6)),
        lever_arm_m=float(raw.get("lever_arm_m", 0.65)),
        target_angle_rad=target_angle_rad,
        damping_ratio=float(raw.get("damping_ratio", 0.55)),
        effective_inertia_kgm2=float(raw.get("effective_inertia_kgm2", 540.0)),
        armature=float(raw.get("armature", 0.05)),
        include_wheel_load=bool(raw.get("include_wheel_load", False)),
        stiffness_scale=float(raw.get("stiffness_scale", 5.0)),
        servo_stiffness_nm_rad=(
            None
            if raw.get("servo_stiffness_nm_rad") is None
            else float(raw["servo_stiffness_nm_rad"])
        ),
        servo_damping_nm_s_rad=(
            None
            if raw.get("servo_damping_nm_s_rad") is None
            else float(raw["servo_damping_nm_s_rad"])
        ),
        servo_max_torque_nm=float(raw.get("servo_max_torque_nm", 50000.0)),
        stiffness=None if stiffness is None else float(stiffness),
        damping=None if damping is None else float(damping),
    )


def parse_wheel_spin_gain_spec(raw: Mapping[str, float] | None) -> WheelSpinGainSpec:
    raw = raw or {}
    return WheelSpinGainSpec(
        armature=float(raw.get("armature", 0.01)),
    )
