"""One-articulation kinematic anchor + dynamic target, linked by a free D6.

Soft constraint force is applied by a lazy object-template extension that
writes the articulation control-force buffer (player-action path), not by
joint stiffness. Nested ``anchor`` / ``target`` primitives are optional
``box`` / ``sphere`` parts — not a second role-level object field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import newton
import warp as wp

from script.role.objects.base_object import BaseObject, BaseObjectModel
from script.simulate.mesh_builder import MeshBuilder
from pydantic import BaseModel, Field


class RigidSoftAnchorPartModel(BaseModel):
    """One primitive body (kinematic anchor or dynamic target)."""

    shape: Literal["box", "sphere"] = "box"
    size: List[float] = Field(default_factory=lambda: [0.025, 0.025, 0.025])
    radius: float = 0.12
    filter_all_collisions: bool = False
    object_friction: float = 0.5
    object_elasticity: float = 0.5
    color: Optional[List[float]] = None


class RigidSoftAnchorTargetModel(RigidSoftAnchorPartModel):
    shape: Literal["box", "sphere"] = "sphere"
    object_mass: float = 0.8
    lock_inertia: bool = True


class RigidSoftAnchorModel(BaseObjectModel):
    type: Literal["rigid_soft_anchor"] = "rigid_soft_anchor"

    target_offset: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.22])
    max_linear_force: float = 250.0
    linear_stiffness: float = 600.0
    linear_damping: float = 40.0
    anchor: RigidSoftAnchorPartModel = Field(default_factory=RigidSoftAnchorPartModel)
    target: RigidSoftAnchorTargetModel = Field(default_factory=RigidSoftAnchorTargetModel)


def _as_dict(value: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        merged = dict(fallback)
        merged.update(value)
        return merged
    return dict(fallback)


def _vec3(value: Any, fallback: List[float]) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return [float(fallback[0]), float(fallback[1]), float(fallback[2])]
    return [float(value[0]), float(value[1]), float(value[2])]


def _shape_color(raw: Any):
    if not raw:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    vals = [float(raw[0]), float(raw[1]), float(raw[2])]
    peak = max(vals)
    if peak > 1.0:
        vals = [v / 255.0 for v in vals]
    return vals


def _part_shape_cfg(base_cfg, part: Dict[str, Any], *, density: float) -> newton.ModelBuilder.ShapeConfig:
    collide = not bool(part.get("filter_all_collisions", False))
    cfg = newton.ModelBuilder.ShapeConfig(
        density=float(density),
        mu=float(part.get("object_friction", base_cfg.mu)),
        restitution=float(part.get("object_elasticity", 0.0)),
        has_shape_collision=collide,
        has_particle_collision=collide,
        is_visible=True,
        collision_group=int(base_cfg.collision_group),
    )
    return cfg


def _add_part_shape(
    builder_env: newton.ModelBuilder,
    body: int,
    part: Dict[str, Any],
    cfg: newton.ModelBuilder.ShapeConfig,
    label: str,
) -> int:
    shape = str(part.get("shape", "box")).lower()
    color = _shape_color(part.get("color"))
    if shape == "sphere":
        radius = float(part["radius"])
        return builder_env.add_shape_sphere(
            body,
            radius=radius,
            cfg=cfg,
            color=color,
            label=label,
        )
    size = _vec3(part.get("size"), RigidSoftAnchorPartModel().size)
    return builder_env.add_shape_box(
        body=body,
        hx=size[0],
        hy=size[1],
        hz=size[2],
        cfg=cfg,
        color=color,
        label=label,
    )


def _part_aabb(part: Dict[str, Any], fallback: RigidSoftAnchorPartModel) -> List[float]:
    shape = str(part.get("shape", fallback.shape)).lower()
    if shape == "sphere":
        radius = float(part.get("radius", fallback.radius))
        return [radius, radius, radius]
    return _vec3(part.get("size"), fallback.size)


class RigidSoftAnchorObject(BaseObject):
    object_key = "rigid_soft_anchor"
    model_cls = RigidSoftAnchorModel
    object_type_id: int = 8

    @staticmethod
    def add_physics(builder_env: newton.ModelBuilder, label: str, data: RigidSoftAnchorModel, cfg, **kwargs):
        defaults = RigidSoftAnchorModel()
        data_dict = _as_dict(data, defaults.model_dump())
        anchor = _as_dict(data_dict.get("anchor"), defaults.anchor.model_dump())
        target = _as_dict(data_dict.get("target"), defaults.target.model_dump())
        offset = _vec3(data_dict.get("target_offset"), defaults.target_offset)

        joint_start = builder_env.joint_count
        unlimited = newton.ModelBuilder.JointDofConfig.create_unlimited

        anchor_body = builder_env.add_link(
            mass=0.0,
            lock_inertia=True,
            is_kinematic=True,
            label=f"{label}_anchor",
        )
        free_joint = builder_env.add_joint_free(
            child=anchor_body,
            label=f"{label}_anchor_free_joint",
            collision_filter_parent=False,
        )

        target_body = builder_env.add_link(
            xform=wp.transform(wp.vec3(offset[0], offset[1], offset[2]), wp.quat_identity()),
            mass=float(target.get("object_mass", defaults.target.object_mass)),
            lock_inertia=bool(target.get("lock_inertia", True)),
            is_kinematic=False,
            label=f"{label}_target",
        )
        d6_joint = builder_env.add_joint_d6(
            parent=anchor_body,
            child=target_body,
            linear_axes=[unlimited(newton.Axis.X), unlimited(newton.Axis.Y), unlimited(newton.Axis.Z)],
            angular_axes=[unlimited(newton.Axis.X), unlimited(newton.Axis.Y), unlimited(newton.Axis.Z)],
            label=f"{label}_target_soft_joint",
            collision_filter_parent=True,
        )
        q_start = int(builder_env.joint_q_start[d6_joint])
        builder_env.joint_q[q_start + 0] = offset[0]
        builder_env.joint_q[q_start + 1] = offset[1]
        builder_env.joint_q[q_start + 2] = offset[2]

        builder_env.add_articulation(
            [free_joint, d6_joint],
            label=f"{label}_articulation",
        )

        anchor_cfg = _part_shape_cfg(cfg, anchor, density=0.0)
        target_density = 0.0 if bool(target.get("lock_inertia", True)) else float(cfg.density)
        target_cfg = _part_shape_cfg(cfg, target, density=target_density)
        anchor_shape = _add_part_shape(
            builder_env, anchor_body, anchor, anchor_cfg, f"{label}_anchor"
        )
        target_shape = _add_part_shape(
            builder_env, target_body, target, target_cfg, f"{label}_target"
        )

        return {
            "path_body_map": {
                "/anchor": int(anchor_body),
                "/target": int(target_body),
            },
            "path_joint_map": {
                "/anchor_free_joint": int(free_joint),
                "/target_soft_joint": int(d6_joint),
            },
            "path_shape_map": {
                "/anchor": int(anchor_shape),
                "/target": int(target_shape),
            },
            "joint_start": int(joint_start),
            "joint_end": int(builder_env.joint_count),
            "passive_constraint": {
                "target_offset": offset,
                "max_linear_force": float(data_dict.get("max_linear_force", defaults.max_linear_force)),
                "linear_stiffness": float(data_dict.get("linear_stiffness", defaults.linear_stiffness)),
                "linear_damping": float(data_dict.get("linear_damping", defaults.linear_damping)),
                "anchor_friction": float(anchor.get("object_friction", defaults.anchor.object_friction)),
                "anchor_elasticity": float(anchor.get("object_elasticity", defaults.anchor.object_elasticity)),
                "target_friction": float(target.get("object_friction", defaults.target.object_friction)),
                "target_elasticity": float(target.get("object_elasticity", defaults.target.object_elasticity)),
            },
        }

    @staticmethod
    def add_visual(mesh_builder: MeshBuilder, data: RigidSoftAnchorModel, pos):
        defaults = RigidSoftAnchorModel()
        data_dict = _as_dict(data, defaults.model_dump())
        anchor = _as_dict(data_dict.get("anchor"), defaults.anchor.model_dump())
        target = _as_dict(data_dict.get("target"), defaults.target.model_dump())
        offset = _vec3(data_dict.get("target_offset"), defaults.target_offset)
        origin = wp.vec3(*pos) if not isinstance(pos, wp.vec3) else pos

        def _add(part: Dict[str, Any], at):
            shape = str(part.get("shape", "box")).lower()
            if shape == "sphere":
                mesh_builder.add_sphere(pos=at, radius=float(part.get("radius", defaults.target.radius)))
                return
            size = _vec3(part.get("size"), defaults.anchor.size)
            mesh_builder.add_box(pos=at, size=size)

        _add(anchor, origin)
        _add(target, wp.vec3(origin[0] + offset[0], origin[1] + offset[1], origin[2] + offset[2]))

    @staticmethod
    def get_size(data: RigidSoftAnchorModel) -> List[float]:
        defaults = RigidSoftAnchorPartModel()
        data_dict = _as_dict(data, RigidSoftAnchorModel().model_dump())
        return _part_aabb(_as_dict(data_dict.get("anchor"), defaults.model_dump()), defaults)
