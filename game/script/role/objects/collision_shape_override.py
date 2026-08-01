"""Per-body collision shape overrides for articulation imports.

Consumed from the object config key ``body_collision_shape_overrides``
(e.g. ``{"wheel_*": "cylinder"}``). Body names are matched case-insensitively
against the body label's basename (and full path) using fnmatch-style
wildcards (``*`` / ``?``). Only shapes added by the object currently being
imported are modified, so one object's override never leaks onto another
object sharing the same builder.
"""

from __future__ import annotations

import fnmatch
import math
from typing import Dict, Optional

import numpy as np
import warp as wp

import newton
from newton import GeoType, ShapeFlags

_SUPPORTED_OVERRIDE_TYPES = frozenset({"cylinder", "box", "sphere"})


def _body_basename(body_label: str) -> str:
    return str(body_label).rstrip("/").split("/")[-1]


def _match_override(body_label: str, overrides: Dict[str, str]) -> Optional[str]:
    label = str(body_label)
    basename = _body_basename(label)
    for pattern, shape_type in overrides.items():
        pat = str(pattern)
        if fnmatch.fnmatchcase(basename.lower(), pat.lower()) or fnmatch.fnmatchcase(
            label.lower(), pat.lower()
        ):
            return str(shape_type).lower()
    return None


def _mesh_vertices(builder_env, shape_idx: int) -> Optional[np.ndarray]:
    mesh = builder_env.shape_source[shape_idx]
    if mesh is None:
        return None
    scale = builder_env.shape_scale[shape_idx]
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return vertices * np.asarray([scale[0], scale[1], scale[2]], dtype=np.float64)


def _quat_align_z(axis: np.ndarray) -> wp.quat:
    """Build a quaternion rotating the local +Z axis onto ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return wp.quat_identity()
    axis = axis / norm
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z, axis), -1.0, 1.0))
    if abs(dot) > 1.0 - 1e-6:
        if dot > 0:
            return wp.quat_identity()
        return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)
    cross = np.cross(z, axis)
    cross = cross / np.linalg.norm(cross)
    angle = math.acos(dot)
    return wp.quat_from_axis_angle(
        wp.vec3(float(cross[0]), float(cross[1]), float(cross[2])), float(angle)
    )


def _replace_shape_with_cylinder(builder_env, shape_idx: int) -> bool:
    """Replace a mesh collision shape with a fitted cylinder.

    Uses a PCA-based OBB of the mesh: the smallest principal axis is treated
    as the cylinder axis (the wheel axle), and the two larger extents give the
    radius. The cylinder's local +Z is aligned to that axle direction.
    """
    vertices = _mesh_vertices(builder_env, shape_idx)
    if vertices is None or len(vertices) < 3:
        return False

    center = vertices.mean(axis=0)
    cov = np.cov((vertices - center).T)
    evals, evecs = np.linalg.eigh(cov)
    proj = (vertices - center) @ evecs
    extents = proj.max(axis=0) - proj.min(axis=0)

    axle_idx = int(np.argmin(extents))
    axle = evecs[:, axle_idx]
    radius = float(max(extents[i] for i in range(3) if i != axle_idx)) * 0.5
    half_height = float(extents[axle_idx]) * 0.5
    radius = max(radius, 1e-4)
    half_height = max(half_height, 1e-4)
    quat = _quat_align_z(axle)

    shape_tf = builder_env.shape_transform[shape_idx]
    builder_env.shape_type[shape_idx] = int(GeoType.CYLINDER)
    builder_env.shape_source[shape_idx] = None
    builder_env.shape_scale[shape_idx] = wp.vec3(
        float(radius), float(half_height), 0.0
    )
    builder_env.shape_transform[shape_idx] = shape_tf * wp.transform(
        wp.vec3(float(center[0]), float(center[1]), float(center[2])), quat
    )
    return True


def _replace_shape_with_box(builder_env, shape_idx: int) -> bool:
    """Replace a mesh collision shape with its axis-aligned bounding box."""
    vertices = _mesh_vertices(builder_env, shape_idx)
    if vertices is None or len(vertices) < 1:
        return False

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) * 0.5
    half = np.maximum((vmax - vmin) * 0.5, 1e-4)

    shape_tf = builder_env.shape_transform[shape_idx]
    builder_env.shape_type[shape_idx] = int(GeoType.BOX)
    builder_env.shape_source[shape_idx] = None
    builder_env.shape_scale[shape_idx] = wp.vec3(
        float(half[0]), float(half[1]), float(half[2])
    )
    builder_env.shape_transform[shape_idx] = shape_tf * wp.transform(
        wp.vec3(float(center[0]), float(center[1]), float(center[2])),
        wp.quat_identity(),
    )
    return True


def _replace_shape_with_sphere(builder_env, shape_idx: int) -> bool:
    """Replace a mesh collision shape with its bounding sphere."""
    vertices = _mesh_vertices(builder_env, shape_idx)
    if vertices is None or len(vertices) < 1:
        return False

    center = vertices.mean(axis=0)
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
    return True


def apply_body_collision_shape_overrides(
    builder_env,
    shape_start: int,
    overrides: Optional[Dict[str, str]],
) -> int:
    """Apply per-body collision shape type overrides to mesh shapes.

    Args:
        builder_env: The Newton ModelBuilder.
        shape_start: Shape index where this object's shapes begin (only shapes
            at ``>= shape_start`` are considered, keeping overrides scoped to
            the object currently being imported).
        overrides: Mapping of body-name patterns to override shape type.

    Returns:
        Number of shapes that were replaced.
    """
    if not overrides:
        return 0

    applied = 0
    # Snapshot the shape count: preserving visuals appends new shapes.
    shape_count = builder_env.shape_count
    for shape_idx in range(shape_start, shape_count):
        flags = int(builder_env.shape_flags[shape_idx])
        if not (flags & int(ShapeFlags.COLLIDE_SHAPES)):
            continue
        if int(builder_env.shape_type[shape_idx]) != int(GeoType.MESH):
            continue

        body_idx = int(builder_env.shape_body[shape_idx])
        if body_idx < 0 or body_idx >= builder_env.body_count:
            continue
        body_label = str(builder_env.body_label[body_idx])

        override_type = _match_override(body_label, overrides)
        if override_type is None or override_type not in _SUPPORTED_OVERRIDE_TYPES:
            continue

        # Keep the authored visual mesh visible while swapping the collider.
        flags = int(builder_env.shape_flags[shape_idx])
        if flags & int(ShapeFlags.VISIBLE):
            mesh = builder_env.shape_source[shape_idx]
            if mesh is not None:
                cfg = newton.ModelBuilder.ShapeConfig(
                    density=0.0,
                    margin=builder_env.shape_margin[shape_idx],
                    is_solid=builder_env.shape_is_solid[shape_idx],
                    has_shape_collision=False,
                    has_particle_collision=False,
                    is_visible=True,
                )
                builder_env.add_shape_mesh(
                    body=body_idx,
                    xform=builder_env.shape_transform[shape_idx],
                    cfg=cfg,
                    mesh=mesh,
                    color=builder_env.shape_color[shape_idx],
                    label=f"{builder_env.shape_label[shape_idx]}_visual",
                    scale=builder_env.shape_scale[shape_idx],
                )
                builder_env.shape_flags[shape_idx] &= ~int(ShapeFlags.VISIBLE)

        if override_type == "cylinder":
            replaced = _replace_shape_with_cylinder(builder_env, shape_idx)
        elif override_type == "box":
            replaced = _replace_shape_with_box(builder_env, shape_idx)
        else:
            replaced = _replace_shape_with_sphere(builder_env, shape_idx)
        if replaced:
            applied += 1

    return applied
