"""Per-body collision shape overrides for articulation imports.

Consumed from the object config key ``body_collision_shape_overrides``.
Body names are matched case-insensitively against the body label's basename
(and full path) using fnmatch-style wildcards (``*`` / ``?``). Only shapes
added by the object currently being imported are modified, so one object's
override never leaks onto another object sharing the same builder.

Override values can be either:

* a plain type name (``"cylinder"`` / ``"box"`` / ``"sphere"`` / ``"capsule"``
  / ``"disable"``): the shape is re-derived from the mesh geometry
  (PCA / AABB / bounding sphere / fitted capsule);
* an explicit spec dict for exact (e.g. mjlab-style) shapes::

      {"left_thigh": {"type": "capsule", "radius": 0.055,
                      "fromto": [[0.0, 0.0, -0.03], [-0.06, 0.0, -0.17]]}}
      {"torso":     {"type": "capsule", "radius": 0.09,
                      "fromto": [[0.01, 0.0, 0.08], [0.01, 0.0, 0.2]],
                      "also_add": [{"type": "sphere", "radius": 0.06,
                                    "pos": [0.0, 0.0, 0.43]}]}}
      {"hand":      {"type": "disable"}}
      {"pelvis":    {"add": true, "type": "sphere", "radius": 0.07,
                     "pos": [0.0, 0.0, -0.08]}}
* a list of specs on one body (disable leftover meshes, then add several
  primitives — e.g. mjlab's 7 foot capsules).

``fromto`` and ``pos`` are expressed in the body's local frame (matching MuJoCo
geom semantics). ``disable`` turns the body's collision shapes off entirely
(clears the ``COLLIDE_SHAPES`` flag), which is useful for dropping limbs that
should never self-collide (e.g. finger links) so they never enter the
broadphase / CCD path.

``add: true`` inserts a brand-new primitive collider on the matched body
instead of replacing an existing mesh. ``also_add`` on a replace spec does
the same for extra primitives on that body (e.g. head sphere on torso).
Added shapes carry no mass (``density=0``) and are not visible, so dynamics
stay governed by the asset's inertials.
"""

from __future__ import annotations

import fnmatch
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import warp as wp

import newton
from newton import GeoType, ShapeFlags

_SUPPORTED_OVERRIDE_TYPES = frozenset({"cylinder", "box", "sphere", "capsule", "disable"})


def _body_basename(body_label: str) -> str:
    return str(body_label).rstrip("/").split("/")[-1]


def _parse_override(value: Any) -> Optional[Dict[str, Any]]:
    """Normalize an override value into a spec dict (or None if unsupported)."""
    if isinstance(value, str):
        t = value.lower()
        if t in _SUPPORTED_OVERRIDE_TYPES:
            return {"type": t}
        return None
    if isinstance(value, dict):
        t = str(value.get("type", "")).lower()
        if t not in _SUPPORTED_OVERRIDE_TYPES:
            return None
        spec = dict(value)
        spec["type"] = t
        return spec
    return None


def _body_matches(body_label: str, pattern: str) -> bool:
    """fnmatch-style case-insensitive match against a body label (basename or path)."""
    label = str(body_label)
    basename = _body_basename(label)
    pat = str(pattern)
    return fnmatch.fnmatchcase(basename.lower(), pat.lower()) or fnmatch.fnmatchcase(
        label.lower(), pat.lower()
    )


def _iter_override_specs(raw: Any) -> List[Dict[str, Any]]:
    """Normalize a YAML override value into a list of spec dicts."""
    items = raw if isinstance(raw, list) else [raw]
    specs: List[Dict[str, Any]] = []
    for item in items:
        spec = _parse_override(item)
        if spec is not None:
            specs.append(spec)
    return specs


def _specs_for_body(body_label: str, overrides: Dict[str, Any]) -> List[Dict[str, Any]]:
    for pattern, raw in overrides.items():
        if _body_matches(body_label, pattern):
            return _iter_override_specs(raw)
    return []


def _additive_specs_from(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect ``add: true`` and ``also_add`` primitives from one spec."""
    out: List[Dict[str, Any]] = []
    if spec.get("add"):
        out.append(spec)
    for extra in spec.get("also_add") or []:
        parsed = _parse_override(extra)
        if parsed is None:
            continue
        added = dict(parsed)
        added["add"] = True
        out.append(added)
    return out


def _match_override(body_label: str, overrides: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    specs = _specs_for_body(body_label, overrides)
    if not specs:
        return None
    for spec in specs:
        if not spec.get("add") and spec["type"] != "disable":
            return spec
    return specs[0]


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


def _set_primitive_shape(
    builder_env,
    shape_idx: int,
    geom_type: int,
    scale: Sequence[float],
    center: Sequence[float],
    quat: wp.quat,
    compose_existing: bool = True,
) -> bool:
    """Rewrite a shape slot to a primitive geometry in the body-local frame.

    Fitted overrides (mesh PCA / AABB) keep ``compose_existing=True`` because
    vertex centres live in the leftover mesh frame. Explicit mjlab-style
    ``pos`` / ``fromto`` are already body-local, so they replace the transform.
    """
    new_tf = wp.transform(
        wp.vec3(float(center[0]), float(center[1]), float(center[2])), quat
    )
    if compose_existing:
        new_tf = builder_env.shape_transform[shape_idx] * new_tf
    builder_env.shape_type[shape_idx] = int(geom_type)
    builder_env.shape_source[shape_idx] = None
    builder_env.shape_scale[shape_idx] = wp.vec3(
        float(scale[0]), float(scale[1]), float(scale[2])
    )
    builder_env.shape_transform[shape_idx] = new_tf
    return True


def _fitted_cylinder(builder_env, shape_idx: int) -> bool:
    """Fit a cylinder from a PCA OBB of the mesh (smallest axis = axle)."""
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

    return _set_primitive_shape(
        builder_env,
        shape_idx,
        GeoType.CYLINDER,
        (radius, half_height, 0.0),
        center,
        _quat_align_z(axle),
    )


def _fitted_box(builder_env, shape_idx: int) -> bool:
    """Fit the axis-aligned bounding box of the mesh."""
    vertices = _mesh_vertices(builder_env, shape_idx)
    if vertices is None or len(vertices) < 1:
        return False

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) * 0.5
    half = np.maximum((vmax - vmin) * 0.5, 1e-4)

    return _set_primitive_shape(
        builder_env,
        shape_idx,
        GeoType.BOX,
        half,
        center,
        wp.quat_identity(),
    )


def _fitted_sphere(builder_env, shape_idx: int) -> bool:
    """Fit the bounding sphere of the mesh."""
    vertices = _mesh_vertices(builder_env, shape_idx)
    if vertices is None or len(vertices) < 1:
        return False

    center = vertices.mean(axis=0)
    radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
    radius = max(radius, 1e-4)

    return _set_primitive_shape(
        builder_env,
        shape_idx,
        GeoType.SPHERE,
        (radius, 0.0, 0.0),
        center,
        wp.quat_identity(),
    )


def _explicit_capsule(builder_env, shape_idx: int, spec: Dict[str, Any]) -> bool:
    """Build a capsule from an explicit mjlab-style spec.

    Supports either ``fromto=[[ax,ay,az],[bx,by,bz]]`` or an explicit
    ``axis=[dx,dy,dz]`` + ``half_length`` pair, plus ``radius``.
    """
    radius = float(spec.get("radius", 0.0))
    if radius <= 0.0:
        return False

    fromto = spec.get("fromto")
    if fromto is not None:
        if len(fromto) != 2:
            return False
        a = np.asarray(fromto[0], dtype=np.float64)
        b = np.asarray(fromto[1], dtype=np.float64)
        axis = b - a
        center = (a + b) * 0.5
        half_length = float(np.linalg.norm(axis)) * 0.5
    else:
        axis = np.asarray(spec.get("axis", [0.0, 0.0, 1.0]), dtype=np.float64)
        center = np.asarray(spec.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        half_length = float(spec.get("half_length", 0.0))

    if half_length <= 0.0:
        return False
    radius = max(radius, 1e-4)
    half_length = max(half_length, 1e-4)

    return _set_primitive_shape(
        builder_env,
        shape_idx,
        GeoType.CAPSULE,
        (radius, half_length, 0.0),
        center,
        _quat_align_z(axis),
        compose_existing=False,
    )


def _explicit_sphere(builder_env, shape_idx: int, spec: Dict[str, Any]) -> bool:
    radius = float(spec.get("radius", 0.0))
    if radius <= 0.0:
        return False
    pos = np.asarray(spec.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    return _set_primitive_shape(
        builder_env,
        shape_idx,
        GeoType.SPHERE,
        (max(radius, 1e-4), 0.0, 0.0),
        pos,
        wp.quat_identity(),
        compose_existing=False,
    )


def _disable_shape(builder_env, shape_idx: int) -> bool:
    """Turn a shape's collision off without removing it."""
    builder_env.shape_flags[shape_idx] = int(
        builder_env.shape_flags[shape_idx]
    ) & ~int(ShapeFlags.COLLIDE_SHAPES)
    return True


def _is_replaceable_geometry(builder_env, shape_idx: int) -> bool:
    gtype = int(builder_env.shape_type[shape_idx])
    return gtype in (int(GeoType.MESH), int(GeoType.CONVEX_MESH))


def _additive_shape_cfg() -> "newton.ModelBuilder.ShapeConfig":
    """Collider-only config for additive primitives (massless, invisible)."""
    return newton.ModelBuilder.ShapeConfig(
        density=0.0,
        margin=0.0,
        is_solid=True,
        has_shape_collision=True,
        has_particle_collision=False,
        is_visible=False,
    )


def _apply_additive_spec(builder_env, body_idx: int, spec: Dict[str, Any]) -> bool:
    """Insert a brand-new primitive collider on ``body_idx`` from an ``add`` spec."""
    override_type = spec["type"]
    cfg = _additive_shape_cfg()

    if override_type == "sphere":
        radius = float(spec.get("radius", 0.0))
        if radius <= 0.0:
            return False
        pos = np.asarray(spec.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        builder_env.add_shape_sphere(
            body=body_idx,
            xform=wp.transform(
                wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                wp.quat_identity(),
            ),
            radius=max(radius, 1e-4),
            cfg=cfg,
        )
        return True

    if override_type == "capsule":
        radius = float(spec.get("radius", 0.0))
        if radius <= 0.0:
            return False
        fromto = spec.get("fromto")
        if fromto is not None:
            if len(fromto) != 2:
                return False
            a = np.asarray(fromto[0], dtype=np.float64)
            b = np.asarray(fromto[1], dtype=np.float64)
            axis = b - a
            center = (a + b) * 0.5
            half_length = float(np.linalg.norm(axis)) * 0.5
        else:
            axis = np.asarray(spec.get("axis", [0.0, 0.0, 1.0]), dtype=np.float64)
            center = np.asarray(spec.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
            half_length = float(spec.get("half_length", 0.0))
        if half_length <= 0.0:
            return False
        builder_env.add_shape_capsule(
            body=body_idx,
            xform=wp.transform(
                wp.vec3(float(center[0]), float(center[1]), float(center[2])),
                _quat_align_z(axis),
            ),
            radius=max(radius, 1e-4),
            half_height=max(half_length, 1e-4),
            cfg=cfg,
        )
        return True

    return False


def _apply_spec(
    builder_env, shape_idx: int, body_idx: int, spec: Dict[str, Any]
) -> bool:
    override_type = spec["type"]

    if override_type == "disable":
        return _disable_shape(builder_env, shape_idx)

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
        replaced = _fitted_cylinder(builder_env, shape_idx)
    elif override_type == "box":
        replaced = _fitted_box(builder_env, shape_idx)
    elif override_type == "sphere":
        if "radius" in spec:
            replaced = _explicit_sphere(builder_env, shape_idx, spec)
        else:
            replaced = _fitted_sphere(builder_env, shape_idx)
    elif override_type == "capsule":
        if "radius" in spec and ("fromto" in spec or "half_length" in spec):
            replaced = _explicit_capsule(builder_env, shape_idx, spec)
        else:
            # Fit a capsule from the mesh's PCA OBB along the longest axis.
            replaced = _fitted_capsule(builder_env, shape_idx)
    else:
        return False
    return replaced


def _fitted_capsule(builder_env, shape_idx: int) -> bool:
    """Fit a capsule from a PCA OBB of the mesh (longest axis = capsule axis)."""
    vertices = _mesh_vertices(builder_env, shape_idx)
    if vertices is None or len(vertices) < 3:
        return False

    center = vertices.mean(axis=0)
    cov = np.cov((vertices - center).T)
    evals, evecs = np.linalg.eigh(cov)
    proj = (vertices - center) @ evecs
    extents = proj.max(axis=0) - proj.min(axis=0)

    axis_idx = int(np.argmax(extents))
    axis = evecs[:, axis_idx]
    radius = float(max(extents[i] for i in range(3) if i != axis_idx)) * 0.5
    half_length = float(extents[axis_idx]) * 0.5
    radius = max(radius, 1e-4)
    half_length = max(half_length, 1e-4)

    return _set_primitive_shape(
        builder_env,
        shape_idx,
        GeoType.CAPSULE,
        (radius, half_length, 0.0),
        center,
        _quat_align_z(axis),
    )


def apply_body_collision_shape_overrides(
    builder_env,
    shape_start: int,
    overrides: Optional[Dict[str, Any]],
) -> int:
    """Apply per-body collision shape overrides to (convex) mesh shapes.

    Args:
        builder_env: The Newton ModelBuilder.
        shape_start: Shape index where this object's shapes begin (only shapes
            at ``>= shape_start`` are considered, keeping overrides scoped to
            the object currently being imported).
        overrides: Mapping of body-name patterns to a shape type name or an
            explicit spec dict (see module docstring).

    Returns:
        Number of shapes that were replaced or disabled.
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

        body_idx = int(builder_env.shape_body[shape_idx])
        if body_idx < 0 or body_idx >= builder_env.body_count:
            continue
        body_label = str(builder_env.body_label[body_idx])

        specs = _specs_for_body(body_label, overrides)
        if not specs:
            continue
        disable = any(
            (not s.get("add")) and s["type"] == "disable" for s in specs
        )
        if disable:
            if _disable_shape(builder_env, shape_idx):
                applied += 1
            continue
        replace_spec = next(
            (s for s in specs if not s.get("add") and s["type"] != "disable"),
            None,
        )
        if replace_spec is None:
            continue
        # Replacement requires a (convex) mesh source to swap out.
        if not _is_replaceable_geometry(builder_env, shape_idx):
            continue
        if _apply_spec(builder_env, shape_idx, body_idx, replace_spec):
            applied += 1

    # Additive pass: insert brand-new primitive colliders on matched bodies
    # (``add: true`` and ``also_add``). Scoped to bodies, so it works even
    # when the body has no COLLIDE_SHAPES shape to replace.
    for pattern, raw in overrides.items():
        for spec in _iter_override_specs(raw):
            for add_spec in _additive_specs_from(spec):
                for body_idx in range(builder_env.body_count):
                    body_label = str(builder_env.body_label[body_idx])
                    if not _body_matches(body_label, pattern):
                        continue
                    if _apply_additive_spec(builder_env, body_idx, add_spec):
                        applied += 1

    return applied


def apply_body_collision_exclude_pairs(
    builder_env,
    shape_start: int,
    exclude_pairs: Optional[Sequence[Sequence[str]]],
) -> int:
    """Add shape-level collision filter pairs for excluded body pairs.

    Mirrors mjlab's ``<contact><exclude>`` semantics: the given body pairs
    never collide with each other even though their (usually adjacent) capsule
    colliders overlap at nominal joint positions (e.g. elbow-wrist,
    pelvis-hip_roll). Body names are matched case-insensitively against the
    body label's basename (and full path) with fnmatch wildcards, scoped to the
    shapes this object imported (``>= shape_start``).

    Returns:
        Number of shape filter pairs added.
    """
    if not exclude_pairs:
        return 0

    # Body basename -> colliding shape indices belonging to this object.
    body_shapes: Dict[str, List[int]] = {}
    shape_count = builder_env.shape_count
    for shape_idx in range(shape_start, shape_count):
        flags = int(builder_env.shape_flags[shape_idx])
        if not (flags & int(ShapeFlags.COLLIDE_SHAPES)):
            continue
        body_idx = int(builder_env.shape_body[shape_idx])
        if body_idx < 0 or body_idx >= builder_env.body_count:
            continue
        body_shapes.setdefault(
            _body_basename(str(builder_env.body_label[body_idx])).lower(), []
        ).append(shape_idx)

    def _expand(name: str) -> List[int]:
        shapes: List[int] = []
        for basename, idxs in body_shapes.items():
            if fnmatch.fnmatchcase(basename, name.lower()):
                shapes.extend(idxs)
        return shapes

    added = 0
    for pair in exclude_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        shapes_a = _expand(str(pair[0]))
        shapes_b = _expand(str(pair[1]))
        if not shapes_a or not shapes_b:
            continue
        for sa in shapes_a:
            for sb in shapes_b:
                if sa == sb:
                    continue
                builder_env.add_shape_collision_filter_pair(sa, sb)
                added += 1
    return added
