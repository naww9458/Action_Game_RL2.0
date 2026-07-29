"""USD anchor prim lookup and transform utilities for tool mounting."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import warp as wp

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:  # pragma: no cover - optional at import time
    Gf = None
    Usd = None
    UsdGeom = None


_USD_STAGE_CACHE: dict = {}


def _require_usd():
    if Usd is None:
        raise ImportError("pxr.Usd is required for tool anchor utilities")


def open_usd_stage(asset_path: str):
    """Open a USD stage, reusing a process-level cache keyed by path."""
    _require_usd()
    key = str(asset_path)
    cached = _USD_STAGE_CACHE.get(key)
    if cached is not None:
        return cached
    stage = Usd.Stage.Open(asset_path)
    if stage is None:
        raise FileNotFoundError(f"USD stage not found: {asset_path}")
    _USD_STAGE_CACHE[key] = stage
    return stage


def clear_usd_stage_cache() -> None:
    """Drop cached USD stages (e.g. after asset hot-reload)."""
    _USD_STAGE_CACHE.clear()


def find_anchor_prim(stage, anchor_name: str):
    """Find a prim by exact name anywhere in the stage."""
    _require_usd()
    for prim in stage.Traverse():
        if prim.GetName() == anchor_name:
            return prim
    raise KeyError(f"Anchor prim '{anchor_name}' not found in USD stage")


def find_body_prim_path(path_body_map: dict, suffix: str) -> str:
    """Resolve a body prim path from *path_body_map* using a suffix match."""
    matches = [path for path in path_body_map if path.rstrip("/").endswith(suffix)]
    if not matches:
        raise KeyError(
            f"No body prim ending with '{suffix}' in path_body_map keys: {list(path_body_map)[:8]}..."
        )
    if len(matches) > 1:
        matches.sort(key=len)
    return matches[0]


def _gf_to_wp_transform(m: "Gf.Matrix4d") -> wp.transform:
    translation = m.ExtractTranslation()
    rotation = m.ExtractRotationQuat()
    return wp.transform(
        wp.vec3(float(translation[0]), float(translation[1]), float(translation[2])),
        wp.quat(
            float(rotation.GetImaginary()[0]),
            float(rotation.GetImaginary()[1]),
            float(rotation.GetImaginary()[2]),
            float(rotation.GetReal()),
        ),
    )


def get_world_transform(prim) -> wp.transform:
    _require_usd()
    xformable = UsdGeom.Xformable(prim)
    world_mat = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return _gf_to_wp_transform(world_mat)


def get_anchor_local_to_body(anchor_prim, body_prim_path: str) -> wp.transform:
    """Return anchor transform expressed in the owning body frame."""
    _require_usd()
    stage = anchor_prim.GetStage()
    body_prim = stage.GetPrimAtPath(body_prim_path)
    if not body_prim or not body_prim.IsValid():
        raise KeyError(f"Body prim not found: {body_prim_path}")

    anchor_world = get_world_transform(anchor_prim)
    body_world = get_world_transform(body_prim)
    body_inv = wp.transform_inverse(body_world)
    return wp.transform_multiply(body_inv, anchor_world)


def resolve_anchor_pair(
    host_asset_path: str,
    tool_asset_path: str,
    host_anchor_name: str,
    tool_anchor_name: str,
    host_path_body_map: dict,
    tool_path_body_map: dict,
    host_body_prim_suffix: str,
    tool_base_body_prim_suffix: str,
) -> Tuple[str, str, wp.transform, wp.transform]:
    host_body_path = find_body_prim_path(host_path_body_map, host_body_prim_suffix)
    tool_body_path = find_body_prim_path(tool_path_body_map, tool_base_body_prim_suffix)

    host_stage = open_usd_stage(host_asset_path)
    tool_stage = open_usd_stage(tool_asset_path)

    host_anchor = find_anchor_prim(host_stage, host_anchor_name)
    tool_anchor = find_anchor_prim(tool_stage, tool_anchor_name)

    host_local = get_anchor_local_to_body(host_anchor, host_body_path)
    tool_local = get_anchor_local_to_body(tool_anchor, tool_body_path)
    return host_body_path, tool_body_path, host_local, tool_local


def _vec3_parts(v) -> Tuple[float, float, float]:
    if hasattr(v, "x"):
        return float(v.x), float(v.y), float(v.z)
    return float(v[0]), float(v[1]), float(v[2])


def _quat_parts(q) -> Tuple[float, float, float, float]:
    if hasattr(q, "w"):
        return float(q.x), float(q.y), float(q.z), float(q.w)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _transform_parts(xform) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    if hasattr(xform, "p") and hasattr(xform, "q"):
        return _vec3_parts(xform.p), _quat_parts(xform.q)
    if isinstance(xform, np.ndarray):
        flat = xform.reshape(-1)
        if flat.size >= 7:
            return (float(flat[0]), float(flat[1]), float(flat[2])), (
                float(flat[3]),
                float(flat[4]),
                float(flat[5]),
                float(flat[6]),
            )
    if isinstance(xform, (tuple, list)) and len(xform) == 2:
        return _vec3_parts(xform[0]), _quat_parts(xform[1])
    raise TypeError(f"Unsupported transform type: {type(xform)!r}")


def _quat_rotate(q: Tuple[float, float, float, float], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def _transform_multiply_np(
    parent: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
    child: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    p0, q0 = parent
    p1, q1 = child
    q0x, q0y, q0z, q0w = q0
    q1x, q1y, q1z, q1w = q1
    r1 = _quat_rotate(q0, p1)
    p = (p0[0] + r1[0], p0[1] + r1[1], p0[2] + r1[2])
    q = (
        q0w * q1x + q0x * q1w + q0y * q1z - q0z * q1y,
        q0w * q1y - q0x * q1z + q0y * q1w + q0z * q1x,
        q0w * q1z + q0x * q1y - q0y * q1x + q0z * q1w,
        q0w * q1w - q0x * q1x - q0y * q1y - q0z * q1z,
    )
    return p, q


def _transform_inverse_np(
    xform: Tuple[Tuple[float, float, float], Tuple[float, float, float, float]],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    p, q = xform
    qx, qy, qz, qw = q
    inv_q = (-qx, -qy, -qz, qw)
    inv_p = _quat_rotate(inv_q, (-p[0], -p[1], -p[2]))
    return inv_p, inv_q


def compute_anchor_world_position(
    body_transform,
    anchor_local: wp.transform,
) -> Tuple[float, float, float]:
    body = _transform_parts(body_transform)
    local_p, _ = _transform_parts(anchor_local)
    world_p = _transform_multiply_np(body, (local_p, (0.0, 0.0, 0.0, 1.0)))[0]
    return world_p


def resolve_possess_offset_above_anchor(
    asset_path: str,
    anchor_name: str,
    path_body_map: dict,
    body_prim_suffix: str,
    height_above: float,
) -> Tuple[float, float, float]:
    """
    Camera possess offset in root/chassis body frame: anchor local position + world-up height.
    """
    body_path = find_body_prim_path(path_body_map, body_prim_suffix)
    stage = open_usd_stage(asset_path)
    anchor_prim = find_anchor_prim(stage, anchor_name)
    anchor_local = get_anchor_local_to_body(anchor_prim, body_path)
    local_p, _ = _transform_parts(anchor_local)
    return (
        float(local_p[0]),
        float(local_p[1]),
        float(local_p[2]) + float(height_above),
    )


def anchor_distance(body_a, local_a: wp.transform, body_b, local_b: wp.transform) -> float:
    pos_a = compute_anchor_world_position(body_a, local_a)
    pos_b = compute_anchor_world_position(body_b, local_b)
    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]
    dz = pos_b[2] - pos_a[2]
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def anchor_horizontal_distance(body_a, local_a: wp.transform, body_b, local_b: wp.transform) -> float:
    pos_a = compute_anchor_world_position(body_a, local_a)
    pos_b = compute_anchor_world_position(body_b, local_b)
    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]
    return float(math.sqrt(dx * dx + dy * dy))


def anchor_vertical_separation(body_a, local_a: wp.transform, body_b, local_b: wp.transform) -> float:
    pos_a = compute_anchor_world_position(body_a, local_a)
    pos_b = compute_anchor_world_position(body_b, local_b)
    return abs(float(pos_b[2] - pos_a[2]))


def anchor_within_mount_proximity(
    body_a,
    local_a: wp.transform,
    body_b,
    local_b: wp.transform,
    horizontal_threshold: float,
    vertical_threshold: float,
) -> bool:
    """True when mount anchors are close in XY and within vertical height tolerance."""
    horizontal = anchor_horizontal_distance(body_a, local_a, body_b, local_b)
    vertical = anchor_vertical_separation(body_a, local_a, body_b, local_b)
    return horizontal <= float(horizontal_threshold) and vertical <= float(vertical_threshold)


def compose_body_snap_transform(host_body, host_anchor_local: wp.transform, tool_anchor_local: wp.transform):
    """Return flat body_q row [px, py, pz, qx, qy, qz, qw] for snapping tool base to host mount."""
    return compose_mounted_tool_transform(
        host_body,
        host_anchor_local,
        tool_anchor_local,
        yaw_rad=0.0,
    )


def compute_weld_relpose_from_anchors(
    host_anchor_local: wp.transform,
    tool_anchor_local: wp.transform,
) -> wp.transform:
    """Build Newton/MuJoCo WELD ``relpose`` for coinciding mount anchors.

    MuJoCo ``mjEQ_WELD`` stores:
    - ``eq_data[0:3]`` (Newton ``anchor``): weld point in **body2** frame
    - ``eq_data[3:6]`` (Newton ``relpose`` translation): weld point in **body1** frame
    - ``eq_data[6:10]`` (Newton ``relpose`` rotation): body2 orientation relative to body1

    So translation must be the host-anchor position, **not**
    ``host_anchor * inv(tool_anchor)`` (that double-counts the tool offset).
    """
    return compose_mounted_weld_relpose(
        host_anchor_local,
        tool_anchor_local,
        yaw_rad=0.0,
    )


def compose_mounted_weld_relpose(
    host_anchor_local: wp.transform,
    tool_anchor_local: wp.transform,
    yaw_rad: float,
    mount_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> wp.transform:
    """MuJoCo WELD relpose: body1 anchor translation + relative orientation (optional yaw)."""
    host_anchor = _transform_parts(host_anchor_local)
    tool_anchor = _transform_parts(tool_anchor_local)
    host_p, host_q = host_anchor

    # Orientation when body1*host_anchor*R_yaw and body2*tool_anchor frames coincide:
    # body2_q = body1_q * host_q * yaw_q * inv(tool_q)
    if abs(float(yaw_rad)) > 1.0e-8:
        yaw_q = _quat_from_axis_angle(_normalize3(mount_axis), float(yaw_rad))
        orient_parent = (host_p, _quat_multiply(host_q, yaw_q))
    else:
        orient_parent = host_anchor

    _, rel_q = _transform_multiply_np(orient_parent, _transform_inverse_np(tool_anchor))
    return wp.transform(
        wp.vec3(host_p[0], host_p[1], host_p[2]),
        wp.quat(rel_q[0], rel_q[1], rel_q[2], rel_q[3]),
    )


def _quat_multiply(
    q0: Tuple[float, float, float, float],
    q1: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    q0x, q0y, q0z, q0w = q0
    q1x, q1y, q1z, q1w = q1
    return (
        q0w * q1x + q0x * q1w + q0y * q1z - q0z * q1y,
        q0w * q1y - q0x * q1z + q0y * q1w + q0z * q1x,
        q0w * q1z + q0x * q1y - q0y * q1x + q0z * q1w,
        q0w * q1w - q0x * q1x - q0y * q1y - q0z * q1z,
    )


def _normalize3(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length <= 1.0e-8:
        return 0.0, 0.0, 1.0
    return v[0] / length, v[1] / length, v[2] / length


def _quat_from_axis_angle(axis: Tuple[float, float, float], angle: float) -> Tuple[float, float, float, float]:
    ax, ay, az = _normalize3(axis)
    half = angle * 0.5
    s = math.sin(half)
    return ax * s, ay * s, az * s, math.cos(half)


def compose_mounted_tool_transform(
    host_body,
    host_anchor_local: wp.transform,
    tool_anchor_local: wp.transform,
    yaw_rad: float,
    mount_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
):
    """Snap tool base to host mount, then apply yaw rotation about *mount_axis* in mount frame."""
    host = _transform_parts(host_body)
    host_anchor = _transform_parts(host_anchor_local)
    tool_anchor = _transform_parts(tool_anchor_local)
    mount_world = _transform_multiply_np(host, host_anchor)
    axis_local = _normalize3(mount_axis)
    axis_world = _quat_rotate(mount_world[1], axis_local)
    yaw_transform = ((0.0, 0.0, 0.0), _quat_from_axis_angle(axis_world, yaw_rad))
    rotated_mount = _transform_multiply_np(mount_world, yaw_transform)
    desired = _transform_multiply_np(rotated_mount, _transform_inverse_np(tool_anchor))
    p, q = desired
    return np.array([p[0], p[1], p[2], q[0], q[1], q[2], q[3]], dtype=np.float32)
