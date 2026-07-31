"""turret_110mm exclusive third-person human aim overlay (screen-space geometry).

Pure geometry only: the overlay is built from explicit numbers and drawn with
pyglet shapes. Deciding *when* it should appear (which host, which limits)
lives in the turret's aim action (``functions/aim.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Turret110mmAimViewStyle:
    circle_radius_px: float = 14.0
    circle_segments: int = 24
    circle_color: Tuple[int, int, int, int] = (255, 0, 0, 220)
    limit_line_color: Tuple[int, int, int, int] = (255, 0, 0, 200)
    limit_line_half_width_px: float = 90.0
    line_width: float = 2.0

    @classmethod
    def from_mapping(cls, raw: dict | None) -> "Turret110mmAimViewStyle":
        base = cls()
        if not raw:
            return base

        def _rgba(value, default: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
            if value is None:
                return default
            vals = [int(v) for v in value]
            if len(vals) == 3:
                return (vals[0], vals[1], vals[2], default[3])
            if len(vals) >= 4:
                return (vals[0], vals[1], vals[2], vals[3])
            return default

        return cls(
            circle_radius_px=float(raw.get("circle_radius_px", base.circle_radius_px)),
            circle_segments=int(raw.get("circle_segments", base.circle_segments)),
            circle_color=_rgba(raw.get("circle_color"), base.circle_color),
            limit_line_color=_rgba(raw.get("limit_line_color"), base.limit_line_color),
            limit_line_half_width_px=float(
                raw.get("limit_line_half_width_px", base.limit_line_half_width_px)
            ),
            line_width=float(raw.get("line_width", base.line_width)),
        )


@dataclass(frozen=True)
class ScreenSegment:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class Turret110mmAimViewOverlay:
    """Screen-space geometry for CustomViewerGL UI overlay."""

    center_x: float
    center_y: float
    circle_radius_px: float
    circle_segments: int
    pitch_limit_lines: Tuple[ScreenSegment, ...]
    circle_color: Tuple[int, int, int, int]
    limit_line_color: Tuple[int, int, int, int]
    line_width: float


def joint_pitch_limits_to_camera_pitch_deg(
    limit_lower_rad: float,
    limit_upper_rad: float,
    pitch_joint_sign: float,
) -> Tuple[float, float]:
    """
    Map joint-q limits to camera pitch (deg) that aligns barrel on a level host.

    With pitch_joint_sign=-1 (+q = depression): ±10° joint → ±10° camera pitch.
    """
    sign = float(pitch_joint_sign)
    if abs(sign) < 1e-8:
        sign = -1.0
    cam_a = math.degrees(float(limit_lower_rad) / sign)
    cam_b = math.degrees(float(limit_upper_rad) / sign)
    return (min(cam_a, cam_b), max(cam_a, cam_b))


def pitch_delta_to_screen_offset_y(
    delta_pitch_deg: float,
    window_h: float,
    fov_deg: float,
) -> float:
    """Perspective map: camera-relative pitch delta → screen Y offset from center."""
    half_fov = max(1e-3, abs(float(fov_deg)) * 0.5)
    half_h = float(window_h) * 0.5
    return half_h * math.tan(math.radians(float(delta_pitch_deg))) / math.tan(
        math.radians(half_fov)
    )


def build_turret_110mm_third_person_aim_view(
    *,
    window_w: float,
    window_h: float,
    camera_pitch_deg: float,
    camera_fov_deg: float,
    pitch_limit_cam_deg: Tuple[float, float],
    style: Turret110mmAimViewStyle | None = None,
) -> Turret110mmAimViewOverlay:
    """
    Build overlay where each pitch-limit line sits at screen center when
    camera_pitch equals that limit (pitch-only; no yaw limit lines).
    """
    style = style or Turret110mmAimViewStyle()
    cx = float(window_w) * 0.5
    cy = float(window_h) * 0.5
    half_w = float(style.limit_line_half_width_px)

    limit_lines: list[ScreenSegment] = []
    for limit_deg in pitch_limit_cam_deg:
        dy = pitch_delta_to_screen_offset_y(
            float(limit_deg) - float(camera_pitch_deg),
            window_h,
            camera_fov_deg,
        )
        y = cy + dy
        limit_lines.append(ScreenSegment(cx - half_w, y, cx + half_w, y))

    return Turret110mmAimViewOverlay(
        center_x=cx,
        center_y=cy,
        circle_radius_px=float(style.circle_radius_px),
        circle_segments=int(style.circle_segments),
        pitch_limit_lines=tuple(limit_lines),
        circle_color=style.circle_color,
        limit_line_color=style.limit_line_color,
        line_width=style.line_width,
    )


def draw_turret_110mm_aim_view_overlay(
    overlay: Turret110mmAimViewOverlay,
    *,
    shapes_module,
    batch=None,
) -> tuple:
    """
    Build pyglet shape primitives for the overlay.

    Returns ``(batch, shapes)``; the caller draws ``batch`` under screen-space
    ortho and keeps ``shapes`` alive until the frame is drawn.
    """
    draw_batch = batch if batch is not None else shapes_module.Batch()
    keep: list = []
    thickness = float(max(1.0, overlay.line_width))

    for seg in overlay.pitch_limit_lines:
        keep.append(
            shapes_module.Line(
                seg.x0,
                seg.y0,
                seg.x1,
                seg.y1,
                thickness=thickness,
                color=overlay.limit_line_color,
                batch=draw_batch,
            )
        )

    keep.append(
        shapes_module.Arc(
            float(overlay.center_x),
            float(overlay.center_y),
            max(1.0, float(overlay.circle_radius_px)),
            segments=max(3, int(overlay.circle_segments)),
            thickness=thickness,
            color=overlay.circle_color,
            batch=draw_batch,
        )
    )

    return draw_batch, keep
