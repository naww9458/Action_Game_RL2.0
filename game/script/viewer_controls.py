from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pyglet.window import key

_CFG_PATH = Path(__file__).with_name("viewer_controls_cfg.yaml")

_KEY_DISPLAY = {
    "w": "W",
    "a": "A",
    "s": "S",
    "d": "D",
    "r": "R",
    "b": "B",
    "h": "H",
    "i": "I",
    "p": "P",
    "y": "Y",
    "q": "Q",
    "space": "Space",
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
    "tab": "Tab",
    "return": "Enter",
    "enter": "Enter",
    "backspace": "Backspace",
    "esc": "Esc",
    "shift": "Shift",
    "ctrl": "Ctrl",
    "lalt": "Left Alt",
    "ralt": "Right Alt",
    "left_mouse_click": "Left Click",
    "left_mouse_drag": "Left Drag",
    "right_mouse_drag": "Right Drag",
    "middle_mouse_drag": "Middle Drag",
    "scroll": "Scroll",
}


def create_keyboard_mapping() -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for attr in dir(key):
        if len(attr) == 2 and attr.startswith("_") and attr[1].isdigit():
            mapping[attr[1]] = getattr(key, attr)
            continue
        if attr.startswith("_"):
            continue
        value = getattr(key, attr)
        if len(attr) == 1:
            mapping[attr.lower()] = value
        elif attr.startswith("NUMBER_"):
            mapping[attr.split("_")[1]] = value
        else:
            mapping[attr.lower()] = value
            if attr == "ESCAPE":
                mapping["esc"] = value
            if attr == "ENTER":
                mapping["return"] = value
            if attr == "LSHIFT":
                mapping["shift"] = value
            if attr == "LCTRL":
                mapping["ctrl"] = value
            if attr == "LALT":
                mapping["lalt"] = value
            if attr == "RALT":
                mapping["ralt"] = value
    return mapping


_KEYBOARD_MAPPING = create_keyboard_mapping()


def resolve_keys(key_names: List[str]) -> List[int]:
    symbols: List[int] = []
    for name in key_names:
        normalized = name.lower().strip()
        if normalized.endswith("+scroll"):
            continue
        symbol = _KEYBOARD_MAPPING.get(normalized)
        if symbol is not None:
            symbols.append(symbol)
    return symbols


def format_keys(key_names: List[str]) -> str:
    parts: List[str] = []
    for name in key_names:
        normalized = name.lower().strip()
        if "+" in normalized:
            segments = [_KEY_DISPLAY.get(seg, seg.upper()) for seg in normalized.split("+")]
            parts.append(" + ".join(segments))
        else:
            parts.append(_KEY_DISPLAY.get(normalized, normalized.upper()))
    return ", ".join(parts)


@dataclass
class ControlBinding:
    category: str
    id: str
    keys: List[str]
    description: str
    context: Optional[str] = None
    symbols: List[int] = field(default_factory=list)


@dataclass
class DebugGeometrySpinConfig:
    min: float = 0.01
    max: float = 50.0
    step: float = 0.1
    decimals: int = 3


@dataclass
class DebugGeometryConfig:
    line_length: float = 1.0
    line_width: float = 0.03
    surface_offset: float = 0.5
    circle_radius: float = 0.01
    circle_lift: float = 0.002
    num_segments: int = 16
    default_forward_local: tuple[float, float, float] = (1.0, 0.0, 0.0)
    shoot_forward_shape_key_prefixes: List[str] = field(default_factory=lambda: ["rigid_"])
    length_spin: DebugGeometrySpinConfig = field(default_factory=DebugGeometrySpinConfig)
    width_spin: DebugGeometrySpinConfig = field(
        default_factory=lambda: DebugGeometrySpinConfig(
            min=0.001, max=1.0, step=0.01, decimals=4
        )
    )
    extension_line_color: tuple[float, float, float] = (0.0, 1.0, 0.0)
    circle_line_color: tuple[float, float, float] = (1.0, 0.0, 0.0)


@dataclass
class ViewerControlsConfig:
    defaults: Dict[str, Any]
    category_labels: Dict[str, str]
    bindings: List[ControlBinding]
    debug_geometry: DebugGeometryConfig = field(default_factory=DebugGeometryConfig)

    def binding_by_id(self, binding_id: str) -> Optional[ControlBinding]:
        for binding in self.bindings:
            if binding.id == binding_id:
                return binding
        return None

    def bindings_by_category(self) -> Dict[str, List[ControlBinding]]:
        grouped: Dict[str, List[ControlBinding]] = {}
        for binding in self.bindings:
            grouped.setdefault(binding.category, []).append(binding)
        return grouped

    def category_title(self, category: str) -> str:
        return self.category_labels.get(category, category.replace("_", " ").title())


@dataclass
class SimulationControl:
    paused: bool = False
    auto_reset_on_env_end: bool = False
    manual_reset_enabled: bool = True
    _reset_requested: bool = field(default=False, repr=False)

    @classmethod
    def from_defaults(cls, cfg: ViewerControlsConfig) -> "SimulationControl":
        defaults = cfg.defaults
        return cls(
            paused=False,
            auto_reset_on_env_end=bool(defaults.get("auto_reset_on_env_end", False)),
            manual_reset_enabled=bool(defaults.get("manual_reset_enabled", True)),
        )

    def request_reset(self):
        self._reset_requested = True

    def consume_reset_request(self) -> bool:
        if not self._reset_requested:
            return False
        self._reset_requested = False
        return True

    def to_queue_payload(self) -> Dict[str, Any]:
        return {
            "paused": self.paused,
            "auto_reset_on_env_end": self.auto_reset_on_env_end,
            "manual_reset_enabled": self.manual_reset_enabled,
            "reset_requested": self.consume_reset_request(),
        }


def _parse_forward_local(raw: Any) -> tuple[float, float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    return (1.0, 0.0, 0.0)


def _parse_rgb(raw: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    return default


def _parse_spin_config(raw: Any, default: DebugGeometrySpinConfig) -> DebugGeometrySpinConfig:
    if not isinstance(raw, dict):
        return default
    return DebugGeometrySpinConfig(
        min=float(raw.get("min", default.min)),
        max=float(raw.get("max", default.max)),
        step=float(raw.get("step", default.step)),
        decimals=int(raw.get("decimals", default.decimals)),
    )


def _parse_debug_geometry_config(raw: Any) -> DebugGeometryConfig:
    default = DebugGeometryConfig()
    if not isinstance(raw, dict):
        return default
    prefixes = raw.get("shoot_forward_shape_key_prefixes", default.shoot_forward_shape_key_prefixes)
    if not isinstance(prefixes, list):
        prefixes = list(default.shoot_forward_shape_key_prefixes)
    return DebugGeometryConfig(
        line_length=float(raw.get("line_length", default.line_length)),
        line_width=float(raw.get("line_width", default.line_width)),
        surface_offset=float(raw.get("surface_offset", default.surface_offset)),
        circle_radius=float(raw.get("circle_radius", default.circle_radius)),
        circle_lift=float(raw.get("circle_lift", default.circle_lift)),
        num_segments=int(raw.get("num_segments", default.num_segments)),
        default_forward_local=_parse_forward_local(
            raw.get("default_forward_local", default.default_forward_local)
        ),
        shoot_forward_shape_key_prefixes=[str(p) for p in prefixes],
        length_spin=_parse_spin_config(raw.get("length_spin"), default.length_spin),
        width_spin=_parse_spin_config(raw.get("width_spin"), default.width_spin),
        extension_line_color=_parse_rgb(
            raw.get("extension_line_color"), default.extension_line_color
        ),
        circle_line_color=_parse_rgb(raw.get("circle_line_color"), default.circle_line_color),
    )


def load_viewer_controls(path: Optional[Path] = None) -> ViewerControlsConfig:
    cfg_path = path or _CFG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    bindings: List[ControlBinding] = []
    for item in raw.get("bindings", []):
        key_names = [str(k) for k in item.get("keys", [])]
        bindings.append(
            ControlBinding(
                category=str(item.get("category", "other")),
                id=str(item["id"]),
                keys=key_names,
                description=str(item.get("description", "")),
                context=item.get("context"),
                symbols=resolve_keys(key_names),
            )
        )

    return ViewerControlsConfig(
        defaults=dict(raw.get("defaults", {})),
        category_labels={str(k): str(v) for k, v in raw.get("category_labels", {}).items()},
        bindings=bindings,
        debug_geometry=_parse_debug_geometry_config(raw.get("debug_geometry")),
    )
