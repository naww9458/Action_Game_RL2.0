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
class ViewerControlsConfig:
    defaults: Dict[str, Any]
    category_labels: Dict[str, str]
    bindings: List[ControlBinding]

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
    )
