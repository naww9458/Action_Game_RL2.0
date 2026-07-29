"""Project-root UI strings from {language}.yaml (see app_settings.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LANG_CODE: Optional[str] = None
_LANG_DATA: Optional[Dict[str, str]] = None


def _load_language_code() -> str:
    settings_path = _PROJECT_ROOT / "app_settings.yaml"
    try:
        with settings_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return str(data.get("language", "zh"))
    except Exception:
        return "zh"


def _ensure_lang_data() -> Dict[str, str]:
    global _LANG_CODE, _LANG_DATA
    lang = _load_language_code()
    if _LANG_DATA is not None and _LANG_CODE == lang:
        return _LANG_DATA

    _LANG_CODE = lang
    lang_path = _PROJECT_ROOT / f"{lang}.yaml"
    try:
        with lang_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        _LANG_DATA = {str(k): str(v) for k, v in loaded.items()}
    except Exception:
        _LANG_DATA = {}
    return _LANG_DATA


def translate(key: str, default: Optional[str] = None) -> str:
    data = _ensure_lang_data()
    value = data.get(key)
    if value is not None and value != key:
        return value
    if default is not None:
        return default
    return key
