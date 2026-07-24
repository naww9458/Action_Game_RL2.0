from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

_TEMPLATE_ROOT = Path(__file__).parent
_TEMPLATES_REGISTERED = False


def load_object_templates() -> Dict[str, Dict[str, Any]]:
    templates: Dict[str, Dict[str, Any]] = {}
    for folder in sorted(_TEMPLATE_ROOT.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        template_path = folder / "template.yaml"
        if not template_path.exists():
            continue
        with template_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        template_id = str(data.get("id", folder.name))
        templates[template_id] = data
    return templates


def get_object_template(template_id: str) -> Optional[Dict[str, Any]]:
    return load_object_templates().get(template_id)


def _import_register_callable(module_path: str, callable_name: str) -> Callable[[], None]:
    module = importlib.import_module(module_path)
    register_fn = getattr(module, callable_name, None)
    if register_fn is None or not callable(register_fn):
        raise AttributeError(
            f"Module '{module_path}' has no callable '{callable_name}' for articulation registration."
        )
    return register_fn


def _register_template_folder(folder: Path, template_data: Dict[str, Any]) -> None:
    register_py = folder / "register.py"
    if register_py.exists():
        module = importlib.import_module(
            f"script.role.objects.object_template.{folder.name}.register"
        )
        register_fn = getattr(module, "register", None)
        if register_fn is None or not callable(register_fn):
            raise AttributeError(
                f"Template folder '{folder.name}' has register.py but no register() function."
            )
        register_fn()
        return

    articulation = dict(template_data.get("articulation") or {})
    module_path = articulation.get("register_module")
    if not module_path:
        return

    callable_name = str(articulation.get("register_callable", "register"))
    register_fn = _import_register_callable(str(module_path), callable_name)
    register_fn()


def ensure_object_templates_registered() -> None:
    """Discover object_template/*/register.py (or template.yaml articulation hooks) once."""
    global _TEMPLATES_REGISTERED
    if _TEMPLATES_REGISTERED:
        return

    for folder in sorted(_TEMPLATE_ROOT.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        template_path = folder / "template.yaml"
        if not template_path.exists():
            continue
        with template_path.open("r", encoding="utf-8") as fh:
            template_data = yaml.safe_load(fh) or {}
        if not isinstance(template_data, dict):
            continue
        _register_template_folder(folder, template_data)

    _TEMPLATES_REGISTERED = True
