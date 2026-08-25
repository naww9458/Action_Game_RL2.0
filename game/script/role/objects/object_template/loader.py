from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

_TEMPLATE_ROOT = Path(__file__).parent
_TEMPLATES_REGISTERED = False
_TEMPLATE_SETUP_FNS: List[Callable[[Any], None]] = []
_TEMPLATE_PREPARE_FNS: List[Callable[[Any], None]] = []


def load_object_templates() -> Dict[str, Dict[str, Any]]:
    templates: Dict[str, Dict[str, Any]] = {}
    for folder in sorted(_TEMPLATE_ROOT.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_") or not folder.name.isidentifier():
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


def load_template_control_config(template_id: str) -> Dict[str, Any]:
    """Load an object template's control config file (raw dict).

    The relative path is declared in the template itself
    (``control_config_path``, default ``control_configs.yaml``), so callers
    (e.g. generic abilities) never need to know the template folder layout.
    Returns an empty dict when the template or file is unavailable.
    """
    template = get_object_template(template_id)
    if template is None:
        return {}
    cfg_rel = str(template.get("control_config_path") or "control_configs.yaml")
    cfg_path = _TEMPLATE_ROOT / template_id / cfg_rel
    if not cfg_path.exists():
        return {}
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


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
        setup_fn = getattr(module, "setup", None)
        if callable(setup_fn):
            _TEMPLATE_SETUP_FNS.append(setup_fn)
        prepare_fn = getattr(module, "prepare_object", None)
        if callable(prepare_fn):
            _TEMPLATE_PREPARE_FNS.append(prepare_fn)
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
        if not folder.is_dir() or folder.name.startswith("_") or not folder.name.isidentifier():
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


def run_object_template_setups(environment: Any) -> None:
    """Run per-template ``setup(environment)`` hooks after physics is built.

    ``register.py`` may define ``setup(environment)`` for object-specific runtime
    wiring (sensors, soft forces, ...). Templates without ``setup`` are skipped.
    Each hook is responsible for no-op when its object is not in the environment.
    """
    ensure_object_templates_registered()
    for setup_fn in _TEMPLATE_SETUP_FNS:
        setup_fn(environment)


def prepare_object_config(object_cfg: Any) -> None:
    """Let loaded object templates write their own fields onto an object config.

    Each template's ``register.prepare_object`` no-ops when the object is not
    theirs. Callers do not know which files a template reads.
    """
    if object_cfg is None:
        return
    ensure_object_templates_registered()
    for prepare_fn in _TEMPLATE_PREPARE_FNS:
        prepare_fn(object_cfg)
