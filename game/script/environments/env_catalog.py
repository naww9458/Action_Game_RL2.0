"""Filesystem catalog for nested environment configs.

Layout::

    game/script/environments/
      environment.py, get_environment.py, env_catalog.py, rewards/
      template/{rl,play}/<series>/[<category>/...]/<env>/<env>.yaml
      custom/{rl,play}/...

Series and category folders have ``_category.yaml``. Each environment is its
own folder (no ``_category.yaml``) holding ``<env>.yaml`` and optional
``<env>.py``. Catalog scan indexes YAML + ``_category.yaml``; Python files
are copied with the env folder.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

ENVIRONMENTS_ROOT = Path(__file__).resolve().parent

ROOT_TEMPLATE = "template"
ROOT_CUSTOM = "custom"
BUCKET_RL = "rl"
BUCKET_PLAY = "play"

STRUCTURAL_ROOTS = (ROOT_TEMPLATE, ROOT_CUSTOM)
STRUCTURAL_BUCKETS = (BUCKET_RL, BUCKET_PLAY)
CATEGORY_FILENAME = "_category.yaml"
RESERVED_FILENAMES = {CATEGORY_FILENAME}
SKIP_DIR_NAMES = {"__pycache__", "rewards"}
INIT_FILENAME = "__init__.py"

_SLUG_UNSAFE = re.compile(r'[<>:"/\\|?*]')
_SLUG_SPACES = re.compile(r"\s+")

DEFAULT_ROOT_META = {
    ROOT_TEMPLATE: {
        "name": "範本",
        "intro": "隨專案提供的官方環境。不可用右鍵新增，可刪除或複製到自定義。",
    },
    ROOT_CUSTOM: {
        "name": "自定義",
        "intro": "使用者可新增、修改並上傳到範本的環境。",
    },
}
DEFAULT_BUCKET_META = {
    BUCKET_RL: {
        "name": "強化學習環境",
        "intro": "用於訓練與評估策略的任務環境。",
    },
    BUCKET_PLAY: {
        "name": "娛樂/測試環境",
        "intro": "用於遊玩、解算器與物件插件驗證的環境。",
    },
}


@dataclass
class CatalogNode:
    """One tree entry: structural folder, series, nested category, or env."""

    node_type: str
    path: Path
    name: str
    intro: str = ""
    root: str = ""
    bucket: str = ""
    rel_id: str = ""
    env_id: Optional[str] = None
    environment_class: Optional[str] = None
    children: List["CatalogNode"] = field(default_factory=list)

    @property
    def is_env(self) -> bool:
        return self.node_type == "env"

    @property
    def is_structural(self) -> bool:
        return self.node_type in ("root", "bucket")

    @property
    def is_series(self) -> bool:
        return self.node_type == "series"

    @property
    def is_category(self) -> bool:
        return self.node_type == "category"


def environments_root() -> Path:
    return ENVIRONMENTS_ROOT


def write_category_yaml(directory: Path, name: str, intro: str = "") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"name": str(name), "intro": str(intro or "")}
    (directory / CATEGORY_FILENAME).write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _ensure_init_py(directory)


def read_category_yaml(directory: Path) -> Dict[str, str]:
    path = directory / CATEGORY_FILENAME
    if not path.is_file():
        return {"name": directory.name, "intro": ""}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"name": directory.name, "intro": ""}
    if not isinstance(data, dict):
        return {"name": directory.name, "intro": ""}
    name = str(data.get("name") or directory.name)
    intro = str(data.get("intro") or "")
    return {"name": name, "intro": intro}


def _ensure_init_py(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    init_path = directory / INIT_FILENAME
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")


def _ensure_python_packages(directory: Path) -> None:
    """Touch ``__init__.py`` from ``directory`` up through ``environments/``."""
    root = ENVIRONMENTS_ROOT.resolve()
    current = directory.resolve()
    while True:
        _ensure_init_py(current)
        if current == root:
            break
        if current.parent == current:
            break
        try:
            current.relative_to(root)
        except ValueError:
            break
        current = current.parent


def env_script_path(yaml_path: Path) -> Path:
    return yaml_path.with_suffix(".py")


def resolve_env_yaml(directory: Path) -> Optional[Path]:
    """Return the env YAML inside an environment folder, if any."""
    if not directory.is_dir():
        return None
    preferred = directory / f"{directory.name}.yaml"
    if preferred.is_file():
        return preferred
    preferred_yml = directory / f"{directory.name}.yml"
    if preferred_yml.is_file():
        return preferred_yml
    found = list(_iter_env_files(directory))
    if len(found) == 1:
        return found[0]
    return None


def is_env_dir(directory: Path) -> bool:
    """True when ``directory`` is an env folder, not a series/category."""
    if not directory.is_dir():
        return False
    if (directory / CATEGORY_FILENAME).is_file():
        return False
    return resolve_env_yaml(directory) is not None


def as_catalog_leaf(path: Path) -> Path:
    """Map an env YAML path to its folder so copy/delete move the whole env."""
    resolved = Path(path)
    if resolved.is_file() and resolved.suffix.lower() in (".yaml", ".yml"):
        parent = resolved.parent
        if is_env_dir(parent):
            return parent
    return resolved


def env_script_module_name(py_path: Path) -> str:
    rel = py_path.resolve().relative_to(ENVIRONMENTS_ROOT.parent.resolve())
    return "script." + ".".join(rel.with_suffix("").parts)


def ensure_environment_roots() -> Path:
    """Create the six structural folders and their metadata if missing."""
    ENVIRONMENTS_ROOT.mkdir(parents=True, exist_ok=True)
    _ensure_init_py(ENVIRONMENTS_ROOT)
    for root_id, meta in DEFAULT_ROOT_META.items():
        root_dir = ENVIRONMENTS_ROOT / root_id
        if not (root_dir / CATEGORY_FILENAME).is_file():
            write_category_yaml(root_dir, meta["name"], meta["intro"])
        else:
            _ensure_init_py(root_dir)
        for bucket_id, bucket_meta in DEFAULT_BUCKET_META.items():
            bucket_dir = root_dir / bucket_id
            if not (bucket_dir / CATEGORY_FILENAME).is_file():
                write_category_yaml(bucket_dir, bucket_meta["name"], bucket_meta["intro"])
            else:
                _ensure_init_py(bucket_dir)
    return ENVIRONMENTS_ROOT


def slugify(name: str) -> str:
    text = _SLUG_SPACES.sub("_", _SLUG_UNSAFE.sub("_", str(name).strip())).strip("._")
    return text or "untitled"


def unique_child_path(parent: Path, base_name: str, *, suffix: str = "") -> Path:
    slug = slugify(base_name)
    candidate = parent / f"{slug}{suffix}"
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = parent / f"{slug}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _rel_id(path: Path) -> str:
    try:
        return path.resolve().relative_to(ENVIRONMENTS_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_env_header(path: Path) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    environment_class = data.get("environment_class")
    return {
        "display_name": str(data.get("display_name") or path.stem),
        "intro": str(data.get("intro") or ""),
        "environment_class": str(environment_class) if environment_class else None,
    }


def _has_catalog_content(directory: Path) -> bool:
    """True if the folder is a series/category, an env, or contains either."""
    if (directory / CATEGORY_FILENAME).is_file():
        return True
    if is_env_dir(directory):
        return True
    try:
        children = list(directory.iterdir())
    except OSError:
        return False
    for child in children:
        if child.name.startswith(".") or child.name in SKIP_DIR_NAMES:
            continue
        if child.is_dir() and _has_catalog_content(child):
            return True
        if (
            child.is_file()
            and child.suffix.lower() in (".yaml", ".yml")
            and child.name not in RESERVED_FILENAMES
        ):
            return True
    return False


def _iter_env_files(directory: Path) -> Iterable[Path]:
    for child in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_file():
            continue
        if child.name in RESERVED_FILENAMES:
            continue
        if child.suffix.lower() in (".yaml", ".yml"):
            yield child


def _scan_folder(
    directory: Path,
    *,
    node_type: str,
    root: str,
    bucket: str,
) -> CatalogNode:
    meta = read_category_yaml(directory)
    node = CatalogNode(
        node_type=node_type,
        path=directory,
        name=meta["name"],
        intro=meta["intro"],
        root=root,
        bucket=bucket,
        rel_id=_rel_id(directory),
    )
    child_type = "category" if node_type in ("series", "category") else "series"
    if node_type == "bucket":
        child_type = "series"
    for child_dir in sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name not in SKIP_DIR_NAMES
        ],
        key=lambda p: p.name.lower(),
    ):
        env_yaml = resolve_env_yaml(child_dir) if is_env_dir(child_dir) else None
        if env_yaml is not None:
            header = _read_env_header(env_yaml)
            node.children.append(
                CatalogNode(
                    node_type="env",
                    path=env_yaml,
                    name=header["display_name"],
                    intro=header["intro"],
                    root=root,
                    bucket=bucket,
                    rel_id=_rel_id(env_yaml),
                    env_id=env_yaml.stem,
                    environment_class=header["environment_class"],
                )
            )
            continue
        if not _has_catalog_content(child_dir):
            continue
        node.children.append(
            _scan_folder(child_dir, node_type=child_type, root=root, bucket=bucket)
        )
    for env_path in _iter_env_files(directory):
        header = _read_env_header(env_path)
        node.children.append(
            CatalogNode(
                node_type="env",
                path=env_path,
                name=header["display_name"],
                intro=header["intro"],
                root=root,
                bucket=bucket,
                rel_id=_rel_id(env_path),
                env_id=env_path.stem,
                environment_class=header["environment_class"],
            )
        )
    return node


def scan_catalog() -> List[CatalogNode]:
    ensure_environment_roots()
    roots: List[CatalogNode] = []
    for root_id in STRUCTURAL_ROOTS:
        root_dir = ENVIRONMENTS_ROOT / root_id
        meta = read_category_yaml(root_dir)
        root_node = CatalogNode(
            node_type="root",
            path=root_dir,
            name=meta["name"],
            intro=meta["intro"],
            root=root_id,
            rel_id=root_id,
        )
        for bucket_id in STRUCTURAL_BUCKETS:
            bucket_dir = root_dir / bucket_id
            root_node.children.append(
                _scan_folder(bucket_dir, node_type="bucket", root=root_id, bucket=bucket_id)
            )
        roots.append(root_node)
    return roots


def iter_envs(nodes: Optional[List[CatalogNode]] = None) -> Iterable[CatalogNode]:
    if nodes is None:
        nodes = scan_catalog()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if node.is_env:
            yield node
        stack.extend(reversed(node.children))


def resolve_env_path_by_id(env_id: str) -> Optional[Path]:
    """Return the YAML whose stem equals ``env_id``. Custom overrides template."""
    wanted = str(env_id or "").strip()
    if not wanted:
        return None
    if "/" in wanted or "\\" in wanted:
        return resolve_env_id(wanted)
    matches: Dict[str, Path] = {}
    for env in iter_envs():
        if env.path.stem != wanted:
            continue
        matches[env.root] = env.path
    if ROOT_CUSTOM in matches:
        return matches[ROOT_CUSTOM]
    return matches.get(ROOT_TEMPLATE)


def resolve_env_id(env_id: str) -> Optional[Path]:
    rel = str(env_id or "").replace("\\", "/").strip("/")
    if not rel:
        return None
    path = (ENVIRONMENTS_ROOT / rel).resolve()
    try:
        path.relative_to(ENVIRONMENTS_ROOT.resolve())
    except ValueError:
        return None
    if path.is_file():
        return path
    if path.is_dir():
        return resolve_env_yaml(path)
    return None


def parse_catalog_path(path: Path) -> Optional[Dict[str, Any]]:
    """Split an absolute catalog path into root / bucket / remainder."""
    try:
        rel = path.resolve().relative_to(ENVIRONMENTS_ROOT.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 2:
        return {
            "root": parts[0] if parts else "",
            "bucket": "",
            "rest": (),
            "rel": rel,
        }
    return {
        "root": parts[0],
        "bucket": parts[1],
        "rest": parts[2:],
        "rel": rel,
    }


def counterpart_root(root_id: str) -> str:
    return ROOT_CUSTOM if root_id == ROOT_TEMPLATE else ROOT_TEMPLATE


def destination_for(path: Path) -> Optional[Path]:
    parsed = parse_catalog_path(path)
    if parsed is None or not parsed["root"] or not parsed["bucket"]:
        return None
    dest_root = counterpart_root(str(parsed["root"]))
    return ENVIRONMENTS_ROOT.joinpath(dest_root, parsed["bucket"], *parsed["rest"])


def _copy_category_file(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_meta = src_dir / CATEGORY_FILENAME
    if src_meta.is_file():
        shutil.copy2(src_meta, dst_dir / CATEGORY_FILENAME)
    _ensure_python_packages(dst_dir)


def _class_name_from_environment_class(raw: str) -> str:
    return str(raw).replace(":", ".").rsplit(".", 1)[-1]


def _retarget_environment_class_value(raw: str, yaml_path: Path, src_root: str, dst_root: str) -> Optional[str]:
    src_prefix = f"script.environments.{src_root}."
    dst_prefix = f"script.environments.{dst_root}."
    if raw.startswith(src_prefix):
        return dst_prefix + raw[len(src_prefix):]
    companion = env_script_path(yaml_path)
    if companion.is_file():
        class_name = _class_name_from_environment_class(raw)
        if class_name:
            return f"{env_script_module_name(companion)}.{class_name}"
    return None


def _set_yaml_environment_class(yaml_path: Path, new_value: str) -> None:
    text = yaml_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    replaced = False
    out = []
    for line in lines:
        stripped = line.lstrip()
        if not replaced and stripped.startswith("environment_class:"):
            indent = line[: len(line) - len(stripped)]
            newline = "\n" if line.endswith("\n") else ""
            out.append(f"{indent}environment_class: {new_value}{newline}")
            replaced = True
        else:
            out.append(line)
    if replaced:
        yaml_path.write_text("".join(out), encoding="utf-8")


def _retarget_yaml_environment_class(yaml_path: Path, src_root: str, dst_root: str) -> None:
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return
    if not isinstance(data, dict):
        return
    raw = data.get("environment_class")
    if not isinstance(raw, str) or not raw:
        return
    new_value = _retarget_environment_class_value(raw, yaml_path, src_root, dst_root)
    if new_value and new_value != raw:
        _set_yaml_environment_class(yaml_path, new_value)


def _copy_companion_script(src_yaml: Path, dest_yaml: Path, *, overwrite: bool) -> None:
    src_py = env_script_path(src_yaml)
    if not src_py.is_file():
        return
    dest_py = env_script_path(dest_yaml)
    if dest_py.exists() and not overwrite:
        raise FileExistsError(str(dest_py))
    shutil.copy2(src_py, dest_py)


def _retarget_copied_tree(dest: Path, src_root: str, dst_root: str) -> None:
    if dest.is_file() and dest.suffix.lower() in (".yaml", ".yml"):
        _retarget_yaml_environment_class(dest, src_root, dst_root)
        _ensure_python_packages(dest.parent)
        return
    if not dest.is_dir():
        return
    for yaml_path in dest.rglob("*.yaml"):
        if yaml_path.name in RESERVED_FILENAMES:
            continue
        _retarget_yaml_environment_class(yaml_path, src_root, dst_root)
    _ensure_python_packages(dest)


def _copy_ancestors(src_kind: Path, dst_kind: Path, rest: Tuple[str, ...]) -> None:
    acc_src = src_kind
    acc_dst = dst_kind
    for part in rest:
        acc_src = acc_src / part
        acc_dst = acc_dst / part
        if acc_src.is_dir():
            _copy_category_file(acc_src, acc_dst)


def copy_catalog_node(src: Path, *, overwrite: bool = False) -> Path:
    """Copy a series / category / env across template ↔ custom.

    Ancestor folders are reused; sibling nodes are not copied. ``overwrite``
    replaces the destination leaf only.
    """
    parsed = parse_catalog_path(src)
    if parsed is None:
        raise ValueError(f"Path is not inside the environment catalog: {src}")
    src = as_catalog_leaf(src)
    parsed = parse_catalog_path(src)
    if parsed is None:
        raise ValueError(f"Path is not inside the environment catalog: {src}")
    root_id = str(parsed["root"])
    bucket_id = str(parsed["bucket"])
    rest: Tuple[str, ...] = tuple(parsed["rest"])
    if root_id not in STRUCTURAL_ROOTS or bucket_id not in STRUCTURAL_BUCKETS:
        raise ValueError("Cannot copy structural template/custom/rl/play folders.")
    if not rest:
        raise ValueError("Select a series, category, or environment to copy.")

    dest = destination_for(src)
    if dest is None:
        raise ValueError(f"Cannot map counterpart for {src}")
    src_kind = ENVIRONMENTS_ROOT / root_id / bucket_id
    dst_kind = ENVIRONMENTS_ROOT / counterpart_root(root_id) / bucket_id
    dst_kind.mkdir(parents=True, exist_ok=True)

    dst_root_id = counterpart_root(root_id)
    if src.is_file():
        parent_rest = rest[:-1]
        _copy_ancestors(src_kind, dst_kind, parent_rest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest_py = env_script_path(dest)
        src_py = env_script_path(src)
        if dest.exists() and not overwrite:
            raise FileExistsError(str(dest))
        if src_py.is_file() and dest_py.exists() and not overwrite:
            raise FileExistsError(str(dest_py))
        shutil.copy2(src, dest)
        _copy_companion_script(src, dest, overwrite=overwrite)
        _retarget_copied_tree(dest, root_id, dst_root_id)
        return dest

    if not src.is_dir():
        raise FileNotFoundError(str(src))

    _copy_ancestors(src_kind, dst_kind, rest[:-1])
    if dest.exists():
        if not overwrite:
            raise FileExistsError(str(dest))
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    _retarget_copied_tree(dest, root_id, dst_root_id)
    return dest


def create_folder_node(parent: Path, name: str, intro: str = "") -> Path:
    path = unique_child_path(parent, name)
    write_category_yaml(path, name, intro)
    return path


def create_env_yaml(parent: Path, name: str, intro: str = "") -> Path:
    folder = unique_child_path(parent, name)
    folder.mkdir(parents=True, exist_ok=False)
    _ensure_python_packages(folder)
    path = folder / f"{folder.name}.yaml"
    payload = {
        "display_name": str(name),
        "intro": str(intro or ""),
    }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def delete_catalog_node(path: Path) -> None:
    parsed = parse_catalog_path(path)
    if parsed is None:
        raise ValueError(f"Path is not inside the environment catalog: {path}")
    path = as_catalog_leaf(path)
    parsed = parse_catalog_path(path)
    if parsed is None:
        raise ValueError(f"Path is not inside the environment catalog: {path}")
    rest = parsed["rest"]
    if not rest:
        raise ValueError("Cannot delete structural template/custom/rl/play folders.")
    resolved = path.resolve()
    resolved.relative_to(ENVIRONMENTS_ROOT.resolve())
    if resolved.is_dir():
        shutil.rmtree(resolved)
    elif resolved.is_file():
        companion = env_script_path(resolved)
        resolved.unlink()
        if companion.is_file():
            companion.unlink()
    else:
        raise FileNotFoundError(str(path))


def is_structural_path(path: Path) -> bool:
    parsed = parse_catalog_path(path)
    if parsed is None:
        return False
    return not parsed["rest"]
