from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from newton import Model
from script.game_config import GameConfig

if TYPE_CHECKING:
    from script.game import Game


def _path_suffix(path: str) -> str:
    return path.rstrip("/").split("/")[-1].lower()


@dataclass
class BodyParamSpec:
    path: str
    display_name: str
    global_body_index: int
    body_in_obj_idx: int
    is_base_body: bool = True

    @property
    def can_edit_position(self) -> bool:
        return self.is_base_body

    @property
    def can_edit_orientation(self) -> bool:
        return self.is_base_body


@dataclass
class JointParamSpec:
    path: str
    display_name: str
    dof_in_obj_idx: int
    limit_min: float = -3.14159
    limit_max: float = 3.14159


@dataclass
class ActionDimSpec:
    dim_index: int
    display_name: str
    lo: float
    hi: float
    step: float


@dataclass
class AbilityActionSpec:
    ability_name: str
    dims: List[ActionDimSpec] = field(default_factory=list)


@dataclass
class PlayerActionSpec:
    player_index: int
    rl_action_row: int
    abilities: List[AbilityActionSpec] = field(default_factory=list)

    @property
    def total_dim(self) -> int:
        return sum(len(a.dims) for a in self.abilities)


@dataclass
class ObjectInspectorSpec:
    catalog_key: str
    label: str
    local_role_idx: int
    body_kind: str  # "articulation" | "deformable" | "rigid"
    pattern: str
    view_obj_idx: int
    bodies: List[BodyParamSpec] = field(default_factory=list)
    joints: List[JointParamSpec] = field(default_factory=list)
    particle_count: int = 1
    player_action: Optional[PlayerActionSpec] = None


class InspectorCatalog:
    def __init__(self):
        self.specs: Dict[str, ObjectInspectorSpec] = {}
        self.specs_by_role: Dict[int, ObjectInspectorSpec] = {}

    @classmethod
    def build_from_game(cls, game: "Game") -> "InspectorCatalog":
        catalog = cls()
        pm = game.physics_manager
        ab = game.articulation_body
        db = game.deformable_body
        model: Model = pm.model

        body_keys = [str(k).lower() for k in model.body_label]
        joint_keys = [str(k).lower() for k in model.joint_label]
        duplicate_labels = {
            label
            for label, count in Counter(pm.role_object_labels.values()).items()
            if count > 1
        }

        for local_role_idx, label in sorted(pm.role_object_labels.items()):
            catalog_key = _catalog_key(label, local_role_idx, duplicate_labels)
            spec = ObjectInspectorSpec(
                catalog_key=catalog_key,
                label=label,
                local_role_idx=local_role_idx,
                body_kind="rigid",
                pattern=label,
                view_obj_idx=-1,
            )

            if label in ab.patterns:
                spec.body_kind = "articulation"
                spec.pattern = label
                spec.view_obj_idx = _view_obj_idx(ab, label, local_role_idx)
                meta = pm.object_metadata_by_role.get(
                    local_role_idx,
                    pm.object_metadata.get(label, {}),
                )
                spec.bodies = _resolve_bodies(
                    meta, model, body_keys, ab, label, local_role_idx, spec.view_obj_idx
                )
                spec.joints = _resolve_joints(meta, model, joint_keys, ab, label)
            elif label in db.patterns:
                spec.body_kind = "deformable"
                spec.pattern = label
                spec.view_obj_idx = _view_obj_idx(db, label, local_role_idx)
                spec.particle_count = db.count_particle_per_object.get(label, 1)
                spec.bodies = [
                    BodyParamSpec(
                        path="particle_0",
                        display_name="particle_0",
                        global_body_index=-1,
                        body_in_obj_idx=0,
                    )
                ]
            else:
                spec.bodies = [_make_root_body_spec(ab, label, spec.view_obj_idx, local_role_idx)]

            catalog.specs[catalog_key] = spec
            catalog.specs_by_role[local_role_idx] = spec
            spec.player_action = _resolve_player_action(game, local_role_idx)

        return catalog

    def list_catalog_keys(self) -> List[str]:
        return sorted(
            self.specs.keys(),
            key=lambda key: self.specs[key].local_role_idx,
        )

    def get_by_catalog_key(self, catalog_key: str) -> Optional[ObjectInspectorSpec]:
        return self.specs.get(catalog_key)

    def get_by_role(self, global_role_idx: int, num_objects_env: int) -> Optional[ObjectInspectorSpec]:
        local = global_role_idx % num_objects_env
        return self.specs_by_role.get(local)

    def get_by_label(self, label: str) -> Optional[ObjectInspectorSpec]:
        matches = [spec for spec in self.specs.values() if spec.label == label]
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        return sorted(matches, key=lambda spec: spec.local_role_idx)[0]


def _catalog_key(label: str, local_role_idx: int, duplicate_labels: set[str]) -> str:
    if label in duplicate_labels:
        return f"{label} [{local_role_idx}]"
    return label


def _view_obj_idx(body_mgr, pattern: str, local_role_idx: int) -> int:
    indices = body_mgr.patterns.get(pattern, [])
    try:
        return indices.index(local_role_idx)
    except ValueError:
        return 0


def _resolve_root_body_index(ab, label: str, view_obj_idx: int, fallback: int) -> int:
    """Map a view object slot to the Newton body index within env 0."""
    local_indices = getattr(ab, "view_body_local_indices_gpus", {}).get(label)
    if local_indices is None:
        return fallback
    arr = local_indices.numpy()
    if view_obj_idx < 0 or view_obj_idx >= len(arr):
        return fallback
    return int(arr[view_obj_idx])


def _make_root_body_spec(
    ab,
    label: str,
    view_obj_idx: int,
    local_role_idx: int,
) -> BodyParamSpec:
    return BodyParamSpec(
        path="root",
        display_name="root",
        global_body_index=_resolve_root_body_index(ab, label, view_obj_idx, local_role_idx),
        body_in_obj_idx=0,
    )


def _resolve_bodies(
    meta: dict,
    model,
    body_keys: List[str],
    ab,
    label: str,
    local_role_idx: int,
    view_obj_idx: int,
) -> List[BodyParamSpec]:
    path_body_map: Dict[str, int] = meta.get("path_body_map") or {}
    if not path_body_map:
        return [_make_root_body_spec(ab, label, view_obj_idx, local_role_idx)]

    base_body = _resolve_root_body_index(ab, label, view_obj_idx, local_role_idx)

    bodies: List[BodyParamSpec] = []
    sorted_paths = sorted(path_body_map.items(), key=lambda kv: kv[1])
    base_builder_idx = min(path_body_map.values())
    for body_in_obj_idx, (path, builder_idx) in enumerate(sorted_paths):
        suffix = _path_suffix(path)
        global_body_index = _match_body_index(body_keys, suffix, base_body)
        if global_body_index < 0:
            global_body_index = base_body + body_in_obj_idx
        bodies.append(
            BodyParamSpec(
                path=path,
                display_name=suffix,
                global_body_index=global_body_index,
                body_in_obj_idx=body_in_obj_idx,
                is_base_body=(builder_idx == base_builder_idx),
            )
        )
    return bodies


def _match_body_index(body_keys: List[str], suffix: str, fallback: int) -> int:
    for i, key in enumerate(body_keys):
        if key.endswith(suffix) or suffix in key:
            return i
    return fallback


def _resolve_joints(
    meta: dict,
    model,
    joint_keys: List[str],
    ab,
    label: str,
) -> List[JointParamSpec]:
    path_joint_map: Dict[str, int] = meta.get("path_joint_map") or {}
    if not path_joint_map:
        return []

    view_idx = next((i for i, p in enumerate(ab.patterns.keys()) if p == label), -1)
    dof_names: List[str] = []
    if view_idx >= 0:
        view = ab.views[view_idx]
        dof_names = [n.lower() for n in getattr(view, "joint_dof_names", [])]

    lim_lower = model.joint_limit_lower.numpy() if model.joint_limit_lower is not None else None
    lim_upper = model.joint_limit_upper.numpy() if model.joint_limit_upper is not None else None
    qd_start = model.joint_qd_start.numpy()

    joints: List[JointParamSpec] = []
    sorted_paths = sorted(path_joint_map.items(), key=lambda kv: kv[1])
    for path, _builder_idx in sorted_paths:
        suffix = _path_suffix(path)
        dof_in_obj_idx = -1
        if dof_names:
            for i, name in enumerate(dof_names):
                if name == suffix or suffix in name or name in suffix:
                    dof_in_obj_idx = i
                    break
        if dof_in_obj_idx < 0:
            dof_in_obj_idx = len(joints)

        joint_idx = _match_joint_index(joint_keys, suffix)
        lo, hi = -3.14159, 3.14159
        if joint_idx >= 0 and joint_idx + 1 < len(qd_start):
            dof_idx = int(qd_start[joint_idx])
            if lim_lower is not None and lim_upper is not None and dof_idx < len(lim_lower):
                lo = float(lim_lower[dof_idx])
                hi = float(lim_upper[dof_idx])
                if abs(hi) > 1e5:
                    hi = 3.14159
                if abs(lo) > 1e5:
                    lo = -3.14159
        joints.append(
            JointParamSpec(
                path=path,
                display_name=suffix,
                dof_in_obj_idx=dof_in_obj_idx,
                limit_min=lo,
                limit_max=hi,
            )
        )
    return joints


def _match_joint_index(joint_keys: List[str], suffix: str) -> int:
    for i, key in enumerate(joint_keys):
        if key.endswith(suffix) or suffix in key:
            return i
    return -1


def _action_dims_from_spec(action_spec: dict, start_offset: int) -> List[ActionDimSpec]:
    atype = str(action_spec.get("type", "box")).lower()
    shape = action_spec.get("shape", 1)
    if isinstance(shape, str):
        try:
            shape = int(shape)
        except ValueError:
            shape = 1
    shape = max(int(shape), 1)

    if atype == "discrete":
        n = int(action_spec.get("n", 2))
        lo, hi = 0.0, float(max(n - 1, 1))
        step = 1.0 if n <= 2 else max((hi - lo) / 100.0, 0.01)
    else:
        rng = action_spec.get("range", [-1.0, 1.0])
        if isinstance(rng, list) and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
        else:
            lo, hi = -1.0, 1.0
        step = max((hi - lo) / 200.0, 0.001) if hi > lo else 0.01

    dims: List[ActionDimSpec] = []
    dim_labels = action_spec.get("dims")
    for i in range(shape):
        label = f"action_{i}" if shape > 1 else "action"
        if isinstance(dim_labels, list) and i < len(dim_labels):
            entry = dim_labels[i]
            if isinstance(entry, dict) and entry.get("name"):
                label = str(entry["name"])
            elif isinstance(entry, str):
                label = entry
        dims.append(
            ActionDimSpec(
                dim_index=start_offset + i,
                display_name=label,
                lo=lo,
                hi=hi,
                step=step,
            )
        )
    return dims


def _resolve_player_action(game: "Game", local_role_idx: int) -> Optional[PlayerActionSpec]:
    action_cfg = getattr(GameConfig, "ACTION_SPACE_CONFIG", None)
    if not action_cfg:
        return None

    players = game.players
    num_objects_env = game.num_objects_env
    env0_players = sorted(idx for idx in players.index_obj_role if idx < num_objects_env)
    if local_role_idx not in env0_players:
        return None

    player_index = env0_players.index(local_role_idx)
    if player_index >= len(action_cfg):
        return None

    rl_row = -1
    try:
        role_list_idx = players.index_obj_role.index(local_role_idx)
        mask = getattr(game.level, "is_rl_player_mask", None)
        if mask is not None and role_list_idx < len(mask):
            rl_row = int(mask[role_list_idx])
    except ValueError:
        return None

    player_actions = action_cfg[player_index]
    if not isinstance(player_actions, dict):
        return None

    abilities: List[AbilityActionSpec] = []
    offset = 0
    for ability_name, action_spec in player_actions.items():
        dims = _action_dims_from_spec(action_spec, offset)
        if not dims:
            continue
        abilities.append(AbilityActionSpec(ability_name=ability_name, dims=dims))
        offset += len(dims)

    if not abilities:
        return None

    return PlayerActionSpec(
        player_index=player_index,
        rl_action_row=rl_row,
        abilities=abilities,
    )
