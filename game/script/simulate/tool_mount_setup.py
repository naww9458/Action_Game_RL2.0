"""Level-time setup for tool mount metadata and build-time joint slots."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import warp as wp

from script.role.objects.tool_anchor import resolve_anchor_pair
from script.role.objects.object_template.loader import get_object_template
from script.role.objects.object_template.tool_function_registry import create_tool_action
from script.role.objects.usd import _join_asset_path
from script.role.tool import ToolModel
from script.simulate.mount_joint_builder import (
    build_tool_mount_joint,
    collect_tool_mount_metadata,
    compute_max_mount_joints_per_env,
)
from script.simulate.mount_joint_registry import MountJointRegistry, ToolMountRecord

# Schema defaults are the single source for values a level may omit
# (LevelConfig.tool_configs: List[ToolModel] always fills them via model_dump);
# these references keep runtime safety without re-typing the constants here.
_MOUNT_SCHEMA_DEFAULTS = ToolModel()

if TYPE_CHECKING:
    from script.levels.levels import Levels


def _normalize_optional_str_candidates(
    value: Optional[Union[str, List[Optional[str]]]],
    default_value: Optional[str],
) -> List[str]:
    """Normalize a str / list[str]/None into a concrete candidate list.

    Empty result means neither the level config nor the template provided a value.
    """
    if value is None:
        return [default_value] if default_value else []
    if isinstance(value, list):
        out: List[str] = []
        for v in value:
            if v is None:
                if default_value:
                    out.append(default_value)
            else:
                out.append(str(v))
        if out:
            return out
        return [default_value] if default_value else []
    return [str(value)]


def _require_tool_mount_field(
    field_name: str,
    value: Optional[str],
    *,
    tool_key: str,
    tool_pattern: Optional[str],
) -> str:
    if value is not None and str(value).strip():
        return str(value)
    pattern_hint = tool_pattern or "<pattern>"
    raise ValueError(
        f"Tool '{tool_key}' missing required mount field '{field_name}'. "
        f"Set it in tool_configs or object_template/{pattern_hint}/template.yaml."
    )


def _normalize_optional_int_candidates(
    value: Optional[Union[int, List[Optional[int]]]],
    num_players: int,
) -> List[int]:
    """Normalize int / list[int]/None into candidate player indices."""
    if value is None:
        return list(range(num_players))
    if isinstance(value, list):
        out: List[int] = []
        for v in value:
            if v is None:
                continue
            out.append(int(v))
        return out or list(range(num_players))
    return [int(value)]


def _host_spawn_distance_sq(host: dict, player_configs: List[dict], tool_cfg: dict) -> float:
    """Squared horizontal distance between a host's default spawn and the tool's spawn.

    Used to bind a tool to the host it was placed next to when multiple players
    are compatible (e.g. two armored vehicles + two turrets).
    """
    player_idx = int(host["host_player_index"])
    player_pos = player_configs[player_idx].get("default_position") or (0.0, 0.0, 0.0)
    tool_pos = tool_cfg.get("default_position") or (0.0, 0.0, 0.0)
    dx = float(player_pos[0]) - float(tool_pos[0])
    dy = float(player_pos[1]) - float(tool_pos[1])
    return dx * dx + dy * dy


def setup_tool_mount_joints(level: "Levels") -> Optional[MountJointRegistry]:
    tool_configs: List[dict] = level.level_configs.get("tool_configs") or []
    if not tool_configs:
        # No tools → no mount registry (avoid empty create/bind on non-tool levels).
        level.mount_joint_registry = None
        return None

    max_per_env = compute_max_mount_joints_per_env(tool_configs)
    registry = MountJointRegistry(
        max_mount_joints_per_env=max_per_env,
        num_env=getattr(level, "num_env", 1),
    )

    tools = getattr(level, "tools", None)
    players = level.players
    physics_manager = level.physics_manager
    builder = physics_manager.builder_env

    player_configs = level.level_configs.get("player_configs") or []
    solver_type = str(
        (level.level_configs.get("environment_configs") or {})
        .get("solver_config", {})
        .get("type", "")
    )

    slot_index = 0
    for tool_index, tool_cfg in enumerate(tool_configs):
        # tool_key 以物件 ID（id 欄位）為準，缺省時退回 name / 索引
        tool_key = str(
            tool_cfg.get("id")
            or tool_cfg.get("name")
            or f"tool_{tool_index}"
        )
        tool_role_id = _resolve_tool_role_id(tools, tool_index)

        tool_meta = physics_manager.object_metadata_by_role.get(tool_role_id, {})
        tool_label = physics_manager.role_object_labels.get(tool_role_id, "")

        tool_object = dict(tool_cfg.get("object") or {})
        tool_pattern = tool_object.get("pattern")
        tool_template = (
            get_object_template(str(tool_pattern)) if tool_pattern is not None else None
        )

        pattern_str = str(tool_pattern) if tool_pattern is not None else None

        effective_mount_anchor_name = tool_cfg.get("mount_anchor_name")
        if effective_mount_anchor_name is None and tool_template:
            effective_mount_anchor_name = tool_template.get("mount_anchor_name")
        effective_mount_anchor_name = _require_tool_mount_field(
            "mount_anchor_name",
            effective_mount_anchor_name,
            tool_key=tool_key,
            tool_pattern=pattern_str,
        )

        effective_host_anchor_name_default = (
            tool_template.get("host_anchor_name") if tool_template else None
        )
        effective_host_body_prim_suffix_default = (
            tool_template.get("host_body_prim_suffix") if tool_template else None
        )

        effective_tool_base_body_prim_suffix = tool_cfg.get("tool_base_body_prim_suffix")
        if effective_tool_base_body_prim_suffix is None and tool_template:
            effective_tool_base_body_prim_suffix = tool_template.get(
                "tool_base_body_prim_suffix"
            )
        effective_tool_base_body_prim_suffix = _require_tool_mount_field(
            "tool_base_body_prim_suffix",
            effective_tool_base_body_prim_suffix,
            tool_key=tool_key,
            tool_pattern=pattern_str,
        )

        host_anchor_candidates = _normalize_optional_str_candidates(
            tool_cfg.get("host_anchor_name"),
            default_value=effective_host_anchor_name_default,
        )
        host_body_prim_suffix_candidates = _normalize_optional_str_candidates(
            tool_cfg.get("host_body_prim_suffix"),
            default_value=effective_host_body_prim_suffix_default,
        )
        if not host_anchor_candidates:
            _require_tool_mount_field(
                "host_anchor_name",
                None,
                tool_key=tool_key,
                tool_pattern=pattern_str,
            )
        if not host_body_prim_suffix_candidates:
            _require_tool_mount_field(
                "host_body_prim_suffix",
                None,
                tool_key=tool_key,
                tool_pattern=pattern_str,
            )
        host_player_index_candidates = _resolve_host_player_candidates(
            tool_cfg,
            player_configs=player_configs,
            num_players=len(players.index_obj_role),
            tool_key=tool_key,
        )
        tool_asset = _join_asset_path(
            tool_object.get("file_path_or_source", ""),
            tool_object.get("file_name", ""),
        )

        resolved_host_body_path: Optional[str] = None
        resolved_tool_body_path: Optional[str] = None
        resolved_host_local: Optional[wp.transform] = None
        resolved_tool_local: Optional[wp.transform] = None

        # When several players can host this tool (host_player_index: null),
        # collect every compatible host and bind to the one whose default spawn
        # is nearest to the tool's spawn. Without this, each tool binds to the
        # first compatible player, so a second armored vehicle never sees the
        # mount prompt (level8-1 / level9-2 have two vehicles + two turrets).
        valid_hosts: List[dict] = []
        last_key_error: Optional[Exception] = None
        for host_player_index in host_player_index_candidates:
            if host_player_index >= len(players.index_obj_role) or host_player_index < 0:
                continue

            host_role_id = players.index_obj_role[host_player_index]
            host_meta = physics_manager.object_metadata_by_role.get(host_role_id, {})
            host_label = physics_manager.role_object_labels.get(host_role_id, "")
            host_player_cfg = dict(player_configs[host_player_index])
            host_object = dict(host_player_cfg.get("object") or {})
            host_asset = _join_asset_path(
                host_object.get("file_path_or_source", ""),
                host_object.get("file_name", ""),
            )

            matched = False
            for host_anchor_name in host_anchor_candidates:
                for host_body_prim_suffix in host_body_prim_suffix_candidates:
                    try:
                        (
                            host_body_path,
                            tool_body_path,
                            host_local,
                            tool_local,
                        ) = resolve_anchor_pair(
                            host_asset_path=host_asset,
                            tool_asset_path=tool_asset,
                            host_anchor_name=str(host_anchor_name),
                            tool_anchor_name=str(effective_mount_anchor_name),
                            host_path_body_map=host_meta.get("path_body_map") or {},
                            tool_path_body_map=tool_meta.get("path_body_map") or {},
                            host_body_prim_suffix=str(host_body_prim_suffix),
                            tool_base_body_prim_suffix=str(effective_tool_base_body_prim_suffix),
                        )
                        valid_hosts.append(
                            {
                                "host_player_index": host_player_index,
                                "host_role_id": host_role_id,
                                "host_meta": host_meta,
                                "host_label": host_label,
                                "host_body_path": host_body_path,
                                "tool_body_path": tool_body_path,
                                "host_local": host_local,
                                "tool_local": tool_local,
                            }
                        )
                        matched = True
                        break
                    except KeyError as e:
                        last_key_error = e
                        continue
                if matched:
                    break

        if not valid_hosts:
            raise KeyError(
                f"Tool '{tool_key}' could not resolve a valid host mount anchor/body prim. "
                f"Tried host_player_index={host_player_index_candidates}, "
                f"host_anchor_name={host_anchor_candidates}, host_body_prim_suffix={host_body_prim_suffix_candidates}. "
                f"Last error: {last_key_error!r}"
            )

        if len(valid_hosts) > 1:
            resolved_host = min(
                valid_hosts,
                key=lambda h: _host_spawn_distance_sq(h, player_configs, tool_cfg),
            )
        else:
            resolved_host = valid_hosts[0]
        resolved_host_body_path = resolved_host["host_body_path"]
        resolved_tool_body_path = resolved_host["tool_body_path"]
        resolved_host_local = resolved_host["host_local"]
        resolved_tool_local = resolved_host["tool_local"]

        host_player_index = int(resolved_host["host_player_index"])
        host_role_id = int(resolved_host["host_role_id"])
        host_meta = resolved_host["host_meta"]
        host_label = str(resolved_host["host_label"])
        host_local = resolved_host_local
        tool_local = resolved_tool_local

        host_body_idx = int(host_meta["path_body_map"][resolved_host_body_path])
        tool_body_idx = int(tool_meta["path_body_map"][resolved_tool_body_path])
        tool_joint_start = int(tool_meta.get("joint_start", 0))
        tool_joint_end = int(tool_meta.get("joint_end", builder.joint_count))

        internal_joint_names = list(tool_cfg.get("internal_joint_names") or [])
        if not internal_joint_names and tool_template:
            template_internal = tool_template.get("internal_joint_names")
            if template_internal:
                internal_joint_names = list(template_internal)

        # Load the tool's attached-tool action (lazy: only imported when a
        # level actually uses this object pattern).
        action = create_tool_action(pattern_str)
        if action is not None:
            action.configure(tool_cfg, tool_template)

        metadata = collect_tool_mount_metadata(
            builder=builder,
            host_body_idx=host_body_idx,
            tool_body_idx=tool_body_idx,
            tool_joint_start=tool_joint_start,
            tool_joint_end=tool_joint_end,
            path_joint_map=tool_meta.get("path_joint_map") or {},
            internal_joint_names=internal_joint_names,
        )

        if action is not None:
            action.resolve_mount_refs(
                builder=builder,
                path_joint_map=tool_meta.get("path_joint_map") or {},
                path_body_map=tool_meta.get("path_body_map") or {},
                tool_joint_start=tool_joint_start,
                tool_joint_end=tool_joint_end,
                tool_body_idx=int(tool_body_idx),
            )

        mount_axis = tuple(
            float(v) for v in (tool_cfg.get("mount_joint_axis") or _MOUNT_SCHEMA_DEFAULTS.mount_joint_axis)
        )
        mount_limits = tool_cfg.get("mount_joint_limits") or _MOUNT_SCHEMA_DEFAULTS.mount_joint_limits
        mount_joint_type = str(
            tool_cfg.get("mount_joint_type") or _MOUNT_SCHEMA_DEFAULTS.mount_joint_type
        )
        tool_body_indices = sorted(
            {int(v) for v in (tool_meta.get("path_body_map") or {}).values()}
        )

        build_result = build_tool_mount_joint(
            builder,
            host_body_idx=metadata.host_body_idx,
            tool_root_body_idx=metadata.tool_root_body_idx,
            host_anchor_local=host_local,
            tool_anchor_local=tool_local,
            mount_joint_type=mount_joint_type,
            mount_axis=mount_axis,
            mount_limits=mount_limits,
            label=f"mount_{tool_key}",
            solver_type=solver_type,
        )

        registry.register(
            ToolMountRecord(
                tool_key=tool_key,
                host_player_index=host_player_index,
                host_role_object_id=host_role_id,
                tool_role_object_id=tool_role_id,
                host_body_idx=metadata.host_body_idx,
                tool_body_idx=metadata.tool_body_idx,
                tool_root_body_idx=metadata.tool_root_body_idx,
                tool_free_joint_idx=metadata.tool_free_joint_idx,
                tool_internal_joint_idxs=list(metadata.tool_internal_joint_idxs),
                tool_pattern=str(pattern_str or ""),
                start_attached=bool(tool_cfg.get("start_attached", False)),
                host_anchor_local=host_local,
                tool_anchor_local=tool_local,
                mount_axis=mount_axis,
                mount_yaw_limits=(float(mount_limits[0]), float(mount_limits[1])),
                proximity_threshold=float(
                    tool_cfg.get("proximity_threshold")
                    or _MOUNT_SCHEMA_DEFAULTS.proximity_threshold
                ),
                proximity_height_threshold=float(
                    tool_cfg.get("proximity_height_threshold")
                    or _MOUNT_SCHEMA_DEFAULTS.proximity_height_threshold
                ),
                tool_body_indices=tool_body_indices,
                mount_joint_idx=build_result.mount_joint_idx,
                mount_eq_idx=build_result.mount_eq_idx,
                mount_joint_dof_idx=build_result.mount_joint_dof_idx,
                mount_joint_coord_idx=build_result.mount_joint_coord_idx,
                mount_joint_type=build_result.mount_joint_type,
                uses_weld_fallback=build_result.uses_weld_fallback,
                slot_index=slot_index,
                action=action,
            )
        )
        slot_index += 1
        mount_desc = (
            f"weld_eq={build_result.mount_eq_idx}"
            if build_result.uses_weld_fallback
            else f"joint={build_result.mount_joint_idx} dof={build_result.mount_joint_dof_idx}"
        )
        print(
            f"[ToolMount] '{tool_key}': host={host_label} tool={tool_label} "
            f"root_body={metadata.tool_root_body_idx} free_joint={metadata.tool_free_joint_idx} "
            f"{mount_desc} (disabled until U attach)"
            + (f" [bound to player {resolved_host['host_player_index']}]" if tool_cfg.get("host_player_id") else "")
        )

    level.mount_joint_registry = registry
    return registry


def _resolve_tool_role_id(tools, tool_index: int) -> int:
    if tools is None:
        raise RuntimeError("Tool role was not instantiated but tool_configs is non-empty")
    if 0 <= tool_index < len(tools.index_obj_role):
        return tools.index_obj_role[tool_index]
    raise KeyError(f"Could not resolve tool role object id for tool index '{tool_index}'")


def _resolve_host_player_candidates(
    tool_cfg: dict,
    *,
    player_configs: List[dict],
    num_players: int,
    tool_key: str,
) -> List[int]:
    """Resolve the host player candidate indices for a tool.

    ``host_player_id`` (unique player ``id`` / object ID) takes priority over the
    legacy ``host_player_index``. When ``host_player_id`` is set the tool is
    explicitly bound to that single player.
    """
    host_player_id = tool_cfg.get("host_player_id")
    if host_player_id:
        for index, cfg in enumerate(player_configs):
            cfg_id = str(cfg.get("id") or cfg.get("name") or "")
            if cfg_id == str(host_player_id):
                return [index]
        raise KeyError(
            f"Tool '{tool_key}' host_player_id={host_player_id!r} does not match any "
            f"player_configs id. Available: {sorted(str(c.get('id') or c.get('name') or '') for c in player_configs)}"
        )
    return _normalize_optional_int_candidates(
        tool_cfg.get("host_player_index"),
        num_players=num_players,
    )
