from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from PyQt6.QtWidgets import QApplication

from script.role.abilities.ability import Ability
from script.role.base_role import BaseRole
from script.viewer_controls import format_keys, DebugGeometryConfig

from .control_bridge import ControlBridge
from .inspector_window import InspectorWindow
from .metadata import InspectorCatalog, ObjectInspectorSpec
from .raycast_select import RaycastSelector

if TYPE_CHECKING:
    from script.custom_viewergl import CustomViewerGL
    from script.game import Game


@dataclass(frozen=True)
class DebugGeometryInstance:
    global_body_idx: int
    shape_idx: int = -1
    forward_local: tuple[float, float, float] = (1.0, 0.0, 0.0)
    # 0 = body_q (+ shape offset), 1 = shoot.py forward formula
    forward_convention: int = 0


class ObjectInspectorPlugin:
    def __init__(self):
        self._app: Optional[QApplication] = None
        self._window: Optional[InspectorWindow] = None
        self._game: Optional["Game"] = None
        self._viewer: Optional["CustomViewerGL"] = None
        self._catalog: Optional[InspectorCatalog] = None
        self._bridge: Optional[ControlBridge] = None
        self._selector: Optional[RaycastSelector] = None
        self._debug_geometry_cfg: Optional[DebugGeometryConfig] = None
        self._visible = False
        self._pending_impulse: Optional[dict] = None

    def attach(self, game: "Game", viewer: "CustomViewerGL"):
        self._game = game
        self._viewer = viewer
        self._debug_geometry_cfg = viewer.viewer_controls_cfg.debug_geometry
        self._catalog = InspectorCatalog.build_from_game(game)
        self._bridge = ControlBridge(game)
        self._selector = RaycastSelector(game.physics_manager.model, game.physics_manager.device)
        self._ensure_qt()
        labels = self._catalog.list_catalog_keys()
        self._window.set_labels(labels)
        self._window.set_label_changed_callback(self._on_label_selected)
        self._window.set_world_changed_callback(self._on_world_changed)
        self._window.set_impulse_callback(self._on_impulse)
        self._window.set_gravity_changed_callback(self._on_gravity_changed)
        self._window.set_camera_move_speed_changed_callback(self._on_camera_move_speed_changed)
        if labels:
            self._select_catalog_key(labels[0], 0)
        game.physics_manager.pre_substep_callback = self._on_pre_substep
        self._window.setup_controls_tab(
            simulation_control=viewer.simulation_control,
            viewer_controls_cfg=viewer.viewer_controls_cfg,
            gameplay_bindings=self._collect_gameplay_bindings(game),
            initial_gravity=self._bridge.read_gravity() if self._bridge else None,
            initial_camera_move_speed=float(getattr(viewer, "_cam_speed", 4.0)),
            show_role_name_labels=viewer.show_role_name_labels,
            on_show_role_name_labels_changed=self._on_show_role_name_labels_changed,
        )
        if self._bridge.has_commands():
            self._window.set_command_labels(list(game.level.command_labels))
        self._window.setup_debug_geometry_controls(
            config=viewer.viewer_controls_cfg.debug_geometry,
            on_changed=self._on_debug_geometry_style_changed,
        )

    def _on_show_role_name_labels_changed(self, checked: bool):
        if self._viewer is not None:
            self._viewer.show_role_name_labels = checked

    def _on_debug_geometry_style_changed(self, length: float, width: float):
        if self._viewer is not None:
            self._viewer.set_debug_geometry_style(length=length, width=width)

    def apply_command_pins(self):
        if not self._bridge or not self._window or not self.is_visible or not self._game:
            return
        if not self._bridge.has_commands():
            return
        self._window.flush_pinned_storage()
        for world_idx, values, pinned_dims in self._window.iter_stored_command_pins():
            self._bridge.apply_command_pins(world_idx, values, pinned_dims)

    def _collect_gameplay_bindings(self, game: "Game") -> List[dict]:
        if Ability._default_configs is None:
            Ability._initialize_class_assets()

        bindings: List[dict] = []
        from script.role.base_role import BaseRole
        from script.role.controller_utils import normalize_controller

        for index_player in game.players.index_obj_role:
            params = BaseRole._object_game_params[index_player]
            controller = normalize_controller(params.get("controller"))
            if controller != "Human":
                continue
            for ability_idx in game.players.get_player_abilities(index_player):
                ability = game.players.abilities_instance_list[ability_idx]
                cfg = Ability._default_configs.root.get(ability.ability_name)
                if cfg is None:
                    continue
                key_names: List[str] = []
                for key_list in cfg.key.keyboard.values():
                    key_names.extend(key_list)
                for key_list in cfg.key.mouse.values():
                    key_names.extend(key_list)
                if not key_names:
                    continue
                description = cfg.action_space.description or ability.ability_name
                bindings.append({
                    "keys": format_keys(key_names),
                    "description": f"{ability.ability_name}: {description}",
                })
            break
        return bindings

    def _on_pre_substep(self, substep_idx: int):
        if self.is_visible:
            self.apply_pinned_controls(substep_idx=substep_idx)

    def _ensure_qt(self):
        if QApplication.instance() is None:
            self._app = QApplication(sys.argv if hasattr(sys, "argv") else [])
        else:
            self._app = QApplication.instance()
        if self._window is None:
            self._window = InspectorWindow()

    def toggle_window(self):
        self._ensure_qt()
        if self._window is None:
            return
        if self._window.isVisible():
            self._window.hide()
            self._visible = False
        else:
            self._window.show()
            self._window.raise_()
            self._visible = True
            self.refresh_from_sim()

    @property
    def is_visible(self) -> bool:
        return self._visible and self._window is not None and self._window.isVisible()

    def tick(self):
        if self._app is None:
            return
        self._app.processEvents()
        if self._window is not None and self._viewer is not None and self._game is not None:
            self._window.sync_controls_status(
                game_over=self._game.game_over,
                show_role_name_labels=self._viewer.show_role_name_labels,
            )
        if self.is_visible:
            self.refresh_from_sim()

    def apply_pinned_controls(self, substep_idx: int = 0):
        if not self._bridge or not self._window or not self.is_visible or not self._game or not self._catalog:
            return
        self._window.flush_pinned_storage()
        self._game.physics_manager.clear_inspector_body_f()

        for spec, world, body, values, pinned in self._window.iter_stored_body_pins(self._catalog):
            self._bridge.apply_body_pinned(spec, world, body, values, pinned)

        for spec, world, joint, values, pinned in self._window.iter_stored_joint_pins(self._catalog):
            self._bridge.apply_joint_pinned(spec, world, joint, values, pinned)

        spec = self._window.current_spec()
        if spec is None:
            return
        world = self._window.current_world()
        body = self._window.current_body()
        joint = self._window.current_joint()
        if self._pending_impulse and substep_idx == 0:
            impulse = self._pending_impulse
            self._pending_impulse = None
            if impulse.get("joint") and joint is not None:
                self._bridge.apply_joint_impulse(spec, world, joint, self._window.get_joint_state().torque)
            elif body is not None:
                self._bridge.apply_body_impulse(
                    spec,
                    world,
                    body,
                    self._window.get_body_state(),
                    apply_force=impulse.get("force", False),
                    apply_torque=impulse.get("torque", False),
                )

    def is_rl_action_override_enabled(self) -> bool:
        return (
            self._window is not None
            and self.is_visible
            and self._window.is_rl_action_enabled()
        )

    def apply_rl_action_pins(self, actions_wp):
        if not self._bridge or not self._window or not self.is_visible or not self._game or not self._catalog:
            return
        if not self._window.is_rl_action_enabled():
            return
        self._window.flush_pinned_storage()
        current = self._window.current_spec()
        if current is not None and current.player_action is not None:
            live = self._window.get_rl_action_values()
            if live:
                self._bridge.apply_rl_action_pins_to_buffer(
                    actions_wp,
                    current.player_action,
                    self._window.current_world(),
                    current.local_role_idx,
                    live,
                    list(live.keys()),
                )
        for spec, world_idx, values in self._window.iter_stored_rl_actions(self._catalog):
            pinned_dims = list(values.keys())
            if not pinned_dims:
                continue
            self._bridge.apply_rl_action_pins_to_buffer(
                actions_wp,
                spec.player_action,
                world_idx,
                spec.local_role_idx,
                values,
                pinned_dims,
            )

    def get_tool_attachment_override_values(self, global_role_object_id: int) -> Optional[List[float]]:
        """Return Tool_attachment RL dims when inspector override is enabled."""
        if (
            not self._window
            or not self._catalog
            or not self.is_visible
            or not self._game
            or not self._window.is_rl_action_enabled()
        ):
            return None

        num_objects_env = self._game.num_objects_env
        local_role_idx = int(global_role_object_id) % num_objects_env
        world_idx = int(global_role_object_id) // num_objects_env
        spec = self._catalog.get_by_role(int(global_role_object_id), num_objects_env)
        if spec is None or spec.player_action is None:
            return None

        tool_dims = []
        for ability in spec.player_action.abilities:
            if ability.ability_name == "Tool_attachment":
                tool_dims.extend(ability.dims)
        if not tool_dims:
            return None

        current = self._window.current_spec()
        if (
            current is not None
            and current.local_role_idx == local_role_idx
            and self._window.current_world() == world_idx
        ):
            live = self._window.get_rl_action_values()
            return [float(live.get(dim.dim_index, 0.0)) for dim in tool_dims]

        for iter_spec, widx, stored in self._window.iter_stored_rl_actions(self._catalog):
            if iter_spec.local_role_idx != local_role_idx or widx != world_idx:
                continue
            return [float(stored.get(dim.dim_index, 0.0)) for dim in tool_dims]

        return None

    def select_from_ray(self, x: float, y: float):
        if not self._game or not self._viewer or not self._selector or not self._catalog:
            return
        pm = self._game.physics_manager
        if pm.shape_to_role_np is None or pm.state_0 is None:
            return
        ray_start, ray_dir = self._viewer.camera.get_world_ray(x, y)
        world_offsets = getattr(self._viewer, "world_offsets", None)
        visible_worlds_mask = None
        picking = getattr(self._viewer, "picking", None)
        if picking is not None:
            visible_worlds_mask = getattr(picking, "visible_worlds_mask", None)
        result = self._selector.query(
            state=pm.state_0,
            ray_start=ray_start,
            ray_dir=ray_dir,
            shape_to_role=pm.shape_to_role_np,
            world_offsets=world_offsets,
            num_objects_env=pm.num_objects_env,
            role_labels=pm.role_object_labels,
            visible_worlds_mask=visible_worlds_mask,
        )
        if result.local_role_idx >= 0:
            self._select_role(result.local_role_idx, result.world_idx)
            if not self.is_visible:
                self.toggle_window()

    def _select_catalog_key(self, catalog_key: str, world_idx: int):
        if not self._catalog or not self._window:
            return
        current = self._window.current_spec()
        if (
            current is not None
            and current.catalog_key == catalog_key
            and self._window.current_world() == world_idx
        ):
            self.refresh_from_sim(resync_rl=True)
            return
        spec = self._catalog.get_by_catalog_key(catalog_key)
        if spec is None:
            return
        self._window.blockSignals(True)
        if self._window.label_combo.findText(catalog_key) >= 0:
            self._window.label_combo.setCurrentText(catalog_key)
        self._window.set_spec(spec, world_idx)
        self._window.blockSignals(False)
        self.refresh_from_sim(resync_rl=True)

    def _select_role(self, local_role_idx: int, world_idx: int):
        if not self._catalog or not self._window:
            return
        spec = self._catalog.get_by_role(local_role_idx, self._game.num_objects_env if self._game else 1)
        if spec is None:
            return
        self._select_catalog_key(spec.catalog_key, world_idx)

    def _on_label_selected(self, catalog_key: str, world_idx: int):
        self._select_catalog_key(catalog_key, world_idx)

    def _on_world_changed(self, world_idx: int):
        if not self._window or not self._catalog:
            return
        spec = self._window.current_spec()
        if spec is None:
            return
        self._window.set_spec(spec, world_idx)
        self.refresh_from_sim(resync_rl=True)

    def _on_impulse(self, force: bool = False, torque: bool = False, joint: bool = False):
        self._pending_impulse = {"force": force, "torque": torque, "joint": joint}

    def _on_gravity_changed(self, gx: float, gy: float, gz: float):
        if self._bridge:
            self._bridge.set_gravity([gx, gy, gz])

    def _on_camera_move_speed_changed(self, speed: float):
        if self._viewer is not None:
            self._viewer._cam_speed = float(speed)

    def build_role_name_tag_entries(self) -> List[tuple[str, int, int]]:
        """Return (display_name, global_body_idx, world_idx) for each role instance."""
        if not self._bridge or not self._catalog or not self._game:
            return []

        game = self._game
        name_list = game.name_list
        num_objects_env = game.num_objects_env
        body_count = game.physics_manager.model.body_count
        entries: List[tuple[str, int, int]] = []

        for catalog_key in self._catalog.list_catalog_keys():
            spec = self._catalog.get_by_catalog_key(catalog_key)
            if spec is None or not spec.bodies:
                continue
            base_body = next((b for b in spec.bodies if b.is_base_body), spec.bodies[0])

            for world_idx in range(game.num_env):
                global_role_id = world_idx * num_objects_env + spec.local_role_idx
                if global_role_id < len(name_list) and name_list[global_role_id]:
                    display_name = str(name_list[global_role_id])
                else:
                    display_name = spec.catalog_key

                global_body_idx = self._bridge._resolve_body_global_index(
                    spec, base_body, world_idx
                )
                if global_body_idx < 0 or global_body_idx >= body_count:
                    continue
                entries.append((display_name, global_body_idx, world_idx))

        return entries

    def _resolve_primary_shape_idx(self, global_body_idx: int) -> int:
        if not self._game:
            return -1
        model = self._game.physics_manager.model
        shape_body = model.shape_body.numpy()
        for shape_idx, body_idx in enumerate(shape_body):
            if int(body_idx) == global_body_idx:
                return int(shape_idx)
        return -1

    def _resolve_debug_geometry_forward_local(
        self,
        spec: ObjectInspectorSpec,
        world_idx: int,
    ) -> tuple[float, float, float]:
        default = (1.0, 0.0, 0.0)
        if self._debug_geometry_cfg is not None:
            default = self._debug_geometry_cfg.default_forward_local
        if not self._game:
            return default
        num_objects_env = self._game.num_objects_env
        global_role_id = world_idx * num_objects_env + spec.local_role_idx

        registry = getattr(self._game.level, "mount_joint_registry", None)
        if registry is not None:
            forward = registry.get_tool_forward_local(global_role_id)
            if forward is not None:
                return forward
        return default

    def _resolve_debug_geometry_convention(self, spec: ObjectInspectorSpec, world_idx: int) -> int:
        if not self._game:
            return 0
        prefixes = (
            self._debug_geometry_cfg.shoot_forward_shape_key_prefixes
            if self._debug_geometry_cfg is not None
            else ["rigid_"]
        )
        num_objects_env = self._game.num_objects_env
        global_role_id = world_idx * num_objects_env + spec.local_role_idx
        if global_role_id < len(BaseRole._object_game_params):
            shape_key = str(BaseRole._object_game_params[global_role_id].get("shape_key", ""))
            for prefix in prefixes:
                if shape_key.startswith(prefix):
                    return 1
        return 0

    def get_debug_geometry_instances(self) -> List[DebugGeometryInstance]:
        if not self._bridge or not self._window or not self._catalog or not self._game:
            return []
        body_count = self._game.physics_manager.model.body_count
        instances: List[DebugGeometryInstance] = []
        for key, enabled in self._window.iter_debug_geometry_flags():
            if not enabled:
                continue
            parsed = InspectorWindow._parse_env_suffix_storage_key(key)
            if parsed is None:
                continue
            catalog_key, world_idx, body_name = parsed
            spec = self._catalog.get_by_catalog_key(catalog_key)
            if spec is None:
                continue
            body = next((b for b in spec.bodies if b.display_name == body_name), None)
            if body is None:
                continue
            global_idx = self._bridge._resolve_body_global_index(spec, body, world_idx)
            if not (0 <= global_idx < body_count):
                continue
            instances.append(
                DebugGeometryInstance(
                    global_body_idx=global_idx,
                    shape_idx=self._resolve_primary_shape_idx(global_idx),
                    forward_local=self._resolve_debug_geometry_forward_local(spec, world_idx),
                    forward_convention=self._resolve_debug_geometry_convention(spec, world_idx),
                )
            )
        return instances

    def refresh_from_sim(self, resync_rl: bool = False):
        if not self._bridge or not self._window or not self.is_visible:
            return
        self._window.set_gravity_values(self._bridge.read_gravity())
        spec = self._window.current_spec()
        if spec is None:
            return
        world = self._window.current_world()
        body = self._window.current_body()
        joint = self._window.current_joint()
        if body is not None:
            state = self._bridge.read_body(spec, world, body)
            self._window.set_body_state(state)
        if joint is not None:
            jstate = self._bridge.read_joint(spec, world, joint)
            self._window.set_joint_state(jstate)
        if spec.player_action is not None and not self._window.is_rl_action_enabled():
            rl_values = self._bridge.read_rl_actions(
                spec.player_action,
                world,
                spec.local_role_idx,
            )
            self._window.set_rl_action_values(rl_values, force=resync_rl)
            if resync_rl:
                self._window._save_rl_action_values()
        if self._bridge.has_commands():
            cmd_values = self._bridge.read_commands(world)
            self._window.set_command_values(cmd_values, force=False)
