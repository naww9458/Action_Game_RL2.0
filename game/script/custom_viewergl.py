from __future__ import annotations

import math
import newton
import warp as wp
import numpy as np
import pyglet
import time

from script.game_config import GameConfig
from pyglet import gl
from pyglet import shapes # 用於繪製背景
from pyglet.window import key, mouse
from newton.viewer import ViewerGL
from newton import State
from queue import Queue, Full
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from script.game import Game

from script.viewer_plugins.object_inspector import ObjectInspectorPlugin
from script.viewer_controls import (
    SimulationControl,
    create_keyboard_mapping,
    load_viewer_controls,
)

class CustomViewerGL(ViewerGL):
    def __init__(
        self,
        event_is_window_setup_ready: 'multiprocessing.Event',
        human_input_queue: 'Queue',
        follow_body_index=None,
        num_envs_display: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_envs_display = max(1, num_envs_display)
        self.renderer.window.push_handlers(self)
        self._ui_batch = pyglet.graphics.Batch()

        # 背景遮罩
        self._ui_bg = shapes.Rectangle(x=10, y=10, width=450, height=100, color=(0, 0, 0), batch=self._ui_batch)
        self._ui_bg.opacity = 150

        self.font_size_system_info = 18
        self.font_size_player_info = 15
        # 步數標籤
        self._label_step = pyglet.text.Label('', font_name='Consolas', font_size=self.font_size_system_info, # bold=True,
                                            x=20, y=20, color=(0, 255, 255, 255), batch=self._ui_batch)
        self._label_fps = pyglet.text.Label('', font_name='Consolas', font_size=self.font_size_system_info, # bold=True,
                                            x=20, y=20, color=(0, 255, 255, 255), batch=self._ui_batch)
        self._label_player_color = pyglet.text.Label('Human: red, RL: green, Bot: blue', font_name='Consolas', font_size=self.font_size_system_info, # bold=True,
                                            x=20, y=20, color=(0, 255, 255, 255), batch=self._ui_batch)

        self._label_follow_body_index = pyglet.text.Label('', font_name='Consolas', font_size=self.font_size_system_info, # bold=True,
                                            x=20, y=20, color=(0, 0, 0), batch=self._ui_batch)

        self._label_attach_prompt = pyglet.text.Label(
            '',
            font_name='Consolas',
            font_size=self.font_size_system_info + 2,
            x=20,
            y=20,
            color=(255, 220, 80, 255),
            batch=self._ui_batch,
        )

        # 玩家數據標籤池
        self._player_labels = []
        self._last_health = []
        self._last_rewards = []
        self._last_step = -1
        self.follow_role_index = follow_body_index
        self.follow_body_index_input_cache = ""
        self.viewer_controls_cfg = load_viewer_controls()
        viewer_defaults = self.viewer_controls_cfg.defaults
        self.free_view_mode = bool(viewer_defaults.get("free_view_mode", True))
        self._mouse_look_wanted = False
        self._viewer_window_focused = True
        self.show_role_name_labels = bool(viewer_defaults.get("show_role_name_labels", True))
        self._role_name_label_height = float(viewer_defaults.get("role_name_label_height", 2.0))
        self._role_name_tag_entries: list[tuple[str, int, int]] = []
        self._role_name_tag_labels: list = []
        self._last_body_q_np: Optional[np.ndarray] = None
        debug_geom_cfg = self.viewer_controls_cfg.debug_geometry
        self._debug_geom_line_length = float(debug_geom_cfg.line_length)
        self._debug_geom_line_width = float(debug_geom_cfg.line_width)
        self._debug_geom_extension_color = debug_geom_cfg.extension_line_color
        self._debug_geom_circle_color = debug_geom_cfg.circle_line_color
        self.possess_offsets: list[tuple[float, float, float]] = []
        self.sim_time = 0.0

        # 相機相關屬性
        self.is_mouse_exclusive = False
        self._sync_mouse_exclusive()
        self.mouse_sensitivity = 0.05
        self.look_yaw = 0.0
        self.look_pitch = 0.0

        self.mouse_buttons = np.zeros(shape=4, dtype=np.int32)
        self.keyboard_keys = np.zeros(shape=65536, dtype=np.int32)
        self.human_input_queue = human_input_queue
        self.simulation_control = SimulationControl.from_defaults(self.viewer_controls_cfg)
        self.health_ref = None
        self._binding_symbols: dict[str, set[int]] = {}
        self._camera_move_symbols: dict[str, set[int]] = {}
        self._build_binding_lookup()
        mapping = create_keyboard_mapping()
        manual_reset_binding = self.viewer_controls_cfg.binding_by_id("manual_reset")
        self.human_input_queue.put({
            "Keyboard_Mappings": mapping,
            "Viewer_Controls_Config": {
                "manual_reset_keys": manual_reset_binding.keys if manual_reset_binding else ["y"],
            },
        })
        event_is_window_setup_ready.set()
        self.game: 'Game' = None
        self.num_players: int = None
        
        # 如果沒有這行，那麽按一下鍵盤 self.on_key_press 會運行兩次，導致 self.is_mouse_exclusive 的值被切換兩次，最終還是保持原來的值不變
        # 還有就是設置 self.follow_body_index_input_cache 的時候按一下 1 字串就變成 11，導致無法設置攝像機跟隨物件
        # 運行兩次是因爲一次是 ViewerGL._init_ 的時候通過 self.renderer.register_key_press(self.on_key_press) 注冊的，還有一次不知道怎麽來的反正就有兩次
        # 可能導致未知影響，只是目前測試還沒發現問題
        self.renderer._key_callbacks = []
        # 使用自訂 on_mouse_drag，避免父類更新 camera.yaw/pitch 後被 look_yaw/look_pitch 覆寫
        self.renderer._mouse_drag_callbacks = []
        self.renderer.register_mouse_drag(self.on_mouse_drag)

        # Newton 1.4 moved camera smoothing state off ViewerGL onto ViewerGui;
        # keep local copies for CustomViewerGL follow / free-camera logic.
        self._cam_vel = np.zeros(3, dtype=np.float32)
        self._cam_damp_tau = 0.083
        self._cam_speed = float(viewer_defaults.get("camera_move_speed", 4.0))
        self._pick_depth_scroll_sensitivity = float(
            viewer_defaults.get("pick_depth_scroll_sensitivity", 0.15)
        )
        self._pick_depth_min = float(viewer_defaults.get("pick_depth_min", 0.2))
        self._camera_orbit_sensitivity = float(
            viewer_defaults.get("camera_orbit_sensitivity", 0.1)
        )
        self._camera_dolly_drag_sensitivity = float(
            viewer_defaults.get("camera_dolly_drag_sensitivity", 0.01)
        )
        self._current_fps = 0.0

        self.object_inspector = ObjectInspectorPlugin()
        self._mouse_press_pos = None
        self._action_view_batch = None
        self._action_view_shapes: list = []

    def _build_binding_lookup(self):
        camera_move_ids = {
            "move_forward", "move_back", "move_left", "move_right", "move_up", "move_down",
        }
        for binding in self.viewer_controls_cfg.bindings:
            symbols = set(binding.symbols)
            self._binding_symbols[binding.id] = symbols
            if binding.id in camera_move_ids:
                self._camera_move_symbols[binding.id] = symbols

    def _symbol_in_binding(self, symbol: int, binding_id: str) -> bool:
        return symbol in self._binding_symbols.get(binding_id, set())

    def _is_any_key_down(self, binding_id: str) -> bool:
        symbols = self._binding_symbols.get(binding_id, set())
        return any(self.renderer.is_key_down(sym) for sym in symbols)

    def setup(self, game: 'Game'):
        self.game = game
        self.num_players = game.num_players
        self.num_objects_total = game.num_objects_total
        # 初始化玩家 Label 池
        for i in range(self.game.num_players):
            lbl = pyglet.text.Label('', font_name='Consolas', font_size=self.font_size_player_info,
                                    x=20, y=0, color=(255, 255, 0, 255), batch=self._ui_batch)
            self._player_labels.append(lbl)
            self._last_health.append(-1)
            self._last_rewards.append(-1.0)

        num_segs = int(self.viewer_controls_cfg.debug_geometry.num_segments)
        debug_geom_cfg = self.viewer_controls_cfg.debug_geometry
        self._debug_num_segs = num_segs
        self.SURFACE_OFFSET = wp.array(
            data=[debug_geom_cfg.surface_offset], dtype=wp.float32, device=GameConfig.DEVICE
        )
        self.LINE_LENGTH = wp.array(
            data=[self._debug_geom_line_length], dtype=wp.float32, device=GameConfig.DEVICE
        )
        self.CIRCLE_RADIUS = wp.array(
            data=[debug_geom_cfg.circle_radius], dtype=wp.float32, device=GameConfig.DEVICE
        )
        self.CIRCLE_LIFT = wp.array(
            data=[debug_geom_cfg.circle_lift], dtype=wp.float32, device=GameConfig.DEVICE
        )
        self.num_segments = wp.array(data=[num_segs], dtype=wp.int32, device=GameConfig.DEVICE)

        self.object_inspector.attach(game, self)

        self.geo_type = newton.GeoType.SPHERE
        shape_scale_all = self.model.shape_scale.numpy()
        index_players = self.game.players.index_obj_role

        # i-1 because index 0 in shape_scale_all is ground
        self.geo_scale = [scale[0] for i, scale in enumerate(shape_scale_all) if i-1 in index_players]

        self._configure_display_envs(game.num_env)
        self.camera.fov = 100.0
        self._build_role_name_label_pool()

    def _build_role_name_label_pool(self):
        self._role_name_tag_entries = self.object_inspector.build_role_name_tag_entries()
        self._role_name_tag_labels = [
            pyglet.text.Label(
                "",
                font_name="Consolas",
                font_size=14,
                x=0,
                y=0,
                color=(255, 255, 255, 230),
                anchor_x="center",
                anchor_y="bottom",
                batch=self._ui_batch,
            )
            for _ in self._role_name_tag_entries
        ]

    def _world_up_offset(self, height: float) -> tuple[float, float, float]:
        if self.camera.up_axis == 0:
            return (height, 0.0, 0.0)
        if self.camera.up_axis == 2:
            return (0.0, 0.0, height)
        return (0.0, height, 0.0)

    def _world_to_screen(self, world_pos: tuple[float, float, float]) -> Optional[tuple[float, float]]:
        window_w, window_h = self.renderer.window.get_size()
        if window_w <= 0 or window_h <= 0:
            return None
        self.camera.update_screen_size(window_w, window_h)
        view_mat = np.array(self.camera.get_view_matrix(), dtype=np.float64).reshape((4, 4), order="F")
        proj_mat = np.array(self.camera.get_projection_matrix(), dtype=np.float64).reshape((4, 4), order="F")
        point = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0], dtype=np.float64)
        clip = proj_mat @ (view_mat @ point)
        if clip[3] <= 1e-6:
            return None
        ndc = clip / clip[3]
        if ndc[2] < -1.0 or ndc[2] > 1.0:
            return None
        x = (ndc[0] + 1.0) * 0.5 * window_w
        y = (ndc[1] + 1.0) * 0.5 * window_h
        return float(x), float(y)

    def _update_role_name_labels(self):
        if not self.show_role_name_labels or self._last_body_q_np is None:
            for label in self._role_name_tag_labels:
                label.text = ""
            return

        display_envs = min(self.num_envs_display, self.game.num_env)
        up_offset = self._world_up_offset(self._role_name_label_height)
        body_q = self._last_body_q_np

        for idx, (display_name, body_idx, world_idx) in enumerate(self._role_name_tag_entries):
            label = self._role_name_tag_labels[idx]
            if world_idx >= display_envs:
                label.text = ""
                continue
            if body_idx < 0 or body_idx >= len(body_q):
                label.text = ""
                continue
            pos = body_q[body_idx, 0:3]
            world_pos = (
                float(pos[0]) + up_offset[0],
                float(pos[1]) + up_offset[1],
                float(pos[2]) + up_offset[2],
            )
            screen = self._world_to_screen(world_pos)
            if screen is None:
                label.text = ""
                continue
            label.text = display_name
            label.x = int(screen[0])
            label.y = int(screen[1])

    def set_debug_geometry_style(
        self,
        length: Optional[float] = None,
        width: Optional[float] = None,
    ):
        spin_cfg = self.viewer_controls_cfg.debug_geometry
        if length is not None:
            self._debug_geom_line_length = max(
                spin_cfg.length_spin.min, float(length)
            )
            self.LINE_LENGTH.assign([self._debug_geom_line_length])
        if width is not None:
            self._debug_geom_line_width = max(
                spin_cfg.width_spin.min, float(width)
            )

    def _configure_display_envs(self, num_env: int):
        display_envs = min(self.num_envs_display, num_env)
        if display_envs <= 1:
            self.set_world_offsets((0, 0, 0))
            return

        self.set_world_offsets((0.0, 0.0, 0.0))
        if display_envs >= num_env or self.world_offsets is None:
            return

        offsets = self.world_offsets.numpy().copy()
        for world_idx in range(display_envs, len(offsets)):
            offsets[world_idx] = np.array([0.0, 0.0, -10000.0], dtype=np.float32)
        self.world_offsets = wp.array(offsets, dtype=wp.vec3, device=GameConfig.DEVICE)
        if hasattr(self, "picking") and self.picking is not None:
            self.picking.world_offsets = self.world_offsets

    def _render_ui_overlay(self, player_health):
        if self.game is None: return

        window_w, window_h = self.renderer.window.get_size()
        self._label_step.y = window_h - (self.font_size_system_info * 2)
        self._label_fps.y = self._label_step.y - (self.font_size_system_info + 10)
        self._label_player_color.y = self._label_fps.y - (self.font_size_system_info + 10)
        offset_player_label = self._label_player_color.y - (self.font_size_system_info + 10)

        if self.follow_role_index is not None:
            mode = "Free View" if self.free_view_mode else "Full Follow"
            label_text_follow_body_index = (
                f"Following role: {self.follow_role_index} [{mode}] (Left Alt to toggle)"
            )
        else:
            label_text_follow_body_index = f"Enter role index to follow: {self.follow_body_index_input_cache}"

        self._label_follow_body_index.text = label_text_follow_body_index
        self._label_follow_body_index.x = window_w - self._label_follow_body_index.content_width - 20
        self._label_follow_body_index.y = window_h - (self.font_size_system_info * 2)

        attach_prompt = ""
        if self.game is not None and self.follow_role_index is not None:
            follow_targets = self._resolve_follow_targets()
            host_role_object_id = follow_targets[0] if follow_targets is not None else None
            if host_role_object_id is not None:
                for ability in getattr(self.game.players, "abilities_instance_list", []):
                    if hasattr(ability, "show_attach_prompt_for_host"):
                        if ability.show_attach_prompt_for_host(host_role_object_id):
                            attach_prompt = ability.get_prompt_text()
                        break
            else:
                for ability in getattr(self.game.players, "abilities_instance_list", []):
                    if getattr(ability, "show_attach_prompt", False):
                        attach_prompt = ability.get_prompt_text()
                        break
        self._label_attach_prompt.text = attach_prompt
        if attach_prompt:
            self._label_attach_prompt.x = (window_w - self._label_attach_prompt.content_width) // 2
            self._label_attach_prompt.y = window_h // 2

        self._label_step.text = f"STEP: {self.game.current_step} / {self.game.max_episode_step}"
        self._label_fps.text = f"FPS: {self._current_fps}"
        # 更新玩家標籤 (髒檢查)
        display_envs = min(self.num_envs_display, self.game.num_env)
        env_mapping = getattr(self.game.players, "index_obj_role_to_env_mapping", [])
        for i, index_player in enumerate(self.game.players.index_obj_role):
            if env_mapping and env_mapping[i] >= display_envs:
                self._player_labels[i].text = ""
                continue
            curr_hp = player_health[index_player]
            curr_rew = self.game.episode_total_rewards[index_player] or 0.0

            if curr_hp != self._last_health[i] or abs(curr_rew - self._last_rewards[i]) > 0.01:
                name = self.game.name_list[index_player]
                self._player_labels[i].text = f"{name:<2} HP:{curr_hp:<3} Reward:{curr_rew:>8.2f}"
                self._last_health[i] = curr_hp
                self._last_rewards[i] = curr_rew

            self._player_labels[i].y = offset_player_label - (i * (self.font_size_player_info + 5))
        # TODO 100 is hardcode
        self._ui_bg.height = 100 + (self.num_players * (self.font_size_player_info + 5))
        self._ui_bg.y = window_h - self._ui_bg.height - 10
        self._update_role_name_labels()
        # OpenGL 渲染狀態切換
        from pyglet.math import Mat4
        proj = self.renderer.window.projection
        view = self.renderer.window.view
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.renderer.window.projection = Mat4.orthogonal_projection(0, window_w, 0, window_h, -1, 1)
        self.renderer.window.view = Mat4()
        # 極速繪製
        self._ui_batch.draw()
        self._draw_tool_action_overlays(window_w, window_h)
        # 還原狀態
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.renderer.window.projection = proj
        self.renderer.window.view = view

    def _draw_tool_action_overlays(self, window_w: int, window_h: int) -> None:
        """Draw screen-space overlays produced by attached tool actions.

        Generic: every attached ``ToolAction`` may return overlay geometry via
        :meth:`ToolAction.view_overlay`; the action owns *what* to draw (e.g. the
        turret_110mm third-person aim HUD) so this module stays tool-agnostic.
        """
        if self.game is None or self.follow_role_index is None:
            return
        follow_targets = self._resolve_follow_targets()
        if follow_targets is None:
            return
        host_role_object_id = follow_targets[0]
        level = getattr(self.game, "level", None)
        mount_registry = getattr(level, "mount_joint_registry", None) if level is not None else None
        if mount_registry is None:
            return

        overlays: list = []
        for record in mount_registry.records.values():
            action = record.action
            if action is None or not record.attached:
                continue
            overlay = action.view_overlay(
                mount_registry,
                record,
                host_role_object_id=host_role_object_id,
                window_w=float(window_w),
                window_h=float(window_h),
                camera_pitch_deg=float(self.camera.pitch),
                camera_fov_deg=float(getattr(self.camera, "fov", 100.0) or 100.0),
            )
            if overlay is not None:
                overlays.append((action, overlay))

        if not overlays:
            self._action_view_shapes = []
            return

        # Rebuild each frame: shapes positions change with camera pitch.
        self._action_view_batch = pyglet.graphics.Batch()
        kept: list = []
        for action, overlay in overlays:
            result = action.draw_view_overlay(
                overlay,
                shapes_module=shapes,
                batch=self._action_view_batch,
            )
            if result is None:
                continue
            _batch, shapes_keep = result
            if shapes_keep:
                kept.extend(list(shapes_keep))
        self._action_view_shapes = kept
        self._action_view_batch.draw()

    def _compute_follow_camera_position(
        self,
        state_0: State,
        camera_body_index: int,
        role_object_id: int,
    ) -> wp.vec3 | None:
        """World-space follow camera position (body root + rotated possess offset)."""
        target_transform = state_0.body_q.numpy()[camera_body_index]
        pos = target_transform[:3]
        quat = target_transform[3:]
        offset = (0.0, 0.0, 0.0)
        if (
            self.possess_offsets
            and 0 <= role_object_id < len(self.possess_offsets)
        ):
            offset = self.possess_offsets[role_object_id]

        if any(abs(v) > 1e-6 for v in offset):
            q = wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            offset_world = wp.quat_rotate(
                q, wp.vec3(float(offset[0]), float(offset[1]), float(offset[2]))
            )
            pos = (
                float(pos[0]) + float(offset_world[0]),
                float(pos[1]) + float(offset_world[1]),
                float(pos[2]) + float(offset_world[2]),
            )

        return wp.vec3(float(pos[0]), float(pos[1]), float(pos[2]))

    def _resolve_follow_targets(self) -> tuple[int, int] | None:
        """Resolve current follow role index to (role_object_id, camera_body_index)."""
        if self.follow_role_index is None or self.game is None:
            return None
        return self.game.players.resolve_follow_targets(
            self.follow_role_index,
            self.game.physics_manager,
        )

    def render(self, player_health: list, frame_dt, state_0: State, index_player_gpu):

        if self.is_mouse_exclusive:
            self.renderer.window.set_mouse_position(self.renderer.window.width // 2, self.renderer.window.height // 2)

        self.health_ref = player_health
        self._last_body_q_np = state_0.body_q.numpy()
        follow_targets = self._resolve_follow_targets()
        if follow_targets is not None:
            role_object_id, camera_body_index = follow_targets
            cam_pos = self._compute_follow_camera_position(
                state_0, camera_body_index, role_object_id
            )
            if not self.free_view_mode:
                quat = state_0.body_q.numpy()[camera_body_index][3:]
                pitch, yaw = quat_to_pitch_yaw(quat)
                self.look_pitch = float(pitch)
                self.look_yaw = float(yaw)

            self.set_camera(
                pos=cam_pos,
                pitch=float(self.look_pitch),
                yaw=float(self.look_yaw),
            )
        else:
            if self.follow_role_index is not None:
                self.follow_role_index = None
                self.follow_body_index_input_cache = ""
                self.free_view_mode = False
            self.set_camera(pos=self.camera.pos, pitch=float(self.look_pitch), yaw=float(self.look_yaw))

        try:
            role_object_id = follow_targets[0] if follow_targets is not None else None
            input_data = {
                "follow_role_index": self.follow_role_index,
                "follow_body_index": role_object_id,
                "keyboard_keys": self.keyboard_keys,
                "mouse_buttons": self.mouse_buttons,
                "look_yaw": self.look_yaw,
                "look_pitch": self.look_pitch,
                "camera_yaw": float(self.camera.yaw),
                "camera_pitch": float(self.camera.pitch),
                "simulation_control": self.simulation_control.to_queue_payload(),
            }
            self.human_input_queue.put_nowait(input_data)
        except Full:
            pass

        self.render_debug_geometry(all_transforms=state_0.body_q)
        self.begin_frame(self.sim_time)

        xforms_gpu = wp.zeros(self.num_players, dtype=wp.transform, device=GameConfig.DEVICE)
        wp.launch(
            kernel=gather_transforms_kernel,
            dim=self.num_players,
            inputs=[state_0.body_q, index_player_gpu, xforms_gpu],
            device=GameConfig.DEVICE
        )

        # TODO 暫時禁用，因爲現在屬於玩家角色控制的物件不再只有球體, 如果物件是球體，啓用代碼後也沒顯示顔色，那可能要把大小調大一點
        # # render color Classification human, rl agent and bot
        # if len(self.game.color_player_shape_gpu) > 0:
        #     self.log_shapes(name="", geo_type=self.geo_type, geo_scale=self.geo_scale, xforms=xforms_gpu, colors=self.game.color_player_shape_gpu)

        self.log_state(state_0)
        self.end_frame()
        self.sim_time += frame_dt
    
    def render_debug_geometry(self, all_transforms):
        instances = self.object_inspector.get_debug_geometry_instances()
        if not instances:
            self.log_lines(name="debug_extension_line", starts=None, ends=None, colors=None)
            self.log_lines(name="debug_surface_circle", starts=None, ends=None, colors=None)
            return

        num_bodies = len(instances)
        num_segs = self._debug_num_segs
        body_indices = [inst.global_body_idx for inst in instances]
        shape_indices = [inst.shape_idx for inst in instances]
        forward_locals = [inst.forward_local for inst in instances]
        forward_conventions = [inst.forward_convention for inst in instances]
        body_indices_gpu = wp.array(body_indices, dtype=wp.int32, device=GameConfig.DEVICE)
        shape_indices_gpu = wp.array(shape_indices, dtype=wp.int32, device=GameConfig.DEVICE)
        forward_locals_gpu = wp.array(forward_locals, dtype=wp.vec3, device=GameConfig.DEVICE)
        convention_gpu = wp.array(forward_conventions, dtype=wp.int32, device=GameConfig.DEVICE)
        l_start = wp.zeros(num_bodies, dtype=wp.vec3, device=GameConfig.DEVICE)
        l_end = wp.zeros(num_bodies, dtype=wp.vec3, device=GameConfig.DEVICE)
        c_starts = wp.zeros(num_bodies * num_segs, dtype=wp.vec3, device=GameConfig.DEVICE)
        c_ends = wp.zeros(num_bodies * num_segs, dtype=wp.vec3, device=GameConfig.DEVICE)

        wp.launch(
            kernel=calculate_param_debug_geometry_gpu,
            dim=num_bodies,
            inputs=[
                all_transforms,
                self.model.shape_transform,
                body_indices_gpu,
                shape_indices_gpu,
                forward_locals_gpu,
                convention_gpu,
                l_start,
                l_end,
                c_starts,
                c_ends,
                self.SURFACE_OFFSET,
                self.LINE_LENGTH,
                self.CIRCLE_RADIUS,
                self.CIRCLE_LIFT,
                self.num_segments,
            ],
            device=GameConfig.DEVICE,
        )

        self.log_lines(
            name="debug_extension_line",
            starts=l_start,
            ends=l_end,
            colors=self._debug_geom_extension_color,
            width=self._debug_geom_line_width,
        )
        self.log_lines(
            name="debug_surface_circle",
            starts=c_starts,
            ends=c_ends,
            colors=self._debug_geom_circle_color,
            width=self._debug_geom_line_width,
        )
    
    def end_frame(self):
        self.renderer.update()
        now = time.perf_counter()
        dt = max(0.0, min(0.1, now - self._last_time))
        self._last_time = now
        self._update_camera(dt)
        if self.wind is not None:
            self.wind.update(dt)
        if self.renderer.has_exit():
            return

        self.renderer.render(
            self.camera,
            self.objects,
            self.lines,
            self.wireframe_shapes,
            self.arrows,
        )

        if self.gui is not None:
            self.gui.render_frame(update_fps=True)
            self._current_fps = float(getattr(self.gui, "_current_fps", 0.0))

        gl.glFlush()
        if self.health_ref is not None:
            self._render_ui_overlay(self.health_ref)
        self.object_inspector.tick()
        self.renderer.present()

    def _ui_is_capturing_mouse(self) -> bool:
        if self.gui is not None:
            return self.gui.is_capturing()
        if self.ui is not None and getattr(self.ui, "is_available", False):
            return self.ui.is_capturing()
        return False

    def _ui_is_capturing_keyboard(self) -> bool:
        return self._ui_is_capturing_mouse()
    
    def on_key_press(self, symbol, modifiers):
        self.keyboard_keys[symbol] = 1

        if self._symbol_in_binding(symbol, "mouse_look"):
            self._mouse_look_wanted = not self._mouse_look_wanted
            self._sync_mouse_exclusive()

        elif self._symbol_in_binding(symbol, "toggle_free_view"):
            self._toggle_free_view_mode()

        elif self._symbol_in_binding(symbol, "unfollow_body"):
            self.follow_role_index = None
            self.follow_body_index_input_cache = ""
            self.free_view_mode = False

        elif self._symbol_in_binding(symbol, "follow_confirm") and self.follow_role_index is None:
            if not self.follow_body_index_input_cache:
                return
            role_index = int(self.follow_body_index_input_cache)
            if 0 <= role_index < self.num_players:
                self.follow_role_index = role_index
                self.follow_body_index_input_cache = ""
                self.free_view_mode = True
            else:
                print("Invalid role index entered: ", self.follow_body_index_input_cache)

        elif self._symbol_in_binding(symbol, "follow_backspace") and self.follow_role_index is None:
            self.follow_body_index_input_cache = self.follow_body_index_input_cache[:-1]

        elif self._symbol_in_binding(symbol, "follow_digit") and self.follow_role_index is None:
            self.follow_body_index_input_cache += str(symbol - key._0)

        elif self._symbol_in_binding(symbol, "toggle_newton_ui"):
            self.show_ui = not self.show_ui

        elif self._symbol_in_binding(symbol, "toggle_inspector"):
            self.object_inspector.toggle_window()

        elif self._symbol_in_binding(symbol, "pause"):
            self.simulation_control.paused = not self.simulation_control.paused

        elif self._symbol_in_binding(symbol, "manual_reset"):
            if self.simulation_control.manual_reset_enabled:
                self.simulation_control.request_reset()
       
            
        # 不好解釋，反正按下 ESC 視窗關閉應該就是 on_key_press 控制的，只是不知道是那個地方，就留個記錄
        # elif symbol == pyglet.window.key.ESCAPE:
        #     # Exit with Escape key
        #     self.renderer.close()
       
    def _toggle_free_view_mode(self) -> None:
        if self.follow_role_index is None:
            return
        self.free_view_mode = not self.free_view_mode

    def _sync_mouse_exclusive(self):
        enabled = self._mouse_look_wanted and self._viewer_window_focused
        if self.is_mouse_exclusive == enabled:
            return
        self.is_mouse_exclusive = enabled
        self.renderer.window.set_exclusive_mouse(enabled)

    def on_activate(self):
        self._viewer_window_focused = True
        self._sync_mouse_exclusive()

    def on_deactivate(self):
        self._viewer_window_focused = False
        if self.is_mouse_exclusive:
            self.renderer.window.set_exclusive_mouse(False)
            self.is_mouse_exclusive = False

    def on_key_release(self, symbol, modifiers): self.keyboard_keys[symbol] = 0

    def _is_ctrl_down(self) -> bool:
        return self.renderer.is_key_down(key.LCTRL) or self.renderer.is_key_down(key.RCTRL)

    def _to_framebuffer_coords(self, x: float, y: float) -> tuple[float, float]:
        fb_w, fb_h = self.renderer.window.get_framebuffer_size()
        win_w, win_h = self.renderer.window.get_size()
        if win_w <= 0 or win_h <= 0:
            return float(x), float(y)
        scale_x = fb_w / win_w
        scale_y = fb_h / win_h
        return float(x) * scale_x, float(y) * scale_y

    def _camera_pan_scale(self) -> float:
        height = max(float(self.camera.height), 1.0)
        if hasattr(self.renderer, "window"):
            _, window_height = self.renderer.window.get_size()
            height = max(float(window_height), 1.0)
        distance = max(self.camera.pivot_distance, self.camera.MIN_PIVOT_DISTANCE)
        visible_height = 2.0 * distance * np.tan(np.radians(self.camera.fov) * 0.5)
        return visible_height / height

    def _get_pick_world_offset(self) -> np.ndarray:
        if self.picking is None or self.model is None:
            return np.zeros(3, dtype=np.float64)
        picked_body_idx = int(self.picking.pick_body.numpy()[0])
        if picked_body_idx < 0 or self.model.body_world is None:
            return np.zeros(3, dtype=np.float64)
        body_world_idx = int(self.model.body_world.numpy()[picked_body_idx])
        world_offsets = getattr(self.picking, "world_offsets", None)
        if world_offsets is None or body_world_idx < 0 or body_world_idx >= world_offsets.shape[0]:
            return np.zeros(3, dtype=np.float64)
        offset = world_offsets.numpy()[body_world_idx]
        return np.array([offset[0], offset[1], offset[2]], dtype=np.float64)

    def _adjust_pick_depth_by_scroll(self, scroll_y: float, mouse_x: float, mouse_y: float) -> bool:
        if (
            not self.picking_enabled
            or self.picking is None
            or not self.picking.is_picking()
            or abs(scroll_y) < 1e-6
        ):
            return False

        fb_x, fb_y = self._to_framebuffer_coords(mouse_x, mouse_y)
        ray_start, ray_dir = self.camera.get_world_ray(fb_x, fb_y)
        ray_origin = np.array([ray_start[0], ray_start[1], ray_start[2]], dtype=np.float64)
        ray_direction = np.array([ray_dir[0], ray_dir[1], ray_dir[2]], dtype=np.float64)
        ray_len = float(np.linalg.norm(ray_direction))
        if ray_len < 1e-9:
            return False
        ray_direction /= ray_len

        pick_state_np = self.picking.pick_state.numpy()
        target = np.array(pick_state_np[0]["picking_target_world"], dtype=np.float64)
        world_offset = self._get_pick_world_offset()
        target_offset = target + world_offset

        depth = float(np.dot(target_offset - ray_origin, ray_direction))
        depth = max(self._pick_depth_min, depth + scroll_y * self._pick_depth_scroll_sensitivity)
        new_target = ray_origin + ray_direction * depth - world_offset

        pick_state_np[0]["picking_target_world"] = (
            float(new_target[0]),
            float(new_target[1]),
            float(new_target[2]),
        )
        self.picking.pick_state.assign(pick_state_np)
        return True

    def on_mouse_scroll(self, x: float, y: float, scroll_x: float, scroll_y: float): 
        if self._ui_is_capturing_mouse():
            return

        if self._adjust_pick_depth_by_scroll(scroll_y, x, y):
            return

        if self._is_ctrl_down():
            fov_delta = scroll_y * 2.0
            self.camera.fov -= fov_delta
            self.camera.fov = max(min(self.camera.fov, 120.0), 15.0)

    def on_mouse_press(self, x, y, button, modifiers):
        self.mouse_buttons[button - 1] = 1
        if self._ui_is_capturing_mouse():
            return

        if button == mouse.LEFT and not self.is_mouse_exclusive:
            self._mouse_press_pos = (x, y)
        elif (
            button == mouse.RIGHT
            and not self.is_mouse_exclusive
            and self.picking_enabled
            and self.picking is not None
        ):
            fb_x, fb_y = self._to_framebuffer_coords(x, y)
            ray_start, ray_dir = self.camera.get_world_ray(fb_x, fb_y)
            if self._last_state is not None:
                self.picking.pick(self._last_state, ray_start, ray_dir)

    def on_mouse_release(self, x, y, button, modifiers):
        self.mouse_buttons[button - 1] = 0
        if button == mouse.LEFT and self._mouse_press_pos is not None and not self.is_mouse_exclusive:
            px, py = self._mouse_press_pos
            if abs(x - px) <= 4 and abs(y - py) <= 4:
                self.object_inspector.select_from_ray(x, y)
        self._mouse_press_pos = None
        if self.picking is not None:
            self.picking.release()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self._ui_is_capturing_mouse():
            return

        if self.is_mouse_exclusive:
            return

        if buttons & mouse.MIDDLE:
            if modifiers & key.MOD_CTRL:
                self.camera.dolly(dy * self._camera_dolly_drag_sensitivity)
            elif modifiers & key.MOD_SHIFT:
                pan_scale = self._camera_pan_scale()
                self.camera.pan(-dx * pan_scale, -dy * pan_scale)
            else:
                sensitivity = self._camera_orbit_sensitivity
                self.camera.orbit(delta_yaw=-dx * sensitivity, delta_pitch=dy * sensitivity)
            self.look_yaw = float(self.camera.yaw)
            self.look_pitch = float(self.camera.pitch)
            return

        if buttons & mouse.LEFT:
            self.look_yaw -= dx * self.mouse_sensitivity
            self.look_pitch += dy * self.mouse_sensitivity
            self.look_pitch = max(-89.0, min(89.0, self.look_pitch))
            if self._mouse_press_pos is not None:
                px, py = self._mouse_press_pos
                if abs(x - px) > 4 or abs(y - py) > 4:
                    self._mouse_press_pos = None

        if buttons & mouse.RIGHT and self.picking_enabled:
            fb_x, fb_y = self._to_framebuffer_coords(x, y)
            ray_start, ray_dir = self.camera.get_world_ray(fb_x, fb_y)
            if self.picking is not None and self.picking.is_picking():
                self.picking.update(ray_start, ray_dir)

    def on_mouse_motion(self, x, y, dx, dy):
        if self.is_mouse_exclusive:
            self.look_yaw -= dx * self.mouse_sensitivity
            self.look_pitch += dy * self.mouse_sensitivity
            self.look_pitch = max(-89.0, min(89.0, self.look_pitch))
    
    def _update_camera(self, dt: float):
        """
        Update the camera position and orientation based on user input.

        Args:
            dt: Time delta since last update.
        """
        if self._ui_is_capturing_keyboard():
            return

        if self.follow_role_index is not None:
            self._cam_vel[:] = 0.0
            return

        # camera-relative basis
        forward = np.array(self.camera.get_front(), dtype=np.float32)
        right = np.array(self.camera.get_right(), dtype=np.float32)
        up = np.array(self.camera.get_up(), dtype=np.float32)

        # keep motion in the horizontal plane
        forward -= up * float(np.dot(forward, up))
        right -= up * float(np.dot(right, up))
        # renormalize
        fn = float(np.linalg.norm(forward))
        ln = float(np.linalg.norm(right))
        if fn > 1.0e-6:
            forward /= fn
        if ln > 1.0e-6:
            right /= ln

        desired = np.zeros(3, dtype=np.float32)
        if self.follow_role_index is None:
            if self._is_any_key_down("move_forward"):
                desired += forward
            if self._is_any_key_down("move_back"):
                desired -= forward
            if self._is_any_key_down("move_left"):
                desired -= right
            if self._is_any_key_down("move_right"):
                desired += right
            if self._is_any_key_down("move_down"):
                desired -= up
            if self._is_any_key_down("move_up"):
                desired += up

        dn = float(np.linalg.norm(desired))
        if dn > 1.0e-6:
            desired = desired / dn * float(self._cam_speed)
        else:
            desired[:] = 0.0

        tau = max(1.0e-4, float(self._cam_damp_tau))
        self._cam_vel += (desired - self._cam_vel) * (dt / tau)

        # integrate position
        dv = type(self.camera.pos)(*self._cam_vel)
        self.camera.pos += dv * dt

    def on_update(self, dt): pass

def quat_to_pitch_yaw(q):
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    return math.degrees(pitch), math.degrees(yaw)

@wp.kernel
def calculate_param_debug_geometry_gpu(
    all_transforms: wp.array(dtype=wp.transform),
    shape_transforms: wp.array(dtype=wp.transform),
    body_indices: wp.array(dtype=wp.int32),
    shape_indices: wp.array(dtype=wp.int32),
    forward_locals: wp.array(dtype=wp.vec3),
    forward_conventions: wp.array(dtype=wp.int32),
    l_start: wp.array(dtype=wp.vec3),
    l_end: wp.array(dtype=wp.vec3),
    c_starts: wp.array(dtype=wp.vec3),
    c_ends: wp.array(dtype=wp.vec3),
    SURFACE_OFFSET: wp.array(dtype=wp.float32),
    LINE_LENGTH: wp.array(dtype=wp.float32),
    CIRCLE_RADIUS: wp.array(dtype=wp.float32),
    CIRCLE_LIFT: wp.array(dtype=wp.float32),
    num_segments_arr: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    idx = body_indices[tid]
    trans = all_transforms[idx]
    shape_idx = shape_indices[tid]
    if shape_idx >= 0 and shape_idx < shape_transforms.shape[0]:
        trans = wp.transform_multiply(trans, shape_transforms[shape_idx])
    rot = wp.transform_get_rotation(trans)
    local_axis = forward_locals[tid]
    if forward_conventions[tid] != 0:
        x = rot[0]
        y = rot[1]
        z = rot[2]
        w = rot[3]
        world_normal = wp.vec3(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            -2.0 * (x * z - w * y),
        )
    else:
        world_normal = wp.quat_rotate(rot, local_axis)
    n_len = wp.length(world_normal)
    if n_len > 1.0e-6:
        world_normal = world_normal / n_len
    else:
        world_normal = wp.vec3(1.0, 0.0, 0.0)
    pos = wp.transform_get_translation(trans)
    p_surface = pos + world_normal * SURFACE_OFFSET[0]
    l_start[tid] = p_surface
    l_end[tid] = p_surface + world_normal * LINE_LENGTH[0]
    arbitrary = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(wp.dot(world_normal, arbitrary)) > 0.99: arbitrary = wp.vec3(0.0, 1.0, 0.0)
    tangent = wp.normalize(wp.cross(world_normal, arbitrary))
    bitangent = wp.normalize(wp.cross(world_normal, tangent))
    n_segs = num_segments_arr[0]
    base_idx = tid * n_segs
    for i in range(n_segs):
        t1 = 2.0 * 3.14159265 * float(i) / float(n_segs)
        t2 = 2.0 * 3.14159265 * float(i + 1) / float(n_segs)
        h = world_normal * CIRCLE_LIFT[0]
        c_starts[base_idx + i] = p_surface + (tangent * wp.cos(t1) + bitangent * wp.sin(t1)) * CIRCLE_RADIUS[0] + h
        c_ends[base_idx + i] = p_surface + (tangent * wp.cos(t2) + bitangent * wp.sin(t2)) * CIRCLE_RADIUS[0] + h




@wp.kernel
def gather_transforms_kernel(
    body_q: wp.array(dtype=wp.transform),
    indices: wp.array(dtype=wp.int32),
    out_q: wp.array(dtype=wp.transform)
):
    tid = wp.tid()
    out_q[tid] = body_q[indices[tid]]