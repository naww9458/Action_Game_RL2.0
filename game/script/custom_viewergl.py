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
from pyglet.window import key
from newton.viewer import ViewerGL
from newton import State
from queue import Queue, Full
from typing import TYPE_CHECKING
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
        self.free_view_mode = False
        self.possess_offsets: list[tuple[float, float, float]] = []
        self.sim_time = 0.0

        # 相機相關屬性
        self.is_mouse_exclusive = False
        self.renderer.window.set_exclusive_mouse(self.is_mouse_exclusive)
        self.mouse_sensitivity = 0.05
        self.look_yaw = 0.0
        self.look_pitch = 0.0

        self.mouse_buttons = np.zeros(shape=4, dtype=np.int32)
        self.keyboard_keys = np.zeros(shape=65536, dtype=np.int32)
        self.human_input_queue = human_input_queue
        self.viewer_controls_cfg = load_viewer_controls()
        self.simulation_control = SimulationControl.from_defaults(self.viewer_controls_cfg)
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

        self.object_inspector = ObjectInspectorPlugin()
        self._mouse_press_pos = None

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
        self.object_inspector.attach(game, self)
        self.num_players = game.num_players
        self.num_objects_total = game.num_objects_total
        # 初始化玩家 Label 池
        for i in range(self.game.num_players):
            lbl = pyglet.text.Label('', font_name='Consolas', font_size=self.font_size_player_info,
                                    x=20, y=0, color=(255, 255, 0, 255), batch=self._ui_batch)
            self._player_labels.append(lbl)
            self._last_health.append(-1)
            self._last_rewards.append(-1.0)

        num_segs = 16
        self.l_start = wp.array(shape=self.num_players, dtype=wp.vec3, device=GameConfig.DEVICE)
        self.l_end = wp.array(shape=self.num_players, dtype=wp.vec3, device=GameConfig.DEVICE)
        self.c_starts = wp.array(shape=self.num_players * num_segs, dtype=wp.vec3, device=GameConfig.DEVICE)
        self.c_ends = wp.array(shape=self.num_players * num_segs, dtype=wp.vec3, device=GameConfig.DEVICE)
        self.SURFACE_OFFSET = wp.array(data=[0.5], dtype=wp.float32, device=GameConfig.DEVICE)
        self.LINE_LENGTH = wp.array(data=[1.0], dtype=wp.float32, device=GameConfig.DEVICE)
        self.CIRCLE_RADIUS = wp.array(data=[0.01], dtype=wp.float32, device=GameConfig.DEVICE)
        self.num_segments = wp.array(data=[num_segs], dtype=wp.int32, device=GameConfig.DEVICE)

        self.geo_type = newton.GeoType.SPHERE
        shape_scale_all = self.model.shape_scale.numpy()
        index_players = self.game.players.index_obj_role

        # i-1 because index 0 in shape_scale_all is ground
        self.geo_scale = [scale[0] for i, scale in enumerate(shape_scale_all) if i-1 in index_players]

        self._configure_display_envs(game.num_env)
        self.camera.fov = 100.0

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
        # 還原狀態
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.renderer.window.projection = proj
        self.renderer.window.view = view

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

        # self.render_debug_geometry(all_transforms=state_0.body_q, index_player_gpu=index_player_gpu) # TODO 暫時禁用，因爲現在屬於玩家角色控制的物件不再只有球體, 如果物件是球體，如果是多個剛體組合的物件比如 Unitree g1，身上會出現很多條綫
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
    
    def render_debug_geometry(self, all_transforms, index_player_gpu):
        wp.launch(kernel=calculate_param_debug_geometry_gpu, dim=self.num_players, inputs=[all_transforms, index_player_gpu, self.l_start, self.l_end, self.c_starts, self.c_ends, self.SURFACE_OFFSET, self.LINE_LENGTH, self.CIRCLE_RADIUS, self.num_segments], device=GameConfig.DEVICE)
        self.log_lines(name="debug_extension_line", starts=self.l_start, ends=self.l_end, colors=(0.0, 1.0, 0.0), width=0.03)
        self.log_lines(name="debug_surface_circle", starts=self.c_starts, ends=self.c_ends, colors=(1.0, 0.0, 0.0), width=0.01)
    
    def end_frame(self):
        self.renderer.update()
        now = time.perf_counter()
        dt = max(0.0, min(0.1, now - self._last_time))
        self._last_time = now
        self._update_camera(dt)
        self.wind.update(dt)
        if self.renderer.has_exit(): return
        self.renderer.render(self.camera, self.objects, self.lines)
        self._update_fps()
        if self.ui.is_available and self.show_ui:
            self.ui.begin_frame()
            self._render_ui()
            self.ui.end_frame()
            self.ui.render()
       
        gl.glFlush()
        self._render_ui_overlay(self.health_ref)
        self.object_inspector.tick()
        self.renderer.present()
    
    def on_key_press(self, symbol, modifiers):
        self.keyboard_keys[symbol] = 1

        if self._symbol_in_binding(symbol, "mouse_look"):
            self.is_mouse_exclusive = not self.is_mouse_exclusive
            self.renderer.window.set_exclusive_mouse(self.is_mouse_exclusive)

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

    def on_key_release(self, symbol, modifiers): self.keyboard_keys[symbol] = 0

    def on_mouse_scroll(self, x: float, y: float, scroll_x: float, scroll_y: float): 
        if self._ui_is_capturing_mouse():
            return

        if self._is_ctrl_down():
            fov_delta = scroll_y * 2.0
            self.camera.fov -= fov_delta
            self.camera.fov = max(min(self.camera.fov, 120.0), 15.0)

    def on_mouse_press(self, x, y, button, modifiers):
        self.mouse_buttons[button-1] = 1
        if button == 1 and not self.is_mouse_exclusive:
            self._mouse_press_pos = (x, y)

    def on_mouse_release(self, x, y, button, modifiers):
        self.mouse_buttons[button-1] = 0
        if button == 1 and self._mouse_press_pos is not None and not self.is_mouse_exclusive:
            px, py = self._mouse_press_pos
            if abs(x - px) <= 4 and abs(y - py) <= 4:
                self.object_inspector.select_from_ray(x, y)
        self._mouse_press_pos = None

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
            desired = desired / dn * self._cam_speed
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
def calculate_param_debug_geometry_gpu(all_transforms: wp.array(dtype=wp.transform), index_player_gpu: wp.array(dtype=wp.int32), l_start: wp.array(dtype=wp.vec3), l_end: wp.array(dtype=wp.vec3), c_starts: wp.array(dtype=wp.vec3), c_ends: wp.array(dtype=wp.vec3), SURFACE_OFFSET: wp.array(dtype=wp.float32), LINE_LENGTH: wp.array(dtype=wp.float32), CIRCLE_RADIUS: wp.array(dtype=wp.float32), num_segments_arr: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    idx = index_player_gpu[tid]
    trans = all_transforms[idx]
    pos = wp.transform_get_translation(trans)
    rot = wp.transform_get_rotation(trans)
    world_normal = wp.quat_rotate(rot, wp.vec3(1.0, 0.0, 0.0))
    world_normal = wp.vec3(world_normal[0], world_normal[1], -world_normal[2])
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
        h = world_normal * 0.002
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