"""Proximity detection, attach prompt, and runtime tool mounting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import warp as wp

from script.i18n import translate
from script.role.abilities.abilities_cfg import get_tool_attachment_detail
from script.role.abilities.ability import Ability
from script.role.abilities.key_mapping import KeyMapping
from script.role.controller_utils import normalize_controller
from script.role.objects.object_template.tool_function_registry import (
    resolve_max_rl_action_dim_for_patterns,
    resolve_rl_action_spec_for_max_pattern,
)

if TYPE_CHECKING:
    from script.levels.levels import Levels
    from script.simulate.mount_joint_registry import MountJointRegistry


class Tool_attachment(Ability):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self._keyboard_attach = []
        self._prev_attach_pressed = False
        self.proximity_threshold = 0.75
        self.proximity_height_threshold = 3.5
        self._configured = False
        self.mount_registry = None
        self.prompt_message_key = "tool_attach_prompt"
        self._prompt_text_cache: Optional[str] = None
        self._tool_rl_action_dim = 0
        self._tool_rl_action_spec: dict = {"type": "box", "shape": 0, "range": [-1.0, 1.0]}

    def configure_from_player_configs(self, player_configs, level: "Levels") -> None:
        self.mount_registry: MountJointRegistry = getattr(level, "mount_joint_registry", None)
        self._players = getattr(level, "players", None)
        self._level = level
        attach_cfg = get_tool_attachment_detail(Ability._default_configs)
        if attach_cfg is not None and attach_cfg.proximity_threshold is not None:
            self.proximity_threshold = float(attach_cfg.proximity_threshold)
        if attach_cfg is not None and attach_cfg.proximity_height_threshold is not None:
            self.proximity_height_threshold = float(attach_cfg.proximity_height_threshold)
        # 角色配置中的 abilities.Tool_attachment 字典形式覆寫優先於全域配置
        for role_cfg in self._role_ability_configs.values():
            if "proximity_threshold" in role_cfg:
                self.proximity_threshold = float(role_cfg["proximity_threshold"])
            if "proximity_height_threshold" in role_cfg:
                self.proximity_height_threshold = float(role_cfg["proximity_height_threshold"])
        if Ability._fps is not None:
            self._fps = int(Ability._fps)
        if self.mount_registry is not None:
            for record in self.mount_registry.records.values():
                if record.proximity_threshold <= 0:
                    record.proximity_threshold = self.proximity_threshold
                if record.proximity_height_threshold <= 0:
                    record.proximity_height_threshold = self.proximity_height_threshold

        tool_patterns = self._collect_level_tool_patterns(level)
        self._tool_rl_action_dim = resolve_max_rl_action_dim_for_patterns(tool_patterns)
        self._tool_rl_action_spec = resolve_rl_action_spec_for_max_pattern(tool_patterns)

        # Warm prompt cache once at configure time (avoid per-frame i18n I/O).
        self._prompt_text_cache = translate(
            self.prompt_message_key,
            default="Press U to attach tool",
        )
        self._configured = True

    @staticmethod
    def _collect_level_tool_patterns(level: "Levels") -> List[str]:
        tool_configs = (getattr(level, "level_configs", None) or {}).get("tool_configs") or []
        patterns: List[str] = []
        for entry in tool_configs:
            if not isinstance(entry, dict):
                continue
            pattern = (entry.get("object") or {}).get("pattern")
            if pattern:
                patterns.append(str(pattern))
        return patterns

    def apply_ability_config_overrides(self, overrides: dict) -> None:
        """Apply per-owner proximity overrides from the abilities dict form."""
        super().apply_ability_config_overrides(overrides)
        if not isinstance(overrides, dict) or not overrides:
            return
        if "proximity_threshold" in overrides:
            self.proximity_threshold = float(overrides["proximity_threshold"])
        if "proximity_height_threshold" in overrides:
            self.proximity_height_threshold = float(overrides["proximity_height_threshold"])

    def _world_for_role_object(self, role_object_id: int) -> int:
        players = getattr(self, "_players", None)
        if players is not None:
            try:
                role_list_idx = players.index_obj_role.index(role_object_id)
                mapping = getattr(players, "index_obj_role_to_env_mapping", None) or []
                if role_list_idx < len(mapping):
                    return int(mapping[role_list_idx])
            except ValueError:
                pass
        num_objects_env = getattr(self, "num_objects_env", 1) or 1
        return int(role_object_id) // max(1, int(num_objects_env))

    def _player_config_index_for_role(self, role_object_id: int) -> Optional[int]:
        players = getattr(self, "_players", None)
        if players is None:
            return None
        num_env = int(getattr(players, "num_role_each_env", 1) or 1)
        local = int(role_object_id) % num_env
        env0_players = sorted(idx for idx in players.index_obj_role if idx < num_env)
        if local not in env0_players:
            return None
        return env0_players.index(local)

    def _controller_for_role_object(self, role_object_id: int) -> str:
        from script.role.base_role import BaseRole

        idx = int(role_object_id)
        object_params = BaseRole._object_game_params
        if idx < 0 or idx >= len(object_params):
            return "Bot"
        params = object_params[idx]
        return normalize_controller(params.get("controller"))

    def _action_offset_for_role(self, role_object_id: int) -> int:
        player_idx = self._player_config_index_for_role(role_object_id)
        if player_idx is not None:
            by_player = getattr(self, "_action_offsets_by_player_index", None)
            if by_player is not None and player_idx in by_player:
                return int(by_player[player_idx])
        offset = self.action_shape_offset
        return int(offset) if offset is not None else 0

    def setup_keymapping(self):
        keys = KeyMapping.get(keys=self._ability_keys())
        self._keyboard_attach = keys["keyboard"].get("attach", [])

    def _ability_keys(self) -> dict:
        attach_cfg = get_tool_attachment_detail(Ability._default_configs)
        if attach_cfg is None:
            return {"keyboard": {"attach": ["u"]}, "mouse": {}}
        return {
            "keyboard": dict(attach_cfg.key.keyboard),
            "mouse": dict(attach_cfg.key.mouse),
        }

    def get_action_spec(self) -> dict:
        spec = dict(self.action_space)
        dim = int(self._tool_rl_action_dim)
        if dim > 0:
            spec.update(self._tool_rl_action_spec)
            spec["shape"] = dim
        else:
            spec["shape"] = 0
        return spec

    def _apply_tool_control_for_host(
        self,
        host_role_object_id: int,
        values: Sequence[float],
        *,
        use_camera_if_no_rl: bool = True,
        camera_yaw: float = 0.0,
        camera_pitch: float = 0.0,
        mouse_buttons=None,
    ) -> None:
        if not self._configured or self.mount_registry is None:
            return
        dim = int(self._tool_rl_action_dim)
        if dim <= 0:
            return

        record = self.mount_registry.get_attached_record(host_role_object_id)
        if record is None:
            return

        slice_values = [float(values[i]) for i in range(min(len(values), dim))]
        if slice_values:
            self.mount_registry.apply_rl_control_for_host(host_role_object_id, slice_values)
        elif use_camera_if_no_rl:
            self.mount_registry.clear_rl_control_for_host(host_role_object_id)

        if use_camera_if_no_rl and not self._inspector_rl_override_active():
            self.mount_registry.drive_attached_tools_frame(
                camera_yaw=float(camera_yaw),
                camera_pitch=float(camera_pitch),
                mouse_buttons=mouse_buttons,
                host_role_object_id=host_role_object_id,
            )

    def _inspector_rl_override_active(self) -> bool:
        physics_manager = self.physics_manager
        viewer = getattr(physics_manager, "viewerGL", None)
        if viewer is None:
            return False
        inspector = getattr(viewer, "object_inspector", None)
        if inspector is None:
            return False
        return inspector.is_rl_action_override_enabled()

    def apply_human_inspector_tool_actions(self, actions) -> None:
        """Apply Tool_attachment dims from the action buffer for Human hosts.

        Must run in ``Game.step`` after inspector pins are written to the buffer
        and before physics ``simulate`` (same timing as other ``rl_action`` hooks).
        """
        if not self._configured or self.mount_registry is None:
            return
        if not self._inspector_rl_override_active():
            return
        if getattr(self, "num_rl_players", 0) <= 0:
            return
        dim = int(self._tool_rl_action_dim)
        if dim <= 0:
            return

        actions_np = actions.numpy()
        index_rl = self.index_rl_players_gpu.numpy()

        for tid in range(int(self.num_rl_players)):
            host_obj_idx = int(index_rl[tid])
            if self._controller_for_role_object(host_obj_idx) != "Human":
                continue
            offset = self._action_offset_for_role(host_obj_idx)
            row = actions_np[tid]
            end = offset + dim
            if end > row.shape[0]:
                continue
            slice_values = [float(row[offset + i]) for i in range(dim)]
            self._apply_tool_control_for_host(
                host_obj_idx,
                slice_values,
                use_camera_if_no_rl=False,
                camera_yaw=0.0,
                camera_pitch=0.0,
                mouse_buttons=None,
            )
            self.mount_registry.drive_attached_tools_frame(
                host_role_object_id=host_obj_idx,
            )

    def human_control_interface(
        self,
        keyboard_keys,
        mouse_buttons,
        look_yaw,
        look_pitch,
        index_human_player_gpu: wp.array,
        **kwargs,
    ):
        if not self._configured or self.mount_registry is None:
            return

        physics_manager = self.physics_manager
        human_obj_idx = int(index_human_player_gpu.numpy()[0])
        world = self._world_for_role_object(human_obj_idx)
        state = physics_manager.state_0
        # One body_q sync shared by proximity + aim hot paths.
        body_q_np = state.body_q.numpy()
        self.mount_registry.update_proximity(state.body_q, world=world, body_q_np=body_q_np)

        pressed = self._is_pressed(self._keyboard_attach, [], keyboard_keys, mouse_buttons) == 1
        edge_press = pressed and not self._prev_attach_pressed
        self._prev_attach_pressed = pressed

        attach_kwargs = dict(
            body_f=state.body_f,
            joint_qd=state.joint_qd,
            body_q_prev=state.body_q_prev,
        )

        if edge_press:
            attached_key = self.mount_registry.get_attached_tool_key(human_obj_idx)
            toggled = False
            if attached_key:
                toggled = self.mount_registry.toggle_attachment(
                    attached_key,
                    state.body_q,
                    state.body_qd,
                    state.joint_q,
                    world=world,
                    **attach_kwargs,
                )
            else:
                tool_key = self.mount_registry.prompt_tool_key(human_obj_idx)
                if tool_key:
                    toggled = self.mount_registry.toggle_attachment(
                        tool_key,
                        state.body_q,
                        state.body_qd,
                        state.joint_q,
                        world=world,
                        **attach_kwargs,
                    )
            if toggled:
                # Attach/detach snaps body_q; refresh shared CPU snapshot for aim.
                body_q_np = state.body_q.numpy()

        camera_yaw = float(kwargs.get("camera_yaw", kwargs.get("look_yaw")) or 0.0)
        camera_pitch = float(kwargs.get("camera_pitch", kwargs.get("look_pitch")) or 0.0)
        if self._inspector_rl_override_active():
            return

        self.mount_registry.drive_attached_tools_frame(
            camera_yaw=camera_yaw,
            camera_pitch=camera_pitch,
            mouse_buttons=mouse_buttons,
            host_role_object_id=human_obj_idx,
        )

    def rl_action(self, actions, **kwargs):
        if not self._configured or self.mount_registry is None:
            return
        if getattr(self, "num_rl_players", 0) <= 0:
            return
        dim = int(self._tool_rl_action_dim)
        if dim <= 0:
            return

        physics_manager = self.physics_manager
        state = physics_manager.state_0
        body_q_np = state.body_q.numpy()
        actions_np = actions.numpy()
        index_rl = self.index_rl_players_gpu.numpy()

        for tid in range(int(self.num_rl_players)):
            host_obj_idx = int(index_rl[tid])
            if self._controller_for_role_object(host_obj_idx) == "Human":
                continue

            offset = self._action_offset_for_role(host_obj_idx)
            world = self._world_for_role_object(host_obj_idx)
            row = actions_np[tid]
            end = offset + dim
            if end > row.shape[0]:
                continue
            slice_values = [float(row[offset + i]) for i in range(dim)]
            if not slice_values:
                continue

            self.mount_registry.apply_rl_control_for_host(host_obj_idx, slice_values)
            self.mount_registry.drive_attached_tools_frame(host_role_object_id=host_obj_idx)

    def bot_action(self, **kwargs):
        pass

    def update_index_bot(self, index_rl_players_gpu, num_rl_players, index_bot_players_gpu, num_bot_players):
        self.index_rl_players_gpu = index_rl_players_gpu
        self.num_rl_players = num_rl_players
        self.index_bot_players_gpu = index_bot_players_gpu
        self.num_bot_players = num_bot_players

    def reset(self):
        self._prev_attach_pressed = False

    def setup_cooldown(self, num_objects_total, owners_ability_list):
        super().setup_cooldown(num_objects_total, owners_ability_list)

    def setup_player_to_env_mapping(
        self,
        index_role_offset_env_gpu,
        num_role_each_env,
        *,
        role_type: str = "player",
    ):
        super().setup_player_to_env_mapping(
            index_role_offset_env_gpu,
            num_role_each_env,
            role_type=role_type,
        )
        # Attach is host/player driven; prefer player env sizing when both exist.
        try:
            offset, num = self.get_env_mapping("player")
        except KeyError:
            offset, num = index_role_offset_env_gpu, num_role_each_env
        self.index_role_offset_env_gpu = offset
        self.num_role_each_env = num
        self.num_objects_env = num

    @property
    def show_attach_prompt(self) -> bool:
        if self.mount_registry is None:
            return False
        return self.mount_registry.any_prompt_visible()

    def show_attach_prompt_for_host(self, host_role_object_id: int) -> bool:
        if self.mount_registry is None:
            return False
        return self.mount_registry.any_prompt_visible(host_role_object_id)

    def get_prompt_text(self, translate_fn=None) -> str:
        if translate_fn is not None:
            text = translate_fn(self.prompt_message_key)
            if text and text != self.prompt_message_key:
                return text
        if self._prompt_text_cache is not None:
            return self._prompt_text_cache

        self._prompt_text_cache = translate(
            self.prompt_message_key,
            default="Press U to attach tool",
        )
        return self._prompt_text_cache
