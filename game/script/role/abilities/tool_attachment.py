"""Proximity detection, attach prompt, and runtime tool mounting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import warp as wp

from script.i18n import translate
from script.role.abilities.abilities_cfg import get_tool_attachment_detail
from script.role.abilities.ability import Ability
from script.role.abilities.key_mapping import KeyMapping

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

    def configure_from_player_configs(self, player_configs, level: "Levels") -> None:
        self.mount_registry: MountJointRegistry = getattr(level, "mount_joint_registry", None)
        self._players = getattr(level, "players", None)
        attach_cfg = get_tool_attachment_detail(Ability._default_configs)
        if attach_cfg is not None and attach_cfg.proximity_threshold is not None:
            self.proximity_threshold = float(attach_cfg.proximity_threshold)
        if attach_cfg is not None and attach_cfg.proximity_height_threshold is not None:
            self.proximity_height_threshold = float(attach_cfg.proximity_height_threshold)
        if Ability._fps is not None:
            self._fps = int(Ability._fps)
        if self.mount_registry is not None:
            for record in self.mount_registry.records.values():
                if record.proximity_threshold <= 0:
                    record.proximity_threshold = self.proximity_threshold
                if record.proximity_height_threshold <= 0:
                    record.proximity_height_threshold = self.proximity_height_threshold
        # Warm prompt cache once at configure time (avoid per-frame i18n I/O).
        self._prompt_text_cache = translate(
            self.prompt_message_key,
            default="Press U to attach tool",
        )
        self._configured = True

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

        self.mount_registry.apply_attached_actions(
            state.body_q,
            state.body_qd,
            physics_manager.control,
            camera_yaw=float(kwargs.get("camera_yaw", kwargs.get("look_yaw")) or 0.0),
            camera_pitch=float(kwargs.get("camera_pitch", kwargs.get("look_pitch")) or 0.0),
            world=world,
            dt=1.0 / max(1, int(getattr(self, "_fps", 50) or 50)),
            host_role_object_id=human_obj_idx,
            body_q_np=body_q_np,
            joint_q=state.joint_q,
            joint_qd=state.joint_qd,
        )

    def rl_action(self, actions, **kwargs):
        pass

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
