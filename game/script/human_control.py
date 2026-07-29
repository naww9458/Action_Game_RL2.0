import warp as wp

from script.role.abilities.ability import Ability
from script.role.abilities.key_mapping import KeyMapping
from script.role.base_role import BaseRole
from script.role.controller_utils import normalize_controller

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from script.game import Game

class HumanControl:
    def __init__(self, game: 'Game'):
        self.game = game
        self.players = game.players
        self.index_human_player = None
        self.index_abilities: list[int] = []
        self.is_human_exist = False

        device = game.physics_manager.device
        for index_player in self.players.index_obj_role:
            params = BaseRole._object_game_params[index_player]
            controller = normalize_controller(params.get("controller"))
            if controller == "Human":
                self.index_human_player = index_player
                self.index_abilities = self.players.get_player_abilities(self.index_human_player)
                self.is_human_exist = True
                break

        init_idx = self.index_human_player if self.index_human_player is not None else 0
        self.index_human_player_gpu = wp.array(data=[init_idx], dtype=wp.int32, device=device)

        self.button_reset = {"keyboard": {"reset": ["y"]}, "mouse": {"reset": []}} # TODO

    def get_player_actions(self, follow_body_index, keyboard_keys, mouse_buttons, look_yaw, look_pitch, **kwargs) -> dict:
        """
        這是一個通用的方法，自動遍歷所有能力並獲取其輸入
        """
        if follow_body_index is None:
            return None

        if self.index_human_player != follow_body_index:
            self.index_human_player = follow_body_index
            self.index_abilities = self.players.get_player_abilities(follow_body_index)
            wp.copy(
                self.index_human_player_gpu,
                wp.array(
                    data=[follow_body_index],
                    dtype=wp.int32,
                    device=self.index_human_player_gpu.device,
                ),
            )

        for index_ability in self.index_abilities:
            self.players.abilities_instance_list[index_ability].human_control_interface(
                keyboard_keys=keyboard_keys,
                mouse_buttons=mouse_buttons,
                look_yaw=look_yaw,
                look_pitch=look_pitch,
                camera_yaw=kwargs.get("camera_yaw", look_yaw),
                camera_pitch=kwargs.get("camera_pitch", look_pitch),
                index_human_player_gpu=self.index_human_player_gpu,
                current_game_step=self.game.current_step,
            )

        return None

    def setup_reset_keymapping(self, viewer_controls_config=None):
        if viewer_controls_config and viewer_controls_config.get("manual_reset_keys"):
            self.button_reset = {
                "keyboard": {"reset": list(viewer_controls_config["manual_reset_keys"])},
                "mouse": {"reset": []},
            }
        keys = KeyMapping.get(keys=self.button_reset)
        self._keyboard_reset = keys["keyboard"]["reset"]
        self._mouse_reset = keys["mouse"]["reset"]
    
    def check_is_reset(self, keyboard_keys=None, mouse_buttons=None, **kwargs) -> dict:
        return Ability._is_pressed(self=None, kb_list=self._keyboard_reset, ms_list=self._mouse_reset, kb_state=keyboard_keys, ms_state=mouse_buttons)
