
from typing import Literal
from script.role.base_role import BaseRole, BaseRoleModel
from script.role.controller_utils import (
    CONTROLLER_CHOICES,
    PlayerController,
    normalize_controller,
)

__all__ = [
    "Player",
    "PlayerModel",
    "PlayerController",
    "CONTROLLER_CHOICES",
    "normalize_controller",
]


class PlayerModel(BaseRoleModel):
    type: Literal["player"] = "player"
    name: str = "Player_1"
    controller: PlayerController = "Human"
    team_id: int = 1
    health: float = 5.0


class Player(BaseRole):
    role_key = "player"
    model_cls = PlayerModel
    path = "player_configs"
    container = "list"


    def __init__(self, configs, **kwargs):
        super().__init__(**kwargs)

        self.setup(configs=configs)

    def update_index_rl_and_bot(self, index_rl_players_gpu, num_rl_players, is_rl_player_mask_gpu, index_bot_players_gpu, num_bot_players, is_bot_player_mask_gpu):
        self.index_rl_players_gpu = index_rl_players_gpu
        self.num_rl_players = num_rl_players
        self.is_rl_player_mask_gpu = is_rl_player_mask_gpu

        self.index_bot_players_gpu = index_bot_players_gpu
        self.num_bot_players = num_bot_players
        self.is_bot_player_mask_gpu = is_bot_player_mask_gpu

        for ability in self.abilities_instance_list:
            ability.update_index_bot(index_rl_players_gpu=index_rl_players_gpu, num_rl_players=num_rl_players, index_bot_players_gpu=index_bot_players_gpu, num_bot_players=num_bot_players)

    def rl_action(self, actions, **kwargs):
        if self.num_rl_players > 0:
            for ability in self.abilities_instance_list:
                # if ability.action_shape_offset is None:
                #     raise ValueError(f"{ability.ability_name}.action_shape_offset cannot be None")
                
                ability.rl_action(actions=actions, **kwargs)

    def bot_action(self, **kwargs):
        if self.num_bot_players > 0:
            for ability in self.abilities_instance_list:
                ability.bot_action(**kwargs)
