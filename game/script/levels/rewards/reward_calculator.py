import inspect
import warp as wp

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Dict, Type
from script.simulate.physics_manager import PhysicsManager
from script.game_config import GameConfig
from script.role.base_role import BaseRole
from script.role.bodies.articulation_body import ArticulationBody
from script.role.bodies.deformable_body import DeformableBody

if TYPE_CHECKING:
    from script.game import Game
    from script.role.player import Player
    from script.role.platform import Platform
    from script.levels.levels import Levels

class RewardComponent(ABC):
    """所有獎勵計算元件的抽象基底類別"""
    _registry: Dict[str, Type["RewardComponent"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            RewardComponent._registry[cls.__name__] = cls

    @classmethod
    def get_registered_names(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def resolve(cls, name: str) -> Type["RewardComponent"]:
        if name not in cls._registry:
            from training.reward_imports import ensure_reward_registered
            ensure_reward_registered(name)
        if name not in cls._registry:
            raise KeyError(f"Unknown reward component: {name}. Available: {cls.get_registered_names()}")
        return cls._registry[name]

    def __init__(self, reward_parameters: dict, articulation_body: ArticulationBody, deformable_body: DeformableBody, **kwargs):
        self.params = reward_parameters
        self.articulation_body = articulation_body
        self.deformable_body = deformable_body

        # define reward parameters
        for key, value in reward_parameters.items():
            # print(f"設置獎勵參數 {key}: {value}")
            setattr(self, key, value)

    @abstractmethod
    def calculate(self, num_players, physics_manager: 'PhysicsManager', player_health: wp.array, step_total_rewards: wp.array, **kwargs):
        """
        計算此元件負責的獎勵
        使用 **kwargs 來接收未來可能需要的額外資訊 (如 entities, alive_count 等)。
        """
        pass

    @abstractmethod
    def reset(self, num_players, terminated, index_player_obj_to_env_mapping_gpu, **kwargs):
        pass

class GameEndDetector(RewardComponent):
    """所有獎勵計算元件的抽象基底類別"""
    def __init__(self, reward_parameters: dict):
        self.params = reward_parameters

        # define reward parameters
        for key, value in reward_parameters.items():
            # print(f"設置獎勵參數 {key}: {value}")
            setattr(self, key, value)

    @abstractmethod
    def calculate(self, num_env, env_players_index_offset: wp.array, num_players_each_env: wp.int32, player_health: wp.array, default_player_health: wp.array, current_step: wp.array, max_episode_step: int, step_total_rewards: wp.array, terminated: wp.array, **kwargs):

        
        """
        計算此元件負責的獎勵，判斷是否需要結束這一回合
        使用 **kwargs 來接收未來可能需要的額外資訊 (如 entities, alive_count 等)。
        """
        pass


class RewardCalculator:
    """協調多個獎勵元件來計算總獎勵"""

    def __init__(self, 
                 level: 'Levels',
                 terminated: wp.array,
                 reward_components: list[RewardComponent] = [], 
                 reward_components_diff: list[RewardComponent] = [], 
                 episode_end_detector: GameEndDetector = None,
                ):
        self.level = level
        self.num_env = level.num_env
        self.physics_manager = level.physics_manager
        self.device = self.physics_manager.device

        self.reward_components = reward_components
        self.reward_components_diff = reward_components_diff
        self.episode_end_detector = episode_end_detector

        # --- 預處理索引 (Env) ---
        self.index_player_offset_env_gpu = level.players.index_role_offset_env_gpu
        self.num_players_each_env = level.players.num_role_each_env
        self.index_platform_offset_env_gpu = level.platforms.index_role_offset_env_gpu
        self.num_platform_each_env = level.platforms.num_role_each_env

        # --- 預處理索引 (玩家) ---
        self.index_player_gpu = level.players.index_obj_role_gpu
        self.num_players = level.players.num_total_object_role
        self.index_rl_players_gpu = level.players.index_rl_players_gpu
        self.is_rl_player_mask_gpu = level.players.is_rl_player_mask_gpu
        self.num_rl_players = level.players.num_rl_players

        # --- 預處理索引 (Platform) ---
        self.index_platform_gpu = level.platforms.index_obj_role_gpu
        self.num_platforms = level.platforms.num_total_object_role

        # --- 預分配緩衝區 ---
        health = []
        self.num_total_object = BaseRole._num_objects_total
        for index in range(self.num_total_object):
            health.append(level.players._object_game_params[index]["health"])

        self.default_player_health = wp.array(health, dtype=wp.float32, device=self.device)
        self.player_health = wp.array(health, dtype=wp.float32, device=self.device, requires_grad=GameConfig.requires_grad)

        self.step_total_rewards_all = wp.zeros(BaseRole._num_objects_total, dtype=wp.float32, device=self.device)
        self.step_total_rewards_rl = wp.zeros(self.num_rl_players, dtype=wp.float32, device=self.physics_manager.device)

        if GameConfig.requires_grad:
            self.step_total_rewards_all_diff = wp.zeros(BaseRole._num_objects_total, dtype=wp.float32, device=self.device, requires_grad=GameConfig.requires_grad)
            self.step_total_rewards_rl_diff = wp.zeros(self.num_rl_players, dtype=wp.float32, device=self.physics_manager.device, requires_grad=GameConfig.requires_grad)

        self.terminated = terminated

    def calculate_rewards(self, current_step: int, actions: wp.array2d, max_episode_step: int, command_vel) -> tuple[wp.array, wp.array]:
        """
        執行所有已註冊的獎勵元件，計算總獎勵，並判斷遊戲是否結束。
        注意：為了 CUDA Graph，這裡不應該有 .numpy() 調用。
        """

        self.step_total_rewards_all.zero_()

        for component in self.reward_components:
            component.calculate(
                num_players=self.num_players,
                num_rl_players = self.num_rl_players,
                physics_manager=self.physics_manager,
                actions=actions,
                player_shape_ids_gpu=self.index_player_gpu,
                is_rl_player_mask_gpu=self.is_rl_player_mask_gpu,
                index_player_obj_to_env_mapping_gpu=self.level.index_player_obj_to_env_mapping_gpu,
                _index_obj_to_env_mapping_gpu=self.level._index_obj_to_env_mapping_gpu,

                env_players_index_offset=self.index_player_offset_env_gpu,
                env_platforms_index_offset=self.index_platform_offset_env_gpu,

                num_players_each_env=self.num_players_each_env,
                num_platforms_each_env=self.num_platform_each_env,

                platform_shape_ids_gpu=self.index_platform_gpu,
                player_health=self.player_health,
                step_total_rewards=self.step_total_rewards_all,

                command_vel=command_vel, 
            )

        if self.episode_end_detector is not None:
            self.episode_end_detector.calculate(
                num_env=self.num_env,
                physics_manager=self.physics_manager,
                env_players_index_offset=self.index_player_offset_env_gpu,
                player_shape_ids_gpu=self.index_player_gpu,
                num_players_each_env=self.num_players_each_env,
                player_health=self.player_health,
                default_player_health=self.default_player_health,
                current_step=current_step,
                actions=actions,
                max_episode_step=max_episode_step,
                step_total_rewards=self.step_total_rewards_all,
                terminated=self.terminated,
            )

        if GameConfig.requires_grad:
            self.step_total_rewards_all_diff.zero_()
            for component in self.reward_components_diff:
                component.calculate(
                    num_players=self.num_players,
                    num_rl_players = self.num_rl_players,
                    physics_manager=self.physics_manager,
                    actions=actions,
                    player_shape_ids_gpu=self.index_player_gpu,
                    is_rl_player_mask_gpu=self.is_rl_player_mask_gpu,
                    index_player_obj_to_env_mapping_gpu=self.level.index_player_obj_to_env_mapping_gpu,

                    env_players_index_offset=self.index_player_offset_env_gpu,
                    env_platforms_index_offset=self.index_platform_offset_env_gpu,

                    num_players_each_env=self.num_players_each_env,
                    num_platforms_each_env=self.num_platform_each_env,

                    platform_shape_ids_gpu=self.index_platform_gpu,
                    player_health=self.player_health,
                    step_total_rewards=self.step_total_rewards_all_diff,
                )

            # print("self.num_players: ", self.num_players)
            # print("self.is_rl_player_mask_gpu: ", self.is_rl_player_mask_gpu)
            # print("self.step_total_rewards_all_diff: ", self.step_total_rewards_all_diff.shape)
            # print("self.step_total_rewards_all1: ", self.step_total_rewards_all.shape)

            wp.launch(
                kernel=self.apply_step_reward_to_rl_Diff,
                dim=self.num_players,
                inputs=[
                    self.index_rl_players_gpu,
                    self.index_player_gpu,
                    self.step_total_rewards_all,
                    self.step_total_rewards_all_diff,
                    self.step_total_rewards_rl_diff,

                    self.num_rl_players,
                    self.is_rl_player_mask_gpu,
                ],
                device=self.device
            )

            
        # if self.step_total_rewards_all.numpy().sum() > 0:
        #     print("self.step_total_rewards_all.numpy(): ", self.step_total_rewards_all.numpy())

        wp.launch(
            kernel=self.apply_step_reward_to_rl,
            dim=self.num_rl_players,
            inputs=[
                self.index_rl_players_gpu,
                self.step_total_rewards_all,
                self.step_total_rewards_rl,
            ],
            device=self.device
        )


    def reset_reward(self):

        wp.launch(
            kernel=self.reset_gpu,
            dim=self.num_total_object,
            inputs=[
                self.terminated,
                self.level._index_obj_to_env_mapping_gpu,
                self.player_health,
                self.default_player_health,
            ],
            device=self.device
        )

        for reward_component in self.reward_components:
            reward_component.reset(
                num_players=self.num_players,
                terminated=self.terminated,
                _index_obj_to_env_mapping_gpu=self.level._index_obj_to_env_mapping_gpu,
                index_player_obj_to_env_mapping_gpu=self.level.index_player_obj_to_env_mapping_gpu,
            )

    @wp.kernel
    def reset_gpu(
        terminated: wp.array(dtype=wp.bool), 
        _index_obj_to_env_mapping_gpu: wp.array(dtype=wp.int32), 
        player_health: wp.array(dtype=wp.float32), 
        default_player_health: wp.array(dtype=wp.float32), 
    ):
        tid = wp.tid()
        index_env = _index_obj_to_env_mapping_gpu[tid]

        if terminated[index_env] == False:
            return

        player_health[tid] = default_player_health[tid]

    @wp.kernel
    def apply_step_reward_to_rl(
        index_rl_players_gpu: wp.array(dtype=wp.int32), 
        step_total_rewards_all: wp.array(dtype=wp.float32), 
        step_total_rewards_rl: wp.array(dtype=wp.float32), 
    ):
        tid = wp.tid()

        index_rl_player = index_rl_players_gpu[tid]
        step_total_rewards_rl[tid] = step_total_rewards_all[index_rl_player]

    @wp.kernel
    def apply_step_reward_to_rl_Diff(
        index_rl_players_gpu: wp.array(dtype=wp.int32), 
        index_player_gpu: wp.array(dtype=wp.int32), 
        step_total_rewards_all: wp.array(dtype=wp.float32), 
        step_total_rewards_all_diff: wp.array(dtype=wp.float32), 
        step_total_rewards_rl_diff: wp.array(dtype=wp.float32), 

        num_rl_player: wp.int32,
        is_rl_player_mask_gpu: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()

        index_player = index_player_gpu[tid]
        step_total_rewards_all[index_player] += step_total_rewards_all_diff[index_player]

        if tid >= num_rl_player:
            return

        index_rl_player = index_rl_players_gpu[tid]
        step_total_rewards_rl_diff[tid] = step_total_rewards_all_diff[index_rl_player]

        # if is_rl_player_mask_gpu[tid] >= 0:
        #     index_rl_player = index_rl_players_gpu[tid]
        #     step_total_rewards_rl_diff[tid] = step_total_rewards_all_diff[index_rl_player]
