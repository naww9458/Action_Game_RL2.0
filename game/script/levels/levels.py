import torch
import warp as wp
import numpy as np

from script.game_config import GameConfig
from script.role.base_role import BaseRole
from script.role.controller_utils import normalize_controller
from script.role.player import Player
from script.role.platform import Platform
from script.role.entity import Entity
from script.role.tool import Tool
from script.role.ability_generated_object import AbilityGeneratedObject
from script.simulate.tool_mount_setup import setup_tool_mount_joints
from script.role.objects.object_template.loader import ensure_object_templates_registered
from script.levels.rewards.reward_calculator import RewardCalculator
from script.role.abilities.articulation_control_config.profile_registry import resolve_player_runtime_pattern

from typing import TYPE_CHECKING, List, Optional
if TYPE_CHECKING:
    # 將導致循環導入的 import 語句移到這裡
    import torch
    from script.game import Game
    from script.simulate.physics_manager import PhysicsManager
    from script.simulate.mount_joint_registry import MountJointRegistry


class Levels:
    def __init__(self, game: 'Game', level_configs: dict = None):
        self.game = game
        self.num_env = self.game.num_env

        self.physics_manager = self.game.physics_manager
        self.articulation_body = self.game.articulation_body
        self.deformable_body = self.game.deformable_body
        self.action_shape_offset: int = 0

        self.level_configs = level_configs or {}

        ensure_object_templates_registered()

        # Optional per-env command buffer (e.g. velocity commands). Subclasses allocate in setup().
        self.commands: Optional[wp.array2d] = None
        self.command_labels: List[str] = []
        BaseRole._num_env = self.num_env
        BaseRole._physics_manager = self.physics_manager
        
        BaseRole._articulation_body = self.articulation_body
        BaseRole._deformable_body = self.deformable_body

        print("Setup Player!")
        self.players: Player = Player(configs=self.level_configs.get("player_configs", []))
        print("Setup Platform!")
        self.platforms: Platform = Platform(configs=self.level_configs.get("platform_configs", []))
        print("Setup Entity!")
        self.entities: Entity = Entity(configs=self.level_configs.get("entity_configs", []))
        print("Setup Tool!")
        tool_configs = self.level_configs.get("tool_configs") or []
        self._inherit_host_spawn_pose_for_tools(tool_configs)
        self.tools: Tool | None = Tool(configs=tool_configs) if tool_configs else None
        print("Setup AbilityGeneratedObject!")
        self.abilities_objects: AbilityGeneratedObject = AbilityGeneratedObject(configs=self.level_configs.get("ability_generated_object_configs", []))

        setup_tool_mount_joints(self)

        role_objects = [self.players, self.platforms, self.entities, self.abilities_objects]
        if self.tools is not None:
            role_objects.insert(3, self.tools)
        BaseRole.physics_index_match_to_role(role_objects=role_objects, num_env=self.num_env)
        BaseRole._num_objects_total = BaseRole._num_objects_env * self.num_env
        self.num_objects_total = BaseRole._num_objects_total
        self.mount_joint_registry: MountJointRegistry = getattr(self, "mount_joint_registry", None)
        GameConfig.NUM_PLAYERS = self.players.num_total_object_role
        GameConfig.NUM_OBJECTS_TOTAL = self.num_objects_total
        self.abilities_objects.update_owner(num_object_total=BaseRole._num_objects_total, 
                                            num_players_each_env=self.players.num_role_each_env, 
                                            index_players_offset_env_list=self.players.index_role_offset_env_list,
                                           )
        
        self.fps = GameConfig.FPS_ACTION

    def _inherit_host_spawn_pose_for_tools(self, tool_configs: List[dict]) -> None:
        """For tools bound to a host via ``host_player_id``, inherit the host's
        spawn pose so the tool's own initial pose fields can be omitted in YAML.

        The tool still needs a valid spawn transform at build time (before the
        mount joint snaps it onto the host), so it spawns at its host's pose.
        """
        player_configs = self.level_configs.get("player_configs") or []
        host_by_id = {}
        for cfg in player_configs:
            hid = str(cfg.get("id") or cfg.get("name") or "")
            if hid:
                host_by_id[hid] = cfg
        pose_keys = (
            "default_position",
            "default_rotation",
            "default_velocity",
            "default_angular_velocity",
        )
        for tool_cfg in tool_configs:
            host_id = tool_cfg.get("host_player_id")
            if not host_id:
                continue
            host_cfg = host_by_id.get(str(host_id))
            if host_cfg is None:
                raise ValueError(
                    f"Tool '{tool_cfg.get('name', '')}' host_player_id={host_id!r} "
                    f"does not match any player_configs name. Available: {sorted(host_by_id)}"
                )
            for key in pose_keys:
                if tool_cfg.get(key) is None and host_cfg.get(key) is not None:
                    tool_cfg[key] = host_cfg.get(key)

    def resolve_player_pattern(self, player_index: int = 0) -> str:
        """Resolve articulation-body player pattern from level YAML player_configs."""
        player_configs = self.level_configs.get("player_configs", [])
        if not player_configs:
            raise ValueError("level_configs is missing player_configs.")
        if player_index < 0 or player_index >= len(player_configs):
            raise IndexError(
                f"player_index {player_index} is out of range for {len(player_configs)} player_configs."
            )
        player_config = dict(player_configs[player_index])
        return resolve_player_runtime_pattern(player_config)

    def setup(self):
        """
        通用設置方法，採用模組化流程。
        """

        # 物理引擎初始化
        self.physics_manager.setup(num_env=self.num_env)
        self.calculate_env_offset_and_sort()

        # 🌟 建立並行物理 View, 這個 num_objects_env 指的是高層次的 Object 而不是 model 中的 body，比如一個 Unitree G1 在 model 中有多個 body 但是在環境中只能算一個 Object
        self.articulation_body.build_view(device=GameConfig.DEVICE, model=self.physics_manager.model, num_objects_env=BaseRole._num_objects_env)
        self.deformable_body.build_view(device=GameConfig.DEVICE, model=self.physics_manager.model, num_objects_env=BaseRole._num_objects_env)
        if self.mount_joint_registry is not None and self.mount_joint_registry.records:
            solver_type = str(
                (self.level_configs.get("environment_configs") or {}).get("solver_config", {}).get(
                    "type", ""
                )
            )
            self.mount_joint_registry.bind_model(
                self.physics_manager.model,
                self.physics_manager.device,
                self.articulation_body.num_joint_dofs_env,
                self.articulation_body.num_rigid_bodies_env,
                solver_type=solver_type,
                num_env=self.num_env,
                solver=self.physics_manager.solver_handler.solver if self.physics_manager.solver_handler else None,
            )
        self.initialize_player_roles()

        for i, ability in enumerate(self.players.abilities_instance_list):
            ability.setup_cooldown(num_objects_total=BaseRole._num_objects_total, owners_ability_list=self.players.abilities_owner_list[i])
            ability.setup_player_to_env_mapping(
                index_role_offset_env_gpu=self.players.index_role_offset_env_gpu,
                num_role_each_env=self.players.num_role_each_env,
                role_type="player",
            )

        if self.tools is not None:
            for i, ability in enumerate(self.tools.abilities_instance_list):
                ability.setup_cooldown(
                    num_objects_total=BaseRole._num_objects_total,
                    owners_ability_list=self.tools.abilities_owner_list[i],
                )
                ability.setup_player_to_env_mapping(
                    index_role_offset_env_gpu=self.tools.index_role_offset_env_gpu,
                    num_role_each_env=self.tools.num_role_each_env,
                    role_type="tool",
                )
            self._configure_tool_abilities()

        # 更新全域配置與 Action Space
        self._configure_articulation_abilities()
        self._finalize_config()
        num_total_player = self.players.num_role_each_env * self.num_env
        num_total_platform = self.platforms.num_role_each_env * self.num_env
        num_total_abilities_object = self.abilities_objects.num_role_each_env * self.num_env

        print(f"Created {self.num_env} env and following objects: ")
        print(f"total: {num_total_player} each env: {self.players.num_role_each_env} players, \ntotal: {num_total_platform} each env: {self.platforms.num_role_each_env} platforms, \ntotal: {num_total_abilities_object} each env: {self.abilities_objects.num_role_each_env} abilities objects.")
        return self.players, self.platforms, self.entities, self.abilities_objects

    def _finalize_config(self):
        """處理 Action Space 和全域 GameConfig"""
        action_space_config = []
        print("abilities_name_index_dict: ", self.players.abilities_name_index_dict)
        print("abilities_instance_list: ", self.players.abilities_instance_list)
        print("abilities_owner_list: ", self.players.abilities_owner_list)

        player_configs = self.level_configs.get("player_configs", [])
        rl_action_shape_offset = 0
        num_players = self.players.num_role_each_env

        for index_p in range(num_players):
            player_actions = {}
            controller = (
                normalize_controller(player_configs[index_p].get("controller"))
                if index_p < len(player_configs)
                else "Bot"
            )
            uses_rl_actions = controller == "RL"

            for i, owner in enumerate(self.players.abilities_owner_list):
                if index_p in owner:
                    key = next((k for k, v in self.players.abilities_name_index_dict.items() if v == i), None)

                    abilities_instance = self.players.abilities_instance_list[i]
                    player_actions[key] = abilities_instance.get_action_spec()

                    if uses_rl_actions:
                        abilities_instance.action_shape_offset = rl_action_shape_offset
                        print(
                            f"{abilities_instance.ability_name}.action_shape_offset",
                            abilities_instance.action_shape_offset,
                        )

                    action_shape = player_actions[key].get("shape", 0)
                    if isinstance(action_shape, int) and action_shape > 0:
                        rl_action_shape_offset += action_shape
                    elif action_shape not in (0, "auto", None):
                        rl_action_shape_offset += int(action_shape)

            action_space_config.append(player_actions)

        self.action_shape_offset = rl_action_shape_offset
        try:
            GameConfig.ACTION_SPACE_CONFIG = action_space_config
            GameConfig.ACTION_SHAPE_OFFSET = rl_action_shape_offset
        except AttributeError:
            print("Warning: GameConfig attributes are immutable or missing.")

        print(f"Action Space Config: {action_space_config}")

    def _configure_tool_abilities(self):
        tool_configs = self.level_configs.get("tool_configs") or []
        for ability in self.tools.abilities_instance_list:
            if hasattr(ability, "configure_from_tool_configs"):
                ability.configure_from_tool_configs(tool_configs, self)
            if hasattr(ability, "configure_from_tool_configs_post_indices"):
                ability.configure_from_tool_configs_post_indices(self)

    def _configure_articulation_abilities(self):
        player_configs = self.level_configs.get("player_configs", [])
        for ability in self.players.abilities_instance_list:
            ability.configure_from_player_configs(player_configs, self)
            ability.configure_from_player_configs_post_indices(self)

    def calculate_env_offset_and_sort(self): 
        """
        This function generates an array of environment indices corresponding to object indices, used for quickly resetting the environment.
        """
        index_all_object_sort_env_list = np.arange(0, BaseRole._num_objects_total)

        _index_offset_env = [BaseRole._num_objects_env * i for i in range(self.num_env + 1)] # 這裏加 1 是爲了防止 update_reset_mask_kernel 中出現 GPU 記憶體越界寫入
        self._index_offset_env_gpu = wp.array(data=_index_offset_env, dtype=wp.int32, device=GameConfig.DEVICE)
        self._index_obj_to_env_mapping_gpu = self.physics_manager.model.body_world
        self._index_all_object_sort_env_gpu = wp.array(
            data=index_all_object_sort_env_list, dtype=wp.int32, device=GameConfig.DEVICE
        )
        self.index_player_obj_to_env_mapping_gpu = self.players.index_obj_role_to_env_mapping_gpu

    def action(self):
        """
        shape state changes in the game
        """
        # Noting to do in base level
        pass

    def update_game_status(self, physics_manager: 'PhysicsManager', reward_calculator: 'RewardCalculator', num_env: wp.int32, current_step: wp.array):
        # Noting to do in base level
        pass

    def _get_observation_state_based(self) -> 'torch.Tensor':
        """
        Return the current observation without taking a step
        """

        pass

    def initialize_player_roles(self):
        color_human = wp.vec3(1, 0, 0)
        color_rl = wp.vec3(0, 1, 0)
        color_bot = wp.vec3(0, 0, 1)

        self.num_rl_players = 0
        bot_player_id = "Bot_player"
        self.index_rl_players: list[int] = []
        self.is_rl_player_mask: list[int] = []
        self.index_bot_players: list[int] = []
        self.is_bot_player_mask: list[int] = []
        self.color_player_shape_list = []

        for index in self.players.index_obj_role:
            name = BaseRole._name_list[index]
            if not name:
                name = f"{bot_player_id}{index}"
                BaseRole._name_list[index] = name

            params = BaseRole._object_game_params[index]
            controller = normalize_controller(params.get("controller"))
            params["controller"] = controller

            if controller in ("Human", "RL"):
                self.index_rl_players.append(index)
                self.is_rl_player_mask.append(self.num_rl_players)
                self.is_bot_player_mask.append(-1)
                self.color_player_shape_list.append(
                    color_human if controller == "Human" else color_rl
                )
                self.num_rl_players += 1
            elif controller == "Bot":
                self.index_bot_players.append(index)
                self.is_rl_player_mask.append(-1)
                self.is_bot_player_mask.append(0)
                self.color_player_shape_list.append(color_bot)
            else:
                self.is_rl_player_mask.append(-1)
                self.is_bot_player_mask.append(-1)

        print("players assigned: ", BaseRole._name_list)
        print("self.index_rl_players: ", self.index_rl_players)

        self.index_rl_players_gpu = wp.array(data=self.index_rl_players, dtype=wp.int32, device=GameConfig.DEVICE)
        self.num_rl_players = len(self.index_rl_players)
        self.index_rl_players_torch = wp.to_torch(self.index_rl_players_gpu).to(torch.int32)
        self.is_rl_player_mask_gpu = wp.array(data=self.is_rl_player_mask, dtype=wp.int32, device=GameConfig.DEVICE)
        self.index_bot_players_gpu = wp.array(data=self.index_bot_players, dtype=wp.int32, device=GameConfig.DEVICE)
        self.num_bot_players = len(self.index_bot_players)
        self.is_bot_player_mask_gpu = wp.array(data=self.is_bot_player_mask, dtype=wp.int32, device=GameConfig.DEVICE)
        self.game.color_player_shape_gpu = wp.array(data=self.color_player_shape_list, dtype=wp.vec3, device=GameConfig.DEVICE)

        self.players.update_index_rl_and_bot(
            index_rl_players_gpu=self.index_rl_players_gpu,
            num_rl_players=self.num_rl_players,
            is_rl_player_mask_gpu=self.is_rl_player_mask_gpu,
            index_bot_players_gpu=self.index_bot_players_gpu,
            num_bot_players=self.num_bot_players,
            is_bot_player_mask_gpu=self.is_bot_player_mask_gpu,
        )

    def reset_env(self, terminated, current_step):
        if self.num_objects_total <= 0:
            return
        
        # 1. 偵測哪些環境過期/Terminated，並更新對應的一維剛體 reset_mask
        wp.launch(
            kernel=self.update_reset_mask_kernel,
            dim=self.num_env,
            inputs=[
                terminated,
                current_step,
                self._index_offset_env_gpu,
                self.physics_manager.reset_mask_gpu,
            ],
            device=GameConfig.DEVICE
        )

        # 2. Release every tool in the reset envs so the upcoming reset truly
        #    restores the initial, unattached state.
        self._reset_tool_attachments(terminated)

        # 3. 🌟 一鍵委託物理管理器進行多態重置！無須再傳入一堆重複、臃腫的 position/rotation 陣列
        self.physics_manager.reset_obj()

        # 4. Re-mount tools configured with `start_attached: true` in reset envs.
        self._restore_start_attached_tools(terminated)

    def _reset_worlds(self, terminated) -> Optional[List[int]]:
        """World indices being reset.

        Returns ``None`` when the terminated data is unavailable/empty, which
        callers treat as "reset all worlds" (matching the initial full reset).
        Returns an empty list when no world is actually terminated.
        """
        try:
            terminated_np = terminated.numpy() if hasattr(terminated, "numpy") else np.asarray(terminated)
        except Exception:
            terminated_np = None
        if terminated_np is None or len(terminated_np) == 0:
            return None
        return [
            int(i)
            for i in range(min(len(terminated_np), self.num_env))
            if bool(terminated_np[i])
        ]

    def _reset_tool_attachments(self, terminated) -> None:
        """Detach every tool in the reset envs.

        ``reset_obj`` restores each object to its (randomized) spawn transform,
        so a tool that was mounted during gameplay (e.g. via the U key) must
        also lose its mount joint / equality constraint — otherwise the tool
        stays "connected" to its host even though the world was reset.
        """
        registry = self.mount_joint_registry
        if registry is None or not registry.records:
            return
        if getattr(registry, "_model", None) is None:
            return

        worlds = self._reset_worlds(terminated)
        if worlds is not None and not worlds:
            return
        registry.reset_attachments(worlds=worlds)

    def _restore_start_attached_tools(self, terminated) -> None:
        """After a reset, re-attach tools whose level config has `start_attached: true`.

        ``reset_obj`` restores each object to its (randomized) spawn transform, so a
        pre-attached tool must be snapped back onto its host and its mount joint
        re-enabled in every world that was reset.
        """
        registry = self.mount_joint_registry
        if registry is None or not registry.records:
            return
        if getattr(registry, "_model", None) is None:
            return
        if not any(getattr(r, "start_attached", False) for r in registry.records.values()):
            return

        worlds = self._reset_worlds(terminated)
        if worlds is not None and not worlds:
            return

        state = self.physics_manager.state_0
        registry.attach_start_attached(
            state.body_q,
            state.body_qd,
            state.joint_q,
            worlds=worlds,
            body_f=getattr(state, "body_f", None),
            joint_qd=getattr(state, "joint_qd", None),
            body_q_prev=getattr(state, "body_q_prev", None),
        )

    @wp.kernel
    def update_reset_mask_kernel(
        terminated: wp.array(dtype=wp.bool), 
        current_step: wp.array(dtype=wp.int32), 
        _index_offset_env_gpu: wp.array(dtype=wp.int32), 
        reset_mask: wp.array(dtype=wp.int32), 
    ):
        tid = wp.tid()
        
        # 檢查這個環境是否需要重置
        if terminated[tid] == False:
            return
        
        current_step[tid] = 0
        # 獲取該環境物件在 _index_all_object_sort_env_gpu 中的區間
        start = _index_offset_env_gpu[tid]
        end = _index_offset_env_gpu[tid + 1]
        
        # 遍歷該環境的所有物件，將物理引擎對應的 reset_mask 設為 1
        # 注意：Warp 內核不支援 [start:end] 切片迭代，必須使用 range
        for i in range(start, end):
            reset_mask[i] = 1


class Level_Default(Levels):
    """
    Level default:
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.window_size = (GameConfig.space_x, GameConfig.space_y, GameConfig.space_z)
        self.obs_torch = torch.zeros((self.num_env,), dtype=torch.float32, device=GameConfig.DEVICE)

    def setup(self):
        super().setup()
        
        # 初始重置環境
        self.reset_env(self.game.terminated, self.game.current_step)
        self.physics_manager.simulate()

        self.reward_calculator = RewardCalculator(
            level=self,
            terminated=self.game.terminated,
        )

        self.gravity = self.physics_manager.gravity[2]
        return self.players, self.platforms, self.entities, self.abilities_objects, self.reward_calculator

    def action(self):
        """
        shape state changes in the game
        """

        pass

    def update_game_status(self, physics_manager: 'PhysicsManager', reward_calculator: RewardCalculator, num_env: wp.int32, current_step: wp.array):
        """
        shape state changes in the game
        """

        wp.launch(
            kernel=self.update_game_status_gpu,
            dim=num_env,
            inputs=[
                current_step,
            ],
            device=GameConfig.DEVICE
        )

    def reset_env(self, terminated, current_step):
        """
        Reset the env to its initial state.
        """
        super().reset_env(terminated=terminated, current_step=current_step)

    def _get_observation_state_based(self) -> torch.Tensor:
        return self.obs_torch

    def _get_observation_image_based(self) -> torch.Tensor:
        pass

    @wp.kernel
    def update_game_status_gpu(
        current_step: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        current_step[tid] += 1



