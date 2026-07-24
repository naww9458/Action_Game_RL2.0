
# import faulthandler
# faulthandler.enable()  # 啟動底層錯誤捕獲

import warp as wp

# # Enable boundary checking: If there is an Out of Bounds array access inside a Kernel, 
# # it will immediately report an error and tell you which line it occurred on.
# wp.config.verify_bounds = True 

# # Force CUDA synchronization and checking: Makes CUDA execute synchronously, 
# # so if a Kernel fails, it will stop immediately at that point without affecting subsequent Kernels.
# wp.config.mode = "debug"        # 啟用內建的陣列邊界檢查 (Bounds Checking)
# wp.config.verify_cuda = True

# # Disable compilation cache (optional, prevents old corrupted cache from interfering)
# wp.clear_kernel_cache()
# wp.init()

import torch
import time
import os
import numpy as np
import sys

from PIL import Image

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
 
from queue import Full, Empty
from script.levels.get_levels import get_level
from script.levels.levels import Levels
from script.simulate.physics_manager import PhysicsManager

from script.role.bodies.articulation_body import ArticulationBody
from script.role.bodies.deformable_body import DeformableBody

from script.levels.rewards.reward_calculator import RewardCalculator
from script.game_config import GameConfig
from script.renderer.renderer import get_renderer, isRendererImplemented
from script.role.abilities.ability import Ability
from script.exceptions import GameClosedException
from utils.fps_calculator import fpsCalculator
from script.role.base_role import BaseRole

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    import multiprocessing as mp
    from multiprocessing import Queue
    from script.role.player import Player
    from script.role.platform import Platform
    from script.role.ability_generated_object import AbilityGeneratedObject

class Game:
    BACKGROUND_COLOR: tuple 

    def __init__(self, 
                 render_mode: str, 
                 model_obs_type: str, 
                 obs_width: int, 
                 obs_height: int, 
                 device,
                 BACKGROUND_COLOR = (41, 50, 65), 
                 physics_manager: PhysicsManager = None, 
                 window_size: tuple[int, int]=(1920, 1080), 
                 max_episode_step: int = None, 
                 player_configs: dict = None, 
                 platform_configs: dict = None, 
                 environment_configs: dict = None, 
                 level_config_path: str = None, 
                 num_env: int = 1, 
                 level: int = None, 
                 sub_level: int = 0, 
                 capture_per_second: int = None, 
                 requires_grad: bool = False,
                 player_controllers: list[str] | None = None,
                ):
        """
        Initialize the balancing ball game.

        Args:
            render_mode: "window" for visible window, "headless" for gym env
            model_obs_type: "game_screen", "state_based", "mixed"
            max_episode_step: 1 step = 1/fps, if fps = 120, 1 step = 1/120
            capture_per_second: save game screen as a image every second, None means no capture
        """
        # Game parameters

        self.BACKGROUND_COLOR=BACKGROUND_COLOR
        self.BACKGROUND_COLOR_RL = (0,0,0)
        self.self_color_RL = (0,255,0)
        self.enemy_color_RL = (255,0,0)
        self.obs_width = obs_width
        self.obs_height = obs_height

        try:
            # May cause attribute error when training because seed is already setted in wapper, just ignore.
            # It is still need for model testing
            import random
            GameConfig.SEED = random.randint(1, int(time.time()))
        except AttributeError:
            pass
        GameConfig.requires_grad = requires_grad
        GameConfig.DEVICE = device

        self.max_episode_step = max_episode_step

        self.render_mode = render_mode
        self.model_obs_type = model_obs_type
        print("self.model_obs_type: ", self.model_obs_type)

        self.renderer = None

        self.human_control = None
        self.is_run_game_human = False
        self.physics_manager = physics_manager
        if physics_manager is None:
            self.physics_manager = PhysicsManager(device=device, viewerGL=None)
        self.articulation_body = ArticulationBody()
        self.deformable_body = DeformableBody()

        self.physics_manager.articulation_body = self.articulation_body
        self.physics_manager.deformable_body = self.deformable_body

        Ability.physics_manager = self.physics_manager
        Ability.articulation_body = self.articulation_body
        Ability.deformable_body = self.deformable_body

        self.num_env = num_env
        self.torch_device = wp.device_to_torch(self.physics_manager.device)
        self.terminated = wp.ones(self.num_env, dtype=wp.bool, device=self.physics_manager.device)
        self.step_total_rewards = None
        self.current_step = wp.zeros(shape=self.num_env, dtype=wp.int32)

        self.level: Levels = get_level(
            level=level, 
            sub_level=sub_level,
            game=self,
            level_config_path=level_config_path,
            player_configs=player_configs, 
            platform_configs=platform_configs, 
            environment_configs=environment_configs,
            player_controllers=player_controllers,
        )
        self.window_x = window_size[0]
        self.window_y = window_size[1]
        self.fps = GameConfig.FPS_ACTION
        self.capture_per_second = capture_per_second
        if capture_per_second:
            self.capture_per_second = capture_per_second * self.fps
            os.makedirs("./capture/", exist_ok=True)

        self.players: Player
        self.platforms: Platform
        self.entities: BaseRole
        self.ability_generated_objects: AbilityGeneratedObject = None
        self.reward_calculator: RewardCalculator
        self.players, self.platforms, self.entities, self.ability_generated_objects, self.reward_calculator = self.level.setup()
        self.num_players = self.players.num_total_object_role

        self.num_objects_total = BaseRole._num_objects_total
        self.num_objects_env = BaseRole._num_objects_env

        # Game state tracking
        self.graph = None
        self.capture_graph_after_step = 1
        self.is_graph_capture_begin = False
        self.game_over = False
        self.episode_total_rewards = np.zeros(shape=self.level.num_objects_total, dtype=np.float32) # Total Score for each player

        # Create folders for captures if needed
        # CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        CURRENT_DIR = "."
        os.makedirs(os.path.dirname(CURRENT_DIR + "/capture/"), exist_ok=True)

        self.setup_render()
        
        self.name_list = BaseRole._name_list
        # self.total_reward_for_test: dict = {name + f"_{i}": {"reset_times": 0, "total_reward": 0} for i, name in enumerate(self.name_list)}
        self.total_reward_for_test: dict = {}
        for i, index_player in enumerate(self.players.index_obj_role):
            name = self.name_list[index_player]
            self.total_reward_for_test[name + f"_{i}"] = {"reset_times": 0, "total_reward": 0}

        print("self.total_reward_for_test: ", self.total_reward_for_test)

        if self.physics_manager.viewerGL is not None:
            self.physics_manager.viewerGL.setup(self)

        self.player_health_cpu, self.rewards_cpu, self.current_step_cpu = self.update_status_for_human()

        if self.render_mode == "window":
            self.fps_calculator = fpsCalculator()

        self.default_action = wp.zeros(shape=[self.players.num_rl_players, GameConfig.ACTION_SHAPE_OFFSET], dtype=wp.float32, device=self.physics_manager.device, requires_grad=GameConfig.requires_grad) # TODO Hard code


    def setup_render(self):
        """Set up renderer and window"""

        self.frame_count = 0
        self.render_fps_counter = 0
        self.render_fps_timer = time.time()
        self.current_render_fps = 0.0

        if self.model_obs_type in ["game_screen", "mixed"]:
            if not isRendererImplemented:
                raise NotImplementedError("renderer is not yet implemented.")

            show_window = (self.render_mode == "window")

            self.renderer = get_renderer(
                num_rl_players=self.players.num_rl_players,
                res=(self.obs_height, self.obs_width), 
                device=self.physics_manager.device, 
                show_window=show_window,
                window_name="RL view Window",
            )

            # Ensure type is int32 for Kernel use
            self.obj_to_env_mapping_torch = wp.to_torch(self.level._index_obj_to_env_mapping_gpu).to(torch.int32)
            body_q_torch = wp.to_torch(self.physics_manager.state_0.body_q)

            # === Build local_to_global_mapping tensor, handle interleaved role sorting ===
            self.local_to_global_mapping_torch = torch.zeros((self.num_env, self.num_objects_env), dtype=torch.int32, device=self.torch_device)
            for e in range(self.num_env):
                env_mask = (self.obj_to_env_mapping_torch == e)
                # torch.nonzero returns global indices that match the condition, put them into this mapping table
                self.local_to_global_mapping_torch[e] = torch.nonzero(env_mask).squeeze(-1).to(torch.int32)
            # =====================================================================

            # --- Get properties of all objects ---
            types = wp.to_torch(self.physics_manager.body_shape_types_gpu).to(torch.int32)
            sizes = wp.to_torch(self.physics_manager.body_size_gpu).to(torch.float32)
            if sizes.dim() == 1 and sizes.numel() >= self.num_objects_total * 3:
                sizes = sizes.view(-1, 3)

            # --- Define single-channel grayscale map (importance sorting) ---
            GRAY_MAP = {
                "background": 0.0,
                "platform":   0.2,  # Static map/walls
                "entities":   0.4,  # Dynamic obstacles
                "bot":        0.7,  # Normal enemy Bot
                "bullet":     1.0   # Bullet (highest contrast)
            }

            # Create global color tensor (Total_Objects, 1)
            colors_torch = torch.full((self.num_objects_total, 1), GRAY_MAP["background"], 
                                    dtype=torch.float32, device=self.torch_device)

            for idx in self.platforms.index_obj_role:
                colors_torch[idx] = GRAY_MAP["platform"]

            for idx in self.entities.index_obj_role:
                colors_torch[idx] = GRAY_MAP["entities"]

            for idx in self.players.index_obj_role:
                colors_torch[idx] = GRAY_MAP["bot"]

            if self.ability_generated_objects is not None:
                for idx in self.ability_generated_objects.index_obj_role:
                    colors_torch[idx] = GRAY_MAP["bullet"]

            # --- Extract template data for "a single environment" for the renderer ---
            # Because the object arrangement structure is the same for every environment, we take Env 0 as the Mesh template
            env0_mask = (self.obj_to_env_mapping_torch == 0)
            
            self.all_object_types = types[env0_mask] # TODO 這是自定義 ID 或許應該和 Newtom 的 self.model.shape_type 適配
            sizes_n = sizes[env0_mask]
            
            # Adjust sizes based on shape (Box: half-length -> full-length)
            box_mask = (self.all_object_types == 1).unsqueeze(-1)
            self.all_object_sizes = torch.where(box_mask, sizes_n * 2.0, sizes_n)
            
            # Extract Env 0 colors as rendering template (N_each_env, 1)
            self.all_object_colors = colors_torch[env0_mask]

            # Call renderer's build_scene_mesh
            self.renderer.build_scene_mesh(
                self.num_objects_env, 
                self.all_object_types,
                sizes=self.all_object_sizes,
                colors=self.all_object_colors
            )

            # Used for returning obs on the first reset.
            body_q_torch = wp.to_torch(self.physics_manager.state_0.body_q)
            self.renderer.compute_v_clip(
                body_q=body_q_torch,
                follow_indices=self.level.index_rl_players_torch,  # Use pre-processed Torch Tensor to avoid per-frame conversion
                obj_to_env_mapping=self.obj_to_env_mapping_torch,
                local_to_global_mapping=self.local_to_global_mapping_torch # Pass mapping table to renderer
            )

        elif self.model_obs_type in ["state_based"]:
            pass
        
        else:
            raise ValueError(f"Invalid obs type: {self.model_obs_type}. Choose from 'game_screen', 'state_based' and 'mixed'")

    def reset(self): 
        """Reset the game state and return the initial observation"""
        # Reset physics objects
        self.level.reset_env(terminated=self.terminated, current_step=self.current_step)
        self.reward_calculator.reset_reward()

        # Reset game state
        if self.render_mode == "window":
            self.game_over = False
            for index_env, is_terminated in enumerate(self.terminated.numpy()): 
                if is_terminated == 1:
                    print(f"=====================================================================")
                    print(f"env {index_env} reset, rewards:")
                    num_player = self.players.num_role_each_env
                    if num_player > 0:
                        offset = self.players.index_role_offset_env_list[index_env]
                        for n in range(num_player):
                            index_player = n+offset
                            print(f"Player {self.name_list[index_player]}: Object index {index_player}, total reward: {self.episode_total_rewards[index_player]}")

                            current_state = self.total_reward_for_test[self.name_list[index_player] + f"_{n+(num_player*index_env)}"]
                            current_state["reset_times"] += 1
                            current_state["total_reward"] += self.episode_total_rewards[index_player]
                            self.total_reward_for_test[self.name_list[index_player] + f"_{n+(num_player*index_env)}"] = current_state
                            self.episode_total_rewards[index_player] = 0

        self.terminated.zero_()

    def step_CUDA_Graph(self, actions: torch.Tensor):
        """
        Take a step in the game using the given actions.

        Args:
            actions: List of continuous actions[-1.0, 1.0] for each player controlling horizontal force

        Returns:
            observation: Game state observation
            step_total_rewards: List of rewards for each player
            terminated: Whether episode is done
            info: Additional information
        """

        # print("actions: ", actions)
        # step_total_rewards, terminated = self.level.action()

        # Step the physics simulation

        # # TODO may cause body_qd turn nan when rl action run in CUDA Graph
        if actions is not None:
            actions_wp = wp.from_torch(actions, dtype=wp.float32)
        else:
            actions_wp = self.default_action

        self._apply_inspector_rl_actions(actions_wp)
        self.players.rl_action(actions=actions_wp)

        if self.graph is not None:
            self._apply_inspector_pinned_controls()
            wp.capture_launch(self.graph)
            self._apply_inspector_commands()
        else:
            # Start capturing
            if self.current_step.numpy()[0] >= self.capture_graph_after_step:
                wp.capture_begin(self.physics_manager.device)
                self.is_graph_capture_begin = True

        # =================================================================================================================
            # if actions is not None:
            #     self.players.rl_action(actions=actions)
    
            self.players.bot_action(index_obj_to_env_mapping_gpu=self.level._index_obj_to_env_mapping_gpu)
            self.physics_manager.simulate()

            self.level.update_game_status(physics_manager=self.physics_manager, reward_calculator=self.reward_calculator, num_env=self.num_env, current_step=self.current_step)

            self._apply_inspector_commands()
            
            self.reward_calculator.calculate_rewards(
                current_step=self.current_step, 
                actions=actions_wp, 
                max_episode_step=self.max_episode_step, 
                command_vel=self.level.commands
            )
            if hasattr(self.level, "on_step_actions"):
                self.level.on_step_actions(actions_wp)
            self.step_total_rewards = self.reward_calculator.step_total_rewards_rl
            self.handle_update_sub_step()

            self.physics_manager.reset_obj()
                
            self.obs_state_based = self._get_observation_state_based()
            
            if self.renderer is not None:
                body_q_torch = wp.to_torch(self.physics_manager.state_0.body_q)

                self.renderer.compute_v_clip(
                    body_q=body_q_torch,
                    follow_indices=self.level.index_rl_players_torch,  # Use pre-processed Torch Tensor to avoid per-frame conversion
                    obj_to_env_mapping=self.obj_to_env_mapping_torch,
                    local_to_global_mapping=self.local_to_global_mapping_torch # Pass mapping table to renderer
                )
                
        # =================================================================================================================

            if self.is_graph_capture_begin:
                self.graph = wp.capture_end(self.physics_manager.device)
                print("CUDA Graph captured and compiled!")

        # self.obs_state_based = self._get_observation_state_based()
        if self.renderer is not None:
            self.obs_game_screen = self._get_observation_game_screen()

        if self.model_obs_type == "state_based":
            self.obs = self.obs_state_based
            
        elif self.model_obs_type == "game_screen":
            self.obs = self.obs_game_screen

        elif self.model_obs_type == "mixed":
            self.obs = {
                "visual": self.obs_game_screen,
                "state": self.obs_state_based
            }

        # print("self.step_total_rewards: ", self.step_total_rewards)
        # print("self.obs: ", self.obs)
        # print("self.level.commands: ", self.level.commands)
        # print(" ")

        return self.obs, self.step_total_rewards, self.terminated 

    def step_Diff(self, actions: torch.Tensor):

        if actions is not None:
            # 將 Torch Tensor 轉換為帶梯度的 Warp Array，使外部的全局 Tape 能夠追蹤它
            actions_wp = wp.from_torch(actions, dtype=wp.float32, requires_grad=GameConfig.requires_grad)
        else:
            actions_wp = self.default_action

        self._apply_inspector_rl_actions(actions_wp)
        self.players.rl_action(actions=actions_wp)

        self.players.bot_action(index_obj_to_env_mapping_gpu=self.level._index_obj_to_env_mapping_gpu)
        self.physics_manager.simulate()

        self.level.update_game_status(
            physics_manager=self.physics_manager, 
            reward_calculator=self.reward_calculator, 
            num_env=self.num_env, 
            current_step=self.current_step
        )

        self._apply_inspector_commands()

        self.reward_calculator.calculate_rewards( 
            current_step=self.current_step, 
            actions=actions_wp, 
            max_episode_step=self.max_episode_step, 
            command_vel=self.level.commands,
        )
        if hasattr(self.level, "on_step_actions"):
            self.level.on_step_actions(actions_wp)
        self.step_total_rewards = self.reward_calculator.step_total_rewards_rl
        self.step_total_rewards_diff = self.reward_calculator.step_total_rewards_rl_diff
        # print("self.step_total_rewards: ", self.step_total_rewards)

        self.handle_update_sub_step()

        self.obs_state_based = self._get_observation_state_based()
        
        if self.renderer is not None:
            body_q_torch = wp.to_torch(self.physics_manager.state_0.body_q)

            self.renderer.compute_v_clip(
                body_q=body_q_torch,
                follow_indices=self.level.index_rl_players_torch,  # Use pre-processed Torch Tensor to avoid per-frame conversion
                obj_to_env_mapping=self.obj_to_env_mapping_torch,
                local_to_global_mapping=self.local_to_global_mapping_torch # Pass mapping table to renderer
            )
            self.obs_game_screen = self._get_observation_game_screen()
            
        # 子彈重置等操作，留在這裡沒有問題

        self.physics_manager.reset_obj()

        if self.model_obs_type == "state_based":
            self.obs = self.obs_state_based
            
        elif self.model_obs_type == "game_screen":
            self.obs = self.obs_game_screen

        elif self.model_obs_type == "mixed":
            self.obs = {
                "visual": self.obs_game_screen,
                "state": self.obs_state_based
            }

        return self.obs, self.step_total_rewards, self.step_total_rewards_diff, self.terminated, actions_wp

    def _get_observation_game_screen(self) -> dict[np.ndarray, np.ndarray]:
        """Convert game state to observation for RL agent"""
        # update particles and draw them
        self.obs_game_screen = self.renderer.render()

        return self.obs_game_screen

    def _get_observation_state_based(self) -> 'torch.Tensor':
        """Public method to get the current observation without taking a step"""
        obs = self.level._get_observation_state_based()

        return obs

    def _apply_inspector_pinned_controls(self):
        viewer = self.physics_manager.viewerGL
        if viewer is not None and hasattr(viewer, "object_inspector"):
            viewer.object_inspector.apply_pinned_controls()

    def _apply_inspector_rl_actions(self, actions_wp):
        viewer = self.physics_manager.viewerGL
        if viewer is not None and hasattr(viewer, "object_inspector"):
            viewer.object_inspector.apply_rl_action_pins(actions_wp)

    def _apply_inspector_commands(self):
        viewer = self.physics_manager.viewerGL
        if viewer is not None and hasattr(viewer, "object_inspector"):
            viewer.object_inspector.apply_command_pins()

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "server":
            self.screen_data = self.physics_manager.state_0.body_q
            return None

        if self.render_mode == "window":
            self.fps_calculator.update()
            
            if not self.game_over: # TODO 沒加上最後的獎勵
                self.player_health_cpu, self.rewards_cpu, self.current_step_cpu = self.update_status_for_human()

            if self.physics_manager.viewerGL is not None:
                viewer = self.physics_manager.viewerGL
                viewer.possess_offsets = self.players.get_possess_offsets()
                # ViewerGL usually handles global data, keep as is
                if viewer.renderer.window.has_exit:
                    raise GameClosedException()
                
                viewer.render(
                    player_health=self.player_health_cpu, 
                    frame_dt=self.physics_manager.frame_dt, 
                    state_0=self.physics_manager.state_0, 
                    index_player_gpu=self.players.index_obj_role_gpu, 
                )

        if self.renderer is not None and self.renderer.show_window:
            self.renderer._display_results(self.obs_game_screen)

            if isinstance(self.capture_per_second, int | float):
                if self.frame_count % self.capture_per_second == 0:  # Every second at 60 FPS
                    pixels = self.obs_game_screen # Gets numpy array of (H, W, 1)
                    if isinstance(pixels, dict):
                        for key, pixel in pixels.items():
                            img_to_save = pixel.squeeze() 

                            # Convert to Image object and save (mode='L' means 8-bit grayscale)
                            Image.fromarray(img_to_save).save(f"capture/frame_{key}_{self.frame_count/60}.png")
                            print(f"Image saved as capture/frame_{self.frame_count/60}.png")
                    else:
                        img_to_save = pixel.squeeze() 

                        # Convert to Image object and save (mode='L' means 8-bit grayscale)
                        Image.fromarray(img_to_save).save(f"capture/frame_{key}_{self.frame_count/60}.png")
                        print(f"Image saved as capture/frame_{self.frame_count/60}.png")
                self.frame_count += 1

            pass
        return None

    def close(self):
        """Close the game and clean up resources"""
        print("The Close function in game class is not actually implement.")
        pass

    def run_game_human(self, 
                       event_is_window_setup_ready: 'mp.Event', 
                       event_is_game_logic_keymapping_setup_ready: 'mp.Event', 
                       physics_manager_state_queue: 'Queue', 
                       human_input_queue: 'Queue',
                       is_lock_fps: bool=True
                      ):
        """Run the game in standalone mode with keyboard controls"""

        setup_data = {'body_shape_types': self.physics_manager.body_shape_types, 'body_size': self.physics_manager.body_size}
        physics_manager_state_queue.put(setup_data)
        self.is_run_game_human = True
        from script.role.abilities.key_mapping import KeyMapping
        key_mapping = human_input_queue.get()
        KeyMapping.Keyboard_Mappings = key_mapping["Keyboard_Mappings"]
        viewer_controls_config = key_mapping.get("Viewer_Controls_Config", {})
        for ability in self.players.abilities_instance_list:
            if hasattr(ability, "apply_runtime_keymapping"):
                ability.apply_runtime_keymapping()
            else:
                ability.setup_keymapping()

        from script.human_control import HumanControl
        self.human_control = HumanControl(self)
        self.human_control.setup_reset_keymapping(viewer_controls_config)

        event_is_game_logic_keymapping_setup_ready.set()

        while not event_is_window_setup_ready.is_set():
            time.sleep(0.1)

        target_fps = GameConfig.FPS_ACTION
        frame_duration = 1.0 / target_fps
        self.run = True

        # for debug obs data
        obs_prev = self._get_observation_state_based()
        try:
            while self.run:
                start_time = time.perf_counter()
                obs = None
                step_total_rewards_diff = None
                terminated = None
                try:
                    human_input = human_input_queue.get_nowait()
                except Empty:
                    human_input = {
                        "follow_body_index": None,
                        "keyboard_keys": None,
                        "mouse_buttons": None,
                        "look_yaw": None,
                        "look_pitch": None,
                        "simulation_control": {},
                    }

                sc = human_input.get("simulation_control", {})
                paused = bool(sc.get("paused", False))
                auto_reset_on_env_end = bool(sc.get("auto_reset_on_env_end", False))
                manual_reset_enabled = bool(sc.get("manual_reset_enabled", True))
                reset_requested = bool(sc.get("reset_requested", False))
                should_step = (not self.game_over) and (not paused)

                if should_step:
                    actions = None
                    if human_input["follow_body_index"] is not None:
                        actions = self.human_control.get_player_actions(**human_input)

                    if GameConfig.requires_grad:
                        obs, step_total_rewards, step_total_rewards_diff, terminated, _ = self.step_Diff(actions)
                    else:
                        obs, step_total_rewards, terminated = self.step_CUDA_Graph(actions)

                    try:
                        physics_manager_state_queue.put_nowait(self.physics_manager.state_0.body_q.numpy())
                    except Full:
                        pass

                self.render()

                if should_step and terminated is not None:
                    if terminated.numpy().tolist()[0] == 1:
                        if auto_reset_on_env_end:
                            self.terminated[0:1].fill_(True)
                            self.reset()
                        else:
                            self.game_over = True
                    elif auto_reset_on_env_end:
                        self.reset()

                if manual_reset_enabled and reset_requested:
                    self.terminated[0:1].fill_(True)
                    self.reset()


                # for debug obs data
                index = human_input["follow_body_index"]
                if index is not None:
                    if obs is not None:
                        # if (obs != obs_prev).any():
                        #     print(f"player index: {index} obs: ", obs[index, 3:8].cpu().numpy().tolist())
                        # print(f"obs: ", obs)
                        if isinstance(obs, dict):
                            obs_prev = obs["state"].clone()
                        else:
                            obs_prev = obs.clone()

                    # if self.step_total_rewards.numpy()[index] != 0.0:
                    #     print("step_total_rewards: ", self.step_total_rewards.numpy()[index])

                        
                    # if step_total_rewards_diff.numpy()[index] > 0.0:
                    #     print("step_total_rewards_diff: ", step_total_rewards_diff.numpy()[index])

                # print(f"Step total reward: {step_total_rewards}")

                elapsed_time = time.perf_counter() - start_time

                # If executing too fast, sleep for the remaining time
                if elapsed_time < frame_duration and is_lock_fps:
                    time.sleep(frame_duration - elapsed_time)


        except GameClosedException:
            print("Window closure detected, cleaning up resources...")
        except Exception as e:
            print(f"Exception occurred: {e}")
            print("print traceback: ")
            import traceback
            traceback.print_exc()
        finally:
            # Perform final cleanup (see below)
            self.physics_manager.cleanup() 
            print("Process exited safely")
            sys.exit(0)

        self.close()

    def update_status_for_human(self):
        """
        Calculate and return the reward for the current state.
        """

        # Checks related to player survival must be at the top, because subsequent reward calculations rely on whether the player is alive
        player_health_cpu = self.reward_calculator.player_health.numpy().tolist()
        rewards_cpu = self.reward_calculator.step_total_rewards_all.numpy()
        current_step_cpu = self.current_step.numpy().tolist()
        # Handle collision rewards between player and entities, including state reset for each Step

        self.episode_total_rewards += rewards_cpu

        return player_health_cpu, rewards_cpu, current_step_cpu

    def handle_update_sub_step(self):
        """
        Handle status update events for sub steps
        """
        if self.ability_generated_objects.num_total_object_role > 0:
            self.ability_generated_objects.update_lifetimes()
        for ability in self.players.abilities_instance_list:
            ability.update_cooldown()










