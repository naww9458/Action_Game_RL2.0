import torch
import warp as wp
import numpy as np

from script.game_config import GameConfig

try:
    from levels.rewards.reward_calculator import RewardCalculator
    from levels.rewards.game_end_reward import G1LocomotionTerminator
    from levels.rewards.mjlab.g1_velocity_locomotion_reward import G1VelocityLocomotionReward
    from training.level_defaults import get_default_train_cfg
    from levels.levels import Levels
    from sensors.foot_contact_sensor import FootContactSensor
    from script.role.objects.object_template.mjlab_unitree_g1.g1_velocity_locomotion_provider import (
        G1VelocityLocomotionProvider,
    )
except ImportError:
    from script.levels.rewards.reward_calculator import RewardCalculator
    from script.levels.rewards.game_end_reward import G1LocomotionTerminator
    from script.levels.rewards.mjlab.g1_velocity_locomotion_reward import G1VelocityLocomotionReward
    from training.level_defaults import get_default_train_cfg
    from script.levels.levels import Levels
    from script.sensors.foot_contact_sensor import FootContactSensor
    from script.role.objects.object_template.mjlab_unitree_g1.g1_velocity_locomotion_provider import (
        G1VelocityLocomotionProvider,
    )

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from script.simulate.physics_manager import PhysicsManager


class Level5_0(Levels):
    """Unitree G1 flat velocity tracking (mjlab-aligned)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.window_size = (GameConfig.space_x, GameConfig.space_y, GameConfig.space_z)

        self.obs_torch = None
        self.obs_wp = None
        self.obs_dim = 0
        self.rl_action_dim = 29
        self.history_len = 1
        self.flat_obs_dim = 0
        self.single_obs_wp = None

        self.g1_provider: G1VelocityLocomotionProvider | None = None
        self.foot_sensor = None
        self.push_timer = None
        self.push_seeds = None
        self.push_seed_offsets = None

    def setup(self):
        super().setup()

        pattern = self.resolve_player_pattern()
        self.g1_provider = G1VelocityLocomotionProvider(
            num_env=self.num_env,
            device=GameConfig.DEVICE,
            articulation_body=self.articulation_body,
            pattern=pattern,
            history_len=self.history_len,
        )
        self.g1_provider.setup()

        self.obs_dim = self.g1_provider.obs_dim
        self.rl_action_dim = self.g1_provider.rl_action_dim
        self.flat_obs_dim = self.g1_provider.flat_obs_dim
        self.obs_wp = self.g1_provider.obs_wp
        self.obs_torch = self.g1_provider.obs_torch
        self.single_obs_wp = self.g1_provider.single_obs_wp
        self.commands = self.g1_provider.commands
        self.command_labels = self.g1_provider.command_labels
        self.policy_actions = self.g1_provider.policy_actions
        self.prev_actions = self.g1_provider.prev_actions

        self.push_timer = wp.zeros(self.num_env, dtype=wp.float32, device=GameConfig.DEVICE)
        seed_base = getattr(GameConfig, "SEED", 31415926)
        self.push_seeds = wp.array(
            np.arange(seed_base + 10000, seed_base + 10000 + self.num_env, dtype=np.int32),
            dtype=wp.int32,
            device=GameConfig.DEVICE,
        )
        self.push_seed_offsets = wp.zeros(self.num_env, dtype=wp.int32, device=GameConfig.DEVICE)

        self.foot_sensor = FootContactSensor(self.num_env, device=GameConfig.DEVICE)
        self.view = self.g1_provider.view

        self._apply_foot_friction_randomization()

        self.reset_env(self.game.terminated, self.game.current_step)
        self.physics_manager.simulate()
        self.obs_buf_gpu = wp.zeros(
            shape=(self.players.num_rl_players, self.flat_obs_dim), dtype=float, device=GameConfig.DEVICE
        )

        try:
            reward_components_cls = GameConfig.reward_components
            reward_components_diff_cls = GameConfig.reward_components_diff
        except AttributeError:
            train_cfg = get_default_train_cfg(5, 0)
            reward_components_cls = train_cfg.reward_components
            reward_components_diff_cls = train_cfg.reward_components_diff
            GameConfig.reward_parameters = train_cfg.reward_parameters

        reward_components = []
        for cls in reward_components_cls:
            rc = cls(
                device=GameConfig.DEVICE,
                abilities_objects=self.abilities_objects,
                num_max_players=self.players.num_total_object_role,
                articulation_body=self.articulation_body,
                deformable_body=self.deformable_body,
                reward_parameters=GameConfig.reward_parameters,
                pattern=pattern,
            )
            if hasattr(rc, "bind_level"):
                rc.bind_level(self)
            reward_components.append(rc)

        reward_components_diff = []
        for cls in reward_components_diff_cls:
            reward_components_diff.append(
                cls(
                    device=GameConfig.DEVICE,
                    abilities_objects=self.abilities_objects,
                    num_max_players=self.players.num_total_object_role,
                    articulation_body=self.articulation_body,
                    deformable_body=self.deformable_body,
                    reward_parameters=GameConfig.reward_parameters,
                )
            )

        game_end = G1LocomotionTerminator(
            device=GameConfig.DEVICE,
            articulation_body=self.articulation_body,
            deformable_body=self.deformable_body,
            reward_parameters=GameConfig.reward_parameters,
            pattern=pattern,
        )
        self.reward_calculator = RewardCalculator(
            level=self,
            terminated=self.game.terminated,
            reward_components=reward_components,
            reward_components_diff=reward_components_diff,
            episode_end_detector=game_end,
        )

        print(
            f"[Level5_0] obs_dim={self.obs_dim}, rl_action_dim={self.rl_action_dim}, history={self.history_len}"
        )
        return self.players, self.platforms, self.entities, self.abilities_objects, self.reward_calculator

    def _apply_foot_friction_randomization(self):
        model = self.physics_manager.model
        if not hasattr(model, "shape_material_mu"):
            return
        mu_np = model.shape_material_mu.numpy()
        rng = np.random.default_rng(getattr(GameConfig, "SEED", 42))
        for i in range(len(mu_np)):
            if i == 0:
                continue
            mu_np[i] = rng.uniform(0.3, 1.2)
        model.shape_material_mu.assign(mu_np)

    def on_step_actions(self, actions_wp):
        rl_dim = self.rl_action_dim
        actions_torch = wp.to_torch(actions_wp)[:, :rl_dim].contiguous()
        new_actions = wp.from_torch(actions_torch)
        self.g1_provider.store_low_level_actions(new_actions)

    def _update_velocity_commands(self, dt: float):
        self.g1_provider.update_velocity_commands(self.physics_manager, dt)

    def reset_env(self, terminated, current_step):
        super().reset_env(terminated=terminated, current_step=current_step)

        if isinstance(terminated, torch.Tensor):
            terminated_int = terminated.to(dtype=torch.int32, device=GameConfig.DEVICE)
        else:
            terminated_int = torch.tensor(terminated, dtype=torch.int32, device=GameConfig.DEVICE)
        terminated_wp = wp.from_torch(terminated_int, dtype=wp.int32)

        self.g1_provider.reset_commands(terminated_wp)
        self.g1_provider.reset_policy_actions(terminated_int.bool())

        if self.foot_sensor is not None:
            self.foot_sensor.reset_envs(terminated_wp)

        if self.obs_wp is not None:
            self.g1_provider.compute_single_frame_obs(self.physics_manager)
            self.g1_provider.reset_history(terminated_wp, self.physics_manager)

    def update_game_status(self, physics_manager, reward_calculator, num_env, current_step):
        dt = 1.0 / float(GameConfig.FPS_ACTION)
        self._update_velocity_commands(dt)

        root_tfs = self.view.get_root_transforms(physics_manager.state_0)
        root_vels = self.view.get_root_velocities(physics_manager.state_0)
        if self.foot_sensor is not None:
            self.foot_sensor.update(root_tfs, root_vels, dt)

        wp.launch(
            kernel=self.update_game_status_gpu,
            dim=num_env,
            inputs=[current_step],
            device=GameConfig.DEVICE,
        )

    def _get_observation_state_based(self) -> torch.Tensor:
        self.g1_provider.get_observation(self.physics_manager)
        return self.obs_torch

    @wp.kernel
    def update_game_status_gpu(current_step: wp.array(dtype=wp.int32)):
        tid = wp.tid()
        current_step[tid] += 1
