import inspect
import math

import torch
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
    from script.environments.environment import Environment

class RewardComponent(ABC):
    """所有獎勵計算元件的抽象基底類別"""

    # Public per-term reward log key (e.g. "track_linear_velocity"). ``None``
    # disables per-term logging for this component. Names follow the mjlab
    # convention so iteration logs can be compared line-by-line.
    log_name: str | None = None
    # Raw per-env physics metrics this component writes every step (e.g.
    # "angular_momentum_mean"). Keys are emitted under ``Metrics/<name>``.
    metric_log_names: tuple[str, ...] = ()

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

    def bind_environment(self, environment) -> None:
        """Optional hook to bind environment-specific resources after construction."""
        return None

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
                 environment: 'Environment' = None,
                 terminated: wp.array = None,
                 reward_components: list[RewardComponent] = [], 
                 reward_components_diff: list[RewardComponent] = [], 
                 episode_end_detector: GameEndDetector = None,
                ):
        self.environment = environment
        self.num_env = environment.num_env
        self.physics_manager = environment.physics_manager
        self.device = self.physics_manager.device

        self.reward_components = reward_components
        self.reward_components_diff = reward_components_diff
        self.episode_end_detector = episode_end_detector

        # --- 每步 per-term 獎勵 / 物理指標緩衝區 (per-env) ---
        # 由元件透過 kwargs (``reward_term_bufs`` / ``metric_bufs``) 在各自
        # kernel 內 atomic 累加。統一公開接口 (``get_reward_terms`` /
        # ``get_metric_means``) 對所有環境 / 框架通用。
        self.reward_term_bufs: Dict[str, wp.array] = {
            c.log_name: wp.zeros(self.num_env, dtype=wp.float32, device=self.device)
            for c in self.reward_components
            if c.log_name
        }
        self.metric_bufs: Dict[str, wp.array] = {
            name: wp.zeros(self.num_env, dtype=wp.float32, device=self.device)
            for c in self.reward_components
            for name in c.metric_log_names
        }
        self._log_name_to_component = {
            c.log_name: c for c in self.reward_components if c.log_name
        }
        # GPU 端每步 running-mean 累加器 (避免每步 CPU sync)，供自動診斷使用。
        # ``self.device`` is a Warp device; convert to a torch device for the
        # 0-d accumulator tensors.
        torch_device = torch.device(str(self.device))
        self._term_accum = {
            name: torch.zeros((), device=torch_device, dtype=torch.float32)
            for name in self.reward_term_bufs
        }
        self._term_accum_steps = 0

        # --- 預處理索引 (Env) ---
        self.index_player_offset_env_gpu = environment.players.index_role_offset_env_gpu
        self.num_players_each_env = environment.players.num_role_each_env
        self.index_platform_offset_env_gpu = environment.platforms.index_role_offset_env_gpu
        self.num_platform_each_env = environment.platforms.num_role_each_env

        # --- 預處理索引 (玩家) ---
        self.index_player_gpu = environment.players.index_obj_role_gpu
        self.num_players = environment.players.num_total_object_role
        self.index_rl_players_gpu = environment.players.index_rl_players_gpu
        self.is_rl_player_mask_gpu = environment.players.is_rl_player_mask_gpu
        self.num_rl_players = environment.players.num_rl_players

        # --- 預處理索引 (Platform) ---
        self.index_platform_gpu = environment.platforms.index_obj_role_gpu
        self.num_platforms = environment.platforms.num_total_object_role

        # --- 預分配緩衝區 ---
        health = []
        self.num_total_object = BaseRole._num_objects_total
        for index in range(self.num_total_object):
            health.append(environment.players._object_game_params[index]["health"])

        self.default_player_health = wp.array(health, dtype=wp.float32, device=self.device)
        self.player_health = wp.array(health, dtype=wp.float32, device=self.device, requires_grad=GameConfig.requires_grad)

        self.step_total_rewards_all = wp.zeros(BaseRole._num_objects_total, dtype=wp.float32, device=self.device)
        self.step_total_rewards_rl = wp.zeros(self.num_rl_players, dtype=wp.float32, device=self.physics_manager.device)

        if GameConfig.requires_grad:
            self.step_total_rewards_all_diff = wp.zeros(BaseRole._num_objects_total, dtype=wp.float32, device=self.device, requires_grad=GameConfig.requires_grad)
            self.step_total_rewards_rl_diff = wp.zeros(self.num_rl_players, dtype=wp.float32, device=self.physics_manager.device, requires_grad=GameConfig.requires_grad)

        self.terminated = terminated

    def calculate_rewards(self, current_step: int, actions: wp.array2d, max_episode_step: int, command_vel, truncated: wp.array = None) -> tuple[wp.array, wp.array]:
        """
        執行所有已註冊的獎勵元件，計算總獎勵，並判斷遊戲是否結束。
        注意：為了 CUDA Graph，這裡不應該有 .numpy() 調用。

        Args:
            truncated: Per-env timeout flag array (optional). Forwarded to the
                episode-end detector so it can distinguish timeouts (truncation)
                from falls (termination).
        """

        self.step_total_rewards_all.zero_()
        for buf in self.reward_term_bufs.values():
            buf.zero_()
        for buf in self.metric_bufs.values():
            buf.zero_()

        for component in self.reward_components:
            component.calculate(
                num_players=self.num_players,
                num_rl_players = self.num_rl_players,
                physics_manager=self.physics_manager,
                actions=actions,
                player_shape_ids_gpu=self.index_player_gpu,
                is_rl_player_mask_gpu=self.is_rl_player_mask_gpu,
                index_player_obj_to_env_mapping_gpu=self.environment.index_player_obj_to_env_mapping_gpu,
                _index_obj_to_env_mapping_gpu=self.environment._index_obj_to_env_mapping_gpu,

                env_players_index_offset=self.index_player_offset_env_gpu,
                env_platforms_index_offset=self.index_platform_offset_env_gpu,

                num_players_each_env=self.num_players_each_env,
                num_platforms_each_env=self.num_platform_each_env,

                platform_shape_ids_gpu=self.index_platform_gpu,
                player_health=self.player_health,
                step_total_rewards=self.step_total_rewards_all,

                command_vel=command_vel,
                reward_term_bufs=self.reward_term_bufs,
                metric_bufs=self.metric_bufs,
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
                truncated=truncated,
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
                    index_player_obj_to_env_mapping_gpu=self.environment.index_player_obj_to_env_mapping_gpu,

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
                self.environment._index_obj_to_env_mapping_gpu,
                self.player_health,
                self.default_player_health,
            ],
            device=self.device
        )

        for reward_component in self.reward_components:
            reward_component.reset(
                num_players=self.num_players,
                terminated=self.terminated,
                _index_obj_to_env_mapping_gpu=self.environment._index_obj_to_env_mapping_gpu,
                index_player_obj_to_env_mapping_gpu=self.environment.index_player_obj_to_env_mapping_gpu,
            )

    # ------------------------------------------------------------------
    # 統一公開接口 (所有 level / 框架通用)
    # ------------------------------------------------------------------
    def get_reward_terms(self) -> Dict[str, torch.Tensor]:
        """Return per-step mean over envs of each logged reward term.

        Keys follow the mjlab convention (``Episode_Reward/<term>``). Values are
        0-d GPU tensors; the per-step running mean is accumulated on GPU for the
        diagnostics consumed by :meth:`get_reward_term_diagnostics`.

        The returned value is scaled by the environment's ``reward_log_scale`` (1.0 by
        default; FlatWalk uses FPS=50) so logged ``Episode_Reward/*`` match
        mjlab's per-second units (per-step mean x FPS = sum/20). The diagnostic
        accumulator keeps the raw per-step mean.
        """
        scale = 1.0
        if self.environment is not None:
            scale = float(getattr(self.environment, "reward_log_scale", 1.0))
        out: Dict[str, torch.Tensor] = {}
        for name, buf in self.reward_term_bufs.items():
            mean = torch.mean(wp.to_torch(buf))
            out[f"Episode_Reward/{name}"] = mean * scale
            if name in self._term_accum:
                # Accumulate into the pre-allocated normal tensor (``out=``
                # avoids creating inference tensors during the rollout's
                # inference-mode context, which would break the in-place reset
                # performed by :meth:`get_reward_term_diagnostics`).
                torch.add(self._term_accum[name], mean, out=self._term_accum[name])
        self._term_accum_steps += 1
        return out

    def get_metric_means(self) -> Dict[str, torch.Tensor]:
        """Return per-step mean over envs of raw physics metrics."""
        return {
            f"Metrics/{name}": torch.mean(wp.to_torch(buf))
            for name, buf in self.metric_bufs.items()
        }

    def get_reward_term_diagnostics(self) -> list[str]:
        """Warn about reward terms that read 0 or abnormal over the last window.

        Runs once per iteration (after the rsl_rl logger prints). Checks, per
        logged term: NaN/Inf (divide-by-zero / dimension error), a zero weight
        (disabled by preset — expected, mirrors mjlab's skipped terms), and a
        near-zero mean with a nonzero weight (dead wiring / early-out or sigma
        too large). Accumulators are reset afterwards.
        """
        if self._term_accum_steps <= 0 or not self._term_accum:
            return []

        warnings: list[str] = []
        eps = 1e-6
        for name, accum in self._term_accum.items():
            mean = (accum / float(self._term_accum_steps)).item()
            comp = self._log_name_to_component.get(name)
            weight = getattr(comp, "weight", None)
            std = getattr(comp, "std", None)
            if not math.isfinite(mean):
                warnings.append(
                    f"{name}: mean={mean:.6f} -> NaN/Inf detected; check divide-by-zero "
                    "or tensor-dimension errors in the reward kernel"
                )
            elif weight is not None and abs(weight) < 1e-12:
                warnings.append(
                    f"{name}: weight=0 (disabled in preset) -> term stays 0.0000 (expected)"
                )
            elif abs(mean) < eps:
                hint = f"std/sigma={std}" if std is not None else "n/a"
                warnings.append(
                    f"{name}: mean~0 with nonzero weight (w={weight}, {hint}); check "
                    "early-out wiring, dead buffers, or sigma too large"
                )

        self._term_accum_steps = 0
        torch_device = torch.device(str(self.device))
        for name in self._term_accum:
            self._term_accum[name] = torch.zeros((), device=torch_device, dtype=torch.float32)
        return warnings

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
