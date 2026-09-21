"""RL-assisted articulation control: high-level commands -> low-level policy -> joints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import torch
import warp as wp

from script.game_config import GameConfig
from script.role.abilities.articulation_body_control import (
    Articulation_body_control,
    _resolve_rl_action_dim,
)
from script.role.abilities.articulation_control_config.profile_registry import (
    AxisKeyBindings,
    command_binding_names,
    command_profile_from_bundle,
    find_player_config_for_ability,
    resolve_policy_checkpoint,
    resolve_human_control_bindings,
)
from script.role.policies.policy_bundle import PolicyBundleRegistry, load_policy_runner

if TYPE_CHECKING:
    from script.environments.environment import Environment


class Articulation_body_control_rl_assisted(Articulation_body_control):
    """Runs mjlab policy action processing before writing joint position targets."""

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.command_dim = 0
        self.command_labels: List[str] = []
        self.command_ranges: List[tuple[float, float]] = []
        self.policy_checkpoint: Optional[str] = None

        self._bundle_spec = None
        self._command_profile = None
        self._obs_actor = None
        self._policy_runner = None

        self._player_env_indices: Optional[List[int]] = None
        self._player_env_indices_gpu: Optional[wp.array] = None
        self._controlled_player_indices: List[int] = []
        self._controlled_player_indices_gpu: Optional[wp.array] = None
        self._controlled_player_action_rows_gpu: Optional[wp.array] = None
        self._gather_indices_torch: Optional[torch.Tensor] = None
        self._rl_player_indices_cache: List[int] = []

        self._pending_human_control: dict | None = None
        self._pending_human_command: Optional[tuple[int, List[float]]] = None
        self._command_bindings: Dict[str, AxisKeyBindings] = {}
        self._default_command: List[float] = []
        self._default_has_target: Optional[bool] = None # TODO 應該改進，此項更多代表的是任務目標/類型，但目前沒想好怎麽改

    def configure_from_player_configs(
        self, player_configs: List[Dict[str, Any]], environment: "Environment"
    ) -> None:
        matched_config = find_player_config_for_ability(
            player_configs,
            self.__class__.__name__,
            **self._share_scope_match_kwargs(),
        )
        matched_object = dict(matched_config.get("object") or {})
        super().configure_from_player_configs(player_configs, environment)
        self.policy_checkpoint = resolve_policy_checkpoint(matched_object)

        self._bundle_spec = PolicyBundleRegistry.get(
            self.control_policy_version,
            robot_pattern=self.pattern,
        )
        self._command_profile = command_profile_from_bundle(self._bundle_spec)
        self.command_dim = self._command_profile.command_dim
        self.command_labels = list(self._command_profile.command_labels)
        self.command_ranges = list(self._command_profile.command_ranges)
        self._pending_human_control = self._command_profile.human_control
        self._default_command = list(self._bundle_spec.default_command or ())
        self._default_has_target = self._bundle_spec.default_has_target

        self._build_player_env_mapping(environment)
        self._obs_actor = PolicyBundleRegistry.create_obs_actor(
            self._bundle_spec.obs_actor,
            num_env=environment.num_env,
            device=GameConfig.DEVICE,
            articulation_body=self.articulation_body,
            pattern=self.pattern,
            history_len=int(self._bundle_spec.history_len),
            instance_world_indices=self._player_env_indices,
            instance_view_indices=self._player_view_indices,
        )
        action_dim = _resolve_rl_action_dim(self.articulation_body, self.pattern)
        self._obs_actor.validate_dims(expected_low_level_action_dim=action_dim)

        self._policy_runner = load_policy_runner(
            self.control_policy_version,
            robot_pattern=self.pattern,
            device=self.policy_device,
            checkpoint_override=self.policy_checkpoint,
            expected_obs_dim=self._obs_actor.obs_dim,
            expected_action_dim=action_dim,
        )

        if hasattr(environment, "bind_assisted_obs_actor"):
            environment.bind_assisted_obs_actor(self._obs_actor)
        if self._default_has_target is False:
            clearer = getattr(self._obs_actor, "try_clear_all_command_role_targets", None)
            if callable(clearer):
                clearer()

        self._configured = True
        print(
            f"[{self.__class__.__name__}] mjlab act path enabled: "
            f"policy={self.control_policy_version}, pattern={self.pattern}, "
            f"obs_dim={self._obs_actor.obs_dim}, action_dim={action_dim}, "
            f"checkpoint={self._policy_runner.checkpoint_path}"
        )

    def configure_from_player_configs_post_indices(self, environment: "Environment") -> None:
        super().configure_from_player_configs_post_indices(environment)
        if self._configured:
            self._build_player_env_mapping(environment)

    def apply_runtime_keymapping(self) -> None:
        if not self._pending_human_control or self.command_dim <= 0:
            return
        self._command_bindings = resolve_human_control_bindings(
            self._pending_human_control,
            command_binding_names(self.command_dim),
        )

    def _build_player_env_mapping(self, environment: "Environment") -> None:
        try:
            ability_idx = environment.players.abilities_instance_list.index(self)
            ability_owners = environment.players.abilities_owner_list[ability_idx]
        except ValueError:
            ability_owners = []

        objects_per_env = self.articulation_body.num_objects_env
        pattern_local_indices = list(self.articulation_body.patterns[self.pattern])
        self._controlled_player_indices = [
            player_idx
            for player_idx in ability_owners
            if player_idx % objects_per_env in pattern_local_indices
        ]
        if not self._controlled_player_indices:
            raise RuntimeError(
                f"{self.__class__.__name__} has no owners with articulation pattern "
                f"'{self.pattern}'."
            )

        mapping = environment.players.index_obj_role_to_env_mapping
        self._rl_player_indices_cache = list(self._controlled_player_indices)
        self._player_env_indices = [
            mapping[player_idx] for player_idx in self._rl_player_indices_cache
        ]
        self._player_view_indices = []
        for player_idx, env_idx in zip(
            self._controlled_player_indices, self._player_env_indices
        ):
            local_player_idx = player_idx - env_idx * objects_per_env
            try:
                self._player_view_indices.append(pattern_local_indices.index(local_player_idx))
            except ValueError as exc:
                raise RuntimeError(
                    f"Assisted-control player {player_idx} is not present in "
                    f"articulation pattern '{self.pattern}'."
                ) from exc

        action_rows = [
            environment.is_rl_player_mask[player_idx]
            for player_idx in self._controlled_player_indices
        ]
        self._player_env_indices_gpu = wp.array(
            self._player_env_indices, dtype=wp.int32, device=GameConfig.DEVICE
        )
        self._controlled_player_indices_gpu = wp.array(
            self._controlled_player_indices, dtype=wp.int32, device=GameConfig.DEVICE
        )
        self._controlled_player_action_rows_gpu = wp.array(
            action_rows, dtype=wp.int32, device=GameConfig.DEVICE
        )
        self._gather_indices_torch = torch.tensor(
            range(len(self._controlled_player_indices)),
            dtype=torch.long,
            device=self._torch_device,
        )

        action_dim = _resolve_rl_action_dim(self.articulation_body, self.pattern)
        shape = (len(self._controlled_player_indices), action_dim)
        self._low_level_actions_torch = torch.zeros(
            shape, dtype=torch.float32, device=self._torch_device
        )
        self._low_level_actions_wp = wp.from_torch(
            self._low_level_actions_torch, dtype=wp.float32
        )
        self._mjlab_targets_wp = wp.zeros(
            shape, dtype=wp.float32, device=GameConfig.DEVICE
        )
        self._encoder_bias_wp = wp.zeros(
            shape, dtype=wp.float32, device=GameConfig.DEVICE
        )

    def _commands_from_keyboard(self, keyboard_keys, mouse_buttons) -> List[float]:
        values: List[float] = []
        for i in range(self.command_dim):
            binding = self._command_bindings.get(f"command_{i}")
            if binding is None:
                values.append(0.0)
                continue
            pos = self._is_pressed(
                binding.positive_keyboard, binding.positive_mouse, keyboard_keys, mouse_buttons
            )
            neg = self._is_pressed(
                binding.negative_keyboard, binding.negative_mouse, keyboard_keys, mouse_buttons
            )
            val = float(pos - neg)
            if self.command_ranges and i < len(self.command_ranges):
                lo, hi = self.command_ranges[i]
                if val > 0.0:
                    val = float(hi)
                elif val < 0.0:
                    val = float(lo)
            values.append(val)
        return values

    def _instance_idx_for_player(self, player_idx: int) -> Optional[int]:
        for i, rl_player_idx in enumerate(self._rl_player_indices_cache):
            if rl_player_idx == player_idx:
                return i
        return None

    def _write_rl_commands(self, actions) -> None:
        if self._obs_actor is None or self._controlled_player_action_rows_gpu is None:
            return
        self._obs_actor.write_commands_from_rl_actions(
            actions,
            self.action_shape_offset,
            self._controlled_player_action_rows_gpu,
        )

    def _apply_default_commands(self) -> None:
        actor = self._obs_actor
        if actor is None or not self._default_command or getattr(actor, "commands", None) is None:
            return
        cmd = wp.to_torch(actor.commands)
        default = torch.tensor(
            self._default_command, dtype=cmd.dtype, device=cmd.device
        )
        if int(default.numel()) != int(cmd.shape[1]):
            raise ValueError(
                f"{self.__class__.__name__} default command dim "
                f"{int(default.numel())} != {int(cmd.shape[1])}"
            )
        cmd[:] = default

    def write_command_base(self, actions, **kwargs) -> None:
        """Lowest command layer: yaml default (play) or trainer actions."""
        self._ensure_configured()
        if kwargs.get("apply_default_commands"):
            self._apply_default_commands()
            return
        self._write_rl_commands(actions)

    def apply_commands(
        self,
        player_idx: int,
        values: Dict[int, float],
        dim_indices: Sequence[int],
    ) -> None:
        """Write selected command dims for one owned player. Last writer wins."""
        instance_idx = self._instance_idx_for_player(int(player_idx))
        if instance_idx is None or self._obs_actor is None:
            return
        cmd = wp.to_torch(self._obs_actor.commands)
        for dim_index in dim_indices:
            if dim_index not in values:
                continue
            if dim_index < 0 or dim_index >= int(cmd.shape[1]):
                continue
            cmd[instance_idx, dim_index] = float(values[dim_index])

    def _apply_commands(self, instance_idx: int, values: Sequence[float]) -> None:
        cmd_torch = wp.to_torch(self._obs_actor.commands)
        for i, val in enumerate(values):
            cmd_torch[instance_idx, i] = float(val)

    def _apply_pending_human_commands(self) -> None:
        pending = self._pending_human_command
        self._pending_human_command = None
        if pending is None:
            return
        instance_idx, values = pending
        self._apply_commands(instance_idx, values)

    def _run_policy_and_apply(self) -> None:
        self._ensure_configured()
        if not self._controlled_player_indices:
            return

        obs = self._obs_actor.get_observation(self.physics_manager)
        low_level_all_envs = self._policy_runner.predict(obs, deterministic=True)
        expected_shape = (
            len(self._controlled_player_indices),
            _resolve_rl_action_dim(self.articulation_body, self.pattern),
        )
        if tuple(low_level_all_envs.shape) != expected_shape:
            raise RuntimeError(
                f"Policy returned {tuple(low_level_all_envs.shape)} actions; "
                f"expected {expected_shape} for assisted G1 instances."
            )
        self._low_level_actions_torch.copy_(low_level_all_envs)
        self._obs_actor.store_low_level_actions(self._low_level_actions_wp)

        self._action_applier.launch_to_targets(
            self._low_level_actions_wp,
            self._encoder_bias_wp,
            self._mjlab_targets_wp,
        )
        self._write_targets_to_physics_control()

    def rl_action(self, actions, **kwargs):
        if not kwargs.pop("commands_already_written", False):
            self.write_command_base(actions, **kwargs)
        self._apply_pending_human_commands()
        self._run_policy_and_apply()

    def human_control_interface(
        self, keyboard_keys, mouse_buttons, look_yaw, index_human_player_gpu, **kwargs
    ):
        self._ensure_configured()
        self._pending_human_command = None
        instance_idx = self._instance_idx_for_player(
            int(index_human_player_gpu.numpy()[0])
        )
        if instance_idx is None:
            return
        values = self._commands_from_keyboard(keyboard_keys, mouse_buttons)
        if any(float(v) != 0.0 for v in values):
            self._pending_human_command = (instance_idx, values)

    def bot_action(self, **kwargs):
        # Low-level policy always runs in rl_action (outside the CUDA graph).
        return

    def get_action_spec(self) -> dict:
        self.action_space["shape"] = self.command_dim
        if self.command_ranges:
            lo = min(r[0] for r in self.command_ranges)
            hi = max(r[1] for r in self.command_ranges)
            self.action_space["range"] = [lo, hi]
        return self.action_space

    def uses_command_as_rl_action(self) -> bool:
        return int(self.command_dim or 0) > 0
