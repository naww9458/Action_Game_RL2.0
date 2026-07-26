"""Core articulation joint-target control shared by direct RL and RL-assisted paths."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from newton import JointTargetMode

from script.role.abilities.ability import Ability
from script.role.abilities.articulation_control_config.joint_config_registry import (
    resolve_command_interface_for_pattern,
    resolve_runtime_nominals_gpu_spec,
)
from script.role.abilities.articulation_control_config.runtime_helpers import (
    adjust_runtime_joint_nominals_kernel,
)
from script.role.abilities.articulation_control_config.profile_registry import (
    AxisKeyBindings,
    find_player_config_for_ability,
    resolve_human_control_bindings,
)

if TYPE_CHECKING:
    from script.levels.levels import Levels
    from script.mjlab_components.act.warp_kernels import MjlabWarpActionApplier
    from script.role.bodies.articulation_body import ArticulationBody


_POSITION_MODE = int(JointTargetMode.POSITION)
_VELOCITY_MODE = int(JointTargetMode.VELOCITY)


def _resolve_rl_action_dim(articulation_body: "ArticulationBody", pattern: str) -> int:
    action_dim = articulation_body.control_rl_action_dim.get(pattern)
    if action_dim is None:
        raise RuntimeError(
            f"No RL action dimension registered for articulation pattern '{pattern}'."
        )
    return int(action_dim)


def _gather_rl_action_params(
    articulation_body: "ArticulationBody",
    pattern: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    action_dim = _resolve_rl_action_dim(articulation_body, pattern)
    scales = articulation_body.control_joint_scales_gpus[pattern].numpy()
    nominals = articulation_body.control_joint_nominal_qs_gpus[pattern].numpy()
    rl_indices = articulation_body.control_joint_rl_action_indices_gpus[pattern].numpy()

    action_scales = np.zeros(action_dim, dtype=np.float32)
    action_offsets = np.zeros(action_dim, dtype=np.float32)
    for dof, rl_idx in enumerate(rl_indices):
        if rl_idx >= 0:
            action_scales[rl_idx] = scales[dof]
            action_offsets[rl_idx] = nominals[dof]
    return action_scales, action_offsets, action_dim


def _create_mjlab_action_applier(
    articulation_body: "ArticulationBody",
    pattern: str,
    device: str,
) -> "MjlabWarpActionApplier":
    from script.mjlab_components.act.warp_kernels import MjlabWarpActionApplier

    scales, offsets, action_dim = _gather_rl_action_params(articulation_body, pattern)
    return MjlabWarpActionApplier(
        device=device,
        scale=scales,
        default_joint_pos=offsets,
        action_dim=action_dim,
    )


@wp.kernel
def write_mjlab_targets_to_control_kernel(
    joint_f: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_target_pos: wp.array(dtype=float),
    joint_target_vel: wp.array(dtype=float),
    index_rl_players_gpu: wp.array(dtype=wp.int32),
    view_object_indices: wp.array(dtype=int),
    view_joint_dof_indices: wp.array(dtype=int),
    view_joint_has_free: wp.array(dtype=wp.int32),
    num_objects_env: int,
    num_joint_dof_env: int,
    count_per_world: int,
    joint_dof_count: int,
    cooldown_ability_owners: wp.array(dtype=wp.int32),
    targets: wp.array2d(dtype=wp.float32),
    joint_nominal_qs: wp.array(dtype=float),
    joint_rl_mask: wp.array(dtype=wp.int32),
    joint_rl_action_indices: wp.array(dtype=wp.int32),
    joint_target_modes: wp.array(dtype=wp.int32),
    use_per_dof_targets: int,
    use_direct_joint_torque: int,
    direct_torque_limit: float,
    direct_velocity_gain: float,
    position_mode: int,
    velocity_mode: int,
):
    tid = wp.tid()
    player_idx = index_rl_players_gpu[tid]

    if cooldown_ability_owners[player_idx] != 0:
        return

    world = player_idx // num_objects_env
    local_idx = player_idx % num_objects_env

    obj_idx = wp.int32(-1)
    for i in range(count_per_world):
        if view_object_indices[i] == local_idx:
            obj_idx = i
            break

    if obj_idx == -1:
        return

    for dof in range(joint_dof_count):
        local_dof_idx = view_joint_dof_indices[obj_idx * joint_dof_count + dof]
        global_dof_idx = world * num_joint_dof_env + local_dof_idx
        if view_joint_has_free[obj_idx] == 1:
            global_dof_idx = global_dof_idx + 6

        mode = joint_target_modes[dof]
        rl_idx = joint_rl_action_indices[dof]
        if use_per_dof_targets != 0:
            target_val = float(targets[tid, dof])
        elif joint_rl_mask[dof] != 0 and rl_idx >= 0:
            target_val = float(targets[tid, rl_idx])
        elif mode == velocity_mode:
            target_val = 0.0
        else:
            target_val = float(joint_nominal_qs[dof])

        if use_direct_joint_torque != 0 and joint_rl_mask[dof] != 0:
            # The command is a target wheel angular velocity.  Drive the wheel
            # through direct joint torque, including active braking at zero.
            torque = direct_velocity_gain * (
                target_val - joint_qd[global_dof_idx]
            )
            if torque > direct_torque_limit:
                torque = direct_torque_limit
            elif torque < -direct_torque_limit:
                torque = -direct_torque_limit
            joint_f[global_dof_idx] = torque
        elif mode == velocity_mode:
            joint_target_vel[global_dof_idx] = target_val
        else:
            joint_target_pos[global_dof_idx] = target_val


class Articulation_body_control(Ability):
    """Writes low-level joint position/velocity targets through the mjlab action pipeline."""

    def __init__(self, ability_name: str | None = None):
        super().__init__(ability_name or self.__class__.__name__)

        self.pattern = ""
        self.control_policy_version: Optional[str] = None
        self.policy_device: Optional[str] = None

        self._action_applier: Optional[MjlabWarpActionApplier] = None
        self._configured = False
        self._use_command_expander = False
        self._command_interface = None
        self._command_bindings: Dict[str, AxisKeyBindings] = {}
        self._pending_human_control: dict | None = None
        self._human_control_applied = False
        self._control_task: str | None = None
        self._runtime_nominals_passive_mask_gpu: Optional[wp.array] = None
        self._runtime_nominals_default_gpu: Optional[wp.array] = None
        self._runtime_nominals_upright_threshold: float = 0.0

        self._low_level_actions_torch: Optional[torch.Tensor] = None
        self._low_level_actions_wp: Optional[wp.array2d] = None
        self._mjlab_targets_wp: Optional[wp.array2d] = None
        self._encoder_bias_wp: Optional[wp.array2d] = None
        self._torch_device: Optional[torch.device] = None

    def configure_from_player_configs(
        self, player_configs: List[Dict[str, Any]], level: "Levels"
    ) -> None:
        matched_config = find_player_config_for_ability(
            player_configs, self.__class__.__name__
        )
        matched_object = dict(matched_config.get("object") or {})
        self._control_task = matched_object.get("control_task")

        super().configure_from_player_configs(player_configs, level)

        self._command_interface = resolve_command_interface_for_pattern(
            self.pattern,
            task_name=self._control_task,
        )
        self._use_command_expander = self._command_interface is not None
        if self._use_command_expander:
            self._pending_human_control = dict(
                self._command_interface.human_control or {}
            )
            print(
                f"[{self.__class__.__name__}] command interface enabled: "
                f"pattern={self.pattern}, command_dim={self._command_interface.command_dim}"
            )
        else:
            self._action_applier = _create_mjlab_action_applier(
                self.articulation_body,
                self.pattern,
                self.physics_manager.device,
            )
        self._build_action_buffers()

    def configure_from_player_configs_post_indices(self, level: "Levels") -> None:
        super().configure_from_player_configs_post_indices(level)
        if self._configured:
            self._build_action_buffers()
            self._setup_runtime_nominals_gpu()

    def _setup_runtime_nominals_gpu(self) -> None:
        self._runtime_nominals_passive_mask_gpu = None
        self._runtime_nominals_default_gpu = None
        self._runtime_nominals_upright_threshold = 0.0

        joint_labels = self.articulation_body.control_joint_labels.get(self.pattern)
        if not joint_labels:
            return

        gpu_spec = resolve_runtime_nominals_gpu_spec(
            self.pattern,
            joint_labels=joint_labels,
            task_name=self._control_task,
        )
        if gpu_spec is None:
            return

        nominals_gpu = self.articulation_body.control_joint_nominal_qs_gpus.get(
            self.pattern
        )
        if nominals_gpu is None:
            return

        device = self.physics_manager.device
        self._runtime_nominals_passive_mask_gpu = wp.array(
            gpu_spec.passive_dof_mask,
            dtype=wp.int32,
            device=device,
        )
        self._runtime_nominals_default_gpu = wp.array(
            nominals_gpu.numpy(),
            dtype=nominals_gpu.dtype,
            device=device,
        )
        self._runtime_nominals_upright_threshold = float(
            gpu_spec.upright_dot_threshold
        )

    def _launch_runtime_nominals_adjustment(
        self,
        view,
        controlled_player_indices: wp.array,
        num_controlled_players: int,
    ) -> None:
        if self._runtime_nominals_passive_mask_gpu is None:
            return

        joint_q = view.get_dof_positions(self.physics_manager.model)
        wp.launch(
            kernel=adjust_runtime_joint_nominals_kernel,
            dim=num_controlled_players,
            inputs=[
                self.physics_manager.state_0.body_q,
                joint_q,
                controlled_player_indices,
                self.articulation_body.view_object_indices_gpus[self.pattern],
                self.articulation_body.view_body_local_indices_gpus[self.pattern],
                self.articulation_body.num_objects_env,
                self.articulation_body.num_rigid_bodies_env,
                view.count_per_world,
                view.joint_dof_count,
                self.articulation_body.control_joint_nominal_qs_gpus[self.pattern],
                self._runtime_nominals_default_gpu,
                self._runtime_nominals_passive_mask_gpu,
                self._runtime_nominals_upright_threshold,
            ],
            device=self.physics_manager.device,
        )

    def _direct_joint_torque_params(self) -> tuple[int, float, float]:
        iface = self._command_interface
        if not self._use_command_expander or iface is None:
            return 0, 0.0, 0.0
        if not getattr(iface, "uses_direct_joint_torque", False):
            return 0, 0.0, 0.0
        return (
            1,
            float(getattr(iface, "direct_torque_limit", 0.0)),
            float(getattr(iface, "direct_velocity_gain", 0.0)),
        )

    def apply_runtime_keymapping(self) -> None:
        if not self._use_command_expander or not self._pending_human_control:
            return
        self._command_bindings = resolve_human_control_bindings(
            self._pending_human_control,
            self._command_interface.binding_names,
        )

    def _joint_dof_count(self) -> int:
        ctx = self._primary_view_ctx
        if ctx is not None and ctx.valid:
            return int(self.articulation_body.views[ctx.view_idx].joint_dof_count)

        body = self.articulation_body
        if body is not None and self.pattern:
            view_idx = next(
                (i for i, p in enumerate(body.patterns) if p == self.pattern),
                -1,
            )
            if view_idx >= 0:
                return int(body.views[view_idx].joint_dof_count)

        raise RuntimeError(
            f"{self.__class__.__name__} has no valid articulation view for '{self.pattern}'."
        )

    def _build_action_buffers(self) -> None:
        if self.num_rl_players <= 0 or self._torch_device is None:
            return

        if self._use_command_expander:
            command_dim = int(self._command_interface.command_dim)
            joint_dof_count = self._joint_dof_count()
            command_shape = (self.num_rl_players, command_dim)
            target_shape = (self.num_rl_players, joint_dof_count)
            self._low_level_actions_torch = torch.zeros(
                command_shape, dtype=torch.float32, device=self._torch_device
            )
            self._low_level_actions_wp = wp.from_torch(
                self._low_level_actions_torch, dtype=wp.float32
            )
            self._mjlab_targets_wp = wp.zeros(
                target_shape, dtype=wp.float32, device=self.physics_manager.device
            )
            self._encoder_bias_wp = wp.zeros(
                command_shape, dtype=wp.float32, device=self.physics_manager.device
            )
            return

        action_dim = _resolve_rl_action_dim(self.articulation_body, self.pattern)
        shape = (self.num_rl_players, action_dim)
        self._low_level_actions_torch = torch.zeros(
            shape, dtype=torch.float32, device=self._torch_device
        )
        self._low_level_actions_wp = wp.from_torch(
            self._low_level_actions_torch, dtype=wp.float32
        )
        self._mjlab_targets_wp = wp.zeros(
            shape, dtype=wp.float32, device=self.physics_manager.device
        )
        self._encoder_bias_wp = wp.zeros(
            shape, dtype=wp.float32, device=self.physics_manager.device
        )

    def _ensure_configured(self) -> None:
        if not self._configured:
            raise RuntimeError(
                f"{self.__class__.__name__} must be configured via configure_from_player_configs()."
            )

    def _commands_from_keyboard(self, keyboard_keys, mouse_buttons) -> List[float]:
        values: List[float] = []
        for name in self._command_interface.binding_names:
            binding = self._command_bindings.get(name)
            if binding is None:
                values.append(0.0)
                continue
            pos = self._is_pressed(
                binding.positive_keyboard,
                binding.positive_mouse,
                keyboard_keys,
                mouse_buttons,
            )
            neg = self._is_pressed(
                binding.negative_keyboard,
                binding.negative_mouse,
                keyboard_keys,
                mouse_buttons,
            )
            values.append(float(pos - neg))
        return values

    def _player_tid(self, player_idx: int) -> Optional[int]:
        player_indices = self.index_rl_players_gpu.numpy().tolist()
        try:
            return player_indices.index(player_idx)
        except ValueError:
            return None

    def _expand_commands_for_tid(self, tid: int, commands: Sequence[float]) -> None:
        joint_labels = self.articulation_body.control_joint_labels[self.pattern]
        rl_mask = self.articulation_body.control_joint_rl_mask_gpus[self.pattern].numpy()
        expanded = self._command_interface.expand_commands(
            commands[0],
            commands[1],
            joint_labels=joint_labels,
            rl_mask=rl_mask,
        )
        target_torch = wp.to_torch(self._mjlab_targets_wp)
        target_torch[tid, :] = torch.tensor(
            expanded, dtype=torch.float32, device=target_torch.device
        )

    def _apply_command_actions(self, actions, action_shape_offset: int = 0) -> None:
        command_dim = int(self._command_interface.command_dim)
        actions_torch = wp.to_torch(actions)
        start = int(action_shape_offset)
        end = start + command_dim
        if actions_torch.shape[1] >= end:
            sliced = actions_torch[:, start:end]
            self._low_level_actions_torch.copy_(sliced)
        else:
            self._low_level_actions_torch.zero_()

        for tid in range(self.num_rl_players):
            self._expand_commands_for_tid(
                tid,
                self._low_level_actions_torch[tid].tolist(),
            )
        self._write_targets_to_physics_control()

    def _apply_raw_actions(self, actions, action_shape_offset: int = 0) -> None:
        self._ensure_configured()
        if self.num_rl_players <= 0:
            return

        if self._use_command_expander:
            self._apply_command_actions(actions, action_shape_offset=action_shape_offset)
            return

        action_dim = _resolve_rl_action_dim(self.articulation_body, self.pattern)
        raw_actions = actions
        if action_shape_offset != 0 or actions.shape[1] != action_dim:
            actions_torch = wp.to_torch(actions)
            sliced = actions_torch[
                :, action_shape_offset : action_shape_offset + action_dim
            ]
            self._low_level_actions_torch.copy_(sliced)
            raw_actions = self._low_level_actions_wp

        self._action_applier.launch_to_targets(
            raw_actions,
            self._encoder_bias_wp,
            self._mjlab_targets_wp,
        )
        self._write_targets_to_physics_control()

    def _write_targets_to_physics_control(self) -> None:
        ctx = self._primary_view_ctx
        if ctx is None or not ctx.valid:
            return
        view = self.articulation_body.views[ctx.view_idx]
        num_joint_dof_env = self.articulation_body.num_joint_dofs_env
        controlled_player_indices = getattr(self, "_controlled_player_indices_gpu", None)
        if controlled_player_indices is None:
            controlled_player_indices = self.index_rl_players_gpu
            num_controlled_players = self.num_rl_players
        else:
            num_controlled_players = controlled_player_indices.shape[0]

        if self._runtime_nominals_passive_mask_gpu is not None:
            self._launch_runtime_nominals_adjustment(
                view,
                controlled_player_indices,
                num_controlled_players,
            )

        use_direct_joint_torque, direct_torque_limit, direct_velocity_gain = (
            self._direct_joint_torque_params()
        )

        wp.launch(
            kernel=write_mjlab_targets_to_control_kernel,
            dim=num_controlled_players,
            inputs=[
                self.physics_manager.control.joint_f,
                self.physics_manager.state_0.joint_qd,
                self.physics_manager.control.joint_target_pos,
                self.physics_manager.control.joint_target_vel,
                controlled_player_indices,
                self.articulation_body.view_object_indices_gpus[self.pattern],
                self.articulation_body.view_joint_dof_indices_gpus[self.pattern],
                self.articulation_body.view_joint_has_free_gpus[self.pattern],
                self.articulation_body.num_objects_env,
                num_joint_dof_env,
                view.count_per_world,
                view.joint_dof_count,
                self.cooldown_ability_owners,
                self._mjlab_targets_wp,
                self.articulation_body.control_joint_nominal_qs_gpus[self.pattern],
                self.articulation_body.control_joint_rl_mask_gpus[self.pattern],
                self.articulation_body.control_joint_rl_action_indices_gpus[self.pattern],
                self.articulation_body.control_joint_target_mode_gpus[self.pattern],
                1 if self._use_command_expander else 0,
                use_direct_joint_torque,
                direct_torque_limit,
                direct_velocity_gain,
                _POSITION_MODE,
                _VELOCITY_MODE,
            ],
            device=self.physics_manager.device,
        )

    def rl_action(self, actions, **kwargs):
        if self._human_control_applied:
            self._human_control_applied = False
            return

        offset = self.action_shape_offset if self.action_shape_offset is not None else 0
        self._apply_raw_actions(actions, action_shape_offset=offset)

    def human_control_interface(
        self, keyboard_keys, mouse_buttons, look_yaw, index_human_player_gpu, **kwargs
    ):
        if not self._use_command_expander:
            return

        self._ensure_configured()
        values = self._commands_from_keyboard(keyboard_keys, mouse_buttons)
        player_idx = int(index_human_player_gpu.numpy()[0])
        tid = self._player_tid(player_idx)
        if tid is None:
            return

        self._expand_commands_for_tid(tid, values)
        self._write_targets_to_physics_control()
        self._human_control_applied = True

    def bot_action(self, **kwargs):
        pass

    def update_index_bot(
        self,
        index_rl_players_gpu,
        num_rl_players,
        index_bot_players_gpu,
        num_bot_players,
    ):
        super().update_index_bot(
            index_rl_players_gpu=index_rl_players_gpu,
            num_rl_players=num_rl_players,
            index_bot_players_gpu=index_bot_players_gpu,
            num_bot_players=num_bot_players,
        )
        if self._configured:
            self._build_action_buffers()

    def get_action_spec(self) -> dict:
        self.action_space["shape"] = _resolve_rl_action_dim(
            self.articulation_body, self.pattern
        )
        return self.action_space

    def reset(self):
        pass
