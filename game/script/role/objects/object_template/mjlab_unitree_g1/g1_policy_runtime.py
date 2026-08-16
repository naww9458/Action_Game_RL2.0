"""G1 policy observation / command runtime, attached from this template's setup.

Environment classes never import this module. ``register.setup(environment)`` no-ops when
the loaded scene has no Unitree G1 player with a control-policy version.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import warp as wp

from script.game_config import GameConfig
from script.role.abilities.articulation_control_config.profile_registry import (
    resolve_player_runtime_pattern,
)
from script.role.abilities.articulation_control_config.robot_pattern import (
    normalize_robot_pattern,
)
from script.role.policies.policy_bundle import PolicyBundleRegistry

from .g1_control_config import G1_ROBOT_NAME

_HISTORY_LEN = 1


def _player_object_cfg(player_cfg: dict) -> dict:
    return dict(player_cfg.get("object") or {})


def _find_g1_player(environment: Any) -> Optional[dict]:
    for cfg in environment.config.get("player_configs") or []:
        if not isinstance(cfg, dict):
            continue
        pattern = str(_player_object_cfg(cfg).get("pattern") or "")
        if normalize_robot_pattern(pattern) == G1_ROBOT_NAME:
            return cfg
    return None


def _range_pair(value, default) -> tuple[float, float]:
    if value is None:
        return (float(default[0]), float(default[1]))
    return (float(value[0]), float(value[1]))


def _encoder_bias_range(dr_cfg: dict) -> Optional[tuple[float, float]]:
    enc_cfg = dr_cfg.get("encoder_bias")
    if not isinstance(enc_cfg, dict) or not enc_cfg.get("enabled", True):
        return None
    br = enc_cfg.get("bias_range") or [-0.015, 0.015]
    return (float(br[0]), float(br[1]))


def _base_com_offset_range(dr_cfg: dict) -> Optional[dict]:
    com_cfg = dr_cfg.get("base_com")
    if not isinstance(com_cfg, dict) or not com_cfg.get("enabled", True):
        return None
    cr = com_cfg.get("offset_range") or {}
    return {
        "x": _range_pair(cr.get("x"), (-0.025, 0.025)),
        "y": _range_pair(cr.get("y"), (-0.025, 0.025)),
        "z": _range_pair(cr.get("z"), (-0.03, 0.03)),
    }


def _wire_encoder_bias_into_action(environment: Any, provider: Any) -> None:
    bias = getattr(provider, "encoder_bias_wp", None)
    if bias is None:
        return
    players = getattr(environment, "players", None)
    abilities = getattr(players, "abilities_instance_list", None) if players is not None else None
    if not abilities:
        return
    for ability in abilities:
        setter = getattr(ability, "set_encoder_bias_source", None)
        if setter is not None:
            setter(bias)


def _bind_provider_on_environment(environment: Any, provider: Any) -> None:
    environment.g1_provider = provider
    environment.obs_dim = provider.obs_dim
    environment.rl_action_dim = provider.rl_action_dim
    environment.flat_obs_dim = provider.flat_obs_dim
    environment.obs_wp = provider.obs_wp
    environment.obs_torch = provider.obs_torch
    environment.single_obs_wp = provider.single_obs_wp
    environment.commands = provider.commands
    environment.command_labels = provider.command_labels
    consumers = getattr(environment, "command_consumer_patterns", None)
    if not isinstance(consumers, set):
        consumers = set()
        environment.command_consumer_patterns = consumers
    pattern = getattr(provider, "pattern", None)
    if pattern:
        consumers.add(normalize_robot_pattern(str(pattern)))
    environment.policy_actions = provider.policy_actions
    environment.prev_actions = provider.prev_actions
    environment.view = provider.view
    environment.history_len = int(getattr(provider, "history_len", _HISTORY_LEN) or _HISTORY_LEN)


class G1PolicyRuntime:
    """Per-environment hook: observation, velocity commands, and previous-action history."""

    def __init__(self, environment: Any, provider: Any):
        self._environment = environment
        self.provider = provider

    def get_observation(self):
        self.provider.get_observation(self._environment.physics_manager)
        return self._environment.obs_torch

    def on_reset(self, terminated, current_step) -> None:
        del current_step
        if isinstance(terminated, torch.Tensor):
            terminated_int = terminated.to(dtype=torch.int32, device=GameConfig.DEVICE)
        else:
            terminated_int = torch.tensor(terminated, dtype=torch.int32, device=GameConfig.DEVICE)
        terminated_wp = wp.from_torch(terminated_int, dtype=wp.int32)
        self.provider.reset_commands(terminated_wp)
        self.provider.reset_policy_actions(terminated_int.bool())
        if getattr(self._environment, "obs_wp", None) is not None:
            self.provider.compute_single_frame_obs(self._environment.physics_manager)
            self.provider.reset_history(terminated_wp, self._environment.physics_manager)

    def on_update_game_status(self, physics_manager, reward_calculator, num_env, current_step) -> None:
        del reward_calculator, num_env, current_step
        dt = 1.0 / float(GameConfig.FPS_ACTION)
        self.provider.update_velocity_commands(physics_manager, dt)

    def on_step_actions(self, actions_wp) -> None:
        """Store game actions as G1 joint targets only when they already match.

        ``flat_walk`` writes ``(num_instances, 29)`` joint actions here. Play
        scenes with ``Articulation_body_control_rl_assisted`` send high-level
        commands (e.g. ``(num_rl_players, 5)``); that ability already stores
        the policy's 29-DoF actions on its own provider.
        """
        provider = self.provider
        if provider is None or getattr(provider, "policy_actions", None) is None:
            return
        expected = (int(provider.num_instances), int(provider.rl_action_dim))
        if expected[0] <= 0 or expected[1] <= 0:
            return
        actions_torch = wp.to_torch(actions_wp)
        if tuple(int(d) for d in actions_torch.shape) != expected:
            return
        provider.store_low_level_actions(wp.from_torch(actions_torch.contiguous()))


def attach_if_present(environment: Any) -> Optional[G1PolicyRuntime]:
    """Attach G1 obs/command runtime when a G1 player with a policy version is loaded."""
    player_cfg = _find_g1_player(environment)
    if player_cfg is None:
        return None
    object_cfg = _player_object_cfg(player_cfg)
    version = object_cfg.get("control_policy_version")
    if not version:
        return None
    articulation_body = getattr(environment, "articulation_body", None)
    if articulation_body is None:
        return None

    env_configs = environment.config.get("environment_configs") or {}
    dr_cfg = env_configs.get("domain_randomization") or {} if isinstance(env_configs, dict) else {}
    if not isinstance(dr_cfg, dict):
        dr_cfg = {}
    obs_noise_cfg = env_configs.get("observation_noise") if isinstance(env_configs, dict) else None
    robot_pattern = normalize_robot_pattern(str(object_cfg.get("pattern") or G1_ROBOT_NAME))
    runtime_pattern = resolve_player_runtime_pattern(player_cfg)
    bundle = PolicyBundleRegistry.get(str(version), robot_pattern=robot_pattern)
    provider = PolicyBundleRegistry.create_obs_provider(
        bundle.obs_provider,
        num_env=environment.num_env,
        device=GameConfig.DEVICE,
        articulation_body=articulation_body,
        pattern=runtime_pattern,
        history_len=_HISTORY_LEN,
        enable_obs_noise=bool(getattr(GameConfig, "ENABLE_OBS_NOISE", False)),
        obs_noise_cfg=obs_noise_cfg,
        encoder_bias_range=_encoder_bias_range(dr_cfg),
        base_com_offset_range=_base_com_offset_range(dr_cfg),
    )
    _bind_provider_on_environment(environment, provider)
    _wire_encoder_bias_into_action(environment, provider)
    runtime = G1PolicyRuntime(environment, provider)
    hooks = getattr(environment, "_object_template_runtime_hooks", None)
    if hooks is None:
        hooks = []
        setattr(environment, "_object_template_runtime_hooks", hooks)
    hooks.append(runtime)
    return runtime


def setup(environment: Any) -> None:
    attach_if_present(environment)
