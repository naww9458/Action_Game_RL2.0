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
from script.role.policies.policy_bundle import PolicyBundleRegistry, PolicyBundleSpec

from .g1_control_config import G1_ROBOT_NAME


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


def _require_range_pair(value: Any, *, context: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{context} must be a [min, max] pair")
    return (float(value[0]), float(value[1]))


def _encoder_bias_range(dr_cfg: dict) -> Optional[tuple[float, float]]:
    enc_cfg = dr_cfg.get("encoder_bias")
    if not isinstance(enc_cfg, dict):
        return None
    if "enabled" not in enc_cfg:
        raise KeyError("domain_randomization.encoder_bias.enabled is required")
    if not enc_cfg["enabled"]:
        return None
    if "bias_range" not in enc_cfg:
        raise KeyError(
            "domain_randomization.encoder_bias.bias_range is required when enabled"
        )
    return _require_range_pair(
        enc_cfg["bias_range"],
        context="domain_randomization.encoder_bias.bias_range",
    )


def _base_com_offset_range(dr_cfg: dict) -> Optional[dict]:
    com_cfg = dr_cfg.get("base_com")
    if not isinstance(com_cfg, dict):
        return None
    if "enabled" not in com_cfg:
        raise KeyError("domain_randomization.base_com.enabled is required")
    if not com_cfg["enabled"]:
        return None
    cr = com_cfg.get("offset_range")
    if not isinstance(cr, dict):
        raise KeyError(
            "domain_randomization.base_com.offset_range is required when enabled"
        )
    missing = [axis for axis in ("x", "y", "z") if axis not in cr]
    if missing:
        raise KeyError(
            "domain_randomization.base_com.offset_range missing axes "
            f"{missing}"
        )
    return {
        "x": _require_range_pair(cr["x"], context="base_com.offset_range.x"),
        "y": _require_range_pair(cr["y"], context="base_com.offset_range.y"),
        "z": _require_range_pair(cr["z"], context="base_com.offset_range.z"),
    }


def _wire_encoder_bias_into_action(environment: Any, obs_actor: Any) -> None:
    bias = getattr(obs_actor, "encoder_bias_wp", None)
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


def _bind_obs_actor_on_environment(environment: Any, obs_actor: Any) -> None:
    environment.g1_obs_actor = obs_actor
    environment.obs_dim = obs_actor.obs_dim
    environment.rl_action_dim = obs_actor.rl_action_dim
    environment.flat_obs_dim = obs_actor.flat_obs_dim
    environment.obs_wp = obs_actor.obs_wp
    environment.obs_torch = obs_actor.obs_torch
    environment.single_obs_wp = obs_actor.single_obs_wp
    environment.commands = obs_actor.commands
    environment.command_labels = obs_actor.command_labels
    consumers = getattr(environment, "command_consumer_patterns", None)
    if not isinstance(consumers, set):
        consumers = set()
        environment.command_consumer_patterns = consumers
    pattern = getattr(obs_actor, "pattern", None)
    if pattern:
        consumers.add(normalize_robot_pattern(str(pattern)))
    environment.policy_actions = obs_actor.policy_actions
    environment.prev_actions = obs_actor.prev_actions
    environment.view = obs_actor.view
    environment.history_len = int(obs_actor.history_len)


def _bind_obs_index_describe(environment: Any) -> None:
    """Expose TryGet ``try_describe_obs_index`` on the environment (NaN reports)."""

    def try_describe_obs_index(index: int, group: str = "") -> Optional[str]:
        actor = getattr(environment, "g1_obs_actor", None)
        critic = getattr(environment, "g1_obs_critic", None)
        hosts = (critic, actor) if "critic" in str(group).lower() else (actor, critic)
        for host in hosts:
            fn = getattr(host, "try_describe_obs_index", None)
            if not callable(fn):
                continue
            try:
                text = fn(int(index), group)
            except TypeError:
                try:
                    text = fn(int(index))
                except Exception:
                    continue
            except Exception:
                continue
            if text:
                return str(text)
        return None

    environment.try_describe_obs_index = try_describe_obs_index


def _attach_contact_runtime(environment: Any, bundle: PolicyBundleSpec) -> None:
    """Create the version's foot contact sensor when ``foot_sensor_cfg.py`` exists."""
    foot_mod = PolicyBundleRegistry.import_version_module(bundle, "foot_sensor_cfg")
    if foot_mod is None:
        return

    environment.resolve_body_mj_id = getattr(foot_mod, "resolve_g1_body_mj_id", None)

    pm = getattr(environment, "physics_manager", None)
    if pm is None:
        return

    try:
        from sensors.foot_contact_sensor import FootContactSensor
    except ImportError:
        from script.sensors.foot_contact_sensor import FootContactSensor

    foot_cfg = foot_mod.load_g1_foot_sensor_config()
    foot_mapping = foot_mod.resolve_g1_foot_body_mapping(
        pm, GameConfig.DEVICE, body_suffixes=foot_cfg.bodies
    )
    environment.foot_sensor = FootContactSensor(
        environment.num_env,
        device=GameConfig.DEVICE,
        primary_newton_bodies=foot_mapping.ankle_newton_bodies_wp,
        num_feet=foot_cfg.num_feet,
        ground_geom_id=foot_cfg.ground_geom_id,
        ground_height=foot_cfg.ground_height,
        foot_height_site_offset=foot_cfg.foot_height_site_offset,
    )
    environment.foot_sensor.bind_solver_constants(
        ngeom=foot_mapping.ngeom,
        njmax=foot_mapping.njmax,
        nbody_mj=foot_mapping.nbody_mj,
        naconmax=foot_mapping.naconmax,
        opt_cone=foot_mapping.opt_cone,
    )


def _attach_critic(environment: Any, bundle: PolicyBundleSpec, obs_actor: Any) -> None:
    from script.role.policies.critic_obs import CriticObsRegistry

    critic_id = bundle.obs_critic
    environment.g1_obs_critic = CriticObsRegistry.create(
        critic_id,
        num_instances=obs_actor.num_instances,
        policy_obs_dim=environment.flat_obs_dim,
        foot_sensor=getattr(environment, "foot_sensor", None),
        physics_manager=environment.physics_manager,
        device=GameConfig.DEVICE,
    )
    if environment.g1_obs_critic is not None:
        print(
            f"[G1PolicyRuntime] asymmetric critic extension active: "
            f"critic_obs_dim={environment.g1_obs_critic.critic_obs_dim}"
        )


class G1PolicyRuntime:
    """Per-environment hook: observation, velocity commands, and previous-action history."""

    def __init__(self, environment: Any, obs_actor: Any):
        self._environment = environment
        self.obs_actor = obs_actor

    def get_observation(self):
        self.obs_actor.get_observation(self._environment.physics_manager)
        return self._environment.obs_torch

    def on_reset(self, terminated, current_step) -> None:
        del current_step
        if isinstance(terminated, wp.array):
            terminated_int = wp.to_torch(terminated).to(dtype=torch.int32)
        elif isinstance(terminated, torch.Tensor):
            terminated_int = terminated.to(dtype=torch.int32, device=GameConfig.DEVICE)
        else:
            terminated_int = torch.as_tensor(
                terminated, dtype=torch.int32, device=GameConfig.DEVICE
            )
        terminated_wp = wp.from_torch(terminated_int.contiguous(), dtype=wp.int32)
        self.obs_actor.reset_commands(terminated_wp)
        self.obs_actor.reset_policy_actions(terminated_int.bool())
        if getattr(self._environment, "obs_wp", None) is not None:
            self.obs_actor.compute_single_frame_obs(self._environment.physics_manager)
            self.obs_actor.reset_history(terminated_wp, self._environment.physics_manager)

        env = self._environment
        pm = getattr(env, "physics_manager", None)
        body_q = getattr(getattr(pm, "state_0", None), "body_q", None) if pm is not None else None
        foot = getattr(env, "foot_sensor", None)
        if foot is not None and body_q is not None:
            foot.reset_envs(terminated_wp, body_q=body_q)
        self_collision = getattr(env, "self_collision_sensor", None)
        if self_collision is not None:
            self_collision.reset_history(terminated_wp)

    def on_update_game_status(self, physics_manager, reward_calculator, num_env, current_step) -> None:
        del reward_calculator, num_env, current_step
        dt = 1.0 / float(GameConfig.FPS_ACTION)
        self.obs_actor.update_velocity_commands(physics_manager, dt)
        clearer = getattr(self.obs_actor, "clear_external_command_hold", None)
        defer = bool(getattr(self.obs_actor, "defer_command_hold_clear", False))
        if callable(clearer) and not defer:
            clearer()
        self._update_foot_sensor(physics_manager, dt)
        self._update_angular_momentum(physics_manager)

    def on_step_actions(self, actions_wp) -> None:
        """Store game actions as G1 joint targets only when they already match.

        Training writes low-level joint actions here. Play scenes with
        ``Articulation_body_control_rl_assisted`` send high-level commands;
        that ability already stores the policy's joint actions on its own obs_actor.
        """
        obs_actor = self.obs_actor
        if obs_actor is None or getattr(obs_actor, "policy_actions", None) is None:
            return
        expected = (int(obs_actor.num_instances), int(obs_actor.rl_action_dim))
        if expected[0] <= 0 or expected[1] <= 0:
            return
        actions_torch = wp.to_torch(actions_wp)
        if tuple(int(d) for d in actions_torch.shape) != expected:
            return
        obs_actor.store_low_level_actions(wp.from_torch(actions_torch.contiguous()))

    def on_post_substep(self, substep_idx: int) -> None:
        """Per-substep contact decode for foot air-time and self-collision history."""
        env = self._environment
        pm = getattr(env, "physics_manager", None)
        solver = getattr(getattr(pm, "solver_handler", None), "solver", None)
        if solver is None or not hasattr(solver, "mjw_data"):
            return
        self_collision = getattr(env, "self_collision_sensor", None)
        if substep_idx == 0 and self_collision is not None:
            self_collision.begin_step()
        mj_data = solver.mjw_data
        if self_collision is not None:
            self_collision.record_substep(
                contact=mj_data.contact,
                efc_force=mj_data.efc.force,
                nacon=mj_data.nacon,
                geom_bodyid=solver.mjw_model.geom_bodyid,
                mjc_body_to_newton=solver.mjc_body_to_newton,
            )
        foot = getattr(env, "foot_sensor", None)
        if foot is not None:
            foot.update_substep_from_solver(
                contact=mj_data.contact,
                efc_force=mj_data.efc.force,
                nacon=mj_data.nacon,
                geom_bodyid=solver.mjw_model.geom_bodyid,
                mjc_body_to_newton=solver.mjc_body_to_newton,
                dt=pm.sim_dt,
            )

    def _update_foot_sensor(self, physics_manager, dt: float) -> None:
        foot = getattr(self._environment, "foot_sensor", None)
        if foot is None:
            return
        foot.refresh_policy_step(
            body_q=physics_manager.state_0.body_q,
            body_qd=physics_manager.state_0.body_qd,
            dt=dt,
        )

    def refresh_contact_policy_step(self, physics_manager, dt: float) -> None:
        """Recompute policy-step foot kinematics after a physics step."""
        self._update_foot_sensor(physics_manager, dt)

    def _update_angular_momentum(self, physics_manager) -> None:
        """Refresh whole-robot subtree angular momentum when the solver exposes it."""
        solver = getattr(getattr(physics_manager, "solver_handler", None), "solver", None)
        if solver is None or not hasattr(solver, "mjw_data"):
            return
        try:
            from mujoco_warp._src.smooth import subtree_vel
        except ImportError:
            return
        subtree_vel(solver.mjw_model, solver.mjw_data)


def _bind_post_substep(environment: Any, runtime: G1PolicyRuntime) -> None:
    pm = getattr(environment, "physics_manager", None)
    if pm is None:
        return
    if (
        getattr(environment, "foot_sensor", None) is None
        and getattr(environment, "self_collision_sensor", None) is None
    ):
        return
    pm.post_substep_callback = runtime.on_post_substep


def attach_if_present(environment: Any) -> Optional[G1PolicyRuntime]:
    """Attach G1 obs/command runtime when a G1 player with a policy version is loaded."""
    if getattr(environment, "g1_obs_actor", None) is not None:
        return None
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
    obs_actor = PolicyBundleRegistry.create_obs_actor(
        bundle.obs_actor,
        num_env=environment.num_env,
        device=GameConfig.DEVICE,
        articulation_body=articulation_body,
        pattern=runtime_pattern,
        history_len=int(bundle.history_len),
        enable_obs_noise=bool(getattr(GameConfig, "ENABLE_OBS_NOISE", False)),
        obs_noise_cfg=obs_noise_cfg,
        encoder_bias_range=_encoder_bias_range(dr_cfg),
        base_com_offset_range=_base_com_offset_range(dr_cfg),
    )
    _bind_obs_actor_on_environment(environment, obs_actor)
    _attach_contact_runtime(environment, bundle)
    _attach_critic(environment, bundle, obs_actor)
    _bind_obs_index_describe(environment)
    _wire_encoder_bias_into_action(environment, obs_actor)
    runtime = G1PolicyRuntime(environment, obs_actor)
    _bind_post_substep(environment, runtime)
    hooks = getattr(environment, "_object_template_runtime_hooks", None)
    if hooks is None:
        hooks = []
        setattr(environment, "_object_template_runtime_hooks", hooks)
    hooks.append(runtime)
    return runtime


def setup(environment: Any) -> None:
    attach_if_present(environment)
