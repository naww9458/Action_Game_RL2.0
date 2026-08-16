import torch
import warp as wp
import numpy as np

from script.game_config import GameConfig

from script.environments.rewards.reward_calculator import RewardCalculator
from script.environments.rewards.game_end_reward import G1LocomotionTerminator
from training.env_defaults import get_default_train_cfg
from script.environments.environment import Environment
from script.role.objects.object_template.mjlab_unitree_g1.g1_foot_sensor_cfg import (
    load_g1_foot_sensor_config,
    resolve_g1_foot_body_mapping,
)
from script.role.policies.critic_obs_provider import CriticObsProviderRegistry

try:
    from sensors.foot_contact_sensor import FootContactSensor
    from sensors.contact_sensor import SelfCollisionSensor
except ImportError:
    from script.sensors.foot_contact_sensor import FootContactSensor
    from script.sensors.contact_sensor import SelfCollisionSensor

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from script.simulate.physics_manager import PhysicsManager


@wp.kernel
def init_push_timer_kernel(
    push_timer: wp.array(dtype=wp.float32),
    push_seeds: wp.array(dtype=wp.int32),
    push_seed_offsets: wp.array(dtype=wp.int32),
    interval_min: float,
    interval_max: float,
):
    tid = wp.tid()
    rng = wp.rand_init(push_seeds[tid], push_seed_offsets[tid])
    push_seed_offsets[tid] = push_seed_offsets[tid] + 1
    push_timer[tid] = wp.randf(rng, interval_min, interval_max)


@wp.kernel
def push_robot_kernel(
    push_timer: wp.array(dtype=wp.float32),
    push_seeds: wp.array(dtype=wp.int32),
    push_seed_offsets: wp.array(dtype=wp.int32),
    root_vels: wp.array2d(dtype=wp.spatial_vector),
    push_vel_out: wp.array2d(dtype=wp.spatial_vector),
    push_mask: wp.array2d(dtype=wp.bool),
    dt: float,
    lin_x_min: float, lin_x_max: float,
    lin_y_min: float, lin_y_max: float,
    lin_z_min: float, lin_z_max: float,
    ang_x_min: float, ang_x_max: float,
    ang_y_min: float, ang_y_max: float,
    ang_z_min: float, ang_z_max: float,
    interval_min: float, interval_max: float,
):
    """mjlab push_by_setting_velocity: interval push, world-frame velocity kick.

    Each env keeps a per-env timer; when it expires the current world-frame root
    velocity is overwritten with itself plus a uniformly sampled delta (linear
    x/y/z, angular roll/pitch/yaw). Applied after rewards (mjlab event timing).
    """
    tid = wp.tid()
    push_mask[tid, 0] = False
    push_timer[tid] = push_timer[tid] - dt
    if push_timer[tid] > 0.0:
        return
    rng = wp.rand_init(push_seeds[tid], push_seed_offsets[tid])
    push_seed_offsets[tid] = push_seed_offsets[tid] + 1
    push_timer[tid] = wp.randf(rng, interval_min, interval_max)

    cur = root_vels[tid, 0]
    out = wp.spatial_vector(
        cur[0] + wp.randf(rng, lin_x_min, lin_x_max),
        cur[1] + wp.randf(rng, lin_y_min, lin_y_max),
        cur[2] + wp.randf(rng, lin_z_min, lin_z_max),
        cur[3] + wp.randf(rng, ang_x_min, ang_x_max),
        cur[4] + wp.randf(rng, ang_y_min, ang_y_max),
        cur[5] + wp.randf(rng, ang_z_min, ang_z_max),
    )
    push_vel_out[tid, 0] = out
    push_mask[tid, 0] = True


class FlatWalk(Environment):
    """Unitree G1 flat velocity tracking (mjlab-aligned)."""

    # Exact iteration-log key order of mjlab's velocity-flat task, so the
    # training log can be compared line-by-line with the reference run
    # (``data_train.md``). ``Game.get_training_log`` reorders its extras dict
    # with this template; keys not produced here (e.g. a disabled reward term)
    # are skipped, extra keys keep their insertion position at the end.
    _MJLAB_LOG_KEY_ORDER: tuple[str, ...] = (
        "Episode_Reward/soft_landing",
        "Episode_Reward/foot_clearance",
        "Episode_Reward/dof_pos_limits",
        "Metrics/angular_momentum_mean",
        "Curriculum/command_vel/lin_vel_y_max",
        "Metrics/twist/error_vel_xy",
        "Episode_Termination/time_out",
        "Metrics/landing_force_mean",
        "Curriculum/command_vel/lin_vel_x_max",
        "Episode_Reward/upright",
        "Curriculum/command_vel/ang_vel_z_max",
        "Episode_Reward/body_ang_vel",
        "Episode_Reward/self_collisions",
        "Episode_Reward/foot_swing_height",
        "Metrics/peak_height_mean",
        "Episode_Reward/track_angular_velocity",
        "Episode_Reward/foot_slip",
        "Episode_Termination/fell_over",
        "Episode_Reward/pose",
        "Episode_Reward/angular_momentum",
        "Curriculum/command_vel/lin_vel_x_min",
        "Episode_Reward/track_linear_velocity",
        "Curriculum/command_vel/ang_vel_z_min",
        "Episode_Metrics/mean_action_acc",
        "Metrics/twist/error_vel_yaw",
        "Episode_Reward/air_time",
        "Episode_Reward/action_rate_l2",
        "Metrics/slip_velocity_mean",
        "Curriculum/command_vel/lin_vel_y_min",
    )

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

        # Asymmetric critic: discovered from the robot's object template via
        # ``CriticObsProviderRegistry``. ``None`` (no registered extension)
        # degrades to the symmetric critic (policy obs only).
        self.critic_obs_provider = None

        # Observation/command provider discovered via PolicyBundleRegistry
        # (object_template register.py + control_policy.yaml observation.provider).
        self.g1_provider = None
        self.foot_sensor = None
        self.self_collision_sensor = None
        self.push_timer = None
        self.push_seeds = None
        self.push_seed_offsets = None
        self.push_enabled = False
        self.push_vel_out = None
        self.push_mask_2d = None
        self._push_interval = (1.0, 3.0)
        self._push_lin = ((-0.5, 0.5), (-0.5, 0.5), (-0.4, 0.4))
        self._push_ang = ((-0.52, 0.52), (-0.52, 0.52), (-0.78, 0.78))

        # mjlab common_step_counter: global policy-step count driving curriculum.
        self._global_step_count = 0
        self._curriculum_stages: list[dict] = []
        self._curriculum_base_ranges: dict[str, tuple[float, float]] = {
            "lin_vel_x": (-1.0, 1.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": (-0.5, 0.5),
        }
        self._curriculum_ranges_applied: Optional[dict] = None

        # Log units: mjlab logs Episode_Reward per-second (sum/20); the game logs
        # per-step means, so multiply by FPS to align (per-step mean x 50 = sum/20).
        self.reward_log_scale = float(getattr(GameConfig, "FPS_ACTION", 50))

        # Per-env twist-error metric buffers (mjlab ``Metrics/twist/*``).
        self._twist_err_xy_buf = wp.zeros(self.num_env, dtype=wp.float32, device=GameConfig.DEVICE)
        self._twist_err_yaw_buf = wp.zeros(self.num_env, dtype=wp.float32, device=GameConfig.DEVICE)

        # Cached 0-d GPU tensors for ``Curriculum/command_vel/*`` (constants).
        self._curriculum_metric_tensors: Optional[dict[str, torch.Tensor]] = None

    def setup(self):
        super().setup()

        pattern = self.resolve_player_pattern()
        env_configs = self.config.get("environment_configs") or {}
        dr_cfg = {}
        if isinstance(env_configs, dict):
            dr_cfg = env_configs.get("domain_randomization") or {}

        if self.g1_provider is None:
            raise RuntimeError(
                "Unitree G1 policy runtime was not attached. "
                "player object.pattern must be unitree_g1 with control_policy_version."
            )

        self.push_timer = wp.zeros(self.num_env, dtype=wp.float32, device=GameConfig.DEVICE)
        seed_base = getattr(GameConfig, "SEED", 31415926)
        self.push_seeds = wp.array(
            np.arange(seed_base + 10000, seed_base + 10000 + self.num_env, dtype=np.int32),
            dtype=wp.int32,
            device=GameConfig.DEVICE,
        )
        self.push_seed_offsets = wp.zeros(self.num_env, dtype=wp.int32, device=GameConfig.DEVICE)

        # C2: push_robot (mjlab interval event). Buffers are persistent and the
        # kernel is seed-driven so CUDA-graph replay stays deterministic.
        push_cfg = dr_cfg.get("push_robot") or {}
        self.push_enabled = bool(push_cfg.get("enabled", True))
        if self.push_enabled:
            interval = push_cfg.get("interval_range_s") or [1.0, 3.0]
            self._push_interval = (float(interval[0]), float(interval[1]))
            vr = push_cfg.get("velocity_range") or {}
            self._push_lin = (
                self._range_of(vr.get("x"), (-0.5, 0.5)),
                self._range_of(vr.get("y"), (-0.5, 0.5)),
                self._range_of(vr.get("z"), (-0.4, 0.4)),
            )
            self._push_ang = (
                self._range_of(vr.get("roll"), (-0.52, 0.52)),
                self._range_of(vr.get("pitch"), (-0.52, 0.52)),
                self._range_of(vr.get("yaw"), (-0.78, 0.78)),
            )
            self.push_vel_out = wp.zeros(
                (self.num_env, 1), dtype=wp.spatial_vector, device=GameConfig.DEVICE
            )
            self.push_mask_2d = wp.zeros(
                (self.num_env, 1), dtype=wp.bool, device=GameConfig.DEVICE
            )
            wp.launch(
                kernel=init_push_timer_kernel,
                dim=self.num_env,
                inputs=[
                    self.push_timer,
                    self.push_seeds,
                    self.push_seed_offsets,
                    self._push_interval[0],
                    self._push_interval[1],
                ],
                device=GameConfig.DEVICE,
            )

        # C3: command-velocity curriculum (mjlab commands_vel stages). Applied
        # by ``update_curriculum`` against the global policy-step counter.
        cur_cfg = dr_cfg.get("curriculum") or {}
        stages_raw = cur_cfg.get("command_vel") or []
        self._curriculum_stages = []
        for s in stages_raw:
            stage = {"step": int(s.get("step", 0))}
            for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                val = s.get(key)
                if val is not None:
                    stage[key] = (float(val[0]), float(val[1]))
            self._curriculum_stages.append(stage)
        self._curriculum_stages.sort(key=lambda s: s["step"])
        self._curriculum_ranges_applied = None

        foot_cfg = load_g1_foot_sensor_config()
        foot_mapping = resolve_g1_foot_body_mapping(
            self.physics_manager, GameConfig.DEVICE, body_suffixes=foot_cfg.bodies
        )
        self.foot_sensor = FootContactSensor(
            self.num_env,
            device=GameConfig.DEVICE,
            primary_newton_bodies=foot_mapping.ankle_newton_bodies_wp,
            num_feet=foot_cfg.num_feet,
            ground_geom_id=foot_cfg.ground_geom_id,
            ground_height=foot_cfg.ground_height,
            foot_height_site_offset=foot_cfg.foot_height_site_offset,
        )
        self.foot_sensor.bind_solver_constants(
            ngeom=foot_mapping.ngeom,
            njmax=foot_mapping.njmax,
            nbody_mj=foot_mapping.nbody_mj,
            naconmax=foot_mapping.naconmax,
            opt_cone=foot_mapping.opt_cone,
        )

        # Self-collision sensor (mjlab ``self_collision`` contact sensor): tracks
        # max robot-on-robot contact force per substep for SelfCollisionCostReward.
        # history width is read from the preset so reward iteration count matches.
        reward_params = getattr(GameConfig, "reward_parameters", None) or {}
        sc_cfg = reward_params.get("SelfCollisionCostReward")
        if isinstance(sc_cfg, dict):
            sc_history = int(sc_cfg.get("num_history", 4))
        else:
            sc_history = 4
        self.self_collision_sensor = SelfCollisionSensor(
            self.num_env,
            device=GameConfig.DEVICE,
            history_length=sc_history,
            ground_geom_id=foot_cfg.ground_geom_id,
        )
        self.self_collision_sensor.bind_solver_constants(
            ngeom=foot_mapping.ngeom,
            njmax=foot_mapping.njmax,
            nbody_mj=foot_mapping.nbody_mj,
            naconmax=foot_mapping.naconmax,
            opt_cone=foot_mapping.opt_cone,
        )
        self.physics_manager.post_substep_callback = self._record_self_collision_substep
        if getattr(self, "view", None) is None:
            self.view = self.g1_provider.view

        # Asymmetric critic extension: the Unitree G1 object template registers
        # a foot-state critic-obs provider under the robot pattern. When it is
        # missing, the environment gracefully falls back to the symmetric critic.
        self.critic_obs_provider = CriticObsProviderRegistry.create(
            pattern,
            num_instances=self.g1_provider.num_instances,
            policy_obs_dim=self.flat_obs_dim,
            foot_sensor=self.foot_sensor,
            physics_manager=self.physics_manager,
            device=GameConfig.DEVICE,
        )
        if self.critic_obs_provider is not None:
            print(
                f"[FlatWalk] asymmetric critic extension active: "
                f"critic_obs_dim={self.critic_obs_provider.critic_obs_dim}"
            )

        self._apply_foot_friction_randomization()

        self.reset_env(self.game.terminated, self.game.current_step)
        self.physics_manager.simulate()
        # Seed real contact buffers after the post-reset physics step so the
        # first reward / critic read is coherent.
        self._update_foot_sensor(
            self.physics_manager, 1.0 / float(GameConfig.FPS_ACTION)
        )
        self.obs_buf_gpu = wp.zeros(
            shape=(self.players.num_rl_players, self.flat_obs_dim), dtype=float, device=GameConfig.DEVICE
        )

        try:
            reward_components_cls = GameConfig.reward_components
            reward_components_diff_cls = GameConfig.reward_components_diff
        except AttributeError:
            train_cfg = get_default_train_cfg("flat_walk")
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
            binder = getattr(rc, "bind_environment", None)
            if binder:
                binder(self)
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
            environment=self,
            terminated=self.game.terminated,
            reward_components=reward_components,
            reward_components_diff=reward_components_diff,
            episode_end_detector=game_end,
        )

        print(
            f"[FlatWalk] obs_dim={self.obs_dim}, rl_action_dim={self.rl_action_dim}, history={self.history_len}"
        )
        return self.players, self.platforms, self.entities, self.abilities_objects, self.reward_calculator

    def _apply_foot_friction_randomization(self):
        """Randomize mu on foot geoms only (mjlab GeomFrictionEvent, shared)."""
        env_configs = self.config.get("environment_configs") or {}
        dr_cfg = env_configs.get("domain_randomization") or {} if isinstance(env_configs, dict) else {}
        fr_cfg = dr_cfg.get("foot_friction") or {}
        if not fr_cfg.get("enabled", False):
            return
        model = self.physics_manager.model
        if not hasattr(model, "shape_material_mu") or not hasattr(model, "shape_body"):
            return
        rng_range = fr_cfg.get("range") or [0.3, 1.2]
        lo, hi = float(rng_range[0]), float(rng_range[1])
        shared = bool(fr_cfg.get("shared_random", True))
        patterns = {str(p).rsplit("/", 1)[-1].lower() for p in (fr_cfg.get("body_patterns") or [])}
        if not patterns:
            return
        mu_np = np.asarray(model.shape_material_mu.numpy()).copy()
        shape_body = np.asarray(model.shape_body.numpy()).reshape(-1)
        raw_labels = model.body_label
        if hasattr(raw_labels, "numpy"):
            raw_labels = raw_labels.numpy()
        labels = [str(x).rstrip("/").split("/")[-1].lower() for x in raw_labels]
        nworld = int(getattr(model, "world_count", 1) or 1)
        bodies_per_world = max(int(model.body_count) // max(nworld, 1), 1)
        foot_idx = []
        for i, b in enumerate(shape_body):
            bi = int(b)
            if bi < 0:
                continue
            loc = bi % bodies_per_world
            if loc < len(labels) and labels[loc] in patterns:
                foot_idx.append(i)
        if not foot_idx:
            return
        rng = np.random.default_rng(getattr(GameConfig, "SEED", 42))
        if shared:
            mu = float(rng.uniform(lo, hi))
            for i in foot_idx:
                mu_np[i] = mu
        else:
            for i in foot_idx:
                mu_np[i] = float(rng.uniform(lo, hi))
        model.shape_material_mu.assign(mu_np)

    def _get_critic_observation(self) -> torch.Tensor:
        """Return the asymmetric critic observation when a provider is registered.

        The provider is defined by the robot's object template (e.g. Unitree G1
        foot-state extras). When no provider is registered (or its buffers are
        not ready), the environment falls back to the symmetric critic (policy obs).
        """
        provider = self.critic_obs_provider
        # Runtime safety: any provider must be fully initialised (setup done)
        # before use; otherwise fall back to the symmetric critic.
        if provider is None or getattr(provider, "critic_obs_wp", None) is None:
            return self.obs_torch
        return provider.get_critic_observation(self.g1_provider.noiseless_obs_wp)

    @wp.kernel
    def update_game_status_gpu(current_step: wp.array(dtype=wp.int32)):
        tid = wp.tid()
        current_step[tid] += 1

    @staticmethod
    def _range_of(value, default):
        """Normalise a YAML [min, max] pair to a (float, float) tuple."""
        if value is None:
            return (float(default[0]), float(default[1]))
        return (float(value[0]), float(value[1]))

    def on_step_actions(self, actions_wp):
        super().on_step_actions(actions_wp)
        # C2: push robot right after rewards (mjlab interval-event timing) so the
        # kick affects the next policy step's simulation.
        self._update_push_robot()

    def _update_push_robot(self):
        if not self.push_enabled or self.view is None or self.push_vel_out is None:
            return
        dt = 1.0 / float(GameConfig.FPS_ACTION)
        root_vels = self.view.get_root_velocities(self.physics_manager.state_0)
        (lx0, lx1), (ly0, ly1), (lz0, lz1) = self._push_lin
        (ax0, ax1), (ay0, ay1), (az0, az1) = self._push_ang
        it0, it1 = self._push_interval
        wp.launch(
            kernel=push_robot_kernel,
            dim=self.num_env,
            inputs=[
                self.push_timer,
                self.push_seeds,
                self.push_seed_offsets,
                root_vels,
                self.push_vel_out,
                self.push_mask_2d,
                dt,
                lx0, lx1, ly0, ly1, lz0, lz1,
                ax0, ax1, ay0, ay1, az0, az1,
                it0, it1,
            ],
            device=GameConfig.DEVICE,
        )
        self.view.set_root_velocities(
            self.physics_manager.state_0, self.push_vel_out, mask=self.push_mask_2d
        )

    def update_curriculum(self):
        """Advance the global policy-step counter and apply curriculum stages.

        Mirrors mjlab ``common_step_counter += 1`` / ``commands_vel``: when the
        counter crosses a stage threshold, the provider's velocity-command
        sampling ranges are updated on device (kernels read them by address, so
        the change is picked up by the replayed CUDA graph on the next launch).
        """
        self._global_step_count += 1
        if not self._curriculum_stages or self.g1_provider is None:
            return
        stage = None
        for s in self._curriculum_stages:
            if self._global_step_count >= s["step"]:
                stage = s
        if stage is None:
            return
        new_ranges = {
            "lin_vel_x": stage.get("lin_vel_x", self._curriculum_base_ranges["lin_vel_x"]),
            "lin_vel_y": stage.get("lin_vel_y", self._curriculum_base_ranges["lin_vel_y"]),
            "ang_vel_z": stage.get("ang_vel_z", self._curriculum_base_ranges["ang_vel_z"]),
        }
        if new_ranges != self._curriculum_ranges_applied:
            self.g1_provider.set_command_velocity_ranges(new_ranges)
            self._curriculum_ranges_applied = new_ranges
            # Invalidate the cached Curriculum/* metric tensors.
            self._curriculum_metric_tensors = None

    def reset_env(self, terminated, current_step):
        super().reset_env(terminated=terminated, current_step=current_step)

        if isinstance(terminated, torch.Tensor):
            terminated_int = terminated.to(dtype=torch.int32, device=GameConfig.DEVICE)
        else:
            terminated_int = torch.tensor(terminated, dtype=torch.int32, device=GameConfig.DEVICE)
        terminated_wp = wp.from_torch(terminated_int, dtype=wp.int32)

        if self.foot_sensor is not None:
            self.foot_sensor.reset_envs(
                terminated_wp, body_q=self.physics_manager.state_0.body_q
            )

        if self.self_collision_sensor is not None:
            self.self_collision_sensor.reset_history(terminated_wp)

    def _update_foot_sensor(self, physics_manager, dt: float) -> None:
        """Policy-step foot-sensor refresh.

        Air-time / contact-time accumulation happens per physics substep in
        ``_record_self_collision_substep`` (mjlab granularity). Here we only
        recompute the first-contact / first-air windows at the policy-step dt
        (mjlab ``compute_first_contact(dt=step_dt)``) and refresh kinematics.
        """
        if self.foot_sensor is None:
            return
        self.foot_sensor.refresh_policy_step(
            body_q=physics_manager.state_0.body_q,
            body_qd=physics_manager.state_0.body_qd,
            dt=dt,
        )

    def _update_angular_momentum(self) -> None:
        """Refresh whole-robot subtree angular momentum (mjlab subtreeangmom).

        mujoco-warp only computes ``subtree_angmom`` during ``sensor_vel`` when
        the model declares a ``subtreeangmom`` sensor; the game's USD-derived
        model does not, so call ``smooth.subtree_vel`` explicitly once per step
        before ``AngularMomentumPenaltyReward`` reads the buffer.
        """
        solver = getattr(getattr(self.physics_manager, "solver_handler", None), "solver", None)
        if solver is None or not hasattr(solver, "mjw_data"):
            return
        try:
            from mujoco_warp._src.smooth import subtree_vel
        except ImportError:
            return
        subtree_vel(solver.mjw_model, solver.mjw_data)

    def _record_self_collision_substep(self, substep_idx: int) -> None:
        """Per-substep hook: decode self-collision force + foot contact state.

        Runs inside ``physics_manager.simulate()`` (also under CUDA-graph
        capture). The self-collision sensor accumulates a per-step max-force
        history; the foot sensor accumulates air/contact times at physics-dt
        granularity (mjlab updates its contact sensors every substep).
        """
        solver = getattr(getattr(self.physics_manager, "solver_handler", None), "solver", None)
        if solver is None or not hasattr(solver, "mjw_data"):
            return
        if substep_idx == 0 and self.self_collision_sensor is not None:
            self.self_collision_sensor.begin_step()
        mj_data = solver.mjw_data
        if self.self_collision_sensor is not None:
            self.self_collision_sensor.record_substep(
                contact=mj_data.contact,
                efc_force=mj_data.efc.force,
                nacon=mj_data.nacon,
                geom_bodyid=solver.mjw_model.geom_bodyid,
                mjc_body_to_newton=solver.mjc_body_to_newton,
            )
        if self.foot_sensor is not None:
            self.foot_sensor.update_substep_from_solver(
                contact=mj_data.contact,
                efc_force=mj_data.efc.force,
                nacon=mj_data.nacon,
                geom_bodyid=solver.mjw_model.geom_bodyid,
                mjc_body_to_newton=solver.mjc_body_to_newton,
                dt=self.physics_manager.sim_dt,
            )

    def update_game_status(self, physics_manager, reward_calculator, num_env, current_step):
        dt = 1.0 / float(GameConfig.FPS_ACTION)
        super().update_game_status(physics_manager, reward_calculator, num_env, current_step)
        self._update_foot_sensor(physics_manager, dt)
        self._update_angular_momentum()

        wp.launch(
            kernel=self.update_game_status_gpu,
            dim=num_env,
            inputs=[current_step],
            device=GameConfig.DEVICE,
        )

    @wp.kernel
    def compute_twist_error_metrics_kernel(
        commands: wp.array2d(dtype=wp.float32),
        root_tfs: wp.array2d(dtype=wp.transform),
        root_vels: wp.array2d(dtype=wp.spatial_vector),
        instance_world_indices: wp.array(dtype=wp.int32),
        instance_view_indices: wp.array(dtype=wp.int32),
        err_xy_buf: wp.array(dtype=wp.float32),
        err_yaw_buf: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        world = instance_world_indices[tid]
        obj_idx = instance_view_indices[tid]

        my_tf = root_tfs[world, obj_idx]
        my_rot = wp.transform_get_rotation(my_tf)
        root_qd = root_vels[world, obj_idx]
        inv_rot = wp.quat_inverse(my_rot)

        # Warp spatial_vector layout is [lin.x, lin.y, lin.z, ang.x, ang.y, ang.z].
        world_lin = wp.vec3(root_qd[0], root_qd[1], root_qd[2])
        world_ang = wp.vec3(root_qd[3], root_qd[4], root_qd[5])
        local_lin = wp.quat_rotate(inv_rot, world_lin)
        local_ang = wp.quat_rotate(inv_rot, world_ang)

        dx = local_lin[0] - commands[tid, 0]
        dy = local_lin[1] - commands[tid, 1]
        err_xy_buf[world] = wp.sqrt(dx * dx + dy * dy)
        err_yaw_buf[world] = wp.abs(local_ang[2] - commands[tid, 2])

    def get_command_metrics(self) -> Optional[dict]:
        """Return per-step twist command-tracking metrics (mjlab ``Metrics/twist/*``).

        Mirrors ``velocity_command.py`` ``_update_metrics``: ``error_vel_xy`` is
        the body-frame xy linear-velocity error and ``error_vel_yaw`` the yaw
        angular-velocity error, both against the resampled velocity command.
        Values are 0-d GPU tensors (mean over envs), consumed by the generic
        ``Game.get_training_log`` public interface.
        """
        if self.g1_provider is None or self.commands is None or self.view is None:
            return None
        root_tfs = self.view.get_root_transforms(self.physics_manager.state_0)
        root_vels = self.view.get_root_velocities(self.physics_manager.state_0)
        wp.launch(
            kernel=self.compute_twist_error_metrics_kernel,
            dim=self.g1_provider.num_instances,
            inputs=[
                self.commands,
                root_tfs,
                root_vels,
                self.g1_provider.instance_world_indices_wp,
                self.g1_provider.instance_view_indices_wp,
                self._twist_err_xy_buf,
                self._twist_err_yaw_buf,
            ],
            device=GameConfig.DEVICE,
        )
        return {
            "Metrics/twist/error_vel_xy": torch.mean(wp.to_torch(self._twist_err_xy_buf)),
            "Metrics/twist/error_vel_yaw": torch.mean(wp.to_torch(self._twist_err_yaw_buf)),
        }

    def get_log_key_order(self) -> Optional[tuple[str, ...]]:
        """Align iteration-log lines with the mjlab reference (exact key order).

        Consumed by ``Game.get_training_log`` to reorder the per-step extras so
        the printed iteration summary matches mjlab's ``data_train.md`` layout.
        """
        return self._MJLAB_LOG_KEY_ORDER

    def _build_curriculum_metric_tensors(self) -> dict[str, torch.Tensor]:
        """Build constant 0-d GPU tensors for the command sampling ranges."""
        ranges = self.g1_provider.get_command_velocity_ranges()
        device = GameConfig.DEVICE
        return {
            "Curriculum/command_vel/lin_vel_x_min": torch.full((), ranges["lin_vel_x"][0], device=device),
            "Curriculum/command_vel/lin_vel_x_max": torch.full((), ranges["lin_vel_x"][1], device=device),
            "Curriculum/command_vel/lin_vel_y_min": torch.full((), ranges["lin_vel_y"][0], device=device),
            "Curriculum/command_vel/lin_vel_y_max": torch.full((), ranges["lin_vel_y"][1], device=device),
            "Curriculum/command_vel/ang_vel_z_min": torch.full((), ranges["ang_vel_z"][0], device=device),
            "Curriculum/command_vel/ang_vel_z_max": torch.full((), ranges["ang_vel_z"][1], device=device),
        }

    def get_curriculum_metrics(self) -> Optional[dict]:
        """Return per-step command-range metrics (mjlab ``Curriculum/command_vel/*``).

        Mirrors the values the command-resample kernels actually sample from.
        The cached tensors are invalidated by ``update_curriculum`` whenever a
        stage boundary is crossed, so logged ranges always match the simulated
        commands. Values are 0-d GPU tensors, consumed by ``Game.get_training_log``.
        """
        if self.g1_provider is None:
            return None
        if self._curriculum_metric_tensors is None:
            self._curriculum_metric_tensors = self._build_curriculum_metric_tensors()
        return self._curriculum_metric_tensors

