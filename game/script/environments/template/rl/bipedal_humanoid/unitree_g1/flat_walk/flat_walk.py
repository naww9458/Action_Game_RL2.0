import torch
import warp as wp
import numpy as np

from script.game_config import GameConfig

from script.environments.rewards.reward_calculator import RewardCalculator
from script.environments.rewards.game_end_reward import G1LocomotionTerminator
from training.env_defaults import get_default_train_cfg
from script.environments.environment import Environment

from typing import Optional


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
        self.rl_action_dim = 0
        self.history_len = 0
        self.flat_obs_dim = 0
        self.single_obs_wp = None

        # Asymmetric critic: attached by the G1 object-template runtime when
        # observation.obs_critic is declared. None degrades to the symmetric critic.
        self.critic_obs_provider = None

        # Observation/command provider attached by object-template register.setup.
        self.g1_provider = None
        self.foot_sensor = None
        self.self_collision_sensor = None
        self.push_timer = None
        self.push_seeds = None
        self.push_seed_offsets = None
        self.push_enabled = False
        self.push_vel_out = None
        self.push_mask_2d = None
        self._push_interval = (0.0, 0.0)
        self._push_lin = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        self._push_ang = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

        # mjlab common_step_counter: global policy-step count driving curriculum.
        self._global_step_count = 0
        self._curriculum_stages: list[dict] = []
        self._curriculum_base_ranges: dict[str, tuple[float, float]] = {}
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
        if "enabled" not in push_cfg:
            raise KeyError("environment_configs.domain_randomization.push_robot.enabled is required")
        self.push_enabled = bool(push_cfg["enabled"])
        if self.push_enabled:
            if "interval_range_s" not in push_cfg:
                raise KeyError(
                    "environment_configs.domain_randomization.push_robot.interval_range_s "
                    "is required when enabled"
                )
            interval = push_cfg["interval_range_s"]
            self._push_interval = (float(interval[0]), float(interval[1]))
            vr = push_cfg.get("velocity_range")
            if not isinstance(vr, dict):
                raise KeyError(
                    "environment_configs.domain_randomization.push_robot.velocity_range "
                    "is required when enabled"
                )
            missing_axes = [axis for axis in ("x", "y", "z", "roll", "pitch", "yaw") if axis not in vr]
            if missing_axes:
                raise KeyError(
                    "environment_configs.domain_randomization.push_robot.velocity_range "
                    f"missing {missing_axes}"
                )
            self._push_lin = (
                self._require_range_pair(vr["x"], "push_robot.velocity_range.x"),
                self._require_range_pair(vr["y"], "push_robot.velocity_range.y"),
                self._require_range_pair(vr["z"], "push_robot.velocity_range.z"),
            )
            self._push_ang = (
                self._require_range_pair(vr["roll"], "push_robot.velocity_range.roll"),
                self._require_range_pair(vr["pitch"], "push_robot.velocity_range.pitch"),
                self._require_range_pair(vr["yaw"], "push_robot.velocity_range.yaw"),
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
        self._curriculum_base_ranges = dict(self.g1_provider.get_command_velocity_ranges())
        cur_cfg = dr_cfg.get("curriculum") or {}
        stages_raw = cur_cfg.get("command_vel") or []
        self._curriculum_stages = []
        for s in stages_raw:
            if "step" not in s:
                raise KeyError(
                    "environment_configs.domain_randomization.curriculum.command_vel "
                    "entries require 'step'"
                )
            stage = {"step": int(s["step"])}
            for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                val = s.get(key)
                if val is not None:
                    stage[key] = (float(val[0]), float(val[1]))
            self._curriculum_stages.append(stage)
        self._curriculum_stages.sort(key=lambda s: s["step"])
        self._curriculum_ranges_applied = None

        if getattr(self, "view", None) is None:
            self.view = self.g1_provider.view

        self._apply_foot_friction_randomization()

        self.reset_env(self.game.terminated, self.game.current_step)
        self.physics_manager.simulate()
        # Seed real contact buffers after the post-reset physics step so the
        # first reward / critic read is coherent.
        runtime_hooks = getattr(self, "_object_template_runtime_hooks", None) or []
        for hook in runtime_hooks:
            refresher = getattr(hook, "refresh_contact_policy_step", None)
            if callable(refresher):
                refresher(self.physics_manager, 1.0 / float(GameConfig.FPS_ACTION))
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
        rng_range = fr_cfg.get("range")
        if rng_range is None:
            raise KeyError(
                "environment_configs.domain_randomization.foot_friction.range "
                "is required when enabled"
            )
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
    def _require_range_pair(value, context: str):
        """Normalise a YAML [min, max] pair to a (float, float) tuple."""
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{context} must be a [min, max] pair")
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

    def update_game_status(self, physics_manager, reward_calculator, num_env, current_step):
        super().update_game_status(physics_manager, reward_calculator, num_env, current_step)

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

