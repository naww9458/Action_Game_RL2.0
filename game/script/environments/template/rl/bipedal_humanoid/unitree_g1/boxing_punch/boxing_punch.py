"""Boxing punch training: 70% boxing / 30% pure-walk fixed world split.

Walk worlds never bind a target: their extras carry the -1 "no target"
sentinel, boxing rewards short-circuit to 0 via ``has_target``, commands stay
randomly sampled, and the striking target spawns ``walk_target_z_raise``
meters higher (kinematic anchor + soft spring holds it far above the robot).
"""

from __future__ import annotations

from typing import Optional

import torch
import warp as wp

from script.environments.custom.rl.bipedal_humanoid.unitree_g1.flat_walk.flat_walk import (
    FlatWalk,
)
from script.game_config import GameConfig
from script.role.base_role import BaseRole
from training.env_defaults import get_default_train_cfg


def _leaf_body_index(path_map, leaf: str) -> int:
    name = str(leaf or "").strip().strip("/")
    if not name or not isinstance(path_map, dict):
        return -1
    matches = [
        int(idx)
        for path, idx in path_map.items()
        if str(path).rstrip("/").split("/")[-1] == name
    ]
    return int(matches[0]) if len(matches) == 1 else -1


class BoxingPunch(FlatWalk):
    """G1 punch training (command overlay + boxing rewards, 70/30 world split)."""

    _MJLAB_LOG_KEY_ORDER: tuple[str, ...] = (
        "Episode_Reward/boxing_distance",
        "Episode_Reward/boxing_ready_pose",
        "Episode_Reward/boxing_punch",
        "Episode_Reward/boxing_retract",
        "Episode_Reward/boxing_hit",
        "Episode_Reward/boxing_pose",
        "Episode_Reward/track_linear_velocity",
        "Episode_Reward/track_angular_velocity",
        "Episode_Reward/upright",
        "Episode_Reward/pose",
        "Episode_Reward/soft_landing",
        "Episode_Reward/foot_clearance",
        "Episode_Reward/foot_swing_height",
        "Episode_Reward/foot_slip",
        "Episode_Reward/air_time",
        "Episode_Reward/body_ang_vel",
        "Episode_Reward/angular_momentum",
        "Episode_Reward/action_rate_l2",
        "Episode_Reward/dof_pos_limits",
        "Episode_Reward/self_collisions",
        "Episode_Termination/time_out",
        "Episode_Termination/fell_over",
        "Metrics/boxing_hit_rate",
        "Metrics/boxing_hit_force",
        "Metrics/twist/error_vel_xy",
        "Metrics/twist/error_vel_yaw",
        "Metrics/twist/follow_vel_x",
        "Metrics/twist/follow_vel_y",
        "Metrics/twist/follow_vel_yaw",
    )

    def setup(self):
        try:
            GameConfig.reward_components
        except AttributeError:
            train_cfg = get_default_train_cfg("boxing_punch")
            GameConfig.reward_components = train_cfg.reward_components
            GameConfig.reward_components_diff = train_cfg.reward_components_diff
            GameConfig.reward_parameters = train_cfg.reward_parameters
        # Masks must exist before FlatWalk.setup's first reset_env, which calls
        # prepare_reset_observations. Spawn Z is raised inside reset_env once
        # articulation_body exists (after Environment.setup, before reset_obj).
        self._init_walk_split_masks()
        result = super().setup()
        self._bind_default_selected_target(world_mask=self._boxing_world_mask)
        self._clear_walk_world_targets()
        return result

    def reset_env(self, terminated, current_step):
        self._apply_walk_target_raise_once()
        super().reset_env(terminated, current_step)

    def _init_walk_split_masks(self) -> None:
        env_configs = self.config.get("environment_configs") or {}
        if "walk_env_ratio" not in env_configs:
            raise KeyError("environment_configs.walk_env_ratio is required")
        if "walk_target_z_raise" not in env_configs:
            raise KeyError("environment_configs.walk_target_z_raise is required")
        ratio = float(env_configs["walk_env_ratio"])
        raise_z = float(env_configs["walk_target_z_raise"])
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"environment_configs.walk_env_ratio must be in [0, 1], got {ratio}")
        if raise_z < 0.0:
            raise ValueError(f"environment_configs.walk_target_z_raise must be >= 0, got {raise_z}")
        self._walk_env_ratio = ratio
        self._walk_target_z_raise = raise_z
        n = int(getattr(self, "num_env", 0) or 0)
        idx = torch.arange(n, dtype=torch.float64)
        # Even interleave: world w is pure-walk iff floor((w+1)*r) > floor(w*r).
        self._walk_world_mask = torch.floor((idx + 1.0) * ratio) > torch.floor(idx * ratio)
        self._boxing_world_mask = ~self._walk_world_mask
        self._walk_spawn_raised = False

    def _apply_walk_target_raise_once(self) -> None:
        """Raise walk-world target spawn Z before the first reset_obj."""
        if getattr(self, "_walk_spawn_raised", False):
            return
        if getattr(self, "articulation_body", None) is None:
            return
        if getattr(self, "_walk_world_mask", None) is None:
            return
        raise_z = float(getattr(self, "_walk_target_z_raise", 0.0) or 0.0)
        if raise_z <= 0.0:
            self._walk_spawn_raised = True
            return
        walk_worlds = torch.nonzero(self._walk_world_mask, as_tuple=False).flatten()
        if walk_worlds.numel() > 0:
            self._raise_walk_world_targets(walk_worlds, raise_z)
        self._walk_spawn_raised = True

    def _target_body_local_index(self) -> int:
        """Body-container local index of the striking target (-1 when unknown)."""
        ab = getattr(self, "articulation_body", None)
        if ab is None:
            return -1
        local_role, _ = self._default_selected_role()
        params = list(getattr(BaseRole, "_object_game_params", None) or [])
        if local_role < 0 or local_role >= len(params):
            return -1
        runtime_pattern = str((params[local_role] or {}).get("runtime_pattern") or "")
        role_list = list(getattr(ab, "patterns", {}).get(runtime_pattern) or [])
        body_locals = list(getattr(ab, "patterns_local_indices", {}).get(runtime_pattern) or [])
        if local_role not in role_list or not body_locals:
            return -1
        return int(body_locals[role_list.index(local_role)])

    def _raise_walk_world_targets(self, walk_worlds: torch.Tensor, raise_z: float) -> None:
        ab = getattr(self, "articulation_body", None)
        local_body_idx = self._target_body_local_index()
        n_body_env = int(getattr(ab, "num_body_object_env", 0) or 0)
        if ab is None or local_body_idx < 0 or n_body_env <= 0:
            raise RuntimeError(
                "BoxingPunch: cannot locate the striking target in articulation_body "
                f"(local_body_idx={local_body_idx}, num_body_object_env={n_body_env})"
            )
        first_buf = getattr(ab, "object_default_position_min_gpu", None)
        if first_buf is None:
            raise RuntimeError("BoxingPunch: object_default_position_min_gpu is not built")
        device = wp.to_torch(first_buf).device
        rows = walk_worlds.to(device=device, dtype=torch.long) * n_body_env + local_body_idx
        for attr in ("object_default_position_min_gpu", "object_default_position_max_gpu"):
            buf = getattr(ab, attr, None)
            if buf is not None:
                wp.to_torch(buf)[rows, 2] += float(raise_z)
        # Default spawn transforms: keep initial placement consistent with the
        # raised reset ranges (translation occupies the first 3 floats).
        trans = getattr(ab, "object_default_trans_gpu", None)
        if trans is not None:
            t = wp.to_torch(trans)
            if t.ndim == 2 and t.shape[1] >= 3:
                t[rows, 2] += float(raise_z)

    def _clear_walk_world_targets(self) -> None:
        """Force has_target=0 on pure-walk worlds so extras stay at the -1 sentinel."""
        obs_actor = getattr(self, "g1_obs_actor", None)
        clearer = getattr(obs_actor, "try_clear_command_role_target", None)
        mask = getattr(self, "_walk_world_mask", None)
        if not callable(clearer) or mask is None:
            return
        worlds = list(getattr(obs_actor, "instance_world_indices", None) or [])
        n = int(mask.numel())
        for inst, world in enumerate(worlds):
            w = int(world)
            if 0 <= w < n and bool(mask[w]):
                clearer(inst)

    def _default_selected_role(self) -> tuple[int, str]:
        entities = getattr(self, "entities", None)
        indices = list(getattr(entities, "index_obj_role", None) or [])
        if not indices:
            return -1, ""
        n_env = max(int(getattr(BaseRole, "_num_objects_env", 1) or 1), 1)
        local = int(indices[0]) % n_env
        names = list(getattr(BaseRole, "_name_list", None) or [])
        label = str(names[local] or "") if 0 <= local < len(names) else ""
        return local, label

    def _preferred_body_name(self) -> str:
        obs_actor = getattr(self, "g1_obs_actor", None)
        getter = getattr(obs_actor, "try_preferred_target_body", None)
        if callable(getter):
            name = str(getter() or "").strip()
            if name:
                return name
        cfg_fn = getattr(obs_actor, "get_boxing_obs_cfg", None)
        if callable(cfg_fn):
            return str((cfg_fn() or {}).get("target_body") or "").strip()
        return ""

    def _boxing_reset_mask(self, terminated=None) -> torch.Tensor:
        """Boxing-world mask, optionally intersected with terminated worlds."""
        mask = self._boxing_world_mask
        if terminated is None:
            return mask
        term = terminated
        if isinstance(term, wp.array):
            term = wp.to_torch(term)
        term = torch.as_tensor(term).reshape(-1).to(dtype=torch.bool).cpu()
        if term.numel() != mask.numel():
            return mask
        return mask & term

    def _bind_default_selected_target(self, world_mask=None) -> None:
        obs_actor = getattr(self, "g1_obs_actor", None)
        select = getattr(obs_actor, "try_select_target", None)
        if not callable(select):
            return
        local_role, label = self._default_selected_role()
        if local_role < 0:
            return
        body_name = self._preferred_body_name()
        if not body_name:
            return
        pm = getattr(self, "physics_manager", None)
        metadata = getattr(pm, "object_metadata_by_role", None) or {}
        path_map = (metadata.get(local_role) or {}).get("path_body_map") or {}
        body = _leaf_body_index(path_map, body_name)
        if body < 0:
            return
        select(
            local_body_idx=int(body),
            local_role_idx=int(local_role),
            label=label,
            world_mask=world_mask,
            physics_manager=pm,
        )

    def prepare_reset_observations(self, terminated, current_step):
        del current_step
        if getattr(self, "_boxing_world_mask", None) is None:
            return
        self._bind_default_selected_target(world_mask=self._boxing_reset_mask(terminated))
        self._clear_walk_world_targets()

    def get_command_metrics(self) -> Optional[dict]:
        """Twist metrics plus mean strike force among envs that hit this step."""
        out = super().get_command_metrics()
        merged = dict(out) if out else {}
        calc = getattr(self, "reward_calculator", None)
        components = getattr(calc, "reward_components", None) if calc is not None else None
        if not components:
            return merged or None
        for component in components:
            getter = getattr(component, "try_get_hit_force_mean", None)
            if not callable(getter):
                continue
            mean = getter()
            if mean is not None:
                merged["Metrics/boxing_hit_force"] = mean
            break
        return merged or None
