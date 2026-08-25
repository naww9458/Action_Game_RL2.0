"""Control-policy version registry for RL-assisted articulation control."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING
import importlib

import torch
import yaml

if TYPE_CHECKING:
    from script.role.bodies.articulation_body import ArticulationBody

GAME_ROOT = Path(__file__).resolve().parents[3]
OBJECT_TEMPLATE_ROOT = Path(__file__).resolve().parents[1] / "objects" / "object_template"

ObsActorFactory = Callable[..., Any]


@dataclass(frozen=True)
class PolicyBundleSpec:
    """Resolved control-policy version (e.g. G1 models/V1)."""

    bundle_id: str
    config_dir: Path
    robot_pattern: str
    control_task: str
    obs_actor: str
    command_dim: int
    default_checkpoint: str = "model.pt"
    policy_module: str = "policy_PPO_g1_velocity"
    description: str = ""
    command_labels: Tuple[str, ...] = ()
    command_ranges: Tuple[Tuple[float, float], ...] = ()
    human_control: Dict[str, Any] = field(default_factory=dict)
    articulation_ability: Dict[str, Any] = field(default_factory=dict)
    obs_dim: Optional[int] = None
    low_level_action_dim: Optional[int] = None
    history_len: int = 1
    obs_critic: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def checkpoint_path(self) -> str:
        """Relative checkpoint filename within ``config_dir`` (legacy alias)."""
        return self.default_checkpoint


def _parse_command_ranges(raw_ranges: Any) -> Tuple[Tuple[float, float], ...]:
    if not raw_ranges:
        return ()
    return tuple(tuple(float(v) for v in item) for item in raw_ranges)


def _read_template_robot_pattern(template_dir: Path) -> Optional[str]:
    template_yaml = template_dir / "template.yaml"
    if not template_yaml.exists():
        return None
    with template_yaml.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    obj = data.get("object") or {}
    pattern = obj.get("pattern")
    return str(pattern) if pattern else None


def _require_str(data: Dict[str, Any], key: str, *, context: str) -> str:
    value = data.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required field '{key}' in {context}")
    return str(value)


def _spec_from_version_yaml(
    version_id: str,
    index_entry: Dict[str, Any],
    policy_data: Dict[str, Any],
    config_path: Path,
    template_dir: Path,
) -> PolicyBundleSpec:
    from script.role.abilities.articulation_control_config.robot_pattern import (
        normalize_robot_pattern,
    )

    commands = dict(policy_data.get("commands") or {})
    action = dict(policy_data.get("action") or {})
    policy = dict(policy_data.get("policy") or {})
    observation = dict(policy_data.get("observation") or {})
    if observation.get("history_len") is None:
        raise ValueError(f"Missing required field 'observation.history_len' in {config_path}")
    history_len = int(observation["history_len"])
    obs_critic = observation.get("obs_critic") or None
    if obs_critic is not None:
        obs_critic = str(obs_critic)

    robot_pattern = (
        index_entry.get("robot_pattern")
        or policy_data.get("robot_pattern")
        or _read_template_robot_pattern(template_dir)
    )
    if not robot_pattern:
        raise ValueError(
            f"Control policy '{version_id}' must declare robot_pattern in policy_versions.yaml, "
            f"control_policy.yaml, or template.yaml under {template_dir}"
        )
    robot_pattern = normalize_robot_pattern(str(robot_pattern))

    if commands.get("dim") is None:
        raise ValueError(f"Missing required field 'commands.dim' in {config_path}")

    return PolicyBundleSpec(
        bundle_id=version_id,
        config_dir=config_path.parent,
        robot_pattern=robot_pattern,
        control_task=_require_str(policy_data, "control_task", context=str(config_path)),
        obs_actor=_require_str(
            observation, "obs_actor", context=f"{config_path}::observation"
        ),
        command_dim=int(commands["dim"]),
        default_checkpoint=_require_str(
            policy, "default_checkpoint", context=f"{config_path}::policy"
        ),
        policy_module=_require_str(policy, "module", context=f"{config_path}::policy"),
        description=str(policy_data.get("description", "")),
        command_labels=tuple(str(label) for label in (commands.get("labels") or ())),
        command_ranges=_parse_command_ranges(commands.get("ranges")),
        human_control=dict(commands.get("human_control") or {}),
        articulation_ability=dict(policy_data.get("articulation_ability") or {}),
        obs_dim=int(observation["obs_dim"]) if observation.get("obs_dim") is not None else None,
        low_level_action_dim=int(action["low_level_dim"])
        if action.get("low_level_dim") is not None
        else None,
        history_len=history_len,
        obs_critic=obs_critic,
        metadata=dict(policy_data.get("metadata") or {}),
    )


class PolicyBundleRegistry:
    _bundles: Dict[Tuple[str, str], PolicyBundleSpec] = {}
    _obs_actor_factories: Dict[str, ObsActorFactory] = {}
    _versions_yaml_loaded = False

    @classmethod
    def register_bundle(cls, spec: PolicyBundleSpec) -> None:
        from script.role.abilities.articulation_control_config.robot_pattern import (
            normalize_robot_pattern,
        )

        key = (normalize_robot_pattern(spec.robot_pattern), spec.bundle_id)
        cls._bundles[key] = spec

    @classmethod
    def register_obs_actor(cls, actor_id: str, factory: ObsActorFactory) -> None:
        cls._obs_actor_factories[actor_id] = factory

    @classmethod
    def _iter_template_policy_version_files(cls):
        if not OBJECT_TEMPLATE_ROOT.exists():
            return
        for template_dir in sorted(OBJECT_TEMPLATE_ROOT.iterdir()):
            if (
                not template_dir.is_dir()
                or template_dir.name.startswith("_")
                or not template_dir.name.isidentifier()
            ):
                continue
            yaml_path = template_dir / "policy_versions.yaml"
            if yaml_path.exists():
                yield template_dir, yaml_path

    @classmethod
    def ensure_loaded(cls) -> None:
        if cls._versions_yaml_loaded:
            return

        for template_dir, yaml_path in cls._iter_template_policy_version_files():
            with yaml_path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}

            for version_id, index_entry in (raw.get("versions") or {}).items():
                if not isinstance(index_entry, dict):
                    continue
                config_rel = str(index_entry.get("config", "")).strip()
                if not config_rel:
                    raise ValueError(
                        f"Control policy version '{version_id}' is missing 'config' in {yaml_path}"
                    )
                config_path = (template_dir / config_rel).resolve()
                if not config_path.exists():
                    raise FileNotFoundError(
                        f"Control policy config for '{version_id}' not found: {config_path}"
                    )
                with config_path.open("r", encoding="utf-8") as policy_fh:
                    policy_data = yaml.safe_load(policy_fh) or {}
                if not isinstance(policy_data, dict):
                    raise ValueError(f"Invalid control policy yaml: {config_path}")
                cls.register_bundle(
                    _spec_from_version_yaml(
                        str(version_id),
                        index_entry,
                        policy_data,
                        config_path,
                        template_dir,
                    )
                )

        cls._versions_yaml_loaded = True

    @classmethod
    def _import_version_py(cls, spec: PolicyBundleSpec, stem: str):
        path = spec.config_dir / f"{stem}.py"
        if not path.is_file():
            return None
        rel = spec.config_dir.resolve().relative_to(OBJECT_TEMPLATE_ROOT.resolve())
        module_name = "script.role.objects.object_template." + ".".join(rel.parts + (stem,))
        return importlib.import_module(module_name)

    @classmethod
    def import_version_module(cls, spec: PolicyBundleSpec, stem: str):
        """Import ``<version_dir>/<stem>.py`` when that file exists, else None."""
        return cls._import_version_py(spec, stem)

    @classmethod
    def _load_version_plugins(cls, spec: PolicyBundleSpec) -> None:
        """Import ``obs_actor.py`` / ``obs_critic.py`` from the version folder on first use."""
        if spec.obs_actor and spec.obs_actor not in cls._obs_actor_factories:
            module = cls._import_version_py(spec, "obs_actor")
            if module is None:
                raise FileNotFoundError(
                    f"{spec.config_dir / 'obs_actor.py'} is required for "
                    f"observation.obs_actor={spec.obs_actor!r}"
                )
            factory = getattr(module, "create_obs_actor", None)
            if factory is None:
                raise AttributeError(
                    f"{spec.config_dir / 'obs_actor.py'} must export create_obs_actor"
                )
            cls.register_obs_actor(spec.obs_actor, factory)

        critic_id = spec.obs_critic
        if not critic_id:
            return
        from script.role.policies.critic_obs_provider import CriticObsProviderRegistry

        if CriticObsProviderRegistry.is_registered(critic_id):
            return
        module = cls._import_version_py(spec, "obs_critic")
        if module is None:
            raise FileNotFoundError(
                f"{spec.config_dir} declares observation.obs_critic={critic_id!r} but has "
                "no obs_critic.py"
            )
        factory = getattr(module, "create_obs_critic", None)
        if factory is None:
            raise AttributeError(
                f"{spec.config_dir / 'obs_critic.py'} must export create_obs_critic"
            )
        CriticObsProviderRegistry.register(critic_id, factory)

    @classmethod
    def _find_spec(
        cls, version_id: str, *, robot_pattern: str | None = None
    ) -> PolicyBundleSpec:
        from script.role.abilities.articulation_control_config.robot_pattern import (
            normalize_robot_pattern,
        )

        cls.ensure_loaded()
        if robot_pattern is not None:
            key = (normalize_robot_pattern(robot_pattern), version_id)
            spec = cls._bundles.get(key)
            if spec is None:
                raise KeyError(
                    f"Control policy version '{version_id}' not found for robot '{robot_pattern}'. "
                    f"Available: {sorted(cls._bundles.keys())}"
                )
            return spec

        matches = [spec for (robot, vid), spec in cls._bundles.items() if vid == version_id]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            robots = sorted({robot for (robot, vid) in cls._bundles if vid == version_id})
            raise KeyError(
                f"Control policy version '{version_id}' is ambiguous across robots {robots}; "
                f"pass robot_pattern explicitly."
            )
        raise KeyError(
            f"Control policy version '{version_id}' not found. "
            f"Available: {sorted(cls._bundles.keys())}"
        )

    @classmethod
    def get(cls, version_id: str, *, robot_pattern: str | None = None) -> PolicyBundleSpec:
        spec = cls._find_spec(version_id, robot_pattern=robot_pattern)
        cls._load_version_plugins(spec)
        return spec

    @classmethod
    def list_version_ids(cls, robot_pattern: str | None = None) -> list[str]:
        """Unique control-policy version ids, optionally filtered by robot pattern."""
        from script.role.abilities.articulation_control_config.robot_pattern import (
            normalize_robot_pattern,
        )

        cls.ensure_loaded()
        wanted = normalize_robot_pattern(robot_pattern) if robot_pattern else None
        ids: list[str] = []
        seen: set[str] = set()
        for pattern, version_id in cls._bundles:
            if wanted is not None and pattern != wanted:
                continue
            if version_id in seen:
                continue
            seen.add(version_id)
            ids.append(version_id)
        return sorted(ids)

    @classmethod
    def has_policy_versions(cls, robot_pattern: str) -> bool:
        return bool(cls.list_version_ids(robot_pattern))

    @classmethod
    def create_obs_actor(
        cls,
        actor_id: str,
        *,
        num_env: int,
        device: str,
        articulation_body: "ArticulationBody",
        pattern: str,
        history_len: int,
        instance_world_indices: list[int] | None = None,
        instance_view_indices: list[int] | None = None,
        **kwargs,
    ):
        cls.ensure_loaded()
        for spec in cls._bundles.values():
            if spec.obs_actor == actor_id:
                cls._load_version_plugins(spec)
                break
        factory = cls._obs_actor_factories.get(actor_id)
        if factory is None:
            raise KeyError(
                f"Observation actor '{actor_id}' not registered. "
                f"Available: {sorted(cls._obs_actor_factories.keys())}"
            )
        return factory(
            num_env=num_env,
            device=device,
            articulation_body=articulation_body,
            pattern=pattern,
            history_len=history_len,
            instance_world_indices=instance_world_indices,
            instance_view_indices=instance_view_indices,
            **kwargs,
        )


def resolve_checkpoint_path(
    *,
    bundle_spec: PolicyBundleSpec,
    checkpoint_name: Optional[str] = None,
    override_path: Optional[str] = None,
) -> Path:
    """Resolve checkpoint under the version directory (``models/<version>/``)."""
    raw = override_path or checkpoint_name or bundle_spec.default_checkpoint
    if not raw:
        raise ValueError("Policy checkpoint name is required but was not provided.")

    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    # Prefer filename relative to the version config directory.
    version_candidate = bundle_spec.config_dir / Path(raw).name
    if version_candidate.exists():
        return version_candidate.resolve()

    search_roots = [
        bundle_spec.config_dir / raw,
        GAME_ROOT / raw,
        Path.cwd() / raw,
    ]
    for path in search_roots:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        f"Policy checkpoint not found: {raw!r} "
        f"(searched under {bundle_spec.config_dir}, {GAME_ROOT})"
    )


class PolicyRunner:
    """Loads a mjlab-compatible checkpoint once and runs batched inference."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str,
        expected_obs_dim: Optional[int] = None,
        expected_action_dim: Optional[int] = None,
    ) -> None:
        from rl_framework.skrl_script.policy_PPO_g1_velocity import (
            Policy,
            infer_dims_from_mjlab_checkpoint,
            load_mjlab_checkpoint,
        )

        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.dims = infer_dims_from_mjlab_checkpoint(str(checkpoint_path), map_location=self.device)

        if expected_obs_dim is not None and self.dims.actor_obs_dim != expected_obs_dim:
            raise ValueError(
                f"Checkpoint obs dim {self.dims.actor_obs_dim} != provider obs dim {expected_obs_dim}"
            )
        if expected_action_dim is not None and self.dims.action_dim != expected_action_dim:
            raise ValueError(
                f"Checkpoint action dim {self.dims.action_dim} != articulation rl dim {expected_action_dim}"
            )

        import numpy as np
        from gymnasium.spaces import Box

        obs_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.dims.actor_obs_dim,),
            dtype=np.float32,
        )
        action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.dims.action_dim,),
            dtype=np.float32,
        )
        self.policy = Policy(
            obs_space,
            action_space,
            str(self.device),
            hidden_dims=self.dims.hidden_dims,
        )
        load_mjlab_checkpoint(str(checkpoint_path), policy=self.policy, strict=True)
        self.policy.to(self.device)
        self.policy.eval()

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=torch.float32, non_blocking=True).contiguous()
        return self.policy.predict_actions(obs, deterministic=deterministic)


def get_policy_bundle(version_id: str, *, robot_pattern: str | None = None) -> PolicyBundleSpec:
    return PolicyBundleRegistry.get(version_id, robot_pattern=robot_pattern)


def load_policy_runner(
    version_id: str,
    *,
    robot_pattern: str | None = None,
    device: str,
    checkpoint_override: Optional[str] = None,
    expected_obs_dim: Optional[int] = None,
    expected_action_dim: Optional[int] = None,
) -> PolicyRunner:
    spec = get_policy_bundle(version_id, robot_pattern=robot_pattern)
    checkpoint = resolve_checkpoint_path(
        bundle_spec=spec,
        checkpoint_name=spec.default_checkpoint,
        override_path=checkpoint_override,
    )
    return PolicyRunner(
        checkpoint,
        device=device,
        expected_obs_dim=expected_obs_dim,
        expected_action_dim=expected_action_dim,
    )
