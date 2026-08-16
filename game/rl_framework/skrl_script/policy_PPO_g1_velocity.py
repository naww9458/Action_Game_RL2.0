"""skrl-compatible policy/value networks aligned with mjlab rsl-rl MLPModel.

Matches ``Mjlab-Velocity-Flat-Unitree-G1`` actor/critic layout:
  - MLP(hidden_dims, ELU) + separate output head
  - ``EmpiricalNormalization`` on observations
  - Gaussian actor with scalar ``std_param`` (``GaussianDistribution``)

Observation and action dimensions are taken from the skrl spaces passed at
construction time, or inferred from an mjlab checkpoint via
``infer_dims_from_mjlab_checkpoint``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

# Matches rsl_rl.modules.distribution.GaussianDistribution defaults.
_STD_RANGE = (1e-6, 1e6)


class EmpiricalNormalization(nn.Module):
  """Running mean/std normalization (same semantics as rsl-rl)."""

  def __init__(self, shape: int, eps: float = 1e-2) -> None:
    super().__init__()
    self.eps = eps
    self.register_buffer("_mean", torch.zeros(1, shape))
    self.register_buffer("_var", torch.ones(1, shape))
    self.register_buffer("_std", torch.ones(1, shape))
    self.register_buffer("count", torch.tensor(0, dtype=torch.long))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self._mean) / (self._std + self.eps)


def _build_encoder(
  input_dim: int, hidden_dims: tuple[int, ...]
) -> tuple[nn.Sequential, int]:
  layers: list[nn.Module] = []
  in_dim = input_dim
  for hidden_dim in hidden_dims:
    layers.append(nn.Linear(in_dim, hidden_dim))
    layers.append(nn.ELU())
    in_dim = hidden_dim
  return nn.Sequential(*layers), in_dim


def _normalize_mjlab_checkpoint(loaded: dict[str, Any]) -> dict[str, Any]:
  """Migrate legacy mjlab / rsl-rl checkpoint layouts to actor/critic dicts."""
  if "model_state_dict" in loaded:
    model_state_dict = loaded.pop("model_state_dict")
    actor_state_dict: dict[str, torch.Tensor] = {}
    critic_state_dict: dict[str, torch.Tensor] = {}

    for key, value in model_state_dict.items():
      if key.startswith("actor."):
        actor_state_dict[key.replace("actor.", "mlp.", 1)] = value
      elif key.startswith("actor_obs_normalizer."):
        actor_state_dict[key.replace("actor_obs_normalizer.", "obs_normalizer.", 1)] = (
          value
        )
      elif key in ("std", "log_std"):
        actor_state_dict[key] = value

      if key.startswith("critic."):
        critic_state_dict[key.replace("critic.", "mlp.", 1)] = value
      elif key.startswith("critic_obs_normalizer."):
        critic_state_dict[
          key.replace("critic_obs_normalizer.", "obs_normalizer.", 1)
        ] = value

    loaded["actor_state_dict"] = actor_state_dict
    loaded["critic_state_dict"] = critic_state_dict

  actor_sd = loaded.get("actor_state_dict", {})
  if "std" in actor_sd:
    actor_sd["distribution.std_param"] = actor_sd.pop("std")
  if "log_std" in actor_sd:
    actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

  return loaded


def _mlp_linear_indices(state_dict: dict[str, torch.Tensor], prefix: str = "mlp.") -> list[int]:
  indices: set[int] = set()
  for key in state_dict:
    if not key.startswith(prefix) or not key.endswith(".weight"):
      continue
    idx = int(key[len(prefix) :].split(".", maxsplit=1)[0])
    if idx % 2 == 0:
      indices.add(idx)
  return sorted(indices)


def _map_mlp_to_local(
  state_dict: dict[str, torch.Tensor],
  *,
  prefix: str = "mlp.",
  output_head: str,
) -> dict[str, torch.Tensor]:
  """Map rsl-rl ``mlp.N`` keys to local ``encoder.*`` / output-head keys."""
  linear_indices = _mlp_linear_indices(state_dict, prefix=prefix)
  if len(linear_indices) < 2:
    raise ValueError(f"No MLP layers found under prefix '{prefix}'")

  *hidden_indices, output_index = linear_indices
  mapped: dict[str, torch.Tensor] = {}

  for enc_idx, mlp_idx in enumerate(hidden_indices):
    local_idx = enc_idx * 2
    for suffix in ("weight", "bias"):
      src = f"{prefix}{mlp_idx}.{suffix}"
      if src in state_dict:
        mapped[f"encoder.{local_idx}.{suffix}"] = state_dict[src]

  for suffix in ("weight", "bias"):
    src = f"{prefix}{output_index}.{suffix}"
    if src in state_dict:
      mapped[f"{output_head}.{suffix}"] = state_dict[src]

  return mapped


def _map_obs_normalizer(
  state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
  mapped: dict[str, torch.Tensor] = {}
  for key, value in state_dict.items():
    if key.startswith("obs_normalizer."):
      mapped[key] = value
  return mapped


@dataclass(frozen=True)
class MjlabCheckpointDims:
  """Inferred tensor shapes from an mjlab rsl-rl checkpoint."""

  actor_obs_dim: int
  critic_obs_dim: int
  action_dim: int
  hidden_dims: tuple[int, ...]


def infer_dims_from_mjlab_checkpoint(
  path: str,
  map_location: str | torch.device = "cpu",
) -> MjlabCheckpointDims:
  """Infer model dimensions from a mjlab ``model_*.pt`` checkpoint."""
  loaded = _normalize_mjlab_checkpoint(
    torch.load(path, map_location=map_location, weights_only=False)
  )
  actor_sd = loaded["actor_state_dict"]
  critic_sd = loaded["critic_state_dict"]

  actor_linears = _mlp_linear_indices(actor_sd)
  critic_linears = _mlp_linear_indices(critic_sd)
  if len(actor_linears) < 2 or len(critic_linears) < 2:
    raise ValueError("Checkpoint does not contain a valid rsl-rl MLP.")

  *actor_hidden_indices, actor_output_idx = actor_linears
  *critic_hidden_indices, _critic_output_idx = critic_linears

  hidden_dims = tuple(
    int(actor_sd[f"mlp.{idx}.weight"].shape[0]) for idx in actor_hidden_indices
  )
  actor_obs_dim = int(actor_sd["mlp.0.weight"].shape[1])
  critic_obs_dim = int(critic_sd["mlp.0.weight"].shape[1])
  action_dim = int(actor_sd[f"mlp.{actor_output_idx}.weight"].shape[0])

  return MjlabCheckpointDims(
    actor_obs_dim=actor_obs_dim,
    critic_obs_dim=critic_obs_dim,
    action_dim=action_dim,
    hidden_dims=hidden_dims,
  )


def load_mjlab_checkpoint(
  path: str,
  policy: nn.Module | None = None,
  value: nn.Module | None = None,
  *,
  map_location: str | torch.device = "cpu",
  strict: bool = True,
) -> dict[str, Any]:
  """Load mjlab rsl-rl weights into local Policy/Value-style modules."""
  loaded = _normalize_mjlab_checkpoint(
    torch.load(path, map_location=map_location, weights_only=False)
  )

  if policy is not None:
    actor_sd = loaded["actor_state_dict"]
    policy_state: dict[str, torch.Tensor] = {
      **_map_obs_normalizer(actor_sd),
      **_map_mlp_to_local(actor_sd, output_head="action_layer"),
    }
    if "distribution.std_param" in actor_sd:
      policy_state["std_param"] = actor_sd["distribution.std_param"]
    policy.load_state_dict(policy_state, strict=strict)

  if value is not None:
    critic_sd = loaded["critic_state_dict"]
    value_state: dict[str, torch.Tensor] = {
      **_map_obs_normalizer(critic_sd),
      **_map_mlp_to_local(critic_sd, output_head="value_layer"),
    }
    value.load_state_dict(value_state, strict=strict)

  return loaded


try:
  from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
except ImportError as _skrl_import_error:  # pragma: no cover - optional dep
  DeterministicMixin = object  # type: ignore[assignment,misc]
  GaussianMixin = object  # type: ignore[assignment,misc]
  Model = nn.Module  # type: ignore[assignment,misc]
  _SKRL_AVAILABLE = False
else:
  _SKRL_AVAILABLE = True


if _SKRL_AVAILABLE:

  class Policy(GaussianMixin, Model):
    """Gaussian actor aligned with mjlab ``MLPModel`` + ``GaussianDistribution``."""

    def __init__(
      self,
      observation_space,
      action_space,
      device,
      hidden_dims: tuple[int, ...] = (512, 256, 128),
      obs_normalization: bool = True,
      init_std: float = 1.0,
      std_range: tuple[float, float] = _STD_RANGE,
      **kwargs,
    ):
      Model.__init__(self, observation_space, action_space, device, **kwargs)
      GaussianMixin.__init__(
        self,
        clip_actions=False,
        clip_log_std=False,
        reduction="mean",
      )

      self.obs_normalization = obs_normalization
      if obs_normalization:
        self.obs_normalizer = EmpiricalNormalization(self.num_observations)
      else:
        self.obs_normalizer = nn.Identity()

      self.encoder, head_in = _build_encoder(self.num_observations, hidden_dims)
      self.action_layer = nn.Linear(head_in, self.num_actions)
      self.std_range = std_range
      self.std_param = nn.Parameter(init_std * torch.ones(self.num_actions))

    def get_specification(self):
      return {}

    def _encode(self, states: torch.Tensor) -> torch.Tensor:
      states = self.obs_normalizer(states)
      return self.encoder(states)

    def _clamped_std(self) -> torch.Tensor:
      low, high = self.std_range
      return self.std_param.clamp(low, high)

    def compute(self, inputs, role):
      x = self._encode(inputs["states"])
      mean_actions = self.action_layer(x)
      log_std = torch.log(self._clamped_std())
      return mean_actions, log_std, {}

    @torch.no_grad()
    def predict_actions(
      self,
      obs: torch.Tensor,
      *,
      deterministic: bool = True,
    ) -> torch.Tensor:
      """Deployment helper: run actor forward without skrl agent overhead."""
      x = self._encode(obs)
      mean = self.action_layer(x)
      if deterministic:
        return mean
      std = self._clamped_std()
      return mean + std * torch.randn_like(mean)

  class Value(DeterministicMixin, Model):
    """Deterministic critic aligned with mjlab ``MLPModel`` (no distribution)."""

    def __init__(
      self,
      observation_space,
      action_space,
      device,
      hidden_dims: tuple[int, ...] = (512, 256, 128),
      obs_normalization: bool = True,
      **kwargs,
    ):
      Model.__init__(self, observation_space, action_space, device, **kwargs)
      DeterministicMixin.__init__(self)

      self.obs_normalization = obs_normalization
      if obs_normalization:
        self.obs_normalizer = EmpiricalNormalization(self.num_observations)
      else:
        self.obs_normalizer = nn.Identity()

      self.encoder, head_in = _build_encoder(self.num_observations, hidden_dims)
      self.value_layer = nn.Linear(head_in, 1)

    def get_specification(self):
      return {}

    def _encode(self, states: torch.Tensor) -> torch.Tensor:
      states = self.obs_normalizer(states)
      return self.encoder(states)

    def compute(self, inputs, role):
      x = self._encode(inputs["states"])
      return self.value_layer(x), {}

    @torch.no_grad()
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
      """Deployment helper: run critic forward without skrl agent overhead."""
      x = self._encode(obs)
      return self.value_layer(x)

else:  # pragma: no cover - optional dep
  Policy = None  # type: ignore[assignment,misc]
  Value = None  # type: ignore[assignment,misc]
