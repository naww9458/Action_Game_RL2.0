"""Expand a 99-dim V1 checkpoint into boxing_v1's packed actor / critic.

Used only when the launcher explicitly resumes from a V1-sized ``.pt``.
Copies the original input columns and zero-fills the new extras+history columns
so a no-target extras vector cannot change the pretrained walking action. Critic
foot-state extras stay after the packed actor columns. Optimizer / iteration
are not part of this transfer.

A 323-dim boxing checkpoint matches the current packed actor size and loads
unchanged (full resume). There is no template-default source checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import yaml

_VERSION_DIR = Path(__file__).resolve().parent


def _expand_cfg() -> dict[str, int]:
    path = _VERSION_DIR / "control_configs.yaml"
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    boxing = ((raw.get("unitree_g1") or {}).get("boxing") or {})
    cfg = boxing.get("checkpoint_expand") or {}
    required = (
        "source_actor_dim",
        "source_critic_dim",
        "target_actor_dim",
        "target_critic_dim",
    )
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"{path}: unitree_g1.boxing.checkpoint_expand missing {missing}")
    out = {key: int(cfg[key]) for key in required}
    if out["target_actor_dim"] <= out["source_actor_dim"]:
        raise ValueError(f"{path}: target_actor_dim must exceed source_actor_dim")
    if out["target_critic_dim"] - out["target_actor_dim"] != out["source_critic_dim"] - out["source_actor_dim"]:
        raise ValueError(
            f"{path}: critic extras dim must stay constant "
            f"(source {out['source_critic_dim'] - out['source_actor_dim']}, "
            f"target {out['target_critic_dim'] - out['target_actor_dim']})"
        )
    return out


def expand_cfg() -> dict[str, int]:
    return dict(_expand_cfg())


def _fill_for_key(key: str) -> float:
    if key.endswith("_var") or key.endswith("_std"):
        return 1.0
    return 0.0


def _expand_feature_tensor(
    value: torch.Tensor,
    *,
    src_full: int,
    dst_full: int,
    src_head: int,
    fill: float,
) -> torch.Tensor:
    if value.ndim == 0 or int(value.shape[-1]) != src_full:
        return value
    new_shape = list(value.shape)
    new_shape[-1] = dst_full
    out = value.new_full(new_shape, float(fill))
    out[..., :src_head] = value[..., :src_head]
    tail = src_full - src_head
    if tail > 0:
        out[..., dst_full - tail :] = value[..., src_head:]
    return out


def _expand_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    src_full: int,
    dst_full: int,
    src_head: int,
) -> dict[str, torch.Tensor]:
    expanded: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            expanded[key] = value
            continue
        expanded[key] = _expand_feature_tensor(
            value,
            src_full=src_full,
            dst_full=dst_full,
            src_head=src_head,
            fill=_fill_for_key(key),
        )
    return expanded


def expand_checkpoint_dict(loaded: dict[str, Any], cfg: Optional[dict[str, int]] = None) -> dict[str, Any]:
    """Return a copy with actor/critic input dims padded; optimizer dropped."""
    dims = cfg or _expand_cfg()
    if "actor_state_dict" not in loaded or "critic_state_dict" not in loaded:
        raise ValueError("Checkpoint must contain actor_state_dict and critic_state_dict")
    expanded = dict(loaded)
    expanded["actor_state_dict"] = _expand_state_dict(
        loaded["actor_state_dict"],
        src_full=dims["source_actor_dim"],
        dst_full=dims["target_actor_dim"],
        src_head=dims["source_actor_dim"],
    )
    expanded["critic_state_dict"] = _expand_state_dict(
        loaded["critic_state_dict"],
        src_full=dims["source_critic_dim"],
        dst_full=dims["target_critic_dim"],
        src_head=dims["source_actor_dim"],
    )
    expanded.pop("optimizer_state_dict", None)
    expanded["iter"] = 0
    return expanded


def _actor_in_dim(loaded: dict[str, Any]) -> int:
    actor_sd = loaded.get("actor_state_dict") or {}
    weight = actor_sd.get("mlp.0.weight")
    if weight is None:
        raise ValueError("Checkpoint actor_state_dict is missing mlp.0.weight")
    return int(weight.shape[1])


def _critic_in_dim(loaded: dict[str, Any]) -> int:
    critic_sd = loaded.get("critic_state_dict") or {}
    weight = critic_sd.get("mlp.0.weight")
    if weight is None:
        raise ValueError("Checkpoint critic_state_dict is missing mlp.0.weight")
    return int(weight.shape[1])


def expand_if_needed(
    loaded: dict[str, Any],
    *,
    target_actor_dim: int,
    target_critic_dim: Optional[int] = None,
) -> tuple[dict[str, Any], bool]:
    """Expand a V1-sized checkpoint; return (dict, did_expand).

    Already-matching target dims are returned unchanged (full resume).
    """
    dims = _expand_cfg()
    actor_in = _actor_in_dim(loaded)
    if actor_in == int(target_actor_dim):
        return loaded, False
    if actor_in != dims["source_actor_dim"]:
        raise ValueError(
            f"Cannot expand checkpoint actor dim {actor_in} to {target_actor_dim} "
            f"(expected source {dims['source_actor_dim']} or target {dims['target_actor_dim']})"
        )
    wanted_critic = int(target_critic_dim or dims["target_critic_dim"])
    if wanted_critic != dims["target_critic_dim"]:
        raise ValueError(
            f"boxing_v1 expander target critic dim is {dims['target_critic_dim']}, "
            f"got {wanted_critic}"
        )
    if int(target_actor_dim) != dims["target_actor_dim"]:
        raise ValueError(
            f"boxing_v1 expander target actor dim is {dims['target_actor_dim']}, "
            f"got {target_actor_dim}"
        )
    critic_in = _critic_in_dim(loaded)
    if critic_in != dims["source_critic_dim"]:
        raise ValueError(
            f"Cannot expand checkpoint critic dim {critic_in} "
            f"(expected source {dims['source_critic_dim']})"
        )
    return expand_checkpoint_dict(loaded, dims), True


def apply_expanded_weights(runner: Any, loaded: dict[str, Any]) -> None:
    """Copy expanded actor/critic weights into an RSL-RL runner (no optimizer)."""
    actor_sd = loaded["actor_state_dict"]
    critic_sd = loaded["critic_state_dict"]
    alg = getattr(runner, "alg", None)
    if alg is None:
        raise RuntimeError("RSL-RL runner has no .alg to receive transferred weights")

    actor = getattr(alg, "actor", None)
    critic = getattr(alg, "critic", None)
    if actor is None or critic is None:
        policy = getattr(alg, "policy", None)
        actor = getattr(policy, "actor", None) if policy is not None else None
        critic = getattr(policy, "critic", None) if policy is not None else None
    if actor is None or critic is None:
        raise RuntimeError(
            "Cannot apply expanded boxing weights: runner.alg has neither "
            "(actor, critic) nor policy.(actor, critic)"
        )
    actor.load_state_dict(actor_sd, strict=True)
    critic.load_state_dict(critic_sd, strict=True)
