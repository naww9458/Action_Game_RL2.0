"""Framework-agnostic NaN / Inf reports for RL env outputs.

Used by ``--nan-check`` guard and abort paths. Reports which env is non-finite
and whether the hit is in observations (column index) or a named reward term.

Obs column names are optional: TryGet ``try_describe_obs_index`` on the env /
object-template obs host. If that mapping is absent, only numeric indices print.

Reward term names come from TryGet ``try_get_reward_term_env_values`` (per-env
buffers on the reward calculator). Total reward is always reported when it is
non-finite.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import torch


_MAX_IDS = 24
_MAX_SAMPLES = 16
_MAX_HOST_HOPS = 16
_HOST_ATTRS = ("environment", "game", "env", "reward_calculator", "unwrapped")


def _as_int_list(indices: torch.Tensor, limit: int) -> list[int]:
    if indices.numel() == 0:
        return []
    return [int(v) for v in indices[:limit].detach().cpu().tolist()]


def _format_id_list(values: list, total: int) -> str:
    text = ", ".join(str(v) for v in values)
    if total > len(values):
        text += f", ... (+{total - len(values)} more)"
    return text


def _format_index_ranges(values: list[int]) -> str:
    if not values:
        return ""
    ranges: list[str] = []
    start = prev = int(values[0])
    for raw in values[1:]:
        i = int(raw)
        if i == prev + 1:
            prev = i
            continue
        ranges.append(f"[{start}:{prev + 1})")
        start = prev = i
    ranges.append(f"[{start}:{prev + 1})")
    return ", ".join(ranges)


def _label_obs_index(
    label_fn: Optional[Callable[[int, str], str]], index: int, group: str
) -> str:
    if label_fn is None:
        return str(index)
    try:
        text = label_fn(index, group)
    except Exception:
        return str(index)
    if not text:
        return str(index)
    return str(text)


def _iter_hosts(env: Any) -> Iterable[Any]:
    """Walk VecEnv / WarpEnv / Game wrappers without assuming a concrete type."""
    if env is None:
        return
    queue: list[Any] = [env]
    seen: set[int] = set()
    hops = 0
    while queue and hops < _MAX_HOST_HOPS:
        current = queue.pop(0)
        hops += 1
        if current is None:
            continue
        ident = id(current)
        if ident in seen:
            continue
        seen.add(ident)
        yield current
        for name in _HOST_ATTRS:
            nxt = getattr(current, name, None)
            if nxt is not None and id(nxt) not in seen:
                queue.append(nxt)


def _resolve_obs_label_fn(env: Any) -> Optional[Callable[[int, str], str]]:
    """TryGet ``try_describe_obs_index`` on any unwrapped host."""
    fn = None
    for host in _iter_hosts(env):
        candidate = getattr(host, "try_describe_obs_index", None)
        if callable(candidate):
            fn = candidate
            break
    if fn is None:
        return None

    def label(index: int, group: str) -> str:
        try:
            text = fn(int(index), group)
        except TypeError:
            try:
                text = fn(int(index))
            except Exception:
                return str(index)
        except Exception:
            return str(index)
        if text:
            return f"{index}:{text}"
        return str(index)

    return label


def _resolve_reward_term_env_values(env: Any) -> dict[str, torch.Tensor]:
    """TryGet per-env last-step reward-term buffers."""
    for host in _iter_hosts(env):
        fn = getattr(host, "try_get_reward_term_env_values", None)
        if callable(fn):
            try:
                values = fn()
            except Exception:
                values = None
            mapped = _as_named_tensors(values)
            if mapped:
                return mapped
        calc = getattr(host, "reward_calculator", None)
        if calc is None:
            continue
        fn = getattr(calc, "try_get_reward_term_env_values", None)
        if not callable(fn):
            continue
        try:
            values = fn()
        except Exception:
            continue
        mapped = _as_named_tensors(values)
        if mapped:
            return mapped
    return {}


def _as_named_tensors(values: Any) -> dict[str, torch.Tensor]:
    if not isinstance(values, dict):
        return {}
    out: dict[str, torch.Tensor] = {}
    for key, tensor in values.items():
        if torch.is_tensor(tensor) and tensor.ndim >= 1:
            out[str(key)] = tensor
    return out


def _first_critic_obs(env: Any) -> Any:
    for host in _iter_hosts(env):
        critic = getattr(host, "critic_obs", None)
        if torch.is_tensor(critic):
            return critic
    return None


def _obs_source_name(key: str) -> str:
    lowered = str(key).lower()
    if lowered in ("policy", "state", "actor"):
        return "obs"
    if lowered in ("critic", "obs.critic"):
        return "obs.critic"
    if lowered in ("obs", "observation"):
        return "obs"
    return f"obs.{key}"


def _report_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    label_fn: Optional[Callable[[int, str], str]] = None,
) -> list[str]:
    if not torch.is_tensor(tensor):
        return []
    if not tensor.is_floating_point() and not tensor.is_complex():
        return []
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    nan_count = int(nan_mask.sum().item())
    inf_count = int(inf_mask.sum().item())
    if nan_count == 0 and inf_count == 0:
        return []

    lines = [
        f"{name} shape={tuple(int(d) for d in tensor.shape)} "
        f"nan={nan_count} inf={inf_count}"
    ]
    bad = nan_mask | inf_mask
    use_obs_labels = name.startswith("obs") and label_fn is not None

    if tensor.ndim == 1:
        ids = bad.nonzero(as_tuple=False).flatten()
        total = int(ids.numel())
        lines.append(
            f"  env_ids ({total}): {_format_id_list(_as_int_list(ids, _MAX_IDS), total)}"
        )
        samples = ids[:_MAX_SAMPLES]
        for idx in samples.detach().cpu().tolist():
            value = tensor[int(idx)].detach()
            lines.append(f"  env={int(idx)} value={value.item()}")
        return lines

    if tensor.ndim == 2:
        env_ids = bad.any(dim=1).nonzero(as_tuple=False).flatten()
        feat_ids = bad.any(dim=0).nonzero(as_tuple=False).flatten()
        env_total = int(env_ids.numel())
        feat_total = int(feat_ids.numel())
        labeled_feats = [
            _label_obs_index(label_fn if use_obs_labels else None, int(i), name)
            for i in _as_int_list(feat_ids, _MAX_IDS)
        ]
        all_feat_ids = _as_int_list(feat_ids, feat_total)
        lines.append(
            f"  env_ids ({env_total}): {_format_id_list(_as_int_list(env_ids, _MAX_IDS), env_total)}"
        )
        feat_header = "feat_ids" if use_obs_labels else "col_ids"
        lines.append(
            f"  {feat_header} ({feat_total}): {_format_id_list(labeled_feats, feat_total)}"
        )
        if all_feat_ids:
            lines.append(f"  nan_col_ranges: {_format_index_ranges(all_feat_ids)}")
        coords = bad.nonzero(as_tuple=False)[:_MAX_SAMPLES]
        for row in coords.detach().cpu().tolist():
            env_i, feat_i = int(row[0]), int(row[1])
            value = tensor[env_i, feat_i].detach()
            feat_label = _label_obs_index(
                label_fn if use_obs_labels else None, feat_i, name
            )
            lines.append(f"  env={env_i} feat={feat_label} value={value.item()}")
        return lines

    coords = bad.nonzero(as_tuple=False)[:_MAX_SAMPLES]
    lines.append(f"  sample coords: {coords.detach().cpu().tolist()}")
    return lines


def _collect_obs_tensors(obs: Any) -> list[tuple[str, torch.Tensor]]:
    if obs is None:
        return []
    if hasattr(obs, "items"):
        out: list[tuple[str, torch.Tensor]] = []
        try:
            items = list(obs.items())
        except Exception:
            return []
        for key, tensor in items:
            if torch.is_tensor(tensor):
                out.append((_obs_source_name(str(key)), tensor))
        return out
    if torch.is_tensor(obs):
        return [("obs", obs)]
    return []


def collect_nan_report_lines(
    obs: Any,
    rewards: torch.Tensor,
    dones: torch.Tensor | None = None,
    *,
    env: Any = None,
    critic_obs: Any = None,
    reward_terms: dict[str, torch.Tensor] | None = None,
    include_reward_terms: bool = True,
) -> list[str]:
    """Return report body lines, or an empty list when every tensor is finite.

    ``include_reward_terms`` is for the guard path (diagnose which term wrote
    NaN). The abort path must only inspect tensors the trainer actually
    received: after a guard reset the leftover per-env term buffers are stale
    and must not raise.
    """
    label_fn = _resolve_obs_label_fn(env)
    if critic_obs is None:
        critic_obs = _first_critic_obs(env)
    if include_reward_terms and reward_terms is None:
        reward_terms = _resolve_reward_term_env_values(env)
    elif not include_reward_terms:
        reward_terms = {}

    lines: list[str] = []
    seen: set[int] = set()

    def append_tensor(name: str, tensor: Any) -> None:
        if not torch.is_tensor(tensor):
            return
        ident = id(tensor)
        if ident in seen:
            return
        chunk = _report_tensor(name, tensor, label_fn=label_fn)
        if not chunk:
            return
        seen.add(ident)
        lines.extend(chunk)

    for name, tensor in _collect_obs_tensors(obs):
        append_tensor(name, tensor)
    append_tensor("obs.critic", critic_obs)
    append_tensor("reward.total", rewards)
    for term_name, tensor in (reward_terms or {}).items():
        short = str(term_name)
        if short.startswith("Episode_Reward/"):
            short = short[len("Episode_Reward/") :]
        append_tensor(f"reward.{short}", tensor)
    if dones is not None:
        append_tensor("dones", dones)
    return lines


def format_env_nan_report(
    obs: Any,
    rewards: torch.Tensor,
    dones: torch.Tensor | None = None,
    *,
    env: Any = None,
    critic_obs: Any = None,
    reward_terms: dict[str, torch.Tensor] | None = None,
) -> Optional[str]:
    """Return a multi-line report, or ``None`` when outputs are finite."""
    lines = collect_nan_report_lines(
        obs,
        rewards,
        dones,
        env=env,
        critic_obs=critic_obs,
        reward_terms=reward_terms,
        include_reward_terms=False,
    )
    if not lines:
        return None
    return "[NaN-Report]\n" + "\n".join(lines)


def print_nan_guard_report(
    obs: Any,
    rewards: torch.Tensor,
    *,
    env: Any = None,
    critic_obs: Any = None,
    count: int | None = None,
    n_bad: int | None = None,
    action: str = "zeroing reward and resetting",
) -> Optional[str]:
    """Print the shared NaN layout on the guard path. Returns the text if printed."""
    lines = collect_nan_report_lines(
        obs,
        rewards,
        None,
        env=env,
        critic_obs=critic_obs,
        include_reward_terms=True,
    )
    if not lines:
        return None
    env_note = f"{n_bad} env(s) " if n_bad is not None else ""
    extras = [action]
    if count is not None:
        extras.append(f"count={count}")
    header = f"[NaN-Guard] {env_note}non-finite obs/reward; {'; '.join(extras)}"
    text = header + "\n" + "\n".join(lines)
    print(text, flush=True)
    return text


def check_nan_with_report(
    obs: Any,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    *,
    env: Any = None,
    critic_obs: Any = None,
) -> None:
    """Raise ``ValueError`` with env/feature ids when any output is NaN or Inf."""
    report = format_env_nan_report(
        obs, rewards, dones, env=env, critic_obs=critic_obs
    )
    if report is None:
        return
    print(report, flush=True)
    raise ValueError(
        report
        + "\nThis usually indicates a bug in the environment's step() or reset() function."
    )
