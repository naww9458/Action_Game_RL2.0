"""Host-side per-step training-log collector.

Non-differentiable: this module only snapshots tensors for TensorBoard /
iteration summaries. It must not run inside a Warp CUDA graph or a
differentiation tape.

``Game`` owns the simulation step and calls the hooks at fixed points;
RL frameworks (RSL-rl, skrl, ...) consume the resulting dict via
``Game.get_training_log()``. Level-specific extras (command metrics,
curriculum ranges, log-key order) are discovered with getattr so this
collector stays unaware of any particular robot or reward term.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import warp as wp

KEY_TIME_OUT = "Episode_Termination/time_out"
KEY_FELL_OVER = "Episode_Termination/fell_over"
KEY_ACTION_ACC = "Episode_Metrics/mean_action_acc"


class TrainingLogCollector:
    """Accumulate per-step extras that must be captured before ``Game.reset()``.

    Two pieces of state live here because they would be lost if read after
    reset / the next action write:

    * action-acceleration history (needs ``a_t, a_{t-1}, a_{t-2}``)
    * termination / truncation counts (``terminated`` / ``truncated`` are
      zeroed in ``Game.reset()``)
    """

    def __init__(self, *, action_dim: int) -> None:
        self.action_dim = int(action_dim)
        self._prev_actions: torch.Tensor | None = None
        self._prev_prev_actions: torch.Tensor | None = None
        self._action_acc_mean: torch.Tensor | None = None
        self._time_out_count: torch.Tensor | None = None
        self._fell_over_count: torch.Tensor | None = None

    def on_actions(self, actions: torch.Tensor | None) -> None:
        """Record action acceleration. Call at the start of each env step.

        ``acc = a_t - 2*a_{t-1} + a_{t-2}`` over the RL action dimensions;
        the stored value is per-env ``mean(|acc|, dim=-1)`` so a framework
        logger can aggregate across envs. The first two steps (missing
        history) report zeros. ``actions is None`` clears the metric.
        """
        if actions is None:
            self._action_acc_mean = None
            return
        act = actions.detach()[:, : self.action_dim].clone()
        if self._prev_actions is not None and self._prev_prev_actions is not None:
            acc = act - 2.0 * self._prev_actions + self._prev_prev_actions
            self._action_acc_mean = torch.mean(torch.abs(acc), dim=-1)
        else:
            self._action_acc_mean = torch.zeros(
                act.shape[0], device=act.device, dtype=act.dtype
            )
        self._prev_prev_actions = self._prev_actions
        self._prev_actions = act

    def on_rewards_done(self, terminated: wp.array, truncated: wp.array) -> None:
        """Snapshot timeout vs. non-timeout termination counts.

        Call after ``calculate_rewards`` and before ``reset()`` zeroes the
        flags. ``time_out`` counts truncated envs; ``fell_over`` counts
        terminated-but-not-truncated envs (the conventional names used by
        this project's training logs).
        """
        terminated_t = wp.to_torch(terminated)
        truncated_t = wp.to_torch(truncated)
        self._time_out_count = torch.count_nonzero(truncated_t).float()
        self._fell_over_count = torch.count_nonzero(terminated_t & ~truncated_t).float()

    def collect(self, *, reward_calculator: Any = None, environment: Any = None) -> dict:
        """Build the unified per-step log dict of GPU tensors.

        Aggregates reward terms, physics metrics, optional environment extras,
        and the snapshots from :meth:`on_actions` / :meth:`on_rewards_done`.
        Missing sources are skipped (no dummy keys).
        """
        log: dict = {}
        if reward_calculator is not None:
            log.update(reward_calculator.get_reward_terms())
            log.update(reward_calculator.get_metric_means())
        self._merge_optional(log, environment, "get_command_metrics")
        self._merge_optional(log, environment, "get_curriculum_metrics")
        if self._time_out_count is not None:
            log[KEY_TIME_OUT] = self._time_out_count
        if self._fell_over_count is not None:
            log[KEY_FELL_OVER] = self._fell_over_count
        if self._action_acc_mean is not None:
            log[KEY_ACTION_ACC] = self._action_acc_mean
        return self._reorder(log, environment)

    def diagnostics(self, reward_calculator: Any = None) -> list:
        """Forward reward-term diagnostics, or an empty list when absent."""
        if reward_calculator is None:
            return []
        return reward_calculator.get_reward_term_diagnostics()

    @staticmethod
    def _merge_optional(log: dict, environment: Any, method_name: str) -> None:
        getter = getattr(environment, method_name, None) if environment is not None else None
        if getter is None:
            return
        extra = getter()
        if extra:
            log.update(extra)

    @staticmethod
    def _reorder(log: dict, environment: Any) -> dict:
        order_getter = getattr(environment, "get_log_key_order", None) if environment is not None else None
        order: Optional[tuple[str, ...]] = order_getter() if order_getter is not None else None
        if not order:
            return log
        ordered = {k: log[k] for k in order if k in log}
        ordered.update({k: v for k, v in log.items() if k not in ordered})
        return ordered
