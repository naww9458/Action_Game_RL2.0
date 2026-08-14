"""RSL-rl OnPolicyRunner subclass adapted to the project's run tooling.

The stock ``OnPolicyRunner`` saves checkpoints as ``model_<it>.pt`` directly
inside its log directory. The project's ``training.runs_manager.RunsManager``
(used by ``rl_launcher.py list-runs / eval / tensorboard``) expects runs under
``runs/`` with checkpoints mirrored into ``<run>/checkpoints/``. This subclass
mirrors every saved checkpoint there so RSL-rl runs integrate with the existing
launcher tooling.
"""

from __future__ import annotations

import os
import shutil

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class RslRlOnPolicyRunner(OnPolicyRunner):
    """``OnPolicyRunner`` that additionally mirrors checkpoints into ``checkpoints/``."""

    def _resolve_log_dir(self) -> str | None:
        """Return the run log directory for the installed rsl_rl API variant.

        Newer rsl_rl exposes ``self.logger.log_dir``; older builds used
        ``self.log_dir`` directly. Prefer the logger path when present.
        """
        logger = getattr(self, "logger", None)
        if logger is not None:
            log_dir = getattr(logger, "log_dir", None)
            if log_dir is not None:
                return log_dir
        return getattr(self, "log_dir", None)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop, emitting reward diagnostics after each iteration log.

        The project's training environment (``.venv``) ships a newer rsl_rl whose
        ``OnPolicyRunner`` prints through ``self.logger.log(...)`` (a ``Logger``
        instance). Older rsl_rl builds printed through ``self.log(locals())``
        instead. We wrap whichever method exists so that after every iteration's
        log line is printed we emit the automatic reward-term diagnostics from
        the environment (``[Reward-Diagnostics]``); always-zero / abnormal terms
        are then surfaced right below the mjlab-style ``Episode_Reward/*`` lines.

        The ep-string line order is already controlled by the per-step
        ``extras["log"]`` (``Game.get_training_log`` reorders it against the
        level's key template), which rsl_rl preserves in ``ep_extras``. No extra
        keys are appended here so the printed section stays aligned with the
        mjlab reference line-by-line.
        """
        if not getattr(self, "_training_log_hooks_installed", False):
            # Locate the iteration-log method for the installed rsl_rl API.
            logger = getattr(self, "logger", None)
            log_method = getattr(logger, "log", None) if logger is not None else None
            if log_method is None:
                log_method = getattr(self, "log", None)

            if log_method is not None:
                def log_with_reward_diagnostics(*args, **kwargs):
                    log_method(*args, **kwargs)
                    diagnostics = self.env.get_training_diagnostics()
                    if diagnostics:
                        print("[Reward-Diagnostics]")
                        for line in diagnostics:
                            print(f"  {line}")

                # Replace the method on the object that owns it.
                if logger is not None and hasattr(logger, "log"):
                    logger.log = log_with_reward_diagnostics
                else:
                    self.log = log_with_reward_diagnostics

            self._training_log_hooks_installed = True

        super().learn(num_learning_iterations, init_at_random_ep_len)

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the checkpoint and mirror it into ``<log_dir>/checkpoints/``."""
        super().save(path, infos)

        log_dir = self._resolve_log_dir()
        if log_dir is None:
            return
        checkpoints_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(checkpoints_dir, os.path.basename(path)))
