from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


from training.runtime_env import ensure_runtime_env, get_project_root, resolve_tensorboard_command

DEVICE = "cuda:0"


def _ensure_game_path():
    ensure_runtime_env()


def _default_project_root() -> Path:
    return get_project_root()


def cmd_train(args: argparse.Namespace) -> int:
    _ensure_game_path()
    from training.loader import TrainingPresetLoader

    loaded = TrainingPresetLoader.load(args.preset)
    num_envs = args.num_envs if args.num_envs is not None else loaded.train_cfg.num_envs_default
    Trainer = loaded.Trainer
    trainer = Trainer(
        device=DEVICE, # TODO Hardcode
        num_envs=num_envs,
        is_training=True,
        loaded_config=loaded,
        enable_window=args.enable_window,
        window_num_envs=args.window_envs if args.enable_window else None,
        checkpoint_path=args.resume,
        preset_path=str(loaded.preset_path) if loaded.preset_path else None,
    )
    if args.mode == "custom":
        trainer.train_custom()
    else:
        trainer.train()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    _ensure_game_path()
    from training.loader import TrainingPresetLoader
    from training.runs_manager import RunsManager

    project_root = _default_project_root()
    runs = RunsManager(project_root=project_root)
    checkpoint_path = runs.checkpoint_path(args.run, args.checkpoint)
    run_info = runs.get_run_info(args.run)

    loaded = None
    if args.preset:
        loaded = TrainingPresetLoader.load(args.preset)
    else:
        run_dir = runs.resolve_run_dir(args.run)
        try:
            loaded = TrainingPresetLoader.load_from_run_config(run_dir)
        except FileNotFoundError:
            loaded = None

    if loaded is None:
        from skrl_script.trainer_base import Trainer_base
        try:
            _, _, _, loaded = Trainer_base().load_config_from_checkpoint(str(checkpoint_path))
        except Exception:
            loaded = None

    if loaded is None:
        raise RuntimeError(
            f"Could not resolve training config for run '{args.run}'. "
            "Ensure config/preset.yaml or config/model_cfg.pkl exists."
        )

    if run_info and run_info.algorithm.upper() == "APG":
        trainer_module = getattr(loaded.Trainer, "__module__", "")
        if "trainer_APG" not in trainer_module:
            from skrl_script.trainer_base import Trainer_base
            _, _, _, loaded = Trainer_base().load_config_from_checkpoint(str(checkpoint_path))

    Trainer = loaded.Trainer
    print(f"Eval trainer: {Trainer.__module__}.{Trainer.__name__} (algorithm={loaded.meta.algorithm})")
    trainer = Trainer(
        device=DEVICE,
        num_envs=args.num_envs,
        is_training=False,
        loaded_config=loaded,
        enable_window=args.enable_window,
        window_num_envs=args.window_envs if args.enable_window else None,
        checkpoint_path=str(checkpoint_path),
    )

    trainer.evaluate_custom(args.episodes)
    return 0


def cmd_tensorboard(args: argparse.Namespace) -> int:
    _ensure_game_path()
    from training.runs_manager import RunsManager

    runs = RunsManager(project_root=_default_project_root())
    proc = runs.launch_tensorboard(args.run, port=args.port)
    print(f"TensorBoard started (PID {proc.pid}) at http://localhost:{args.port}")
    if proc.stdout is not None:
        for line in proc.stdout:
            print(line, end="")
    return proc.wait()


def cmd_list_runs(args: argparse.Namespace) -> int:
    _ensure_game_path()
    from training.runs_manager import RunsManager

    runs = RunsManager(project_root=_default_project_root())
    for run in runs.list_runs():
        ckpts = ", ".join(c.name for c in run.checkpoints[:3])
        print(f"{run.name} | preset={run.preset_id} | checkpoints=[{ckpts}]")
    return 0


def cmd_list_presets(args: argparse.Namespace) -> int:
    _ensure_game_path()
    from training.registry import TrainingPresetRegistry

    for preset in TrainingPresetRegistry.list_presets():
        line = f"{preset['id']} | level={preset['level']}_{preset['sub_level']} | {preset['display_name']}"
        print(line.encode("utf-8", errors="replace").decode("utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL training launcher")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Start training from a preset")
    train_p.add_argument("--preset", required=True, help="Preset id or yaml path")
    train_p.add_argument("--num-envs", type=int, default=None)
    train_p.add_argument("--enable-window", action="store_true")
    train_p.add_argument("--window-envs", type=int, default=1, help="Environments to display when window is enabled")
    train_p.add_argument("--resume", default=None, help="Checkpoint .pt path to resume from")
    train_p.add_argument("--mode", choices=["sequential", "custom"], default="custom")
    train_p.set_defaults(func=cmd_train)

    eval_p = sub.add_parser("eval", help="Evaluate a checkpoint")
    eval_p.add_argument("--run", required=True, help="Run folder name or path")
    eval_p.add_argument("--checkpoint", default="best_agent", help="Checkpoint name without .pt")
    eval_p.add_argument("--preset", default=None, help="Optional preset override")
    eval_p.add_argument("--num-envs", type=int, default=2)
    eval_p.add_argument("--episodes", type=int, default=50)
    eval_p.add_argument("--enable-window", action="store_true")
    eval_p.add_argument("--window-envs", type=int, default=1, help="Environments to display when window is enabled")
    eval_p.add_argument("--level", type=int, default=5)
    eval_p.add_argument("--sub-level", type=int, default=0)
    eval_p.add_argument("--obs-type", default="state_based")
    eval_p.set_defaults(func=cmd_eval)

    tb_p = sub.add_parser("tensorboard", help="Launch TensorBoard for a run")
    tb_p.add_argument("--run", required=True)
    tb_p.add_argument("--port", type=int, default=6006)
    tb_p.set_defaults(func=cmd_tensorboard)

    runs_p = sub.add_parser("list-runs", help="List experiment runs")
    runs_p.set_defaults(func=cmd_list_runs)

    presets_p = sub.add_parser("list-presets", help="List training presets")
    presets_p.set_defaults(func=cmd_list_presets)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
