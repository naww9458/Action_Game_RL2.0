"""Training configuration and experiment management."""

__all__ = [
    "TrainingPresetRegistry",
    "TrainingPresetLoader",
    "LoadedTrainingConfig",
    "RunsManager",
]

def __getattr__(name: str):
    if name == "TrainingPresetRegistry":
        from training.registry import TrainingPresetRegistry
        return TrainingPresetRegistry
    if name in ("TrainingPresetLoader", "LoadedTrainingConfig"):
        from training.loader import TrainingPresetLoader, LoadedTrainingConfig
        return TrainingPresetLoader if name == "TrainingPresetLoader" else LoadedTrainingConfig
    if name == "RunsManager":
        from training.runs_manager import RunsManager
        return RunsManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
