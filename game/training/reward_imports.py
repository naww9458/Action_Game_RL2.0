"""Lazy-import reward modules by class name for registry population."""

import importlib
from functools import lru_cache
from pathlib import Path


REWARD_PACKAGE_CANDIDATES = (
    "script.levels.rewards",
    "levels.rewards",
)


@lru_cache(maxsize=1)
def import_reward_modules() -> tuple[str, ...]:
    """Import every reward Python module, including modules in subdirectories."""
    imported_modules = []
    import_errors = []

    for package_name in REWARD_PACKAGE_CANDIDATES:
        try:
            package = importlib.import_module(package_name)
        except ModuleNotFoundError as exc:
            import_errors.append(f"{package_name}: {exc}")
            continue

        for package_path in package.__path__:
            for module_path in sorted(Path(package_path).rglob("*.py")):
                relative_path = module_path.relative_to(package_path).with_suffix("")
                if (
                    relative_path.name == "__init__"
                    or relative_path.parts == ("reward_calculator",)
                    or any(part.startswith("_") for part in relative_path.parts)
                ):
                    continue

                module_name = f"{package_name}.{'.'.join(relative_path.parts)}"
                importlib.import_module(module_name)
                imported_modules.append(module_name)

        if imported_modules:
            return tuple(imported_modules)

    raise ModuleNotFoundError(
        "Unable to import reward modules from known packages: "
        + "; ".join(import_errors)
    )


def ensure_reward_registered(name: str) -> None:
    from script.levels.rewards.reward_calculator import RewardComponent

    if name in RewardComponent._registry:
        return
    import_reward_modules()
    if name not in RewardComponent._registry:
        raise KeyError(f"Unknown reward component: {name}. Available: {RewardComponent.get_registered_names()}")


def ensure_all_rewards_registered() -> None:
    import_reward_modules()
