import importlib
from pathlib import Path

from script.environments.environment import DefaultEnvironment, Environment
from script.environments.environment_cfg import EnvironmentDefinition
from script.game_config import GameConfig

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from script.game import Game

from typing import Optional, Type


def _import_class(class_path: str) -> Type:
    module_name, _, class_name = class_path.replace(":", ".").rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"Invalid environment_class path: {class_path}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _environment_class_path(config_obj: EnvironmentDefinition) -> Optional[str]:
    return getattr(config_obj, "environment_class", None) or None


def _try_load_sibling_class(config_path, config_obj: EnvironmentDefinition) -> Optional[Type]:
    """Load ``<yaml-stem>.py`` next to the env YAML when it exists."""
    from script.environments.env_catalog import env_script_module_name

    yaml_path = EnvironmentDefinition.resolve_path(config_path)
    py_path = yaml_path.with_suffix(".py")
    if not py_path.is_file():
        return None
    class_path = _environment_class_path(config_obj)
    class_name = None
    if class_path:
        class_name = str(class_path).replace(":", ".").rsplit(".", 1)[-1]
    if not class_name:
        return None
    module_name = env_script_module_name(py_path)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or (exc.name and module_name.startswith(str(exc.name))):
            return None
        raise
    return getattr(module, class_name)


def _is_missing_target_module(exc: ModuleNotFoundError, module_name: str) -> bool:
    """True when ``import_module(module_name)`` failed because that module is absent."""
    missing = str(exc.name or "")
    if not missing:
        return True
    if missing == module_name or module_name.startswith(missing + ".") or missing.startswith(module_name + "."):
        return True
    return missing in module_name.split(".")


def _resolve_environment_class(
    config_obj: EnvironmentDefinition,
    config_path=None,
) -> Type:
    class_path = _environment_class_path(config_obj)
    if class_path:
        try:
            return _import_class(class_path)
        except ModuleNotFoundError as exc:
            module_name, _, _ = str(class_path).replace(":", ".").rpartition(".")
            if not _is_missing_target_module(exc, module_name):
                raise
        except AttributeError:
            pass

    if config_path is not None:
        try:
            sibling = _try_load_sibling_class(config_path, config_obj)
            if sibling is not None:
                return sibling
        except AttributeError:
            pass

    print("No environment_class in environment YAML; using DefaultEnvironment")
    return DefaultEnvironment


def get_environment(
    env_id: str = None,
    game: 'Game' = None,
    environment_config_path=None,
    player_controllers: list[str] | None = None,
    control_policy_version: str | None = None,
    **runtime_overrides
) -> Environment:
    """Load and instantiate an environment from catalog path or ``env_id`` (YAML stem)."""

    config_path = environment_config_path

    if config_path:
        resolved_path = config_path
    else:
        if not env_id:
            raise ValueError(
                "get_environment requires environment_config_path or env_id."
            )
        from script.environments.env_catalog import resolve_env_path_by_id

        resolved_path = resolve_env_path_by_id(str(env_id))
        if resolved_path is None:
            raise FileNotFoundError(
                f"No environment catalog entry for env_id={env_id}"
            )

    from script.environments.env_catalog import prefer_custom_env_path

    resolved_path = prefer_custom_env_path(Path(resolved_path))

    print("config_path: ", resolved_path)
    try:
        config_obj = EnvironmentDefinition.load(
            resolved_path,
            overrides=runtime_overrides,
            player_controllers=player_controllers,
            control_policy_version=control_policy_version,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

    env_cls = _resolve_environment_class(config_obj, resolved_path)

    env_data = config_obj.environment_configs.model_dump()
    try:
        GameConfig.init_from_configs(env_data)
    except AttributeError:
        print("Warning: GameConfig attributes are already initialized and immutable.")

    if game and hasattr(game, 'physics_manager'):
        game.physics_manager.set_env_params(
            tuple(config_obj.environment_configs.gravity),
            config_obj.environment_configs.damping
        )

    return env_cls(
        game=game,
        config=config_obj.model_dump(),
    )
