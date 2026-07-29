import os
import glob
import importlib
from typing import Optional

from .ability import Ability

# 1. 獲取所有模組名稱 (保持不變)
module_names = [
    os.path.basename(f)[:-3]
    for f in glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
    if not f.endswith("__init__.py") and not f.endswith("ability.py")
]

# 2. 單純導入模組，這會觸發子類的 __init_subclass__ 進行註冊
# 此時不會執行 Ability() 的 __init__，所以不會報 GameConfig 的錯誤
for module_name in module_names:
    importlib.import_module(f".{module_name}", __name__)

# 3. 建立一個內部快取字典
# Key: ability class name, or "ClassName@scope" for pattern-scoped abilities.
_ABILITY_CACHE = {}


def _cache_key(name: str, share_key: Optional[str] = None) -> str:
    if share_key:
        return f"{name}@{share_key}"
    return name


def get_shared_ability(name: str, share_key: Optional[str] = None) -> Ability:
    """
    獲取共享能力物件。

    - 無 ``share_key``：全進程單例（Shoot、Jump 等）。
    - 有 ``share_key``：同一 scope 共用一個實例（例如每種關節體一個
      ``Articulation_body_control``）。
    """
    global _ABILITY_CACHE

    key = _cache_key(name, share_key)
    if key in _ABILITY_CACHE:
        return _ABILITY_CACHE[key]

    if name not in Ability._registry:
        raise KeyError(
            f"Ability '{name}' not found in registry. Check class name or module import."
        )

    try:
        instance = Ability._registry[name]()
        instance._ability_share_key = share_key
        _ABILITY_CACHE[key] = instance
        scope_msg = f" scope={share_key}" if share_key else ""
        print(
            f"\033[38;5;34m [Lazy Load] Shared ability created: {name}{scope_msg} \033[0m"
        )
        return instance
    except AttributeError as e:
        print(
            f"\033[31m Error instantiating {name}: {e}. "
            f"Make sure GameConfig is initialized before creating Roles. \033[0m"
        )
        raise e


# 為了兼容性，保留一個空字典或包裝器，但建議統一使用 get_shared_ability
__all__ = list(Ability._registry.keys()) + ["get_shared_ability"]
