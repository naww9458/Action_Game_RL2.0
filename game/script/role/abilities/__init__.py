import os
import glob
import importlib
from .ability import Ability

# 1. 獲取所有模組名稱 (保持不變)
module_names = [
    os.path.basename(f)[:-3] 
    for f in glob.glob(os.path.join(os.path.dirname(__file__), "*.py")) 
    if not f.endswith('__init__.py') and not f.endswith('ability.py')
]

# 2. 單純導入模組，這會觸發子類的 __init_subclass__ 進行註冊
# 此時不會執行 Ability() 的 __init__，所以不會報 GameConfig 的錯誤
for module_name in module_names:
    importlib.import_module(f".{module_name}", __name__)

# 3. 建立一個內部快取字典
_ABILITY_CACHE = {}

def get_shared_ability(name: str) -> Ability:
    """
    獲取單例能力物件。
    如果是第一次獲取，則實例化它；之後則回傳同一個指標。
    """
    global _ABILITY_CACHE
    
    # 如果已經實例化過，直接回傳指標
    if name in _ABILITY_CACHE:
        return _ABILITY_CACHE[name]
    
    # 如果還沒實例化，檢查註冊表裡有沒有這個類別
    if name in Ability._registry:
        # 在這裡才真正呼叫 cls()。
        # 只要你在遊戲啟動、角色建立時才呼叫此函數，GameConfig 絕對已經準備好了。
        try:
            instance = Ability._registry[name]()
            _ABILITY_CACHE[name] = instance
            print(f"\033[38;5;34m [Lazy Load] Singleton instance created: {name} \033[0m")
            return instance
        except AttributeError as e:
            # 輔助除錯：如果還是噴 Config 錯誤，列印出提示
            print(f"\033[31m Error instantiating {name}: {e}. Make sure GameConfig is initialized before creating Roles. \033[0m")
            raise e
            
    raise KeyError(f"Ability '{name}' not found in registry. Check class name or module import.")


# 為了兼容性，保留一個空字典或包裝器，但建議統一使用 get_shared_ability
__all__ = list(Ability._registry.keys())