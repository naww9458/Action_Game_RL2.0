import os
import importlib
import pkgutil

# 先導入基底類，確保 Registry 存在
from .base_object import ObjectRegistry, BaseObject

# 自動遍歷當前資料夾下的所有模組並導入
def load_all_objects():
    pkg_dir = os.path.dirname(__file__)
    for _, module_name, is_pkg in pkgutil.iter_modules([pkg_dir]):
        if module_name in ("base_object", "object_template"):
            continue
        importlib.import_module(f".{module_name}", package=__name__)

# 執行加載
load_all_objects()