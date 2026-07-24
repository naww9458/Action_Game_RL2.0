import os
import importlib
import pkgutil

# 先導入基底類，確保 Registry 存在
from .base_solver import SolverRegistry, BaseSolver

# 自動遍歷當前資料夾下的所有模組並導入
def load_all_solvers():
    pkg_dir = os.path.dirname(__file__)
    for _, module_name, is_pkg in pkgutil.iter_modules([pkg_dir]):
        if module_name != "base_solver":  # 避開基底類文件
            # 動態導入模組，這會觸發類別定義，從而執行 __init_subclass__
            importlib.import_module(f".{module_name}", package=__name__)

# 執行加載
load_all_solvers()

