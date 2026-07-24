import warp as wp
import newton

from pydantic import BaseModel, Field
from typing import Type, Dict, List, Any, Literal, Union, Annotated

class BaseSolverModel(BaseModel):
    type: Literal["base"] = "base"

class BaseSolver:
    solver_key = "BASE"
    model_cls: Type[BaseSolverModel] = BaseSolverModel
    solver_type_id: int = -1  # 用於 GPU kernel 的 ID

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        SolverRegistry.register(cls)

    def __init__(self, config: dict, **kwargs):
        self.config = config.copy()
        # 移除 "type" 鍵（如果不存在則返回 None，防止報錯）
        self.config.pop("type", None)
        self.solver = None

    def setup(self, model):
        raise NotImplementedError

    def step(self, state_in, state_out, control, contacts, dt):
        raise NotImplementedError

    def post_teleport_sync(self, state):
        """當外部 Kernel 強行修改了 state 的位置/旋轉後，調用此方法同步求解器內部狀態"""
        pass

    def reset_history(self):
        """當重置環境時，調用此方法清除歷史快取"""
        pass


    @property
    def body_q_prev(self):
        """默認返回 None，代表不需要額外同步（如 XPBD）"""
        return None


class SolverRegistry:
    _registry: Dict[str, Type['BaseSolver']] = {}

    @classmethod
    def register(cls, solver_cls: Type['BaseSolver']):
        if solver_cls.solver_key and solver_cls.solver_key != "BASE":
            cls._registry[solver_cls.solver_key] = solver_cls
            # print(f"[*] 求解器註冊成功: {solver_cls.solver_key}")

    @classmethod
    def get_solver_union(cls):
        """
        核心邏輯：從註冊表中提取所有 model_cls，動態生成 Union 類型
        """
        models = [handler.model_cls for handler in cls._registry.values()]
        if not models:
            # 防止註冊表為空時報錯，至少給一個基礎模型
            return BaseSolverModel
        
        # 動態構建 Annotated[Union[M1, M2, ...], Field(discriminator="type")]
        return Annotated[Union[tuple(models)], Field(discriminator="type")]

    @classmethod
    def get_handler(cls, key: str):
        return cls._registry.get(key)

    @classmethod
    def get_all_keys(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def get_all_models(cls) -> List[str]:
        models = [handler.model_cls for handler in cls._registry.values()]
        return models


