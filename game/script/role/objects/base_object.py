from pydantic import BaseModel, Field
from typing import Type, Dict, List, Any, Literal, Union, Annotated

# --- 1. 每個形狀對應的 Pydantic 模型 ---
class BaseObjectModel(BaseModel):
    type: Literal["base"] = "base"
    pattern: str = "default"
    object_friction: float = 0.5
    object_elasticity: float = 0.5

# --- 2. 形狀邏輯父類 ---
class BaseObject:
    object_key = "BASE"
    model_cls: Type[BaseObjectModel] = BaseObjectModel
    object_type_id: int = -1  # 用於 GPU kernel 的 ID

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ObjectRegistry.register(cls)

    @staticmethod
    def add_physics(builder_env, label, data, **kwargs):
        """實作 Newton/Warp 的物理形狀添加"""
        raise NotImplementedError

    @staticmethod
    def add_visual(mesh_builder, data: Any, pos):
        """實作渲染用的頂點添加"""
        raise NotImplementedError

    @staticmethod
    def get_size(data: Any) -> List[float]:
        """返回標準化的 XYZ 尺寸供 GPU 使用"""
        raise NotImplementedError


class ObjectRegistry:
    _registry: Dict[str, Type['BaseObject']] = {}

    @classmethod
    def register(cls, object_cls: Type['BaseObject']):
        if object_cls.object_key and object_cls.object_key != "BASE":
            cls._registry[object_cls.object_key] = object_cls
            # print(f"[*] 物理形狀註冊成功: {object_cls.object_key}")

    @classmethod
    def get_object_union(cls):
        """
        核心邏輯：從註冊表中提取所有 model_cls，動態生成 Union 類型
        """
        models = [handler.model_cls for handler in cls._registry.values()]
        if not models:
            # 防止註冊表為空時報錯，至少給一個基礎模型
            return BaseObjectModel
        
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


