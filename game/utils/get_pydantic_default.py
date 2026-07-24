import inspect
from typing import Any, Type
from pydantic import BaseModel


def get_pydantic_default(model: Type[BaseModel]) -> Any:
    """遞歸解析 Pydantic 模型並生成預設值字典"""
    if not inspect.isclass(model) or not issubclass(model, BaseModel):
        return None

    res = {}
    for name, field in model.model_fields.items():
        # ``field.default`` is PydanticUndefined for fields declared with a
        # default_factory.  Store the factory result instead; serializing the
        # sentinel later causes PyYAML to fail when a new object is saved.
        if field.default_factory is not None:
            default_value = field.get_default(call_default_factory=True)
            if isinstance(default_value, BaseModel):
                default_value = default_value.model_dump()
            res[name] = default_value
            continue

        # 1. 如果有定義 default (非 None)，優先使用
        if field.default is not None and field.default != ...:
            if isinstance(field.default, BaseModel):
                res[name] = field.default.model_dump()
            else:
                res[name] = field.default
            continue

        # 2. 如果是嵌套模型
        anno = field.annotation
        # 處理 Annotated 或 Union
        if hasattr(anno, "__metadata__"): # Annotated
            anno = anno.__args__[0]
        
        if inspect.isclass(anno) and issubclass(anno, BaseModel):
            res[name] = get_pydantic_default(anno)
        elif "Union" in str(anno): # 處理 Union，取第一個
            first_type = anno.__args__[0]
            res[name] = get_pydantic_default(first_type)
        else:
            res[name] = None
    return res