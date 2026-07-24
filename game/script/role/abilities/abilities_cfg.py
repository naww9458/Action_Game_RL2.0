from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field, RootModel

class ActionSpaceConfig(BaseModel):
    type: str
    dtype: str
    shape: Union[int, List[int]] | str
    n: Optional[int] = None
    range: Optional[List[float]] = None
    description: Optional[str] = None

class KeyConfig(BaseModel):
    keyboard: Dict[str, List[str]] = Field(default_factory=dict)
    mouse: Dict[str, List[str]] = Field(default_factory=dict)

class AbilityDetail(BaseModel):
    force: float
    speed: float
    cooldown: float
    key: KeyConfig
    action_space: ActionSpaceConfig

# 使用 RootModel 來處理動態的鍵值 (能力名稱)
class AbilitiesConfig(RootModel):
    root: Dict[str, AbilityDetail]