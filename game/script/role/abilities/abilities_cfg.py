from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, RootModel

TOOL_ATTACHMENT_ABILITY_NAME = "Tool_attachment"


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


class AimControlConfig(BaseModel):
    yaw_torque_gain: float = 200.0
    yaw_damping: float = 20.0
    max_yaw_torque: float = 500.0
    pitch_torque_gain: float = 150.0
    pitch_damping: float = 15.0
    max_pitch_torque: float = 300.0
    angle_dead_zone_deg: float = 0.5
    weld_yaw_drive_gain: float = 8.0
    aim_forward_local: List[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0])
    world_up: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])


class AbilityDetail(BaseModel):
    force: float
    speed: float
    cooldown: float
    key: KeyConfig
    action_space: ActionSpaceConfig


class ToolAttachmentDetail(AbilityDetail):
    proximity_threshold: Optional[float] = None
    proximity_height_threshold: Optional[float] = None
    aim_control: Optional[AimControlConfig] = None


def parse_ability_detail(name: str, raw: dict) -> AbilityDetail:
    if name == TOOL_ATTACHMENT_ABILITY_NAME:
        return ToolAttachmentDetail.model_validate(raw)
    return AbilityDetail.model_validate(raw)


def get_tool_attachment_detail(
    configs: Optional["AbilitiesConfig"],
) -> Optional[ToolAttachmentDetail]:
    if configs is None:
        return None
    cfg = configs.root.get(TOOL_ATTACHMENT_ABILITY_NAME)
    return cfg if isinstance(cfg, ToolAttachmentDetail) else None


# 使用 RootModel 來處理動態的鍵值 (能力名稱)
class AbilitiesConfig(RootModel):
    root: Dict[str, AbilityDetail]

    @classmethod
    def model_validate(cls, obj, /, **kwargs):
        if isinstance(obj, dict):
            raw_root = obj.get("root", obj)
            if isinstance(raw_root, dict):
                parsed = {
                    name: (
                        entry
                        if isinstance(entry, (AbilityDetail, ToolAttachmentDetail))
                        else parse_ability_detail(name, entry)
                    )
                    for name, entry in raw_root.items()
                }
                return super().model_validate(parsed, **kwargs)
        return super().model_validate(obj, **kwargs)
