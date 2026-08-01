from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel

TOOL_ATTACHMENT_ABILITY_NAME = "Tool_attachment"
SHOOT_ABILITY_NAME = "Shoot"


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
    model_config = ConfigDict(extra="allow")

    force: float
    speed: float
    cooldown: float
    key: KeyConfig
    action_space: ActionSpaceConfig


class ToolAttachmentDetail(AbilityDetail):
    proximity_threshold: Optional[float] = None
    proximity_height_threshold: Optional[float] = None


class ShootDetail(AbilityDetail):
    # 射擊能力專用參數，與 object_template/*/control_configs.yaml 的 shoot 區段同名，
    # 可在環境配置 (level_*_default_cfg.yaml) 中按環境覆寫。
    forward_force_n: Optional[float] = None
    recoil_force_n: Optional[float] = None
    projectile_generation_point_offset: Optional[List[float]] = None
    cooldown_s: Optional[float] = None
    projectile_mass_kg: Optional[float] = None
    projectile_radius_m: Optional[float] = None


def parse_ability_detail(name: str, raw: dict) -> AbilityDetail:
    if name == TOOL_ATTACHMENT_ABILITY_NAME:
        return ToolAttachmentDetail.model_validate(raw)
    if name == SHOOT_ABILITY_NAME:
        return ShootDetail.model_validate(raw)
    return AbilityDetail.model_validate(raw)


def get_tool_attachment_detail(
    configs: Optional["AbilitiesConfig"],
) -> Optional[ToolAttachmentDetail]:
    if configs is None:
        return None
    cfg = configs.root.get(TOOL_ATTACHMENT_ABILITY_NAME)
    return cfg if isinstance(cfg, ToolAttachmentDetail) else None


# 使用 RootModel 來處理動態的鍵值 (能力名稱)
AbilityDetailT = Union[AbilityDetail, ToolAttachmentDetail, ShootDetail]


class AbilitiesConfig(RootModel):
    root: Dict[str, AbilityDetailT]

    @classmethod
    def model_validate(cls, obj, /, **kwargs):
        if isinstance(obj, dict):
            raw_root = obj.get("root", obj)
            if isinstance(raw_root, dict):
                parsed = {
                    name: (
                        entry
                        if isinstance(entry, AbilityDetailT)
                        else parse_ability_detail(name, entry)
                    )
                    for name, entry in raw_root.items()
                }
                return super().model_validate(parsed, **kwargs)
        return super().model_validate(obj, **kwargs)
