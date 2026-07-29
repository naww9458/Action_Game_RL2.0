"""Attachable tool role (turrets, weapons, etc.)."""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Union

from pydantic import Field

from script.role.base_role import BaseRole, BaseRoleModel

DEFAULT_TOOL_ABILITIES = ["Shoot", "Articulation_body_control"]


class ToolModel(BaseRoleModel):
    type: Literal["tool"] = "tool"
    name: str = "Tool"
    # These defaults are resolved per tool pattern via object_template/*/template.yaml.
    # Keeping them optional allows per-level YAML to omit them.
    mount_anchor_name: Optional[str] = None
    host_anchor_name: Optional[Union[str, List[Optional[str]]]] = None
    host_body_prim_suffix: Optional[Union[str, List[Optional[str]]]] = None
    tool_base_body_prim_suffix: Optional[str] = None
    mount_joint_type: Literal["revolute", "fixed", "ball", "prismatic"] = "revolute"
    mount_joint_axis: List[float] = [0.0, 0.0, 1.0]
    mount_joint_limits: List[float] = Field(default_factory=lambda: [-math.pi, math.pi])
    internal_joint_names: List[str] = []

    # TODO 很可能需要改進名稱，目前的命名像是在表明必須要有俯仰角（有俯仰角對炮塔來説合理，可是車頂關節并非只能支援炮塔）
    # 雖然這個設置代表的是炮塔但結構應該優化比如加一個新屬性 "功能"，然後將 pitch_joint_name 放到 "功能" 裏面
    pitch_joint_name: Optional[str] = None
    # Which host player object(s) to mount onto.
    # - int: use a specific player index
    # - list: try multiple player indices until one matches
    # - None: try all players until one matches
    host_player_index: Optional[Union[int, List[Optional[int]]]] = None
    proximity_threshold: float = 0.75
    proximity_height_threshold: float = 3.5
    aim_body_prim_suffix: Optional[str] = None
    aim_control: Optional[Dict[str, object]] = None
    abilities: List[str] = Field(default_factory=lambda: list(DEFAULT_TOOL_ABILITIES))


class Tool(BaseRole):
    role_key = "tool"
    model_cls = ToolModel
    path = "tool_configs"
    container = "dict"

    def __init__(self, configs: Optional[Dict[str, dict]] = None, **kwargs):
        super().__init__(**kwargs)
        config_list: List[dict] = []
        self.tool_config_keys: List[str] = []

        if configs:
            for key, config in configs.items():
                cfg = dict(config)
                if not cfg.get("abilities"):
                    cfg["abilities"] = list(DEFAULT_TOOL_ABILITIES)
                config_list.append(cfg)
                self.tool_config_keys.append(key)

        self.setup(configs=config_list)
