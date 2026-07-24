
import copy

from typing import List, Optional, Literal
from script.role.base_role import BaseRole, BaseRoleModel

class EntityModel(BaseRoleModel):
    type: Literal["entity"] = "entity"
    name: str = "New_Entity"
    separation: Optional[List[int]] = [1, 1, 1] # XYZ 平均分割為多少份


class Entity(BaseRole):
    role_key = "entity"
    model_cls = EntityModel
    path = "entity_configs"
    container = "dict"


    def __init__(self, configs: dict, **kwargs):
        super().__init__(**kwargs)

        config_list = []
        # 設定一個微小的間隙，防止物理重疊擠壓
        gap = 0.00001 

        if configs:
            for config in configs.values():
                separation = config["separation"]
                if separation == [1, 1, 1]:
                    config_list.append(config)
                    continue

                size_total = config["object"]["size"]
                center_total = config["default_position"]
                total_mass = config["object"]["object_mass"]

                # 計算每個小塊的大小（扣除間隙後的理論大小）
                size_each_part = [
                    size_total[0] / separation[0],
                    size_total[1] / separation[1],
                    size_total[2] / separation[2]
                ]
                
                # 為了讓物理更穩定，實際渲染/碰撞的尺寸可以稍微縮小一點點
                actual_part_size = [
                    size_each_part[0] - gap,
                    size_each_part[1] - gap,
                    size_each_part[2] - gap
                ]

                mass_each_part = total_mass * (separation[0] * separation[1] * separation[2])

                # 起始位置 (左下前角)
                start_pos = [
                    center_total[0] - size_total[0] / 2.0,
                    center_total[1] - size_total[1] / 2.0,
                    center_total[2] - size_total[2] / 2.0
                ]

                for iz in range(separation[2]):
                    # 判斷當前層是否需要偏移（實現交錯）
                    # 偶數層偏移 0，奇數層偏移半個磚塊寬度
                    offset = (size_each_part[0] / 2.0) if (iz % 2 == 1) else -(size_each_part[0] / 2.0)
                    
                    # 如果是交錯層，這一層的磚塊數量可以稍微調整，或者直接讓它超出/縮進
                    # 這裡採用的簡單做法：直接偏移，不處理邊緣截斷
                    for iy in range(separation[1]):
                        for ix in range(separation[0]):
                            
                            if separation[0] > separation[1]:
                                pos_part = [
                                    start_pos[0] + ix * size_each_part[0] * 2 + offset,
                                    start_pos[1] + iy * size_each_part[1] * 2,
                                    start_pos[2] + iz * size_each_part[2] * 2
                                ]
                            else:
                                pos_part = [
                                    start_pos[0] + ix * size_each_part[0] * 2,
                                    start_pos[1] + iy * size_each_part[1] * 2 + offset,
                                    start_pos[2] + iz * size_each_part[2] * 2
                                ]

                            config_part = copy.deepcopy(config)
                            config_part["object"]["size"] = actual_part_size
                            config_part["default_position"] = pos_part
                            config_part["object"]["object_mass"] = mass_each_part
                            
                            config_list.append(config_part)

        self.setup(configs=config_list)