import gymnasium as gym
import numpy as np

import numpy as np
import gymnasium as gym

def schema_to_gym_space(schemas: list):
    """
    將 schema 轉換為 gymnasium.spaces 列表。
    """
    spaces = []
    for schema in schemas:
        low_parts = []
        high_parts = []
        # 預設 dtype，通常 Box 內所有維度共用同一個 dtype
        target_dtype = np.float32 

        for skill_name, config in schema.items():
            # print("skill_name: ", skill_name)
            
            # 取得範圍並確保順序 (low, high)
            space_range = config["range"]
            r_low = float(min(space_range))
            r_high = float(max(space_range))
            
            # 取得形狀 (假設 config["shape"] 是一個 list, 如 [3])
            shape_val = config["shape"]
            
            # 更新 dtype (如果 schema 中有指定)
            if "dtype" in config:
                target_dtype = np.dtype(config["dtype"])

            # 使用 np.full 快速建立向量，避免迴圈 append
            low_parts.append(np.full(shape_val, r_low, dtype=target_dtype))
            high_parts.append(np.full(shape_val, r_high, dtype=target_dtype))

        # 將所有技能的向量拼接成一個長向量
        final_low = np.concatenate(low_parts)
        final_high = np.concatenate(high_parts)

        # 建立 Box 空間
        space = gym.spaces.Box(
            low=final_low,
            high=final_high,
            shape=final_low.shape,
            dtype=target_dtype
        )
                
        spaces.append(space)

    return spaces





