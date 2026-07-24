import newton
import math
import warp as wp
import random
import numpy as np

from abc import ABC, abstractmethod
from script.game_config import GameConfig
from typing import TYPE_CHECKING, Type, Dict, List, Any

from newton.selection import ArticulationView
from .deformable_view import DeformableView

class BaseBody(ABC):
    body_key = "BASE"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BodyRegistry.register(cls)


    @abstractmethod
    def __init__(self):
        
        self.object_default_position_list: List[List[tuple]] = [] # 儲存所有物件的初始位置
        self.object_default_position_min_gpu = None # 儲存所有物件的初始位置範圍中的最小值
        self.object_default_position_max_gpu = None # 儲存所有物件的初始位置範圍中的最大值

        self.object_default_rotation_list: List[List[tuple]] = [] # 儲存所有物件的初始旋轉
        self.object_default_rotation_min_gpu = None # 儲存所有物件的初始位置範圍中的最小值
        self.object_default_rotation_max_gpu = None # 儲存所有物件的初始位置範圍中的最大值

        self.object_default_trans_gpu = None # 儲存所有物件的初始位姿 (Warp Transforms)

        self.patterns: dict[str, list] = {}
        self.patterns_index: list[list[int]] = [] # 對應 pattern 的物件在整個環境中的索引
        self.patterns_local_indices: dict[str, list] = {} # 記錄本體內部 0 基準局部索引
        self.views: list[ArticulationView, DeformableView] = []

        self.num_body_object_env = 0
        self.num_body_objects_total = 0

        # For Setup only
        self.is_ability_generated_object_type_changed = False
        self.current_ability_generated_object_type = ""
        self.ability_generated_object_offset_X = 0.0
        self.ability_generated_object_offset_X_prev = 0.0

    def add_object(self, 
                   label: str,
                   index: int,
                   default_position: List[Any],
                   default_rotation: List[Any],
                   ):
        if label not in self.patterns:
            self.patterns[label] = []
            self.patterns_local_indices[label] = [] 
        self.patterns[label].append(index)
        self.patterns_local_indices[label].append(self.num_body_object_env) # 記錄該物件在此 Body 中的局部序號

        self.num_body_object_env += 1
        self.object_default_position_list.append(default_position)
        self.object_default_rotation_list.append(default_rotation)

    def build_view(self, device, model, num_objects_env):
        self.device = device
        self.model = model
        self.num_objects_env=num_objects_env # 這個 Object 指的是高層次的 Object 而不是 model 中的 body，比如一個 Unitree G1 在 model 中有多個 body 但是在環境中只能算一個 Object

    def finalize_position_ranges(self, world_xy_offset_list, device, num_env): 
        """將最終擴展後的世界座標 list 轉換為 GPU 數組，並自動初始化預設初始位姿"""
        self.num_body_objects_total = self.num_body_object_env * num_env
        self.object_default_position_list *= num_env
        self.object_default_rotation_list *= num_env
        current_world = -1 

        mins_pos_np = np.zeros((self.num_body_objects_total, 3), dtype=np.float32)
        maxs_pos_np = np.zeros((self.num_body_objects_total, 3), dtype=np.float32)
        mins_rot_np = np.zeros((self.num_body_objects_total, 2), dtype=np.float32)
        maxs_rot_np = np.zeros((self.num_body_objects_total, 2), dtype=np.float32)
        
        # 儲存預設位姿
        default_transforms_list = []

        for i in range(self.num_body_objects_total):
            if i % self.num_body_object_env == 0:
                current_world += 1
            world_xy_offset = world_xy_offset_list[current_world]

            # 1. 位置解析與世界偏移
            obj_pos_range = self.object_default_position_list[i]
            mins_pos_np[i] = [obj_pos_range[0][0] + world_xy_offset[0], obj_pos_range[1][0] + world_xy_offset[1], obj_pos_range[2][0]]
            maxs_pos_np[i] = [obj_pos_range[0][1] + world_xy_offset[0], obj_pos_range[1][1] + world_xy_offset[1], obj_pos_range[2][1]]

            # 2. 旋轉解析 (Yaw, Pitch)
            obj_rot_range = self.object_default_rotation_list[i]
            mins_rot_np[i] = [obj_rot_range[0][0], obj_rot_range[1][0]]
            maxs_rot_np[i] = [obj_rot_range[0][1], obj_rot_range[1][1]]

            # 3. 🌟 動態構建預設初始位姿 (Transforms) [4.4.4]
            # 位置均值作為預設點
            pos_x = 0.5 * (mins_pos_np[i][0] + maxs_pos_np[i][0])
            pos_y = 0.5 * (mins_pos_np[i][1] + maxs_pos_np[i][1])
            pos_z = 0.5 * (mins_pos_np[i][2] + maxs_pos_np[i][2])
            pos_vec = wp.vec3(pos_x, pos_y, pos_z)

            # 旋轉角度均值作為預設朝向
            yaw = 0.5 * (mins_rot_np[i][0] + maxs_rot_np[i][0])
            pitch = 0.5 * (mins_rot_np[i][1] + maxs_rot_np[i][1])
            
            q_yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
            q_pitch = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), pitch)
            rot_quat = wp.mul(q_yaw, q_pitch)

            default_transforms_list.append(wp.transform(pos_vec, rot_quat))

        self.object_default_position_min_gpu = wp.from_numpy(mins_pos_np, dtype=wp.vec3, device=device)
        self.object_default_position_max_gpu = wp.from_numpy(maxs_pos_np, dtype=wp.vec3, device=device)
        self.object_default_rotation_min_gpu = wp.from_numpy(mins_rot_np, dtype=wp.vec2, device=device)
        self.object_default_rotation_max_gpu = wp.from_numpy(maxs_rot_np, dtype=wp.vec2, device=device)

        # 將預設 Transforms 載入 GPU 陣列，供 Kernel 讀取
        self.object_default_trans_gpu = wp.array(default_transforms_list, dtype=wp.transform, device=device)

    @wp.kernel
    def generate_random_transforms_kernel(
        reset_mask: wp.array(dtype=wp.int32),                           # 全域一維剛體重置遮罩 (body_count,)
        object_default_position_min_gpu: wp.array(dtype=wp.vec3),       # 全域隨機位置下限 (num_body_objects_total,)
        object_default_position_max_gpu: wp.array(dtype=wp.vec3),       # 全域隨機位置上限 (num_body_objects_total,)
        object_default_rotation_min_gpu: wp.array(dtype=wp.vec2),       # 全域隨機旋轉下限 [Yaw, Pitch] (num_body_objects_total,)
        object_default_rotation_max_gpu: wp.array(dtype=wp.vec2),       # 全域隨機旋轉上限 [Yaw, Pitch] (num_body_objects_total,)
        body_q_default: wp.array(dtype=wp.transform),                   # 全域預設初始位姿 (num_body_objects_total,)
        
        view_object_indices_gpu: wp.array(dtype=int),                   # 本 View 的物件在單世界中的局部索引 (count_per_world,)
        view_body_local_indices_gpu: wp.array(dtype=int),               # 本 View 的物件在當前 Body 中的 0 基準局部索引 
        num_objects_env: int,                                           # 單世界內所有種類的物件總數
        num_body_object_env: int,                                       # 單世界內所有關節體或者軟體的物件總數
        
        seed: wp.int32,
        offset_random: wp.array(dtype=wp.int32),                        # 隨機數偏移量 (world_count,)

        # 輸出變量
        out_transforms: wp.array2d(dtype=wp.transform),                 # 輸出給 View 的並行 3D 姿態陣列 (world_count, count_per_world)
        view_reset_mask: wp.array2d(dtype=bool)                         # 輸出給 View 控制的 2D 遮罩 (world_count, count_per_world) [4.4.4]
    ):
        world, obj_idx = wp.tid()
        
        # 對齊全域 reset_mask (使用環境全域物件總數 num_objects_env)
        local_idx = view_object_indices_gpu[obj_idx]
        flat_idx_env = world * num_objects_env + local_idx
        
        # 僅當該物件被標記為重置時才執行
        if reset_mask[flat_idx_env] == 1:
            view_reset_mask[world, obj_idx] = True
            
            # 對齊本體預設座標陣列 (使用正確的本體局部 0 基準索引)
            local_body_idx = view_body_local_indices_gpu[obj_idx]
            flat_idx_body = world * num_body_object_env + local_body_idx

            state = wp.rand_init(seed, offset_random[world] + world * 10 + obj_idx)
            offset_random[world] += 1

            # 使用 flat_idx_body 來讀取位置
            min_p = object_default_position_min_gpu[flat_idx_body]
            max_p = object_default_position_max_gpu[flat_idx_body]
            rand_pos = wp.vec3(
                wp.randf(state, min_p[0], max_p[0]),
                wp.randf(state, min_p[1], max_p[1]),
                wp.randf(state, min_p[2], max_p[2])
            )

            # 預設初始位姿也必須使用 flat_idx_body 讀取
            default_tf = body_q_default[flat_idx_body]
            default_rot = wp.transform_get_rotation(default_tf)
            
            m_rot = object_default_rotation_min_gpu[flat_idx_body]
            M_rot = object_default_rotation_max_gpu[flat_idx_body]

            # 雙軸旋轉合成
            rand_yaw = wp.randf(state, m_rot[0], M_rot[0])
            q_yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), rand_yaw)

            rand_pitch = wp.randf(state, m_rot[1], M_rot[1])
            q_pitch = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), rand_pitch)

            rand_rot = wp.mul(q_yaw, q_pitch)
            new_rot = wp.mul(rand_rot, default_rot)

            if min_p[0] != max_p[0] or m_rot[0] != M_rot[0] or m_rot[1] != M_rot[1]:
                out_transforms[world, obj_idx] = wp.transform(rand_pos, new_rot)
            else:
                out_transforms[world, obj_idx] = wp.transform(rand_pos, default_rot)
        else:
            view_reset_mask[world, obj_idx] = False



class BodyRegistry:
    _registry: Dict[str, Type['BaseBody']] = {}

    @classmethod
    def register(cls, body_cls: Type['BaseBody']):
        if body_cls.body_key and body_cls.body_key != "BASE":
            cls._registry[body_cls.body_key] = body_cls
            # print(f"[*] body註冊成功: {body_cls.body_key}")

    @classmethod
    def get_handler(cls, key: str):
        return cls._registry.get(key)

    @classmethod
    def get_all_keys(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def get_all_infos(cls) -> dict:
        infos = {}
        # for key, handler in cls._registry.items():
        #     infos[key] = {"path": handler.path, "container": handler.container}

        return infos


