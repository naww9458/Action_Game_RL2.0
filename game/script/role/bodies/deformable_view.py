# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import functools
from fnmatch import fnmatch
from types import NoneType
from typing import TYPE_CHECKING, Any
import numpy as np

import warp as wp
from warp.types import is_array

from newton import Model, State

# ========================================================================================
# GPU Gather/Scatter 核心：實現極致高效的並行記憶體收集與寫回
# ========================================================================================

@wp.kernel
def gather_deformable_attribute_kernel(
    flat_attrib: Any,                 # 底層平坦的粒子陣列 (total_particles, ...)
    offsets_gpu: wp.array[int],       # 該 View 中所有物件的第一世界粒子起點偏移量 (count_per_world,)
    stride_between_worlds: int,       # 跨世界粒子步長
    count_particle_per_object: int,   # 單個物件內部的粒子數
    dst: Any,                         # 輸出的並行 3D 視圖 (world_count, count_per_world, count_particle_per_object, ...)
    slice_offset: int,                # 切片偏移量
):
    world, obj_idx, particle_idx = wp.tid()
    
    # 計算該粒子在全域一維陣列中的精確物理索引
    global_idx = offsets_gpu[obj_idx] + world * stride_between_worlds + (particle_idx + slice_offset)
    dst[world, obj_idx, particle_idx] = flat_attrib[global_idx]


@wp.kernel
def scatter_deformable_attribute_kernel(
    view_mask: wp.array2d(dtype=bool),  # 2D 世界與物件雙重遮罩 (world_count, count_per_world) [4.4.4]
    values: Any,                        # 輸入的值 (world_count, count_per_world, count_particle_per_object, ...)
    offsets_gpu: wp.array[int],         # 物件粒子起點偏移量 (count_per_world,)
    stride_between_worlds: int,         # 跨世界粒子步長
    count_particle_per_object: int,     # 單個物件內部的粒子數
    flat_attrib: Any,                   # 寫回目標物理平坦陣列 (total_particles, ...)
    slice_offset: int,                  # 切片偏移量
):
    world, obj_idx, particle_idx = wp.tid()
    
    # 判定該特定世界中的特定軟體物件是否需要被重設
    if view_mask[world, obj_idx]:
        global_idx = offsets_gpu[obj_idx] + world * stride_between_worlds + (particle_idx + slice_offset)
        flat_attrib[global_idx] = values[world, obj_idx, particle_idx]

@wp.kernel
def localize_template_positions_kernel(
    world_positions: wp.array(dtype=wp.vec3),
    default_transform: wp.transform,
    local_positions: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    inv_tf = wp.transform_inverse(default_transform)
    local_positions[tid] = wp.transform_point(inv_tf, world_positions[tid])


@wp.kernel
def reset_deformable_velocities_direct_kernel(
    view_mask: wp.array2d(dtype=wp.bool),
    offsets_gpu: wp.array(dtype=int),
    stride_between_worlds: int,
    flat_particle_qd: wp.array(dtype=wp.vec3),
    flat_particle_f: wp.array(dtype=wp.vec3),
):
    world, obj_idx, particle_idx = wp.tid()

    if view_mask[world, obj_idx]:
        global_idx = offsets_gpu[obj_idx] + world * stride_between_worlds + particle_idx
        flat_particle_qd[global_idx] = wp.vec3(0.0, 0.0, 0.0)
        flat_particle_f[global_idx] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def reset_deformable_direct_kernel(
    reset_mask: wp.array(dtype=wp.int32),               # 全域一維剛體重置遮罩 (body_count,)
    view_object_indices_gpu: wp.array(dtype=int),       # 本 View 的物件在單世界中的局部索引 (count_per_world,)
    num_objects_env: int,                               # 單世界內所有種類的物件總數
    offsets_gpu: wp.array(dtype=int),                   # 本 View 粒子起點偏移量 (count_per_world,)
    stride_between_worlds: int,                         # 跨世界粒子步長
    template_positions: wp.array2d(dtype=wp.vec3),      # 物理範本形狀 (count_per_world, count_particle_per_object)
    random_transforms: wp.array2d(dtype=wp.transform),  # 隨機生成的剛體位姿 (world_count, count_per_world)
    flat_particle_q: wp.array(dtype=wp.vec3),           # 直接寫入目標：Newton 全局粒子位置 q 
):
    world, obj_idx, particle_idx = wp.tid()
    
    # 對齊一維全域剛體索引，判斷是否重置
    local_idx = view_object_indices_gpu[obj_idx]
    flat_idx_env = world * num_objects_env + local_idx
    
    if reset_mask[flat_idx_env] == 1:
        # 進行剛體變換 (相對距離不變，不形變不爆炸)
        local_pos = template_positions[obj_idx, particle_idx]
        tf = random_transforms[world, obj_idx]
        global_pos = wp.transform_point(tf, local_pos)
        
        # 計算該粒子在全域一維陣列中的絕對物理索引並直接寫入
        global_particle_idx = offsets_gpu[obj_idx] + world * stride_between_worlds + particle_idx
        flat_particle_q[global_particle_idx] = global_pos


# 明確的多載（Overloads）宣告，確保 Warp 的 Cuda 編譯鏈穩定
for dtype in [wp.vec3, float]:
    wp.overload(
        gather_deformable_attribute_kernel,
        {"dst": wp.array3d[dtype], "flat_attrib": wp.array[dtype]}
    )
    wp.overload(
        scatter_deformable_attribute_kernel,
        {"values": wp.array3d[dtype], "flat_attrib": wp.array[dtype]}
    )


class Slice:
    """與 ArticulationView 相同的可雜湊切片封裝，用於快取（Cache）機制的雜湊鍵"""
    def __init__(self, start=None, stop=None):
        self.start = start
        self.stop = stop

    def __hash__(self):
        return hash((self.start, self.stop))

    def __eq__(self, other):
        return isinstance(other, Slice) and self.start == other.start and self.stop == other.stop

    def __str__(self):
        return f"({self.start}, {self.stop})"

    def get(self):
        return slice(self.start, self.stop)


class DeformableView:
    """
    DeformableView 提供了一個專為軟體（Soft Body）、布料（Cloth）、流體粒子設計的高階控制視圖。
    它支持並行環境下的批量觀測與狀態設定，並與 ArticulationView 保持高度一致的 API 設計。
    """

    def __init__(
        self,
        model: Model,
        offset: list[int],                    # 改為接收 list[int] 的物理起點偏移量
        count_particle_per_object: int,       # 單個物件內部的粒子數量
        pattern: str = "deformable_object",   # 代表當前 View 所管理的所有物件的標籤或名稱
        stride_between_worlds: int | None = None,
        verbose: bool | None = None,
    ):
        self.model = model
        self.device = model.device
        self.pattern = pattern

        if verbose is None:
            verbose = wp.config.verbose

        # 解析世界總數
        self.world_count = model.world_count if hasattr(model, "world_count") else 1

        # 儲存粒子計數與偏移列表
        self.offset = offset if isinstance(offset, list) else [offset]
        self.count_particle_per_object = count_particle_per_object

        # 自動解析 count_per_world：每個環境中粒子數相同的同類物件數量
        self.count_per_world = len(self.offset)

        # 將偏移量列表上傳至 GPU 陣列，供 Warp Kernel 在並行查找時調用
        self._offsets_gpu = wp.array(self.offset, dtype=wp.int32, device=self.device)

        # 解析跨世界的粒子步長 (Stride)
        if stride_between_worlds is None:
            if self.world_count > 1 and model.particle_count > 0:
                # 跨世界粒子步長等於：單個世界的所有粒子總數
                self.stride_between_worlds = model.particle_count // self.world_count
            else:
                self.stride_between_worlds = model.particle_count
        else:
            self.stride_between_worlds = stride_between_worlds

        # 建立預設世界 2D 遮罩 (world_count, count_per_world)
        full_mask_np = np.ones((self.world_count, self.count_per_world), dtype=bool)
        self.full_mask = wp.array(full_mask_np, dtype=bool, device=self.device)

        if verbose:
            print(f"DeformableView '{self.pattern}': Resolved")
            print(f"  World Count:              {self.world_count}")
            print(f"  Object Count per World:   {self.count_per_world}")
            print(f"  Particles per Object:     {self.count_particle_per_object}")
            print(f"  Offsets (World 0):        {self.offset}")
            print(f"  Stride between Worlds:    {self.stride_between_worlds}")

    # ========================================================================================
    # 核心底層屬性對接 API (重塑 GPU 3D 視圖)

    @functools.lru_cache(maxsize=None)  # noqa
    def _get_attribute_array(self, name: str, source: Model | State, _slice: Slice | int | None = None):
        attrib = getattr(source, name)
        assert isinstance(attrib, wp.array)

        value_stride = attrib.strides[0]

        # 重塑為規整的 3D 視圖：(world_count, count_per_world, count_particle_per_object)
        shape = (self.world_count, self.count_per_world, self.count_particle_per_object)

        if self.count_per_world == 1:
            # 單一物件通道：零拷貝直接記憶體指針映射 (最高性能)
            strides = (
                self.stride_between_worlds * value_stride,
                self.count_particle_per_object * value_stride,  # 單物件寬度
                value_stride,
            )

            if attrib.ptr is None:
                result = wp.empty(shape, dtype=attrib.dtype, device=attrib.device)
                result.ptr = None
                return result

            # 處理可微分模擬（DiffSim）的梯度傳導鏈
            source_grad = attrib.grad if attrib.requires_grad else None
            grad_view = None
            if source_grad is not None:
                grad_stride = source_grad.strides[0]
                grad_view = wp.array(
                    ptr=int(source_grad.ptr) + self.offset[0] * grad_stride,
                    dtype=source_grad.dtype,
                    shape=shape,
                    strides=(
                        self.stride_between_worlds * grad_stride,
                        self.count_particle_per_object * grad_stride,
                        grad_stride,
                    ),
                    device=source_grad.device,
                    copy=False,
                )

            attrib_view = wp.array(
                ptr=int(attrib.ptr) + self.offset[0] * value_stride,
                dtype=attrib.dtype,
                shape=shape,
                strides=strides,
                device=attrib.device,
                copy=False,
                grad=grad_view,
            )
        else:
            # 多物件通道：建立暫存 Staging 緩衝區 (在呼叫 _get_attribute_values 時進行並行 Gather)
            attrib_view = wp.empty(shape, dtype=attrib.dtype, device=attrib.device)
            attrib_view._staging_array = True  # 標記為 Staging 陣列
            if attrib.requires_grad:
                attrib_view.requires_grad = True

        # 支援自訂粒子子集切片 (切片施加在最後一個粒子維度上)
        if isinstance(_slice, Slice):
            _slice = _slice.get()
        elif not isinstance(_slice, (NoneType, int, slice)):
            raise ValueError(f"不合法的切片類型: {type(_slice)}")

        if _slice is not None:
            attrib_view = attrib_view[:, :, _slice]

        return attrib_view

    def _get_attribute_values(self, name: str, source: Model | State, _slice: slice | None = None):
        attrib_view = self._get_attribute_array(name, source, _slice=_slice)
        
        # 若為多物件模式，需要執行並行的 GPU Gather 收集資料 
        if hasattr(attrib_view, "_staging_array"):
            flat_attrib = getattr(source, name)
            slice_offset = _slice.start if isinstance(_slice, slice) else (0 if _slice is None else _slice)

            # 啟動 Gather Kernel 收集物理狀態
            wp.launch(
                gather_deformable_attribute_kernel,
                dim=attrib_view.shape, # 執行緒網格形狀即為視圖形狀
                inputs=[
                    flat_attrib,
                    self._offsets_gpu,
                    self.stride_between_worlds,
                    self.count_particle_per_object,
                    attrib_view,
                    slice_offset
                ],
                device=self.device
            )
            
            # 同步處理並行梯度
            src_grad = flat_attrib.grad if flat_attrib.requires_grad else None
            dst_grad = attrib_view.grad if attrib_view.requires_grad else None
            if src_grad is not None and dst_grad is not None:
                wp.launch(
                    gather_deformable_attribute_kernel,
                    dim=attrib_view.shape,
                    inputs=[
                        src_grad,
                        self._offsets_gpu,
                        self.stride_between_worlds,
                        self.count_particle_per_object,
                        dst_grad,
                        slice_offset
                    ],
                    device=self.device
                )
            
        return attrib_view

    def _set_attribute_values(
        self, name: str, target: Model | State, values, mask=None, _slice: slice | None = None
    ):
        attrib_view = self._get_attribute_array(name, target, _slice=_slice)

        if not is_array(values) or values.dtype != attrib_view.dtype:
            values = wp.array(values, dtype=attrib_view.dtype, shape=attrib_view.shape, device=self.device, copy=False)
        assert values.shape == attrib_view.shape
        assert values.dtype == attrib_view.dtype

        if values.ptr == attrib_view.ptr:
            return

        # 解決環境遮罩
        if mask is None:
            mask = self.full_mask
        else:
            mask = self._resolve_mask(mask)

        flat_attrib = getattr(target, name)
        slice_offset = _slice.start if isinstance(_slice, slice) else (0 if _slice is None else _slice)

        # 呼叫並行 GPU Scatter Kernel，實現帶有遮罩的安全寫回
        wp.launch(
            scatter_deformable_attribute_kernel,
            dim=attrib_view.shape,
            inputs=[
                mask,
                values,
                self._offsets_gpu,
                self.stride_between_worlds,
                self.count_particle_per_object,
                flat_attrib,
                slice_offset
            ],
            device=self.device,
        )

    # ========================================================================================
    # 遮罩輔助解析器 (核心修復：全面支援 2D 布林遮罩與 1D 自動廣播對齊) 

    def _resolve_mask(self, mask):
        if isinstance(mask, wp.array):
            if mask.dtype is wp.bool:
                # 如果已經是正確的 2D 遮罩，直接通過並使用
                if mask.ndim == 2 and mask.shape == (self.world_count, self.count_per_world):
                    return mask
                # 如果是 1D 世界遮罩，自動將其「廣播（np.repeat）」擴展為健康的 2D 遮罩！
                elif mask.ndim == 1 and mask.shape[0] == self.world_count:
                    mask_np = mask.numpy()
                    mask_2d_np = np.repeat(mask_np[:, np.newaxis], self.count_per_world, axis=1)
                    return wp.array(mask_2d_np, dtype=bool, device=self.device)
        else:
            # 支援傳入原生 Python 列表
            try:
                mask_np = np.asarray(mask, dtype=bool)
                if mask_np.ndim == 1:
                    mask_2d_np = np.repeat(mask_np[:, np.newaxis], self.count_per_world, axis=1)
                    return wp.array(mask_2d_np, dtype=bool, device=self.device)
                elif mask_np.ndim == 2 and mask_np.shape == (self.world_count, self.count_per_world):
                    return wp.array(mask_np, dtype=bool, device=self.device)
            except Exception:
                pass
                
        raise ValueError(
            f"期望傳入形狀為 ({self.world_count}, {self.count_per_world}) 2D 遮罩，"
            f"或形狀為 ({self.world_count},) 1D 世界遮罩的布林遮罩陣列。"
        )

    # ========================================================================================
    # 高階通用屬性 API

    def get_attribute(self, name: str, source: Model | State):
        """讀取指定粒子的物理屬性 (如 particle_q, particle_qd)"""
        return self._get_attribute_values(name, source)

    def set_attribute(self, name: str, target: Model | State, values, mask=None):
        """設定指定粒子的物理屬性 (如 particle_q, particle_qd)"""
        self._set_attribute_values(name, target, values, mask=mask)

    # ========================================================================================
    # 高階語意包裹器 (與 ArticulationView 保持對齊，方便無縫呼叫)

    def get_positions(self, source: Model | State):
        """
        獲取所有並行世界中，該 View 下的所有軟體物件的所有粒子 3D 位置 [m]。
        
        Returns:
            array: 形狀為 (world_count, count_per_world, count_particle_per_object) 的 wp.vec3 陣列
        """
        return self.get_attribute("particle_q", source)

    def set_positions(self, target: Model | State, values, mask=None):
        """
        設定該 View 下的所有軟體物件的所有粒子 3D 位置 [m]。
        """
        self.set_attribute("particle_q", target, values, mask=mask)

    def reset_positions(self, state, random_transforms, template_positions, reset_mask_gpu, view_object_indices_gpu, num_objects_env):
        """
        直接向物理引擎底層的 particle_q 寫入重置後的隨機 3D 座標，避開暫存緩衝區，杜絕重疊與不寫回問題 
        """
        wp.launch(
            reset_deformable_direct_kernel,
            dim=(self.world_count, self.count_per_world, self.count_particle_per_object),
            inputs=[
                reset_mask_gpu,
                view_object_indices_gpu,
                num_objects_env,
                self._offsets_gpu,
                self.stride_between_worlds,
                template_positions,
                random_transforms,
                state.particle_q,
            ],
            device=self.device
        )

    def reset_velocities(self, state, view_reset_mask):
        """直接清零被重置物件的粒子速度與殘留外力，與 reset_positions 走同一套索引邏輯。"""
        wp.launch(
            reset_deformable_velocities_direct_kernel,
            dim=(self.world_count, self.count_per_world, self.count_particle_per_object),
            inputs=[
                view_reset_mask,
                self._offsets_gpu,
                self.stride_between_worlds,
                state.particle_qd,
                state.particle_f,
            ],
            device=self.device,
        )

    def localize_template_positions(self, world_positions, default_transform, local_positions):
        wp.launch(
            localize_template_positions_kernel,
            dim=world_positions.shape[0],
            inputs=[world_positions, default_transform, local_positions],
            device=self.device,
        )

    def get_velocities(self, source: Model | State):
        """
        獲取所有並行世界中，該 View 下的所有軟體物件的所有粒子 3D 速度 [m/s]。
        
        Returns:
            array: 形狀為 (world_count, count_per_world, count_particle_per_object) 的 wp.vec3 陣列
        """
        return self.get_attribute("particle_qd", source)

    def set_velocities(self, target: Model | State, values, mask=None):
        """
        設定該 View 下的所有軟體物件的所有粒子 3D 速度 [m/s]。
        """
        self.set_attribute("particle_qd", target, values, mask=mask)


