import warp as wp
import numpy as np
from .deformable_view import DeformableView
from newton import Model
from .base_body import BaseBody
from typing import List, Any, Dict, Optional

@wp.kernel
def apply_deformable_updates_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    
    offsets_gpu: wp.array(dtype=int),
    stride_between_worlds: int,
    count_particle_per_object: int,
    particles_per_world: int,
    
    # 3D 控制輸入 (world_count, count_per_world, count_particle_per_object)
    in_pos: wp.array3d(dtype=wp.vec3),
    in_vel: wp.array3d(dtype=wp.vec3),
    in_force: wp.array3d(dtype=wp.vec3),
    action_mask: wp.array3d(dtype=wp.int32),
    
    dt: float,
    linear_damping: float,
    particle_inv_mass: wp.array(dtype=float),
):
    tid = wp.tid()
    
    # 解析並行 3D 結構索引
    world = tid // particles_per_world
    local_tid = tid % particles_per_world
    obj_idx = local_tid // count_particle_per_object
    particle_idx = local_tid % count_particle_per_object
    
    # 計算全域一維粒子索引
    global_particle_idx = offsets_gpu[obj_idx] + world * stride_between_worlds + particle_idx
    
    mask = action_mask[world, obj_idx, particle_idx]
    
    # --- 阻尼更新 (保持連續性與可微性) ---
    curr_v = particle_qd[global_particle_idx]
    l_fact = wp.exp(-linear_damping * dt)
    curr_v = curr_v * l_fact
    
    # --- 離散傳送 (斷開梯度，僅在 mask 觸發時執行) ---
    if (mask & 1) != 0:
        particle_q[global_particle_idx] = in_pos[world, obj_idx, particle_idx]
        
    # --- 速度覆寫 ---
    if (mask & 4) != 0:
        curr_v = in_vel[world, obj_idx, particle_idx]
        
    particle_qd[global_particle_idx] = curr_v
    
    if particle_inv_mass[global_particle_idx] == 0.0:
        return
        
    # --- 4. 力累加 (無分支判斷，完美保留梯度鏈) ---
    f = in_force[world, obj_idx, particle_idx]
    particle_f[global_particle_idx] = particle_f[global_particle_idx] + f


class DeformableBody(BaseBody):

    def __init__(self):
        super().__init__()
        self.offset: dict[str, list] = {}
        self.count_particle_per_object: dict[str, int] = {}
        
        self.view_object_indices_gpus: Dict[str, wp.array] = {}
        self.view_body_local_indices_gpus = {} 
        self.random_transforms_gpus: Dict[str, wp.array] = {}
        self.template_positions_gpus: Dict[str, wp.array] = {}
        self.view_reset_mask_gpus: Dict[str, wp.array] = {}
        
        # 為各 View 建立專屬的 3D 控制 Buffer
        self.control_pos_gpus: Dict[str, wp.array] = {}
        self.control_vel_gpus: Dict[str, wp.array] = {}
        self.control_force_gpus: Dict[str, wp.array] = {}
        self.control_mask_gpus: Dict[str, wp.array] = {}

    def add_object(self, 
                   label: str, 
                   index: int, 
                   particle_index: list[int], 
                   default_position: List[Any], 
                   default_rotation: List[Any]
                   ):
        super().add_object(label=label, index=index, default_position=default_position, default_rotation=default_rotation)

        if label not in self.offset:
            self.offset[label] = []
            self.count_particle_per_object[label] = particle_index["count_particle_per_object"]

        self.offset[label].append(particle_index["start"])

    def build_view(self, device, model: Model, num_objects_env):
        super().build_view(device=device, model=model, num_objects_env=num_objects_env)
        self.model = model  # 快取 model 供後續控制使用

        for pattern in self.patterns:
            offset = self.offset[pattern]
            count_particle_per_object = self.count_particle_per_object[pattern]

            view = DeformableView(model, offset=offset, count_particle_per_object=count_particle_per_object, pattern=pattern)
            self.views.append(view)

            # 初始化 3D 控制緩衝區 (world_count, count_per_world, count_particle_per_object)
            shape = (view.world_count, view.count_per_world, count_particle_per_object)
            self.control_pos_gpus[pattern] = wp.zeros(shape=shape, dtype=wp.vec3, device=self.device)
            self.control_vel_gpus[pattern] = wp.zeros(shape=shape, dtype=wp.vec3, device=self.device)
            self.control_force_gpus[pattern] = wp.zeros(shape=shape, dtype=wp.vec3, device=self.device)
            self.control_mask_gpus[pattern] = wp.zeros(shape=shape, dtype=wp.int32, device=self.device)

            # 快取原本的重置輔助資訊
            local_indices = self.patterns[pattern]
            self.view_object_indices_gpus[pattern] = wp.array(
                local_indices, dtype=int, device=self.device
            )

            self.view_body_local_indices_gpus[pattern] = wp.array(
                self.patterns_local_indices[pattern], dtype=int, device=self.device
            )

            # 快取該 View 專屬的隨機輸出位姿緩衝區
            self.random_transforms_gpus[pattern] = wp.empty(
                shape=(view.world_count, view.count_per_world), 
                dtype=wp.transform, 
                device=self.device
            )

            # 快取該 View 專屬的 2D 遮罩緩衝區 
            self.view_reset_mask_gpus[pattern] = wp.zeros(
                shape=(view.world_count, view.count_per_world), 
                dtype=bool, 
                device=self.device
            )

            # 快取並拷貝軟體的靜止範本粒子坐標 
            self.template_positions_gpus[pattern] = wp.empty(
                shape=(view.count_per_world, view.count_particle_per_object), 
                dtype=wp.vec3, 
                device=self.device
            )

            default_tfs_np = self.object_default_trans_gpu.numpy()
            for obj_idx, start_idx in enumerate(offset):
                world_rest = model.particle_q[start_idx : start_idx + count_particle_per_object]
                local_body_idx = self.patterns_local_indices[pattern][obj_idx]
                tf_row = default_tfs_np[local_body_idx]
                default_tf = wp.transform(
                    wp.vec3(float(tf_row[0]), float(tf_row[1]), float(tf_row[2])),
                    wp.quat(float(tf_row[3]), float(tf_row[4]), float(tf_row[5]), float(tf_row[6])),
                )
                view.localize_template_positions(
                    world_rest,
                    default_tf,
                    self.template_positions_gpus[pattern][obj_idx],
                )

    def apply_controls(self, state, dt, linear_damping):
        """
        🌟 呼叫並行控制 Kernel 更新軟體粒子物理狀態 (相容自動微分)
        """
        for i, view in enumerate(self.views):
            pattern = list(self.patterns.keys())[i]
            
            total_particles_in_view = view.world_count * view.count_per_world * view.count_particle_per_object
            particles_per_world = view.count_per_world * view.count_particle_per_object
            
            wp.launch(
                kernel=apply_deformable_updates_kernel,
                dim=total_particles_in_view,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    state.particle_f,
                    view._offsets_gpu,
                    view.stride_between_worlds,
                    view.count_particle_per_object,
                    particles_per_world,
                    self.control_pos_gpus[pattern],
                    self.control_vel_gpus[pattern],
                    self.control_force_gpus[pattern],
                    self.control_mask_gpus[pattern],
                    dt,
                    linear_damping,
                    self.model.particle_inv_mass
                ],
                device=self.device
            )

    def clear_controls(self):
        """
        清空控制 Buffer，避免動作殘留。
        """
        for pattern in self.patterns:
            self.control_mask_gpus[pattern].zero_()
            self.control_pos_gpus[pattern].zero_()
            self.control_vel_gpus[pattern].zero_()
            self.control_force_gpus[pattern].zero_()

    def reset_obj(self, state, reset_mask_gpu, offset_random_gpu, seed, state_alt=None):
        """
        利用 View 完美、安全且高並行地在 GPU 上進行隨機化重置
        """
        view: DeformableView

        for i, view in enumerate(self.views):
            pattern = list(self.patterns.keys())[i]

            # 呼叫並行隨機 Kernel：為需要重置的物件生成隨機位姿，並自動匯出 2D 重置遮罩 
            wp.launch(
                self.generate_random_transforms_kernel,
                dim=(view.world_count, view.count_per_world),
                inputs=[
                    reset_mask_gpu,
                    self.object_default_position_min_gpu,
                    self.object_default_position_max_gpu,
                    self.object_default_rotation_min_gpu,  
                    self.object_default_rotation_max_gpu,  
                    self.object_default_trans_gpu,         

                    self.view_object_indices_gpus[pattern],  
                    self.view_body_local_indices_gpus[pattern],
                    self.num_objects_env,                    
                    self.num_body_object_env,

                    seed,
                    offset_random_gpu,
                ],
                outputs=[
                    self.random_transforms_gpus[pattern],
                    self.view_reset_mask_gpus[pattern] 
                ],
                device=self.device
            )

            view_reset_mask = self.view_reset_mask_gpus[pattern]
            random_transforms = self.random_transforms_gpus[pattern]
            template_positions = self.template_positions_gpus[pattern]
            view_object_indices = self.view_object_indices_gpus[pattern]

            states_to_reset = [state]
            if state_alt is not None:
                states_to_reset.append(state_alt)

            for target_state in states_to_reset:
                view.reset_positions(
                    state=target_state,
                    random_transforms=random_transforms,
                    template_positions=template_positions,
                    reset_mask_gpu=reset_mask_gpu,
                    view_object_indices_gpu=view_object_indices,
                    num_objects_env=self.num_objects_env,
                )
                view.reset_velocities(
                    state=target_state,
                    view_reset_mask=view_reset_mask,
                )



