import newton
import warp as wp
import numpy as np

from newton.selection import ArticulationView
from script.game_config import GameConfig
from newton import Model, State
from .base_body import BaseBody
from typing import List, Any, Dict, Optional

from script.role.abilities.articulation_control_config.joint_config_registry import (
    apply_soft_limits,
    resolve_joint_arrays_for_pattern,
    resolve_rl_action_dim_for_pattern,
)

# =============================================================================
# 剛體級更新 KERNEL (處理整體/局部外部干擾力、重置與傳送)
# =============================================================================
@wp.kernel
def apply_articulation_updates_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    solver_body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    
    view_body_local_indices: wp.array(dtype=int),
    num_rigid_bodies_env: int,
    
    bodies_per_world: int,
    bodies_per_object: int,
    
    # 3D 控制輸入 (world_count, count_per_world, bodies_per_object)
    in_pos: wp.array3d(dtype=wp.vec3),
    in_rot: wp.array3d(dtype=wp.quat),
    in_vel: wp.array3d(dtype=wp.vec3),
    in_omega: wp.array3d(dtype=wp.vec3),
    in_force: wp.array3d(dtype=wp.vec3),
    in_torque: wp.array3d(dtype=wp.vec3),
    action_mask: wp.array3d(dtype=wp.int32),
    
    dt: float,
    linear_damping: float,
    angular_damping: float,
    body_inv_mass: wp.array(dtype=float),
):
    tid = wp.tid()
    
    # 解析 3D 結構索引
    world = tid // bodies_per_world
    local_tid = tid % bodies_per_world
    obj_idx = local_tid // bodies_per_object
    body_in_obj_idx = local_tid % bodies_per_object
    
    # 獲取全域剛體索引
    local_body_idx = view_body_local_indices[local_tid]
    global_body_idx = world * num_rigid_bodies_env + local_body_idx
    
    mask = action_mask[world, obj_idx, body_in_obj_idx]
    
    # --- 阻尼更新 (保持連續性與可微性) ---
    qd = body_qd[global_body_idx]
    l_fact = wp.exp(-linear_damping * dt)
    a_fact = wp.exp(-angular_damping * dt)
    
    curr_v = wp.vec3(qd[0] * l_fact, qd[1] * l_fact, qd[2] * l_fact)
    curr_w = wp.vec3(qd[3] * a_fact, qd[4] * a_fact, qd[5] * a_fact)
    
    # --- 離散傳送 (斷開梯度，用於 Reset) ---
    if (mask & 1) != 0 or (mask & 2) != 0:
        p = body_q[global_body_idx].p
        q = body_q[global_body_idx].q
        if (mask & 1) != 0: 
            p = in_pos[world, obj_idx, body_in_obj_idx]
        if (mask & 2) != 0: 
            q = in_rot[world, obj_idx, body_in_obj_idx]
            
        new_xform = wp.transform(p, q)
        body_q[global_body_idx] = new_xform
        body_q_prev[global_body_idx] = new_xform 
        solver_body_q_prev[global_body_idx] = new_xform
        
    # --- 速度覆寫 ---
    if (mask & 4) != 0:
        curr_v = in_vel[world, obj_idx, body_in_obj_idx]
    if (mask & 8) != 0:
        curr_w = in_omega[world, obj_idx, body_in_obj_idx]
        
    body_qd[global_body_idx] = wp.spatial_vector(
        curr_v[0], curr_v[1], curr_v[2], 
        curr_w[0], curr_w[1], curr_w[2]
    )
    
    if body_inv_mass[global_body_idx] == 0.0:
        return
        
    # --- 外部力與力矩累加 (可微優化，無分支) ---
    f = in_force[world, obj_idx, body_in_obj_idx]
    t = in_torque[world, obj_idx, body_in_obj_idx]
    curr_f = body_f[global_body_idx]
    
    body_f[global_body_idx] = wp.spatial_vector(
        curr_f[0] + f[0], curr_f[1] + f[1], curr_f[2] + f[2],
        curr_f[3] + t[0], curr_f[4] + t[1], curr_f[5] + t[2]
    )


# =============================================================================
# Inspector 浮動基座傳送 (對齊 reset 的 set_root_transforms 流程)
# =============================================================================
@wp.kernel
def build_inspector_root_teleport_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    view_body_local_indices: wp.array(dtype=int),
    num_rigid_bodies_env: int,
    bodies_per_world: int,
    bodies_per_object: int,
    base_body_in_obj_idx: int,
    has_free: wp.array(dtype=wp.int32),
    in_pos: wp.array3d(dtype=wp.vec3),
    in_rot: wp.array3d(dtype=wp.quat),
    in_vel: wp.array3d(dtype=wp.vec3),
    in_omega: wp.array3d(dtype=wp.vec3),
    action_mask: wp.array3d(dtype=wp.int32),
    out_transforms: wp.array2d(dtype=wp.transform),
    out_velocities: wp.array2d(dtype=wp.spatial_vector),
    out_teleport_mask: wp.array2d(dtype=wp.bool),
):
    world, obj_idx = wp.tid()

    if has_free[obj_idx] == 0:
        out_teleport_mask[world, obj_idx] = False
        return

    mask = action_mask[world, obj_idx, base_body_in_obj_idx]
    teleport = (mask & 15) != 0
    out_teleport_mask[world, obj_idx] = teleport
    if not teleport:
        return

    local_tid = obj_idx * bodies_per_object + base_body_in_obj_idx
    local_body_idx = view_body_local_indices[local_tid]
    global_body_idx = world * num_rigid_bodies_env + local_body_idx

    p = body_q[global_body_idx].p
    q = body_q[global_body_idx].q
    if (mask & 1) != 0:
        p = in_pos[world, obj_idx, base_body_in_obj_idx]
    if (mask & 2) != 0:
        q = in_rot[world, obj_idx, base_body_in_obj_idx]
    out_transforms[world, obj_idx] = wp.transform(p, q)

    qd = body_qd[global_body_idx]
    v = wp.vec3(qd[0], qd[1], qd[2])
    w = wp.vec3(qd[3], qd[4], qd[5])
    if (mask & 4) != 0:
        v = in_vel[world, obj_idx, base_body_in_obj_idx]
    if (mask & 8) != 0:
        w = in_omega[world, obj_idx, base_body_in_obj_idx]
    out_velocities[world, obj_idx] = wp.spatial_vector(v[0], v[1], v[2], w[0], w[1], w[2])


@wp.kernel
def sync_articulation_body_q_prev_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    solver_body_q_prev: wp.array(dtype=wp.transform),
    view_body_local_indices: wp.array(dtype=int),
    num_rigid_bodies_env: int,
    bodies_per_world: int,
    bodies_per_object: int,
    teleport_mask: wp.array2d(dtype=wp.bool),
):
    tid = wp.tid()
    world = tid // bodies_per_world
    local_tid = tid % bodies_per_world
    obj_idx = local_tid // bodies_per_object

    if not teleport_mask[world, obj_idx]:
        return

    local_body_idx = view_body_local_indices[local_tid]
    global_body_idx = world * num_rigid_bodies_env + local_body_idx
    xform = body_q[global_body_idx]
    body_q_prev[global_body_idx] = xform
    solver_body_q_prev[global_body_idx] = xform


# =============================================================================
# 全能關節驅動 KERNEL
# =============================================================================
@wp.kernel
def apply_joint_actuation_kernel(
    joint_f: wp.array(dtype=float),            # control.joint_f
    joint_target_vel: wp.array(dtype=float),   # control.joint_target_vel
    joint_target_pos: wp.array(dtype=float),   # control.joint_target_pos
    
    view_joint_dof_indices: wp.array(dtype=int),
    num_joint_dof_env: int,
    dof_per_world: int,
    joint_dof_count: int,
    
    in_joint_torque: wp.array3d(dtype=float),
    in_joint_vel: wp.array3d(dtype=float),
    in_joint_pos: wp.array3d(dtype=float),
    joint_action_mask: wp.array3d(dtype=wp.int32),
    
    view_joint_has_free: wp.array(dtype=wp.int32),  # 判斷物件是否有虛擬基座關節的一維掩碼
):
    tid = wp.tid()
    world = tid // dof_per_world
    local_tid = tid % dof_per_world
    obj_idx = local_tid // joint_dof_count
    dof_in_obj_idx = local_tid % joint_dof_count
    
    # 獲取全局關節自由度索引
    local_dof_idx = view_joint_dof_indices[local_tid]
    global_dof_idx = world * num_joint_dof_env + local_dof_idx
    
    # 若此物件有虛擬基座關節（FREE），則加上 6 DoFs 的偏移量
    if view_joint_has_free[obj_idx] == 1:
        global_dof_idx = global_dof_idx + 6
    
    mask = joint_action_mask[world, obj_idx, dof_in_obj_idx]
    
    # 位元旗標 (bit flag) 三種模式互不重疊: bit0(1)=扭矩, bit1(2)=速度, bit2(4)=位置。
    # Mode 1: 力矩直接驅動 (100% 可微，梯度直通)
    if (mask & 1) != 0:
        joint_f[global_dof_idx] = joint_f[global_dof_idx] + in_joint_torque[world, obj_idx, dof_in_obj_idx]
        
    # Mode 2: 目標速度驅動
    if (mask & 2) != 0:
        joint_target_vel[global_dof_idx] = in_joint_vel[world, obj_idx, dof_in_obj_idx]
        
    # Mode 3: PD 位置偏差控制 (最適合機械手臂與行走運動控制)
    if (mask & 4) != 0:
        joint_target_pos[global_dof_idx] = in_joint_pos[world, obj_idx, dof_in_obj_idx]


# =============================================================================
# ARTICULATION BODY 類實現
# =============================================================================
class ArticulationBody(BaseBody):

    def __init__(self):
        super().__init__()
        self.view_object_indices_gpus: Dict[str, wp.array] = {}
        # Newton root-body indices (local to one world), used for state.body_* access.
        self.view_body_local_indices_gpus: Dict[str, wp.array] = {}
        # High-level object indices, used only for BaseBody randomization buffers.
        self.view_random_object_local_indices_gpus: Dict[str, wp.array] = {}
        self.num_rigid_bodies_env = 0
        self.num_joint_dofs_env = 0
        self.random_transforms_gpus: Dict[str, wp.array] = {}
        self.view_reset_mask_gpus: Dict[str, wp.array] = {}
        self.inspector_root_transforms_gpus: Dict[str, wp.array] = {}
        self.inspector_root_vel_gpus: Dict[str, wp.array] = {}
        self.inspector_teleport_mask_gpus: Dict[str, wp.array] = {}

        # 剛體控制 Buffer (Cartesian Space)
        self.control_pos_gpus: Dict[str, wp.array] = {}
        self.control_rot_gpus: Dict[str, wp.array] = {}
        self.control_vel_gpus: Dict[str, wp.array] = {}
        self.control_omega_gpus: Dict[str, wp.array] = {}
        self.control_force_gpus: Dict[str, wp.array] = {}
        self.control_torque_gpus: Dict[str, wp.array] = {}
        self.control_mask_gpus: Dict[str, wp.array] = {}

        # 關節馬達控制 Buffer (Joint Space)
        self.view_joint_dof_indices_gpus: Dict[str, wp.array] = {}
        self.view_joint_has_free_gpus: Dict[str, wp.array] = {}  # 儲存虛擬基座關節掩碼
        self.control_joint_torque_gpus: Dict[str, wp.array] = {}
        self.control_joint_vel_gpus: Dict[str, wp.array] = {}
        self.control_joint_pos_gpus: Dict[str, wp.array] = {}
        self.control_joint_mask_gpus: Dict[str, wp.array] = {}

        # 機器人特徵解析快取
        self.control_joint_scales_gpus: Dict[str, wp.array] = {}
        self.control_joint_nominal_qs_gpus: Dict[str, wp.array] = {}
        self.control_joint_limits_max_gpus: Dict[str, wp.array] = {}
        self.control_joint_limits_min_gpus: Dict[str, wp.array] = {}
        self.control_joint_dof_offsets_gpus: Dict[str, wp.array] = {}
        self.control_joint_rl_mask_gpus: Dict[str, wp.array] = {}
        self.control_joint_rl_action_indices_gpus: Dict[str, wp.array] = {}
        self.control_joint_target_mode_gpus: Dict[str, wp.array] = {}
        self.control_joint_labels: Dict[str, List[str]] = {}
        self.control_rl_action_dim: Dict[str, int] = {}

        # 快取一開始剛完成加載時的物理預設關節角度 (Shape: world, count, dof)
        self.default_joint_positions_gpus: Dict[str, wp.array] = {}

    def add_object(self, 
                   label: str, 
                   index: int, 
                   default_position: List[Any], 
                   default_rotation: List[Any]
                   ):
        super().add_object(label=label, index=index, default_position=default_position, default_rotation=default_rotation)

    def build_view(self, device, model: Model, num_objects_env):
        super().build_view(device=device, model=model, num_objects_env=num_objects_env)
        self.model = model  # 快取 model 供後續控制使用
        self.num_rigid_bodies_env = 0
        self.num_joint_dofs_env = 0

        articulation_start_np = model.articulation_start.numpy()
        joint_qd_start_np = model.joint_qd_start.numpy()
        joint_type_np = model.joint_type.numpy()
        joint_child_np = model.joint_child.numpy()

        for pattern in self.patterns:
            p = f"{pattern}_articulation"
            print("[ArticulationBody] pattern: ", p)
            view = ArticulationView(
                model=model, 
                pattern=f"{pattern}_articulation", 
                exclude_joint_types=[newton.JointType.FREE]  # 排除虛擬基座關節
            )
            self.views.append(view)

            # [A. 剛體部分]
            registered_count = len(self.patterns[pattern])
            if (
                registered_count != view.count_per_world
                or len(self.patterns_local_indices[pattern]) != view.count_per_world
            ):
                raise ValueError(
                    f"Registered object count does not match ArticulationView for '{pattern}': "
                    f"registered={registered_count}, view={view.count_per_world}"
                )

            # Body-space controls target the root body. This is also valid for a
            # cube/sphere whose only FREE root joint was excluded from the view.
            bodies_per_world = view.count_per_world
            bodies_per_object = 1
            shape = (view.world_count, view.count_per_world, bodies_per_object)

            articulation_ids_cpu = np.asarray(view.articulation_ids.numpy())
            expected_articulation_shape = (view.world_count, view.count_per_world)
            if articulation_ids_cpu.shape != expected_articulation_shape:
                raise ValueError(
                    f"Unexpected articulation_ids shape for '{pattern}': "
                    f"{articulation_ids_cpu.shape}, expected {expected_articulation_shape}"
                )

            root_body_local_indices = []
            for obj_idx, articulation_id_value in enumerate(articulation_ids_cpu[0]):
                articulation_id = int(articulation_id_value)
                start_joint = int(articulation_start_np[articulation_id])
                root_body_idx = int(joint_child_np[start_joint])
                if not 0 <= root_body_idx < model.body_count:
                    raise IndexError(
                        f"Root body index out of range for '{pattern}' object {obj_idx}: "
                        f"{root_body_idx} not in [0, {model.body_count})"
                    )
                root_body_local_indices.append(root_body_idx)

            # Verify that every world has the same body layout before kernels use
            # world * stride + local_index.
            if view.world_count > 1:
                body_strides = []
                for obj_idx, articulation_id_value in enumerate(articulation_ids_cpu[1]):
                    articulation_id = int(articulation_id_value)
                    start_joint = int(articulation_start_np[articulation_id])
                    body_strides.append(
                        int(joint_child_np[start_joint]) - root_body_local_indices[obj_idx]
                    )
                if not body_strides or any(
                    stride <= 0 or stride != body_strides[0] for stride in body_strides
                ):
                    raise ValueError(
                        f"Non-uniform rigid-body world stride for '{pattern}': {body_strides}"
                    )
                pattern_body_stride = body_strides[0]
            else:
                pattern_body_stride = model.body_count

            if self.num_rigid_bodies_env == 0:
                self.num_rigid_bodies_env = pattern_body_stride
            elif view.world_count > 1 and self.num_rigid_bodies_env != pattern_body_stride:
                raise ValueError(
                    f"Inconsistent rigid-body world stride for '{pattern}': "
                    f"{pattern_body_stride} != {self.num_rigid_bodies_env}"
                )

            for world in range(view.world_count):
                for obj_idx, articulation_id_value in enumerate(articulation_ids_cpu[world]):
                    articulation_id = int(articulation_id_value)
                    start_joint = int(articulation_start_np[articulation_id])
                    actual_root = int(joint_child_np[start_joint])
                    expected_root = world * pattern_body_stride + root_body_local_indices[obj_idx]
                    if actual_root != expected_root:
                        raise ValueError(
                            f"Non-uniform root-body layout for '{pattern}' at "
                            f"world={world}, object={obj_idx}: got {actual_root}, "
                            f"expected {expected_root}"
                        )

            # 初始化 3D 剛體控制緩衝區
            self.control_pos_gpus[pattern] = wp.zeros(shape=shape, dtype=wp.vec3, device=self.device, requires_grad=GameConfig.requires_grad) # TODO 理論上坐標旋轉和速度是直接覆寫的，因爲覆寫非綫性應該是不能微分的，需要後面確認
            self.control_rot_gpus[pattern] = wp.zeros(shape=shape, dtype=wp.quat, device=self.device, requires_grad=GameConfig.requires_grad)
            self.control_vel_gpus[pattern] = wp.zeros(shape=shape, dtype=wp.vec3, device=self.device, requires_grad=GameConfig.requires_grad)
            self.control_omega_gpus[pattern] = wp.zeros(shape=shape, dtype=wp.vec3, device=self.device, requires_grad=GameConfig.requires_grad)
            self.control_force_gpus[pattern] = wp.zeros(shape=shape, dtype=wp.vec3, device=self.device, requires_grad=GameConfig.requires_grad)
            self.control_torque_gpus[pattern] = wp.zeros(shape=shape, dtype=wp.vec3, device=self.device, requires_grad=GameConfig.requires_grad)
            self.control_mask_gpus[pattern] = wp.zeros(shape=shape, dtype=wp.int32, device=self.device, requires_grad=GameConfig.requires_grad)

            # [B. 關節馬達部分]
            joint_dof_count = view.joint_dof_count
            if joint_dof_count > 0:
                dof_per_world = joint_dof_count * view.count_per_world
                joint_shape = (view.world_count, view.count_per_world, joint_dof_count)

                # 初始化 3D 關節控制緩衝區
                self.control_joint_torque_gpus[pattern] = wp.zeros(shape=joint_shape, dtype=float, device=self.device, requires_grad=GameConfig.requires_grad)
                self.control_joint_vel_gpus[pattern] = wp.zeros(shape=joint_shape, dtype=float, device=self.device, requires_grad=GameConfig.requires_grad)
                self.control_joint_pos_gpus[pattern] = wp.zeros(shape=joint_shape, dtype=float, device=self.device, requires_grad=GameConfig.requires_grad)
                self.control_joint_mask_gpus[pattern] = wp.zeros(shape=joint_shape, dtype=wp.int32, device=self.device, requires_grad=GameConfig.requires_grad)

                articulation_ids_world0 = articulation_ids_cpu[0]
                has_free_joints = []
                local_dof_indices = []
                for obj_idx in range(view.count_per_world):
                    articulation_id = int(articulation_ids_world0[obj_idx])
                    start_joint = int(articulation_start_np[articulation_id])
                    end_joint = int(articulation_start_np[articulation_id + 1])
                    start_dof = int(joint_qd_start_np[start_joint])
                    end_dof = int(joint_qd_start_np[end_joint])

                    has_free = int(joint_type_np[start_joint] == int(newton.JointType.FREE))
                    controlled_start_dof = start_dof + (6 if has_free else 0)
                    if controlled_start_dof + joint_dof_count > end_dof:
                        raise IndexError(
                            f"Joint DoF range out of articulation bounds for '{pattern}' "
                            f"object {obj_idx}: [{controlled_start_dof}, "
                            f"{controlled_start_dof + joint_dof_count}) exceeds end {end_dof}"
                        )
                    for d in range(joint_dof_count):
                        local_dof_indices.append(start_dof + d)

                    has_free_joints.append(has_free)

                if view.world_count > 1:
                    dof_strides = []
                    for obj_idx, articulation_id_value in enumerate(articulation_ids_cpu[1]):
                        articulation_id = int(articulation_id_value)
                        start_joint = int(articulation_start_np[articulation_id])
                        dof_strides.append(
                            int(joint_qd_start_np[start_joint])
                            - local_dof_indices[obj_idx * joint_dof_count]
                        )
                    if not dof_strides or any(
                        stride <= 0 or stride != dof_strides[0] for stride in dof_strides
                    ):
                        raise ValueError(
                            f"Non-uniform joint-DoF world stride for '{pattern}': {dof_strides}"
                        )
                    pattern_dof_stride = dof_strides[0]
                else:
                    pattern_dof_stride = model.joint_dof_count

                if self.num_joint_dofs_env == 0:
                    self.num_joint_dofs_env = pattern_dof_stride
                elif view.world_count > 1 and self.num_joint_dofs_env != pattern_dof_stride:
                    raise ValueError(
                        f"Inconsistent joint-DoF world stride for '{pattern}': "
                        f"{pattern_dof_stride} != {self.num_joint_dofs_env}"
                    )

                for world in range(view.world_count):
                    for obj_idx, articulation_id_value in enumerate(articulation_ids_cpu[world]):
                        articulation_id = int(articulation_id_value)
                        start_joint = int(articulation_start_np[articulation_id])
                        actual_start = int(joint_qd_start_np[start_joint])
                        expected_start = (
                            world * pattern_dof_stride
                            + local_dof_indices[obj_idx * joint_dof_count]
                        )
                        if actual_start != expected_start:
                            raise ValueError(
                                f"Non-uniform joint-DoF layout for '{pattern}' at "
                                f"world={world}, object={obj_idx}: got {actual_start}, "
                                f"expected {expected_start}"
                            )

                # 將檢測出的虛擬基座掩碼儲存並上傳至 GPU [INDEX]
                self.view_joint_has_free_gpus[pattern] = wp.array(
                    has_free_joints, dtype=wp.int32, device=self.device
                )

                self.view_joint_dof_indices_gpus[pattern] = wp.array(
                    local_dof_indices, dtype=int, device=self.device
                )

                # [C. mjlab 对齐关节 scale / nominal / RL 掩码]
                dof_offset = 0
                joint_labels: List[str] = []

                # Robot configuration is shared by identical instances; resolve
                # labels and limits from the first world/object only.
                for local_dof_idx in local_dof_indices[:joint_dof_count]:
                    offset_dof_idx = local_dof_idx
                    if has_free_joints[0] == 1:
                        offset_dof_idx += 6

                    joint_idx = 0
                    for j in range(model.joint_count):
                        start = joint_qd_start_np[j]
                        end = joint_qd_start_np[j+1] if j+1 < len(joint_qd_start_np) else model.joint_dof_count
                        if start <= offset_dof_idx < end:
                            joint_idx = j
                            break

                    label = model.joint_label[joint_idx].lower() if joint_idx < len(model.joint_label) else ""
                    joint_labels.append(label)

                default_pos_wp = view.get_dof_positions(model)
                default_qs_np = default_pos_wp.numpy()[0, 0, :joint_dof_count]
                default_qs = [float(default_qs_np[i]) for i in range(joint_dof_count)]

                (
                    scales,
                    nominal_qs,
                    limits_max,
                    limits_min,
                    rl_mask,
                    rl_indices,
                    soft_factor,
                ) = resolve_joint_arrays_for_pattern(
                    pattern,
                    joint_labels=joint_labels,
                    default_qs=default_qs,
                )

                joint_limit_upper_np = (
                    model.joint_limit_upper.numpy()
                    if model.joint_limit_upper is not None
                    else None
                )
                joint_limit_lower_np = (
                    model.joint_limit_lower.numpy()
                    if model.joint_limit_lower is not None
                    else None
                )
                joint_target_mode_np = (
                    model.joint_target_mode.numpy()
                    if model.joint_target_mode is not None
                    else None
                )
                target_modes: List[int] = []
                for i, local_dof_idx in enumerate(local_dof_indices[:joint_dof_count]):
                    offset_dof_idx = local_dof_idx
                    if has_free_joints[0] == 1:
                        offset_dof_idx += 6

                    if joint_limit_upper_np is not None and joint_limit_lower_np is not None:
                        lim_u = float(joint_limit_upper_np[offset_dof_idx])
                        lim_l = float(joint_limit_lower_np[offset_dof_idx])
                        if abs(lim_u) < 1e5 and abs(lim_l) < 1e5:
                            limits_max[i] = lim_u
                            limits_min[i] = lim_l

                    limits_min[i], limits_max[i] = apply_soft_limits(
                        limits_min[i], limits_max[i], nominal_qs[i], soft_factor
                    )

                    if joint_target_mode_np is not None:
                        target_modes.append(int(joint_target_mode_np[offset_dof_idx]))
                    else:
                        target_modes.append(0)

                per_dof_rl_dim = max((idx for idx in rl_indices if idx >= 0), default=-1) + 1
                rl_action_dim = resolve_rl_action_dim_for_pattern(
                    pattern,
                    per_dof_rl_dim,
                )
                self.control_rl_action_dim[pattern] = rl_action_dim
                self.control_joint_labels[pattern] = list(joint_labels)

                self.control_joint_scales_gpus[pattern] = wp.array(scales, dtype=float, device=self.device)
                self.control_joint_nominal_qs_gpus[pattern] = wp.array(nominal_qs, dtype=float, device=self.device)
                self.control_joint_limits_max_gpus[pattern] = wp.array(limits_max, dtype=float, device=self.device)
                self.control_joint_limits_min_gpus[pattern] = wp.array(limits_min, dtype=float, device=self.device)
                self.control_joint_rl_mask_gpus[pattern] = wp.array(rl_mask, dtype=wp.int32, device=self.device)
                self.control_joint_rl_action_indices_gpus[pattern] = wp.array(rl_indices, dtype=wp.int32, device=self.device)
                self.control_joint_target_mode_gpus[pattern] = wp.array(
                    target_modes, dtype=wp.int32, device=self.device
                )
                self.control_joint_dof_offsets_gpus[pattern] = wp.array([dof_offset], dtype=int, device=self.device)
                self.default_joint_positions_gpus[pattern] = wp.clone(default_pos_wp)

                print(f"[ArticulationBody] {pattern}: rl_action_dim={rl_action_dim}, total_dof={joint_dof_count}")

            # 快取重置資訊
            local_indices = self.patterns[pattern]
            self.view_object_indices_gpus[pattern] = wp.array(
                local_indices, dtype=int, device=self.device
            )

            self.view_body_local_indices_gpus[pattern] = wp.array(
                root_body_local_indices, dtype=int, device=self.device
            )
            self.view_random_object_local_indices_gpus[pattern] = wp.array(
                self.patterns_local_indices[pattern], dtype=int, device=self.device
            )

            # 快取隨機輸出位姿緩衝區
            self.random_transforms_gpus[pattern] = wp.empty(
                shape=(view.world_count, view.count_per_world), 
                dtype=wp.transform, 
                device=self.device
            )

            # 快取 2D 遮罩緩衝區 
            self.view_reset_mask_gpus[pattern] = wp.zeros(
                shape=(view.world_count, view.count_per_world), 
                dtype=bool, 
                device=self.device
            )
            self.inspector_root_transforms_gpus[pattern] = wp.empty(
                shape=(view.world_count, view.count_per_world),
                dtype=wp.transform,
                device=self.device,
            )
            self.inspector_root_vel_gpus[pattern] = wp.empty(
                shape=(view.world_count, view.count_per_world),
                dtype=wp.spatial_vector,
                device=self.device,
            )
            self.inspector_teleport_mask_gpus[pattern] = wp.zeros(
                shape=(view.world_count, view.count_per_world),
                dtype=bool,
                device=self.device,
            )

    def _apply_inspector_root_teleport(
        self,
        view: ArticulationView,
        pattern: str,
        state: State,
        body_q_prev,
        solver_body_q_prev,
    ):
        """對浮動基座關節體套用 Inspector 釘選的位姿/速度（與 reset 相同路徑，純 GPU mask）。"""
        if view.joint_count <= 0 or pattern not in self.view_joint_has_free_gpus:
            return

        bodies_per_world = view.count_per_world
        bodies_per_object = 1
        base_body_in_obj_idx = 0
        teleport_mask = self.inspector_teleport_mask_gpus[pattern]

        wp.launch(
            kernel=build_inspector_root_teleport_kernel,
            dim=(view.world_count, view.count_per_world),
            inputs=[
                state.body_q,
                state.body_qd,
                self.view_body_local_indices_gpus[pattern],
                self.num_rigid_bodies_env,
                bodies_per_world,
                bodies_per_object,
                base_body_in_obj_idx,
                self.view_joint_has_free_gpus[pattern],
                self.control_pos_gpus[pattern],
                self.control_rot_gpus[pattern],
                self.control_vel_gpus[pattern],
                self.control_omega_gpus[pattern],
                self.control_mask_gpus[pattern],
                self.inspector_root_transforms_gpus[pattern],
                self.inspector_root_vel_gpus[pattern],
                teleport_mask,
            ],
            device=self.device,
        )

        view.set_root_transforms(
            state,
            self.inspector_root_transforms_gpus[pattern],
            mask=teleport_mask,
        )
        view.set_root_velocities(
            state,
            self.inspector_root_vel_gpus[pattern],
            mask=teleport_mask,
        )
        view.eval_fk(state, mask=teleport_mask)

        if body_q_prev is not None and solver_body_q_prev is not None:
            wp.launch(
                kernel=sync_articulation_body_q_prev_kernel,
                dim=bodies_per_world * view.world_count,
                inputs=[
                    state.body_q,
                    body_q_prev,
                    solver_body_q_prev,
                    self.view_body_local_indices_gpus[pattern],
                    self.num_rigid_bodies_env,
                    bodies_per_world,
                    bodies_per_object,
                    teleport_mask,
                ],
                device=self.device,
            )

    def apply_controls(self, state: State, control, body_q_prev, solver_body_q_prev, dt, linear_damping, angular_damping):
        """
        呼叫並行控制 Kernel 同時更新剛體和關節馬達控制 (保持自動微分傳導)
        """
        for i, view in enumerate(self.views):
            pattern = list(self.patterns.keys())[i]

            self._apply_inspector_root_teleport(
                view=view,
                pattern=pattern,
                state=state,
                body_q_prev=body_q_prev,
                solver_body_q_prev=solver_body_q_prev,
            )

            # 執行剛體級（Body Space）笛卡爾更新
            bodies_per_world = view.count_per_world
            bodies_per_object = 1
            total_body_count = bodies_per_world * view.world_count
            
            wp.launch(
                kernel=apply_articulation_updates_kernel,
                dim=total_body_count,
                inputs=[
                    state.body_q, body_q_prev, solver_body_q_prev,
                    state.body_qd, state.body_f,
                    self.view_body_local_indices_gpus[pattern],
                    self.num_rigid_bodies_env,
                    bodies_per_world,
                    bodies_per_object,
                    self.control_pos_gpus[pattern],
                    self.control_rot_gpus[pattern],
                    self.control_vel_gpus[pattern],
                    self.control_omega_gpus[pattern],
                    self.control_force_gpus[pattern],
                    self.control_torque_gpus[pattern],
                    self.control_mask_gpus[pattern],
                    dt,
                    linear_damping,
                    angular_damping,
                    self.model.body_inv_mass
                ],
                device=self.device
            )
            
            # 執行馬達級（Joint Space）關節控制更新
            if view.joint_dof_count > 0:
                joint_dof_count = view.joint_dof_count
                dof_per_world = joint_dof_count * view.count_per_world
                total_dof_count = dof_per_world * view.world_count
                
                # 計算單個環境中的總關節自由度
                num_joint_dof_env = self.num_joint_dofs_env
                
                # 呼叫擴展後的 3D 關節驅動 (並傳入虛擬基座檢測掩碼) [INDEX]
                wp.launch(
                    kernel=apply_joint_actuation_kernel,
                    dim=total_dof_count,
                    inputs=[
                        control.joint_f, control.joint_target_vel, control.joint_target_pos,
                        self.view_joint_dof_indices_gpus[pattern],
                        num_joint_dof_env,
                        dof_per_world,
                        joint_dof_count,
                        self.control_joint_torque_gpus[pattern],
                        self.control_joint_vel_gpus[pattern],
                        self.control_joint_pos_gpus[pattern],
                        self.control_joint_mask_gpus[pattern],
                        self.view_joint_has_free_gpus[pattern]
                    ],
                    device=self.device
                )

    def clear_controls(self):
        """
        重置控制緩衝區，確保每幀輸入是瞬時力，避免殘留
        """
        for pattern in self.patterns:
            # 清除剛體控制
            self.control_mask_gpus[pattern].zero_()
            self.control_pos_gpus[pattern].zero_()
            self.control_rot_gpus[pattern].zero_()
            self.control_vel_gpus[pattern].zero_()
            self.control_omega_gpus[pattern].zero_()
            self.control_force_gpus[pattern].zero_()
            self.control_torque_gpus[pattern].zero_()
            
            # 清除關節馬達控制
            if pattern in self.control_joint_mask_gpus:
                self.control_joint_mask_gpus[pattern].zero_()
                self.control_joint_torque_gpus[pattern].zero_()
                self.control_joint_vel_gpus[pattern].zero_()
                self.control_joint_pos_gpus[pattern].zero_() # 清除位置緩衝區

    def reset_obj(self, state: State, control, reset_mask_gpu, offset_random_gpu, seed):
        """
        利用 View 安全且高並行地在 GPU 上進行隨機化與姿態重置
        """
        for i, view in enumerate(self.views):
            pattern = list(self.patterns.keys())[i]

            # 這裏有問題，很可能是這個 Kernel 導致非法内存存取
            # 呼叫隨機 Kernel：生成隨機世界坐標
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
                    self.view_random_object_local_indices_gpus[pattern],
                    self.num_objects_env,
                    self.num_body_object_env,

                    seed,
                    offset_random_gpu,
                ],
                outputs=[
                    self.random_transforms_gpus[pattern],
                    self.view_reset_mask_gpus[pattern],
                ],
                device=self.device,
            )

            # FREE root is excluded from the view's joint selection, so
            # joint_count may be 0 while is_floating_base is still True.
            # Always use set_root_transforms for floating bases (writes joint_q[0:7]).
            if view.is_floating_base or view.joint_count > 0:
                if view.is_floating_base:
                    view.set_root_transforms(
                        state,
                        self.random_transforms_gpus[pattern],
                        mask=self.view_reset_mask_gpus[pattern],
                    )
                    zero_root_vel = wp.zeros(
                        shape=(view.world_count, view.count_per_world),
                        dtype=wp.spatial_vector,
                        device=self.device,
                    )
                    view.set_root_velocities(
                        state, zero_root_vel, mask=self.view_reset_mask_gpus[pattern]
                    )

                if view.joint_dof_count > 0:
                    view.set_dof_positions(
                        state,
                        self.default_joint_positions_gpus[pattern],
                        mask=self.view_reset_mask_gpus[pattern],
                    )
                    zero_dof_vel = wp.zeros(
                        shape=(view.world_count, view.count_per_world, view.joint_dof_count),
                        dtype=float,
                        device=self.device,
                    )
                    view.set_dof_velocities(
                        state, zero_dof_vel, mask=self.view_reset_mask_gpus[pattern]
                    )

                # 重新傳導骨骼姿勢 
                view.eval_fk(state, mask=self.view_reset_mask_gpus[pattern])
                
                # 啟動並行掩碼控制重置 Kernel (只重置 Terminate 的環境，並傳入偏移掩碼) [INDEX]
                if view.joint_dof_count > 0:
                    dof_per_world = view.joint_dof_count * view.count_per_world
                    total_dof_count = dof_per_world * view.world_count
                    num_joint_dof_env = self.num_joint_dofs_env

                    wp.launch(
                        kernel=reset_control_masked_kernel,
                        dim=total_dof_count,
                        inputs=[
                            reset_mask_gpu,
                            self.view_object_indices_gpus[pattern],
                            self.num_objects_env,
                            self.view_joint_dof_indices_gpus[pattern],
                            num_joint_dof_env,
                            
                            view.world_count,
                            view.count_per_world,
                            view.joint_dof_count,
                            control.joint_target_pos,
                            control.joint_target_vel,
                            control.joint_f,
                            control.joint_act,
                            
                            self.model.joint_target_pos,
                            self.model.joint_target_vel,
                            self.view_joint_has_free_gpus[pattern],
                        ],
                        device=self.device,
                    )

            else:
                # 基礎剛體（如方塊、球體）的重置維持不變
                reshaped_transforms = self.random_transforms_gpus[pattern].reshape(
                    (view.world_count, view.count_per_world, 1)
                )
                view.set_attribute(
                    "body_q", state, reshaped_transforms, mask=self.view_reset_mask_gpus[pattern]
                )

                zero_body_vel = wp.zeros(
                    shape=(view.world_count, view.count_per_world, 1),
                    dtype=wp.spatial_vector,
                    device=self.device,
                )
                view.set_attribute(
                    "body_qd", state, zero_body_vel, mask=self.view_reset_mask_gpus[pattern]
                )

@wp.kernel
def reset_control_masked_kernel(
    reset_mask: wp.array(dtype=wp.int32),         # 全域一維重置遮罩 (body_count,)
    view_object_indices: wp.array(dtype=int),     # 本 View 的物件局部索引 (count_per_world,)
    num_objects_env: int,                         # 單環境物件總數
    view_joint_dof_indices: wp.array(dtype=int),  # 關節局部索引
    num_joint_dof_env: int,                       # 單環境關節自由度總數
    
    world_count: int,
    count_per_world: int,
    joint_dof_count: int,
    
    # Newton 全局控制目標
    control_joint_target_pos: wp.array(dtype=float),
    control_joint_target_vel: wp.array(dtype=float),
    control_joint_f: wp.array(dtype=float),
    control_joint_act: wp.array(dtype=float),
    
    # Newton 模型物理預設目標
    model_joint_target_pos: wp.array(dtype=float),
    model_joint_target_vel: wp.array(dtype=float),
    
    view_joint_has_free: wp.array(dtype=wp.int32),  # 判斷物件是否有虛擬基座關節的一維掩碼
):
    tid = wp.tid() 
    
    dof_per_world = count_per_world * joint_dof_count
    world = tid // dof_per_world
    local_tid = tid % dof_per_world
    obj_idx = local_tid // joint_dof_count
    
    # 對齊一維全域重置索引，判斷該特定環境是否需要重置 
    local_idx = view_object_indices[obj_idx]
    flat_idx_env = world * num_objects_env + local_idx
    
    if reset_mask[flat_idx_env] == 1:
        # 僅在環境真正 Terminate 時，才將該環境的馬達控制目標與力矩歸零/恢復預設
        local_dof_idx = view_joint_dof_indices[local_tid]
        global_dof_idx = world * num_joint_dof_env + local_dof_idx
        
        # 若此物件有虛擬基座關節（FREE），則加上 6 DoFs 的偏移量 [INDEX]
        if view_joint_has_free[obj_idx] == 1:
            global_dof_idx = global_dof_idx + 6
        
        control_joint_target_pos[global_dof_idx] = model_joint_target_pos[global_dof_idx]
        control_joint_target_vel[global_dof_idx] = model_joint_target_vel[global_dof_idx]
        control_joint_f[global_dof_idx] = 0.0
        control_joint_act[global_dof_idx] = 0.0