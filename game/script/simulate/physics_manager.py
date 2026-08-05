import newton
import warp as wp
import numpy as np
import newton.solvers

from script.simulate.mesh_builder import MeshBuilder
from script.role.objects.base_object import ObjectRegistry
from script.simulate.solvers.base_solver import SolverRegistry
from script.game_config import GameConfig
from script.exceptions import GameClosedException
from script.sensors.contact_sensor import ContactSensor, build_shape_to_role_map
from script.simulate.coupling_index_builder import CouplingIndexBuilder

from script.role.bodies.articulation_body import ArticulationBody
from script.role.bodies.deformable_body import DeformableBody
from utils.warp_math import suspend_tape

from typing import Callable, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from game.script.custom_viewergl import CustomViewerGL


# =============================================================================
# PHYSICS MANAGER CLASS
# =============================================================================
class PhysicsManager:

    def __init__(self, device, viewerGL: 'CustomViewerGL'=None):
        self.builder = newton.ModelBuilder()
        self.builder_env = newton.ModelBuilder()
        self.device = wp.get_device(device) # TODO Hard code
        wp.set_device(device)

        self.mesh_builder = MeshBuilder()
        
        # 🌟 由外部傳入或初始化的三大物理大類管理對象
        self.articulation_body: ArticulationBody
        self.deformable_body: DeformableBody

        # Add a ground plane (infinite static plane at z=0)
        self.builder.add_ground_plane()
        self.is_finalized = False

        self.body_shape_types = []
        self.body_size = []
        self.mass_default_list = [] 

        # 用來暫存初始材質參數的列表 (friction, elasticity)
        self.initial_materials = [] 

        self.viewerGL = viewerGL

        # Cuda graph capture is not allowed on the first run of simulate(). 
        # The first run includes parameter resets and initialization operations; the normal simulation only begins on the second run. 
        # Capturing on the first run will cause all objects to become misaligned and the system to freeze.
        self.current_step = 0
        self.capture_graph_after_step = 1
        self.pre_substep_callback: Optional[Callable[[int], None]] = None
        self.inspector_body_f = None

        self.contact_sensor: ContactSensor | None = None
        self._role_shape_ranges: list[tuple[int, int, int]] = []
        self._template_shape_count = 0

        self.object_metadata: dict[str, dict] = {}
        self.object_metadata_by_role: dict[int, dict] = {}
        self.role_object_labels: dict[int, str] = {}
        self.shape_to_role_np: np.ndarray | None = None
        self.num_objects_env: int = 0

        # 耦合求解器索引收集器（僅在 solver 為 "coupled" 時使用）
        self._coupling_builder = CouplingIndexBuilder()

    def add_shape(self, label: str, object_config: dict, collision_group, pos = None, rot = None, vel = None, role_object_id: int = -1):
        """
        現在傳入的 object_config 是一個 Pydantic 對象 
        """
        if self.is_finalized:
            raise RuntimeError("Cannot add shape after builded a model!")

        object_key = object_config["type"]
        object_handler = ObjectRegistry.get_handler(object_key)
        if not object_handler:
            raise ValueError(f"Object key '{object_key}' not registered!")

        cfg = self.builder_env.ShapeConfig()
        cfg.collision_group = collision_group

        # 耦合索引收集：記錄 add_physics 前的 builder_env 狀態
        body_before = self.builder_env.body_count
        joint_before = self.builder_env.joint_count
        particle_before = self.builder_env.particle_count

        shape_begin = self.builder_env.shape_count
        # 調用形狀專屬邏輯 (多型調用)
        add_result = object_handler.add_physics(builder_env=self.builder_env, label=label, data=object_config, cfg=cfg, pos=pos, rot=rot, vel=vel)
        shape_end = self.builder_env.shape_count

        # 耦合索引收集：記錄 add_physics 後的 builder_env 狀態
        self._coupling_builder.record_object(
            label=label,
            body_start=body_before,
            body_end=self.builder_env.body_count,
            joint_start=joint_before,
            joint_end=self.builder_env.joint_count,
            particle_start=particle_before,
            particle_end=self.builder_env.particle_count,
        )

        if isinstance(add_result, dict):
            meta = {
                "path_body_map": dict(add_result.get("path_body_map") or {}),
                "path_joint_map": dict(add_result.get("path_joint_map") or {}),
                "path_shape_map": dict(add_result.get("path_shape_map") or {}),
                "joint_start": add_result.get("joint_start"),
                "joint_end": add_result.get("joint_end"),
            }
            self.object_metadata[label] = meta
            if role_object_id >= 0:
                self.object_metadata_by_role[role_object_id] = meta
            add_result["_path_body_map"] = meta["path_body_map"]
            index = add_result
        else:
            index = add_result

        if role_object_id >= 0:
            self.role_object_labels[role_object_id] = label

        if role_object_id >= 0 and shape_end > shape_begin:
            self._role_shape_ranges.append((shape_begin, shape_end, role_object_id))
        # print("builder_env.articulation_label[-1]: ", self.builder_env.articulation_label[-1])

        # 記錄 GPU 需要的通用數據
        self.body_shape_types.append(object_handler.object_type_id)

        # TODO 剛體物件返回的是大小，關節體未實現，軟體未實現
        shape_size = object_handler.get_size(object_config)
        self.body_size.append(shape_size)

        # # 其他通用參數
        # self.mass_default_list.append(object_config["object_mass"]) # TODO 需要改進，應該添加對 lock_inertia 的判斷，或者移動到 setup 函數直接拿最後的質量
        #                                                             # 只是暫時不會在環境運行的時候修改質量，因此暫時可以不改
        self.initial_materials.append((object_config["object_friction"], object_config["object_elasticity"]))

        return index, shape_size
    
    def add_mesh(self, object_config, position):
        
        object_key = object_config["type"]
        object_handler = ObjectRegistry.get_handler(object_key)
        if not object_handler:
            raise ValueError(f"Object key '{object_key}' not registered!")
        
        pos = wp.vec3(*position)
        object_handler.add_visual(mesh_builder=self.mesh_builder, data=object_config, pos=pos)

    def setup(self, num_env):
        # self.builder.shape_collision_filter_pairs.append((1, 61)) # +1 是因為 0 是地板

        solver_config = GameConfig.SOLVER_CONFIG
        solver_key = solver_config["type"].lower()
        solver_handler_cls = SolverRegistry.get_handler(solver_key)
        if not solver_handler_cls:
            raise ValueError(f"Solver key '{solver_key}' not registered!")

        # MPM 求解器需要額外的 per-particle 材質/狀態自訂屬性 (mpm: namespace)。
        # 必須在 add_world / finalize 之前註冊，屬性會隨 builder 合併傳播。
        from script.simulate.solvers.mpm import (
            filter_mpm_collider_shapes,
            solver_requires_mpm_attributes,
        )
        from newton.solvers import SolverImplicitMPM

        if solver_requires_mpm_attributes(solver_config):
            SolverImplicitMPM.register_custom_attributes(self.builder_env)
            # MPM collider mesh 建構僅支援特定 GeoType；對其餘形狀（如 USD 匯入的
            # CONVEX_MESH）關閉 COLLIDE_PARTICLES，避免建立求解器時拋出 NotImplementedError。
            filter_mpm_collider_shapes(self.builder_env)
            # MPM 求解器每個 step 都會依粒子位置動態分配稀疏網格 (allocate_by_voxels)，
            # CUDA Graph 捕獲期間禁止動態記憶體分配，故含 MPM 的環境停用圖形捕獲，
            # step_CUDA_Graph 退化成直接逐幀模擬。
            self.capture_graph_after_step = 2**31 - 1

        for n in range(num_env):
            self.builder.add_world(self.builder_env)

        # 耦合求解器索引結構鎖定（在 add_world 之後、solver 創建之前）
        self._coupling_builder.finalize_structure(
            env_body_count=self.builder_env.body_count,
            env_joint_count=self.builder_env.joint_count,
            env_particle_count=self.builder_env.particle_count,
            num_env=num_env,
        )

        # Finalize the model - this creates the simulation-ready Model object
        self.solver_handler = solver_handler_cls(
            config=solver_config,
            builder=self.builder,
            coupling_builder=self._coupling_builder,
        )
        self.model = self.builder.finalize(device=self.device)

        self.mesh = self.mesh_builder.finalize(device=self.device)
        print(f"Model finalized for device {self.model.device}:")
        print(f"  Bodies: {self.model.body_count}")
        print(f"  Shapes: {self.model.shape_count}")
        print(f"  Joints: {self.model.joint_count}")
        self.is_finalized = True

        self.body_shape_types = self.body_shape_types * num_env
        self.body_size = self.body_size * num_env
        self.mass_default_list = self.mass_default_list * num_env
        self.initial_materials = self.initial_materials * num_env

        self._template_shape_count = self.builder_env.shape_count
        num_roles = GameConfig.NUM_OBJECTS_TOTAL
        num_objects_env = num_roles // num_env if num_env > 0 else num_roles
        shape_to_role_np = build_shape_to_role_map(
            shape_count=self.model.shape_count,
            role_shape_ranges=self._role_shape_ranges,
            template_shape_count=self._template_shape_count,
            num_env=num_env,
            num_objects_env=num_objects_env,
        )

        nconmax = int(solver_config.get("nconmax", 150) or 150)
        raw_capacity = max(nconmax * GameConfig.SUB_STEPS, 256) * 2

        self.shape_to_role_np = shape_to_role_np
        self.num_objects_env = num_objects_env

        self.contact_sensor = ContactSensor(
            num_roles=num_roles,
            shape_count=self.model.shape_count,
            shape_to_role_np=shape_to_role_np,
            raw_capacity=raw_capacity,
            device=self.device,
        )

        # Setting gravity and damping
        gravity = [wp.vec3(*self.gravity) for i in range(num_env)]
        self.model.gravity = wp.array(gravity, dtype=wp.vec3, device=self.device)
        print("gravity: ", self.model.gravity)
        print("linear_damping: ", self.linear_damping)
        print("angular_damping: ", self.angular_damping)
        print("ground_friction: ", GameConfig.GROUND_FRICTION)

        self.solver_handler.setup(self.model)
        print(f"Solver created: {type(self.solver_handler.solver).__name__}")

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        # Create two state objects for time integration
        self.state_0 = self.model.state(requires_grad=GameConfig.requires_grad)  # Current state
        self.state_1 = self.model.state(requires_grad=GameConfig.requires_grad)  # Next state

        self.state_default = self.model.state() # For reset env
        self.mass_default = wp.array(data=self.mass_default_list, dtype=wp.float32, device=self.device)

        if self.state_0.body_q is not None:
            n = len(self.state_0.body_q)
            # 手動建立並賦值
            self.state_0.body_q_prev = wp.zeros(n, dtype=wp.transform, device=self.device)
            self.state_1.body_q_prev = wp.zeros(n, dtype=wp.transform, device=self.device)
            self.solver_body_q_prev = self.solver_handler.body_q_prev if self.solver_handler.body_q_prev is not None else self.state_0.body_q_prev
            self.state_default.body_q_prev = wp.zeros(n, dtype=wp.transform, device=self.device)

            # 初始化：讓 prev 等於當前坐標，防止第一幀速度爆炸
            wp.copy(self.state_0.body_q_prev, self.state_0.body_q)
            wp.copy(self.state_1.body_q_prev, self.state_1.body_q)
            wp.copy(self.state_default.body_q_prev, self.state_default.body_q)

            print("Successfully manually allocated body_q_prev.")
            self.inspector_body_f = wp.zeros(n, dtype=wp.spatial_vector, device=self.device)

        # Set the model (this logs the static geometry)
        if self.viewerGL is not None:
            self.viewerGL.set_model(self.model)

        # The control object is not used in this example, but we create it for completeness
        self.control = self.model.control()

        self.graph = None

        # Simulation parameters
        fps = GameConfig.FPS_ACTION  # Frames per second for visualization
        self.frame_dt = 1.0 / fps  # Time step per frame
        self.sim_substeps = GameConfig.SUB_STEPS  # Number of physics substeps per frame
        self.sim_dt = self.frame_dt / self.sim_substeps  # Physics time step
        print("Simulation configured:")
        print(f"  Frame rate: {fps} Hz")
        print(f"  Frame dt: {self.frame_dt:.4f} s")
        print(f"  Physics substeps: {self.sim_substeps}")
        print(f"  Physics dt: {self.sim_dt:.4f} s")

        # 建立材質 GPU 緩衝區
        n = self.model.shape_count

        # 將暫存的 Python List 轉為 Numpy Array
        init_mats = np.array(self.initial_materials, dtype=np.float32) # shape (N, 2)

        full_friction_np = np.zeros(n, dtype=np.float32)
        full_elasticity_np = np.zeros(n, dtype=np.float32)

        # 2. 設定第一位 (索引 0) 為地面的材質
        full_friction_np[0] = GameConfig.GROUND_FRICTION
        full_elasticity_np[0] = 0.5  # 假設地面的彈性係數，可依需求修改

        if len(init_mats) > 0:
            # 確保填入的長度不會超過 buffer 總長度
            num_to_fill = min(len(init_mats), n - 1)
            full_friction_np[1 : 1 + num_to_fill] = init_mats[:num_to_fill, 0]
            full_elasticity_np[1 : 1 + num_to_fill] = init_mats[:num_to_fill, 1]

            # 4. 將處理好的 Numpy 資料一次性上傳到 GPU 緩衝區
            self.buf_friction = wp.array(full_friction_np, dtype=wp.float32, device=self.device)
            self.buf_elasticity = wp.array(full_elasticity_np, dtype=wp.float32, device=self.device)

            wp.launch(
                kernel=apply_materials_kernel,
                dim=self.model.shape_count,
                inputs=[
                    self.model.shape_material_mu,
                    self.buf_friction
                ],
                device=self.device
            )

        self.body_shape_types_gpu = wp.array(data=self.body_shape_types, dtype=wp.int32, device=self.device)
        self.body_size_gpu = wp.array(data=self.body_size, dtype=wp.vec3, device=self.device)

        object_count = self.articulation_body.num_body_objects_total + self.deformable_body.num_body_objects_total
        self.reset_mask_gpu = wp.zeros(shape=object_count, dtype=wp.int32, device=self.device)
        self.offset_random_gpu = wp.zeros(shape=num_env, dtype=wp.int32, device=self.device)

    def simulate(self):
        """Run multiple physics substeps for one frame."""
        self.contact_sensor.reset_frame()

        for substep_idx in range(self.sim_substeps):
            # Clear forces in input state
            self.state_0.clear_forces()

            if self.viewerGL is not None:
                self.viewerGL.apply_forces(self.state_0)

            if self.pre_substep_callback is not None:
                self.pre_substep_callback(substep_idx)

            if self.inspector_body_f is not None and substep_idx == 0:
                wp.launch(
                    add_inspector_body_forces_kernel,
                    dim=len(self.state_0.body_f),
                    inputs=[self.state_0.body_f, self.inspector_body_f],
                    device=self.device,
                )

            # Apply articulation-level updates
            self.articulation_body.apply_controls(
                state=self.state_0,
                control=self.control,
                body_q_prev=self.state_0.body_q_prev,
                solver_body_q_prev=self.solver_body_q_prev,
                dt=self.sim_dt,
                linear_damping=self.linear_damping,
                angular_damping=self.angular_damping
            )

            # Apply deformable-level updates
            self.deformable_body.apply_controls(
                state=self.state_0,
                dt=self.sim_dt,
                linear_damping=self.linear_damping
            )

            # Detect collisions
            self.collision_pipeline.collide(self.state_0, self.contacts)

            # 記錄當前 sub-step 的所有碰撞
            self.contact_sensor.record_rigid_contacts(self.contacts)

            self.solver_handler.step(
                state_in=self.state_0,
                state_out=self.state_1,
                control=self.control,
                contacts=self.contacts,
                dt=self.sim_dt
            )
            # Swap states (next becomes current)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # 清除本幀控制緩衝區
        with suspend_tape(): # TODO 並不確定是否真的起作用，有沒有這行實際上并不影響梯度流通，但這有可能會把不需要錄制的代碼排除在外進而優化性能，需要測試
            self.articulation_body.clear_controls()
            self.deformable_body.clear_controls()


    @property
    def collision_matrix(self):
        """Backward-compat alias for role-level contact matrix."""
        return self.contact_sensor.role_contact_matrix

    @property
    def ground_contact_flags(self):
        return self.contact_sensor.ground_contact_flags
    def set_env_params(self, gravity, damping):
        self.gravity = gravity
        self.linear_damping = damping[0]
        self.angular_damping = damping[1]

    def set_runtime_gravity(self, gravity):
        from newton import ModelFlags

        self.gravity = list(gravity)
        self.model.set_gravity(tuple(gravity))
        if self.solver_handler is not None and self.solver_handler.solver is not None:
            self.solver_handler.solver.notify_model_changed(ModelFlags.MODEL_PROPERTIES)

    def read_runtime_gravity(self) -> list[float]:
        gravity_np = self.model.gravity.numpy()
        if gravity_np.ndim >= 1 and len(gravity_np) > 0:
            vec = gravity_np[0]
            return [float(vec[0]), float(vec[1]), float(vec[2])]
        if isinstance(self.gravity, (list, tuple)):
            return [float(self.gravity[0]), float(self.gravity[1]), float(self.gravity[2])]
        return gravity_np

    def cleanup(self):
        if self.viewerGL and hasattr(self.viewerGL, 'renderer'):
            self.viewerGL.close()
        
        # 2. 清理 CUDA Graph (雖然進程結束會自動釋放，但這是一個好習慣)
        self.graph = None

    # CPU Action, only for debugging
    def check_collision(self, role_a, role_b):
        return self.contact_sensor.check_role_collision(role_a, role_b)

    def reset_obj(self):
        """
        委託 ArticulationBody 和 DeformableBody 自行完成高階重置

        Suspend the active Warp tape: APG records ``step_Diff`` (which calls
        this) inside ``wp.Tape``, but discrete teleports must not be differentiated.
        """
        with suspend_tape(): # TODO 並不確定是否真的起作用，有沒有這行實際上并不影響梯度流通，但這有可能會把不需要錄制的代碼排除在外進而優化性能，需要測試
            # 委託 Articulation 關節體進行 GPU 並行隨機重置
            self.articulation_body.reset_obj(
                state=self.state_0,
                control=self.control,  # 傳遞全局控制對象
                reset_mask_gpu=self.reset_mask_gpu,
                offset_random_gpu=self.offset_random_gpu,
                seed=GameConfig.SEED,
            )

            # 委託 Deformable 軟體進行 GPU 並行隨機重置
            self.deformable_body.reset_obj(
                state=self.state_0,
                reset_mask_gpu=self.reset_mask_gpu,
                offset_random_gpu=self.offset_random_gpu,
                seed=GameConfig.SEED,
                state_alt=self.state_1,
            )

            # 雙緩衝剛體狀態也必須與 state_0 對齊，否則 CUDA Graph 重放時可能讀到舊 body_q。
            if self.state_0.body_q is not None and self.state_1.body_q is not None:
                wp.copy(self.state_1.body_q, self.state_0.body_q)
                wp.copy(self.state_1.body_qd, self.state_0.body_qd)

            # 關節體（如 free-joint 剛體）的位姿存放在 joint_q，同樣需要雙緩衝對齊
            if self.state_0.joint_q is not None and self.state_1.joint_q is not None:
                wp.copy(self.state_1.joint_q, self.state_0.joint_q)
                if self.state_1.joint_qd is not None:
                    wp.copy(self.state_1.joint_qd, self.state_0.joint_qd)

            # 一鍵清空環境重置遮罩，防子步驟重置信號漏失
            self.reset_mask_gpu.zero_()

            # state.body_q_prev 是 VBD 等求解器計算速度的回退來源；teleport 後若不同步，
            # 下一幀會把位姿差誤當成速度，表現為剛體突然加速。
            if self.state_0.body_q is not None and self.state_0.body_q_prev is not None:
                wp.copy(self.state_0.body_q_prev, self.state_0.body_q)
                if self.state_1.body_q_prev is not None:
                    wp.copy(self.state_1.body_q_prev, self.state_0.body_q)

            # 物理求解器後置位姿同步與歷史快取清理
            self.solver_handler.post_teleport_sync(self.state_0)
            self.solver_handler.reset_history()

    def clear_inspector_body_f(self):
        if self.inspector_body_f is not None:
            self.inspector_body_f.zero_()

    # =============================================================================
    # WARP KERNEL: Omnipotent state update kernel
    # =============================================================================
    @wp.kernel
    def apply_body_updates_kernel(
        body_q: wp.array(dtype=wp.transform),
        body_q_prev: wp.array(dtype=wp.transform),
        solver_body_q_prev: wp.array(dtype=wp.transform),
        
        body_qd: wp.array(dtype=wp.spatial_vector),
        body_f: wp.array(dtype=wp.spatial_vector),
        dt: float,
        linear_damping: float,
        angular_damping: float,

        body_inv_mass: wp.array(dtype=float),       # 質量倒數 (Newton 內部使用)

        # --- 緩衝區輸入 ---
        in_pos: wp.array(dtype=wp.vec3),
        in_rot: wp.array(dtype=wp.quat),
        in_vel: wp.array(dtype=wp.vec3),
        in_omega: wp.array(dtype=wp.vec3),
        in_force: wp.array(dtype=wp.vec3),
        in_torque: wp.array(dtype=wp.vec3),
        action_mask: wp.array(dtype=wp.int32)
    ):
        tid = wp.tid()

        mask = action_mask[tid]
        
        # --- 阻尼 (連續函數，保留梯度) ---
        qd = body_qd[tid]
        l_fact = wp.exp(-linear_damping * dt)
        a_fact = wp.exp(-angular_damping * dt)
        
        # 局部更新速度變量
        curr_v = wp.vec3(qd[0] * l_fact, qd[1] * l_fact, qd[2] * l_fact)
        curr_w = wp.vec3(qd[3] * a_fact, qd[4] * a_fact, qd[5] * a_fact)

        # --- 離散傳送 (斷開梯度，用於重置) ---
        # 定義：1:Pos, 2:Rot, 4:Vel, 8:Omega
        if (mask & 1) != 0 or (mask & 2) != 0:
            p = body_q[tid].p
            q = body_q[tid].q
            if (mask & 1) != 0: p = in_pos[tid]
            if (mask & 2) != 0: q = in_rot[tid]
            
            new_xform = wp.transform(p, q)
            body_q[tid] = new_xform
            # 防止 XPBD 產生巨大的瞬間速度
            body_q_prev[tid] = new_xform 
            solver_body_q_prev[tid] = new_xform
        
        # --- 速度覆寫 ---
        if (mask & 4) != 0:
            curr_v = in_vel[tid]
        if (mask & 8) != 0:
            curr_w = in_omega[tid]

        # 最後統一寫回速度
        body_qd[tid] = wp.spatial_vector(
            curr_v[0], curr_v[1], curr_v[2], 
            curr_w[0], curr_w[1], curr_w[2]
        )

        if body_inv_mass[tid] == 0.0: 
            return
        
        # --- 連續力/力矩 (BPTT 優化核心) ---
        # 不使用 mask 判斷，讓網絡輸出 0 來代表不施力，確保梯度流暢
        f = in_force[tid]
        t = in_torque[tid]
        curr_f = body_f[tid]
        body_f[tid] = wp.spatial_vector(
            curr_f[0] + f[0], curr_f[1] + f[1], curr_f[2] + f[2],
            curr_f[3] + t[0], curr_f[4] + t[1], curr_f[5] + t[2]
        )


@wp.kernel
def add_inspector_body_forces_kernel(
    body_f: wp.array(dtype=wp.spatial_vector),
    inspector_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    curr = body_f[tid]
    extra = inspector_f[tid]
    body_f[tid] = wp.spatial_vector(
        curr[0] + extra[0], curr[1] + extra[1], curr[2] + extra[2],
        curr[3] + extra[3], curr[4] + extra[4], curr[5] + extra[5],
    )


@wp.kernel
def apply_materials_kernel(
    shape_material_mu: wp.array(dtype=wp.float32),
    buf_friction: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    
    # Newton 中 shape 0 是靜態地板
    shape_idx = tid
    shape_material_mu[shape_idx] = buf_friction[tid]






