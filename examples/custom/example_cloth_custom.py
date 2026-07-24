import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import Mesh, ParticleFlags
# 只保留 VBD 求解器
from newton.solvers import SolverVBD

class Example:
    def __init__(self, viewer, args):
        # --- 1. 模擬基礎設定 ---
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        # VBD 模擬複雜模型時，子步數建議維持在 40-50
        self.sim_substeps = 10 
        self.sim_time = 0.0
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 8 # VBD 內部迭代次數

        self.viewer = viewer
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        
        # 僅註冊 VBD 求解器屬性
        SolverVBD.register_custom_attributes(builder)

        # --- 2. 加載模型資產 ---
        # 雖然不使用 style3d 求解器，但我們仍可以下載它的模型資產來用
        asset_path = newton.utils.download_asset("style3d")

        # 加載衛衣 USD
        # garment_usd_name = "Women_Skirt"
        # garment_usd_name = "Female_T_Shirt"
        garment_usd_name = "Women_Sweatshirt"
        usd_stage = Usd.Stage.Open(str(asset_path / "garments" / (garment_usd_name + ".usd")))
        usd_prim_garment = usd_stage.GetPrimAtPath(str("/Root/" + garment_usd_name + "/Root_Garment"))

        # 獲取 Mesh 數據
        garment_mesh_data = newton.usd.get_mesh(usd_prim_garment)

        # --- 3. 使用 Newton 原生方法添加布料 ---
        # 這裡用 builder.add_cloth_mesh 替代 style3d.add_cloth_mesh
        # VBD 會自動根據這裡添加的 Mesh 生成距離約束 (Distance Constraints)
        
        #   elasticity
        self.tri_ke = 1e4
        self.tri_ka = 1e4
        # self.tri_kd = 1.5e-6
        self.tri_kd = -100.0

        self.bending_ke = 5
        self.bending_kd = 1e-2
        self.cloth_particle_radius = 0.003

        builder.add_cloth_mesh(
            pos=wp.vec3(0, 0, 0.0), # 稍微抬高一點
            rot=wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=wp.pi / 2.0),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=garment_mesh_data.vertices.tolist(),
            indices=garment_mesh_data.indices.tolist(),
            density=0.02,
            # 以下參數會被轉化為 VBD 的約束剛度
            # 數值需要根據 VBD 的量綱進行調整
            
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
            tri_kd=self.tri_kd,
            edge_ke=self.bending_ke,
            edge_kd=self.bending_kd,
            particle_radius=self.cloth_particle_radius,
        )

        # 加載人體模型 (Avatar) 作為碰撞體
        usd_stage_avatar = Usd.Stage.Open(str(asset_path / "avatars" / "Female.usd"))
        usd_prim_avatar = usd_stage_avatar.GetPrimAtPath("/Root/Female/Root_SkinnedMesh_Avatar_0_Sub_2")
        avatar_mesh = newton.usd.get_mesh(usd_prim_avatar)

        builder.add_shape_mesh(
            body=builder.add_body(is_kinematic=True), # 靜態碰撞體
            xform=wp.transform(
                p=wp.vec3(0, 0, 0),
                q=wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=wp.pi / 2.0),
            ),
            mesh=Mesh(avatar_mesh.vertices, avatar_mesh.indices),
        )

        # 地面
        builder.add_ground_plane()
        
        # --- 4. 初始化模型與求解器 ---
        builder.color() 
        self.model = builder.finalize()

        # 碰撞與物理穩定性設定
        self.model.soft_contact_radius = 0.005 
        self.model.soft_contact_margin = 0.008
        self.model.soft_contact_ke = 1.0e4 # 增加剛度防止穿透
        self.model.soft_contact_kd =0.01
        self.model.set_gravity((0.0, 0.0, -9.81))

        # 使用 VBD 求解器
        self.solver = SolverVBD(
            model=self.model,
            iterations=self.iterations,
        )
        
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.0, -1.8, 1.2), 0.0, -270.0)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        p_lower = wp.vec3(-0.5, -0.2, 0.9)
        p_upper = wp.vec3(0.5, 0.2, 1.6)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)