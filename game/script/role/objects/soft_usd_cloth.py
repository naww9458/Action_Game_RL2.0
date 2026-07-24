import newton
import warp as wp

from typing import Any, List, Literal
from pxr import Usd
from script.simulate.mesh_builder import MeshBuilder
from script.role.objects.base_object import BaseObjectModel, BaseObject


class SoftUsdClothModel(BaseObjectModel):
    type: Literal["soft_usd_cloth"] = "soft_usd_cloth"

    # file_name: str = "garments/Women_Sweatshirt.usd" # USD 文件名稱
    # file_name: str = "garments/Women_Sweatshirt.usd" # USD 文件名稱
    file_name: str = "garments/Women_Sweatshirt.usd" # USD 文件名稱
    file_path_or_source: str = "Nvidia" # USDC 路徑或來源
    mesh_path: str = "/Root/Women_Sweatshirt/Root_Garment" # USDC Mesh 或者 TetMesh 路徑

    scale: float = 1.0
    # vertices: list[Vec3]  # from the usd file
    # indices: list[int]    # from the usd file
    density: float = 0.02

    tri_ke: float | None = 1e4
    tri_ka: float | None = 1e4
    tri_kd: float | None = -100.0
    tri_drag: float | None = None
    tri_lift: float | None = None
    edge_ke: float | None = 5
    edge_kd: float | None = 1e-2
    add_springs: bool = False
    spring_ke: float | None = None
    spring_kd: float | None = None
    particle_radius: float | None = 0.003
    custom_attributes_particles: dict[str, Any] | None = None
    custom_attributes_edges: dict[str, Any] | None = None
    custom_attributes_triangles: dict[str, Any] | None = None
    custom_attributes_springs: dict[str, Any] | None = None


class SoftUsdClothObject(BaseObject):
    object_key = "soft_usd_cloth"
    model_cls = SoftUsdClothModel
    object_type_id: int = 4  # TODO 用於 GPU kernel 的 ID, 這是自定義 ID 或許應該和 Newtom 的 self.model.shape_type 適配

    @staticmethod
    def add_physics(builder_env: newton.ModelBuilder, label: str, data: SoftUsdClothModel, pos, rot, vel, **kwargs):

        if data["file_path_or_source"] == "Nvidia":
            asset_path = newton.utils.download_asset("style3d") # TODO 這等於 data["file_path_or_source"] == "Nvidia"，只是做不做這個判斷都一樣
            usd_stage = Usd.Stage.Open(str(asset_path / data["file_name"]))

        else: 
            usd_stage = Usd.Stage.Open(f"{data["file_path_or_source"]}/{data["file_name"]}")

        prim = usd_stage.GetPrimAtPath(data["mesh_path"])
        garment_mesh_data = newton.usd.get_mesh(prim=prim)

        current_particle_count = builder_env.particle_count
        builder_env.add_cloth_mesh(
            pos=wp.vec3(pos[0], pos[1], pos[2]),
            rot=wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=wp.pi / 2.0), # TODO Hard code
            scale=data["scale"],
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=garment_mesh_data.vertices.tolist(),
            indices=garment_mesh_data.indices.tolist(),
            density=data["density"],
            # 以下參數會被轉化為 VBD 的約束剛度
            # 數值需要根據 VBD 的量綱進行調整
            
            tri_ke=data["tri_ke"],
            tri_ka=data["tri_ka"],
            tri_kd=data["tri_kd"],
            tri_drag=data["tri_drag"],
            tri_lift=data["tri_lift"],
            edge_ke=data["edge_ke"],
            edge_kd=data["edge_kd"],

            add_springs=data["add_springs"],
            spring_ke=data["spring_ke"],
            spring_kd=data["spring_kd"],
            particle_radius=data["particle_radius"],

            custom_attributes_particles=data["custom_attributes_particles"],
            custom_attributes_edges=data["custom_attributes_edges"],
            custom_attributes_triangles=data["custom_attributes_triangles"],
            custom_attributes_springs=data["custom_attributes_springs"],
        )
        object_particle_count = builder_env.particle_count - current_particle_count

        return {"start": current_particle_count, "count_particle_per_object": object_particle_count}

    @staticmethod
    def add_visual(mesh_builder: MeshBuilder, data: SoftUsdClothModel, pos):
        raise NotImplementedError()

    @staticmethod
    def get_size(data: SoftUsdClothModel) -> List[float]: # TODO
        # return data.size
        print(f"{__class__.__name__}.get_size not implemented")
        return [0.0, 0.0, 0.0]


