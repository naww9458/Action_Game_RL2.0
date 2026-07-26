import newton
import warp as wp

from typing import List, Literal
from pxr import Usd
from script.simulate.mesh_builder import MeshBuilder
from script.role.objects.base_object import BaseObjectModel, BaseObject


class SoftUsdGeneralModel(BaseObjectModel):
    type: Literal["soft_usd_general"] = "soft_usd_general"

    file_name: str = str("hollow_sphere_TetMesh.usdc") # USD 文件名稱
    file_path_or_source: str = "./Action_Game_RL_Assets/assets/" # USDC 路徑或來源
    mesh_path: str = str("/root/sphere/TetMesh") # USDC Mesh 或者 TetMesh 路徑

    scale: float = 1.0
    # vertices: list[Vec3]  # from the usd file
    # indices: list[int]    # from the usd file
    density: float | None = None
    k_mu: float | None = None    # float | ndarray[_AnyShape, dtype[Any]] | None
    k_lambda: float | None = None # float | ndarray[_AnyShape, dtype[Any]] | None
    k_damp: float | None = None # float | ndarray[_AnyShape, dtype[Any]] | None

    tri_ke: float = 0
    tri_ka: float = 0
    tri_kd: float = 0
    tri_drag: float = 0
    tri_lift: float = 0
    add_surface_mesh_edges: bool = True
    edge_ke: float = 0
    edge_kd: float = 0
    particle_radius: float | None = 0.15


class SoftUsdGeneralObject(BaseObject):
    object_key = "soft_usd_general"
    model_cls = SoftUsdGeneralModel
    object_type_id: int = 5  # TODO 用於 GPU kernel 的 ID, 這是自定義 ID 或許應該和 Newtom 的 self.model.shape_type 適配

    @staticmethod
    def add_physics(builder_env: newton.ModelBuilder, label: str, data: SoftUsdGeneralModel, pos, rot, vel, **kwargs):
        
        if data["file_path_or_source"] == "Nvidia":
            asset_path = newton.utils.download_asset("style3d") # TODO 這等於 data["file_path_or_source"] == "Nvidia"，只是做不做這個判斷都一樣
            usd_stage = Usd.Stage.Open(str(asset_path / data["file_name"]))

        else: 
            usd_stage = Usd.Stage.Open(f"{data["file_path_or_source"]}/{data["file_name"]}")

        prim = usd_stage.GetPrimAtPath(data["mesh_path"])
        tetmesh = newton.TetMesh.create_from_usd(prim)

        current_particle_count = builder_env.particle_count
        builder_env.add_soft_mesh(
            pos=wp.vec3(pos[0], pos[1], pos[2]),
            rot=wp.quat(0.0, 0.0, 0.0, 1.0),
            scale=data["scale"],  # already in meters
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=tetmesh,

            # vertices: list[Vec3] | None = None,
            # indices: list[int] | None = None,
            density=data["density"],
            k_mu=data["k_mu"],
            k_lambda=data["k_lambda"],
            k_damp=data["k_damp"],

            tri_ke=data["tri_ke"],
            tri_ka=data["tri_ka"],
            tri_kd=data["tri_kd"],
            tri_drag=data["tri_drag"],
            tri_lift=data["tri_lift"],
            add_surface_mesh_edges=data["add_surface_mesh_edges"],
            edge_ke=data["edge_ke"],
            edge_kd=data["edge_kd"],
            particle_radius=data["particle_radius"],
        )
        object_particle_count = builder_env.particle_count - current_particle_count

        return {"start": current_particle_count, "count_particle_per_object": object_particle_count}

    @staticmethod
    def add_visual(mesh_builder: MeshBuilder, data: SoftUsdGeneralModel, pos):
        raise NotImplementedError()

    @staticmethod
    def get_size(data: SoftUsdGeneralModel) -> List[float]: # TODO
        # return data.size
        print(f"{__class__.__name__}.get_size not implemented")
        return [0.0, 0.0, 0.0]


