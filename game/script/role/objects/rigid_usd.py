import newton
import warp as wp

from typing import List, Literal
from pxr import Usd
from script.simulate.mesh_builder import MeshBuilder
from script.role.objects.base_object import BaseObjectModel, BaseObject


class RigidUsdModel(BaseObjectModel):
    type: Literal["rigid_usd"] = "rigid_usd"

    object_mass: float = 0.0
    lock_inertia: bool = False
    is_kinematic: bool = False

    file_name: str = str("avatars/Female.usd") # USD 文件名稱
    file_path_or_source: str = "Nvidia" # USDC 路徑或來源
    mesh_path: str = str("/Root/Female/Root_SkinnedMesh_Avatar_0_Sub_2") # USDC Mesh 或者 TetMesh 路徑
    
    scale: float | None = None # Vec3
    # cfg: ShapeConfig | None = None
    color: int | None = None # Vec3
    # custom_attributes: dict[str, Any] | None = None


class RigidUsdObject(BaseObject):
    object_key = "rigid_usd"
    model_cls = RigidUsdModel
    object_type_id: int = 2  # TODO 用於 GPU kernel 的 ID, 這是自定義 ID 或許應該和 Newtom 的 self.model.shape_type 適配

    @staticmethod
    def add_physics(builder_env: newton.ModelBuilder, label: str, data: RigidUsdModel, cfg, **kwargs):
        body = builder_env.add_body(
            mass=data["object_mass"], 
            lock_inertia=data["lock_inertia"], 
            is_kinematic=data["is_kinematic"],
            label=label,
            )
        
        if data["file_path_or_source"] == "Nvidia":
            asset_path = newton.utils.download_asset("style3d") # TODO 這等於 data["file_path_or_source"] == "Nvidia"，只是做不做這個判斷都一樣
            usd_stage = Usd.Stage.Open(str(asset_path / data["file_name"]))

        else: 
            usd_stage = Usd.Stage.Open(f"{data["file_path_or_source"]}/{data["file_name"]}")

        prim = usd_stage.GetPrimAtPath(data["mesh_path"])
        mesh = newton.Mesh.create_from_usd(prim)

        index = builder_env.add_shape_mesh(
            body=body, 
            xform=wp.transform(
                p=wp.vec3(0, 0, 0),
                q=wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=wp.pi / 2.0),
            ),
            mesh=mesh, 
            scale=data["scale"],
            cfg=cfg,
            color=data["color"],
            label=label,
        )

        return index

    @staticmethod
    def add_visual(mesh_builder: MeshBuilder, data: RigidUsdModel, pos):
        # mesh_builder.add_box(pos=pos, size=data.size)
        raise NotImplementedError()

    @staticmethod
    def get_size(data: RigidUsdModel) -> List[float]: # TODO
        # return data.size
        print(f"{__class__.__name__}.get_size not implemented")
        return [0.0, 0.0, 0.0]


