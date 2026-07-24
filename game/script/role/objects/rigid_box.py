import newton

from typing import List, Literal
from script.simulate.mesh_builder import MeshBuilder
from script.role.objects.base_object import BaseObjectModel, BaseObject
from script.game_config import GameConfig


class RigidBoxModel(BaseObjectModel):
    type: Literal["rigid_box"] = "rigid_box"
    size: List[float] = [1.0, 1.0, 1.0]
    object_mass: float = 0.0
    lock_inertia: bool = False
    is_kinematic: bool = False
    as_site: bool = False
    color: None = None # Vec3 | None


class RigidBoxObject(BaseObject):
    object_key = "rigid_box"
    model_cls = RigidBoxModel
    object_type_id: int = 1  # TODO 用於 GPU kernel 的 ID, 這是自定義 ID 或許應該和 Newtom 的 self.model.shape_type 適配

    @staticmethod
    def add_physics(builder_env: newton.ModelBuilder, label: str, data: RigidBoxModel, cfg, **kwargs):
        body = builder_env.add_body(
            mass=data["object_mass"], 
            lock_inertia=data["lock_inertia"], 
            is_kinematic=data["is_kinematic"],
            label=label,
            )

        size = data["size"]
        index = builder_env.add_shape_box(
            body, 
            hx=size[0], 
            hy=size[1], 
            hz=size[2], 
            cfg=cfg,
            as_site=data["as_site"],
            color=data["color"],
            label=label,
            )

        return index

    @staticmethod
    def add_visual(mesh_builder: MeshBuilder, data: RigidBoxModel, pos):
        size = data["size"]
        mesh_builder.add_box(pos=pos, size=size)

    @staticmethod
    def get_size(data: RigidBoxModel) -> List[float]:
        # 獲取尺寸 List [w, h, l]
        size = data["size"]
        return size


