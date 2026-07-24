import newton

from typing import List, Literal
from script.simulate.mesh_builder import MeshBuilder
from script.role.objects.base_object import BaseObjectModel, BaseObject
from script.game_config import GameConfig


class RigidSphereModel(BaseObjectModel):
    type: Literal["rigid_sphere"] = "rigid_sphere"
    radius: float = 0.025
    object_mass: float = 0.0
    lock_inertia: bool = False
    is_kinematic: bool = False
    as_site: bool = False
    color: None = None # Vec3 | None


class RigidSphereObject(BaseObject):
    object_key = "rigid_sphere"
    model_cls = RigidSphereModel
    object_type_id: int = 0  # TODO 用於 GPU kernel 的 ID, 這是自定義 ID 或許應該和 Newtom 的 self.model.shape_type 適配

    @staticmethod
    def add_physics(builder_env: newton.ModelBuilder, label: str, data: RigidSphereModel, cfg, **kwargs):
        body = builder_env.add_body(
            mass=data["object_mass"], 
            lock_inertia=data["lock_inertia"], 
            is_kinematic=data["is_kinematic"],
            label=label,
            )
        
        radius = data["radius"]
        index = builder_env.add_shape_sphere(
            body, 
            radius=radius, 
            cfg=cfg,
            as_site=data["as_site"],
            color=data["color"],
            label=label,
            )

        return index

    @staticmethod
    def add_visual(mesh_builder: MeshBuilder, data: RigidSphereModel, pos):
        radius = data["radius"]
        mesh_builder.add_sphere(pos=pos, radius=radius)

    @staticmethod
    def get_size(data: RigidSphereModel) -> List[float]:
        radius = data["radius"]
        return [radius, radius, radius]
