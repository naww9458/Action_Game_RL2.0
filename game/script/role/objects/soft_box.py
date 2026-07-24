import newton
import warp as wp

from typing import List, Literal
from script.simulate.mesh_builder import MeshBuilder
from script.role.objects.base_object import BaseObjectModel, BaseObject
from script.game_config import GameConfig


class SoftBoxModel(BaseObjectModel):
    type: Literal["soft_box"] = "soft_box"

    dim_x: int = 2
    dim_y: int = 2
    dim_z: int = 2
    cell_x: float = 0.1
    cell_y: float = 0.1
    cell_z: float = 0.1

    density: float = 1.0e3
    k_mu: float = 1.0e5
    k_lambda: float = 1.0e5
    k_damp: float = 1e-1

    fix_left: bool = False
    fix_right: bool = False
    fix_top: bool = False
    fix_bottom: bool = False

    tri_ke: float = 0
    tri_ka: float = 0
    tri_kd: float = 0
    tri_drag: float = 0
    tri_lift: float = 0

    add_surface_mesh_edges: bool = True
    edge_ke: float = 0
    edge_kd: float = 0
    particle_radius: float | None = None


class SoftBoxObject(BaseObject):
    object_key = "soft_box"
    model_cls = SoftBoxModel
    object_type_id: int = 3  # TODO 用於 GPU kernel 的 ID, 這是自定義 ID 或許應該和 Newtom 的 self.model.shape_type 適配

    @staticmethod
    def add_physics(builder_env: newton.ModelBuilder, label: str, data: SoftBoxModel, pos, rot, vel, **kwargs):
        current_particle_count = builder_env.particle_count
        builder_env.add_soft_grid(
            pos=wp.vec3(pos[0], pos[1], pos[2]),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0), # TODO Hard code

            dim_x=data["dim_x"],
            dim_y=data["dim_y"],
            dim_z=data["dim_z"],
            cell_x=data["cell_x"],
            cell_y=data["cell_y"],
            cell_z=data["cell_z"],

            density=data["density"],
            k_mu=data["k_mu"],
            k_lambda=data["k_lambda"],
            k_damp=data["k_damp"],

            fix_left=data["fix_left"],
            fix_right=data["fix_right"],
            fix_top=data["fix_top"],
            fix_bottom=data["fix_bottom"],

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
    def add_visual(mesh_builder: MeshBuilder, data: SoftBoxModel, pos):
        # raise NotImplementedError()
        pass

    @staticmethod
    def get_size(data: SoftBoxModel) -> List[float]: # TODO
        # return data.size
        print(f"{__class__.__name__}.get_size not implemented")
        return [0.0, 0.0, 0.0]


