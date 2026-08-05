"""
MPMParticles — MPM 粒子物件。

以 ``add_particle_grid`` 在 (pos + bounds_lo) ~ (pos + bounds_hi) 的長方體區域內
產生規則格點粒子，並透過 custom_attributes 直接寫入 ``mpm:`` namespace 的
per-particle 材質參數 (young_modulus、friction、yield_pressure 等)。

物件產生的粒子屬於 soft domain，會被 CouplingIndexBuilder 記入 soft 粒子索引，
供 coupled 求解器的 MPM entry 使用；粒子位置/速度的重置由 DeformableBody 處理。
"""

import numpy as np
import warp as wp

from typing import List, Literal, Optional

import newton
from newton.solvers import SolverImplicitMPM

from script.simulate.mesh_builder import MeshBuilder
from script.role.objects.base_object import BaseObjectModel, BaseObject


class MPMParticlesModel(BaseObjectModel):
    type: Literal["mpm_particles"] = "mpm_particles"

    # 生成區域 (相對於 default_position 的偏移)
    bounds_lo: List[float] = [-0.5, -0.5, 0.0]
    bounds_hi: List[float] = [0.5, 0.5, 0.5]

    voxel_size: float = 0.1
    particles_per_cell: int = 2
    density: float = 2500.0
    radius_mean: Optional[float] = None

    # MPM 材質參數 (寫入 mpm: namespace 的自訂屬性)
    young_modulus: float = 1.0e6
    poisson_ratio: float = 0.3
    damping: float = 0.0
    friction: float = 0.6
    yield_pressure: float = 5.0e4
    tensile_yield_ratio: float = 0.2
    yield_stress: float = 0.0
    hardening: float = 10.0
    hardening_rate: float = 1.0
    softening_rate: float = 1.0
    dilatancy: float = 0.5
    viscosity: float = 0.0


class MPMParticlesObject(BaseObject):
    object_key = "mpm_particles"
    model_cls = MPMParticlesModel
    object_type_id: int = 4  # TODO 用於 GPU kernel 的 ID

    @staticmethod
    def add_physics(builder_env: newton.ModelBuilder, label: str, data: MPMParticlesModel, pos, rot, vel, **kwargs):
        # MPM 自訂屬性 (mpm: namespace) 為冪等註冊，在加入粒子前必須宣告。
        SolverImplicitMPM.register_custom_attributes(builder_env)

        bounds_lo = np.asarray(data["bounds_lo"], dtype=np.float64)
        bounds_hi = np.asarray(data["bounds_hi"], dtype=np.float64)
        voxel_size = float(data["voxel_size"])
        particles_per_cell = int(data["particles_per_cell"])

        # 依 particles_per_cell 計算每個方向的粒子數 (與 NVIDIA 範例一致)
        res = np.ceil(particles_per_cell * (bounds_hi - bounds_lo) / voxel_size).astype(int)
        cell_size = (bounds_hi - bounds_lo) / np.maximum(res, 1)
        cell_volume = float(np.prod(cell_size))
        radius = float(np.max(cell_size) * 0.5)
        mass = cell_volume * float(data["density"])

        current_particle_count = builder_env.particle_count

        origin = np.asarray(pos, dtype=np.float64) + bounds_lo
        builder_env.add_particle_grid(
            pos=wp.vec3(float(origin[0]), float(origin[1]), float(origin[2])),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(res[0]) + 1,
            dim_y=int(res[1]) + 1,
            dim_z=int(res[2]) + 1,
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=data["radius_mean"] if data["radius_mean"] is not None else radius,
            flags=newton.ParticleFlags.ACTIVE,
            custom_attributes={
                "mpm:young_modulus": data["young_modulus"],
                "mpm:poisson_ratio": data["poisson_ratio"],
                "mpm:damping": data["damping"],
                "mpm:friction": data["friction"],
                "mpm:yield_pressure": data["yield_pressure"],
                "mpm:tensile_yield_ratio": data["tensile_yield_ratio"],
                "mpm:yield_stress": data["yield_stress"],
                "mpm:hardening": data["hardening"],
                "mpm:hardening_rate": data["hardening_rate"],
                "mpm:softening_rate": data["softening_rate"],
                "mpm:dilatancy": data["dilatancy"],
                "mpm:viscosity": data["viscosity"],
            },
        )

        object_particle_count = builder_env.particle_count - current_particle_count

        return {"start": current_particle_count, "count_particle_per_object": object_particle_count}

    @staticmethod
    def add_visual(mesh_builder: MeshBuilder, data: MPMParticlesModel, pos):
        pass

    @staticmethod
    def get_size(data: MPMParticlesModel) -> List[float]:
        bounds_lo = np.asarray(data["bounds_lo"], dtype=np.float64)
        bounds_hi = np.asarray(data["bounds_hi"], dtype=np.float64)
        return list((bounds_hi - bounds_lo).astype(float))
