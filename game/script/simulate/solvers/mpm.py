"""
MPMSolver — 基於 newton.solvers.SolverImplicitMPM 的隱式 MPM 求解器。

適用於顆粒/流體/彈塑性連續體等材質。配置參數對應
``SolverImplicitMPM.Config``；粒子材質屬性 (young_modulus、friction、
yield_pressure 等) 透過 ``register_custom_attributes`` 註冊為
``mpm:`` namespace 的自訂屬性，並由 mpm_particles 物件在
add_physics 階段透過 add_particle_grid 的 custom_attributes 寫入。
"""

from __future__ import annotations

import numpy as np
import warp as wp

from typing import List, Literal, Optional, Union

import newton
from newton.solvers import SolverImplicitMPM

from script.simulate.solvers.base_solver import BaseSolverModel, BaseSolver


def solver_requires_mpm_attributes(solver_config: dict) -> bool:
    """判斷 solver_config 是否需要註冊 MPM 自訂屬性。

    適用於 type == "mpm" 或 coupled 且 solvers 列表包含 "mpm" 的情況。
    """
    if not isinstance(solver_config, dict):
        return False
    solver_type = str(solver_config.get("type", "")).lower()
    if solver_type == "mpm":
        return True
    if solver_type == "coupled":
        from script.simulate.solvers.coupled import resolve_coupled_domains

        _, soft = resolve_coupled_domains(solver_config)
        return soft == "mpm"
    return False


# newton 的 MPM collider mesh 建構流程 (_get_shape_mesh) 支援的形狀類型白名單。
# 白名單之外的形狀 (例如 CONVEX_MESH、ELLIPSOID、HFIELD、GAUSSIAN) 在建構
# collider 時會拋出 NotImplementedError，導致含該形狀物件的場景無法建立 MPM 求解器。
# 因此必須在模型 finalize 前清除這些形狀的 COLLIDE_PARTICLES 旗標，使它們不參與
# MPM 粒子碰撞（剛體碰撞 COLLIDE_SHAPES 不受影響）。
_MPM_COLLIDER_SUPPORTED_GEO_TYPES = frozenset(
    {
        newton.GeoType.MESH,
        newton.GeoType.PLANE,
        newton.GeoType.SPHERE,
        newton.GeoType.CAPSULE,
        newton.GeoType.CYLINDER,
        newton.GeoType.CONE,
        newton.GeoType.BOX,
    }
)


def filter_mpm_collider_shapes(builder_env) -> None:
    """清除不支援 MPM 碰撞網格的形狀之 COLLIDE_PARTICLES 旗標。

    需在 add_world / finalize 之前對 builder_env 呼叫。預設所有形狀都會帶上
    COLLIDE_PARTICLES 旗標，此函數僅對 GeoType 不在 MPM 支援白名單內的形狀
    關閉該旗標，容器、地板等由 BOX / PLANE / MESH 組成的剛體不受影響。
    """
    if not hasattr(builder_env, "shape_flags") or not hasattr(builder_env, "shape_type"):
        return

    flags = builder_env.shape_flags
    shape_types = builder_env.shape_type
    for shape_idx in range(len(flags)):
        if int(flags[shape_idx]) & int(newton.ShapeFlags.COLLIDE_PARTICLES):
            geo_type = newton.GeoType(int(shape_types[shape_idx]))
            if geo_type not in _MPM_COLLIDER_SUPPORTED_GEO_TYPES:
                flags[shape_idx] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)


class MPMSolverModel(BaseSolverModel):
    type: Literal["mpm"] = "mpm"

    # numerics
    max_iterations: int = 250
    tolerance: float = 1.0e-4
    solver: Union[str, List[str]] = "auto"
    warmstart_mode: str = "auto"
    collider_velocity_mode: str = "forward"

    # grid
    voxel_size: float = 0.1
    grid_type: str = "sparse"
    grid_padding: int = 0
    max_active_cell_count: int = -1
    transfer_scheme: str = "apic"
    integration_scheme: str = "pic"

    # material / background
    critical_fraction: float = 0.0
    air_drag: float = 1.0

    # experimental
    collider_normal_from_sdf_gradient: bool = False
    collider_basis: str = "S2"
    strain_basis: str = "P0"
    velocity_basis: str = "Q1"


def build_mpm_config(config: dict) -> SolverImplicitMPM.Config:
    """將 dict 配置轉換為 SolverImplicitMPM.Config dataclass。

    僅複製 Config 中存在且已定義的欄位，避免把額外業務參數誤傳入。
    """
    mpm_config = SolverImplicitMPM.Config()
    allowed = {f for f in mpm_config.__dataclass_fields__}
    for key, value in (config or {}).items():
        if key in allowed and value is not None:
            setattr(mpm_config, key, value)
    return mpm_config


def reset_mpm_particle_state(state) -> None:
    """重置 MPM 粒子的內部變量狀態 (彈性應變/塑性體積/應力/變形梯度)。

    particle_q / particle_qd 由 DeformableBody 的重置 kernel 處理，
    此處僅清理 MPM 專屬的隱藏狀態，避免重置後殘留塑性應變造成「彈跳」。
    該函式對無 mpm 屬性的 state 是安全空操作。
    """
    mpm_state = getattr(state, "mpm", None)
    if mpm_state is None:
        return

    identity = wp.mat33(np.eye(3))
    mpm_state.particle_qd_grad.zero_()
    mpm_state.particle_stress.zero_()
    mpm_state.particle_Jp.fill_(1.0)
    mpm_state.particle_elastic_strain.fill_(identity)
    mpm_state.particle_transform.fill_(identity)


class MPMSolver(BaseSolver):
    solver_key = "mpm"
    model_cls = MPMSolverModel
    solver_type_id: int = 4

    def __init__(self, config: dict, builder: "newton.ModelBuilder", **kwargs):
        super().__init__(config)

    def setup(self, model):
        self.model = model

        self.solver = SolverImplicitMPM(
            model,
            config=build_mpm_config(self.config),
        )

    def step(self, state_in, state_out, control, contacts, dt):
        # MPM 的碰撞由求解器內部光柵化 collider 處理，contacts 參數未使用。
        self.solver.step(state_in, state_out, control, contacts, dt)
        # 將穿透 collider 的粒子投影回外部，並同步速度/速度梯度。
        self.solver.project_outside(state_out, state_out, dt)

    def post_teleport_sync(self, state):
        # Teleport 後粒子位置已被修改，MPM 內部變形狀態不再有效，全部重置。
        reset_mpm_particle_state(state)

    def reset_history(self):
        pass

    @property
    def body_q_prev(self):
        """MPM 不維護剛體歷史位置，無需外部同步。"""
        return None
