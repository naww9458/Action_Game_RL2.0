import warp as wp
import newton

from typing import Literal
from script.simulate.solvers.base_solver import BaseSolverModel, BaseSolver

class VBDSolverModel(BaseSolverModel):
    type: Literal["vbd"] = "vbd"
    iterations: int = 10
    friction_epsilon: float = 0.01
    integrate_with_external_rigid_solver: bool = False
    particle_enable_self_contact: bool = False
    particle_self_contact_radius: float = 0.2
    particle_self_contact_margin: float = 0.2
    particle_conservative_bound_relaxation: float = 0.85
    particle_vertex_contact_buffer_size: int = 32
    particle_edge_contact_buffer_size: int = 64
    particle_collision_detection_interval: int = 0
    particle_edge_parallel_epsilon: float = 0.00001
    particle_enable_tile_solve: bool = True
    particle_topological_contact_filter_threshold: int = 2
    particle_rest_shape_contact_exclusion_radius: float = 0
    particle_external_vertex_contact_filtering_map: dict | None = None
    particle_external_edge_contact_filtering_map: dict | None = None
    rigid_avbd_alpha: float = 0.95
    rigid_avbd_joint_alpha: float | None = None
    rigid_avbd_contact_alpha: float | None = None
    rigid_avbd_beta: float = 0
    rigid_avbd_linear_beta: float | None = None
    rigid_avbd_angular_beta: float | None = None
    rigid_avbd_gamma: float = 0.999
    rigid_contact_hard: bool = True
    rigid_contact_history: bool = False
    rigid_contact_stick_motion_eps: float = 0.0001
    rigid_contact_stick_freeze_translation_eps: float = 0.0001
    rigid_contact_stick_freeze_angular_eps: float = 0.0001
    rigid_contact_k_start: float = 100
    rigid_body_contact_buffer_size: int = 64
    rigid_body_particle_contact_buffer_size: int = 256
    rigid_joint_linear_ke: float = 100000
    rigid_joint_angular_ke: float = 100000
    rigid_joint_linear_k_start: float = 100
    rigid_joint_angular_k_start: float = 10
    rigid_joint_linear_kd: float = 0
    rigid_joint_angular_kd: float = 0
    rigid_enable_dahl_friction: bool | None = None


class VBDSolver(BaseSolver):
    solver_key = "vbd"
    model_cls = VBDSolverModel
    solver_type_id: int = 1

    def __init__(self, config, builder: "newton.ModelBuilder", **kwargs):
        super().__init__(config)
        builder.color()

    def setup(self, model):
        self.model = model

        self.model.soft_contact_ke = 10000.0  
        self.model.soft_contact_kd = -100.0  
        self.model.soft_contact_mu = 0.5

        if self.config["rigid_contact_history"] is True:
            self.pipeline = newton.CollisionPipeline(self.model, contact_matching="latest")


        self.solver = newton.solvers.SolverVBD(
            model,
            **self.config
        )

    def step(self, state_in, state_out, control, contacts, dt):

        # self.pipeline.collide(state_in, contacts)
        self.solver.step(state_in, state_out, control, contacts, dt)

    def post_teleport_sync(self, state):
        # 關鍵：將 teleport 後的最新位置同步到 VBD 內部維護的 body_q_prev 中
        if self.model.body_count > 0:
            wp.copy(self.solver.body_q_prev, state.body_q)

    def reset_history(self):
        # 重置接觸歷史更新標記 [1]
        self.solver.set_rigid_history_update(True)


    @property
    def body_q_prev(self):
        """返回 VBD 求解器內部維護的前一影格位置陣列"""
        return None

