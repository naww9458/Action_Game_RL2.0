import newton

from typing import Literal
from script.simulate.solvers.base_solver import BaseSolverModel, BaseSolver

class MuJoCoSolverModel(BaseSolverModel):
    type: Literal["mujoco"] = "mujoco"
    
    separate_worlds: bool | None = None
    njmax: int | None = None
    nconmax: int | None = None
    iterations: int | None = None
    ls_iterations: int | None = None
    ccd_iterations: int | None = None
    sdf_iterations: int | None = None
    sdf_initpoints: int | None = None
    solver: int | str | None = "newton"
    integrator: int | str | None = None
    cone: int | str | None = None
    jacobian: int | str | None = None
    impratio: float | None = None
    tolerance: float | None = None
    ls_tolerance: float | None = None
    ccd_tolerance: float | None = None
    density: float | None = None
    viscosity: float | None = None
    wind: tuple | None = None
    magnetic: tuple | None = None
    use_mujoco_cpu: bool = False
    enable_multiccd: bool = False
    disable_contacts: bool = False
    update_data_interval: int = 1
    save_to_mjcf: str | None = None
    use_mujoco_contacts: bool = True
    include_sites: bool = True
    skip_visual_only_geoms: bool = True



class MuJoCoSolver(BaseSolver):
    solver_key = "mujoco"
    model_cls = MuJoCoSolverModel
    solver_type_id: int = 2

    def __init__(self, config, builder: "newton.ModelBuilder", **kwargs):
        super().__init__(config, **kwargs)

    def setup(self, model):
        self.solver = newton.solvers.SolverMuJoCo(
            model, 
            **self.config
        )

    def step(self, state_in, state_out, control, contacts, dt):
        self.solver.step(state_in, state_out, control, contacts, dt)

    def post_teleport_sync(self, state):
        # XPBD 主要依賴 state.body_q_prev，通常已在您的 Kernel 中處理完畢
        pass

