from typing import Literal

import newton
import warp as wp

from script.simulate.solvers.base_solver import BaseSolver, BaseSolverModel

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
        # Teleport writes Newton joint_q; MuJoCo qpos is synced on the next
        # step() when update_data_interval == 1. Warm-start lives in mjw_data
        # and is cleared in on_env_reset (not here: reset_obj runs inside the
        # CUDA graph every step).
        del state

    def on_env_reset(self, state, terminated) -> None:
        """Zero MuJoCo warm-start / applied-force buffers for reset worlds.

        Newton ``SolverMuJoCo.reset`` documents that after a NaN divergence
        these buffers poison the next step even once joint_q is teleported.
        ``flags=0`` clears buffers only — it must not overwrite the spawn
        pose ArticulationView just wrote.
        """
        solver = getattr(self, "solver", None)
        reset = getattr(solver, "reset", None)
        if not callable(reset) or state is None or terminated is None:
            return
        mask = terminated
        if not isinstance(mask, wp.array):
            mask = wp.from_torch(mask.contiguous(), dtype=wp.bool)
        try:
            reset(state, world_mask=mask, flags=0)
        except (TypeError, ValueError):
            try:
                reset(state, flags=0)
            except (TypeError, ValueError):
                return

