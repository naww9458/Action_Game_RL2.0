import warp as wp
import newton

from typing import Literal
from script.simulate.solvers.base_solver import BaseSolverModel, BaseSolver


@wp.kernel
def _copy_spatial_vector_kernel(
    src: wp.array(dtype=wp.spatial_vector),
    dst: wp.array(dtype=wp.spatial_vector),
):
    """Differentiable elementwise copy (wp.clone / assign break reverse-mode AD)."""
    tid = wp.tid()
    dst[tid] = src[tid]


@wp.kernel
def _copy_transform_kernel(
    src: wp.array(dtype=wp.transform),
    dst: wp.array(dtype=wp.transform),
):
    """Differentiable body_q copy for XPBD apply_body_deltas under Tape."""
    tid = wp.tid()
    dst[tid] = src[tid]


class XPBDSolverModel(BaseSolverModel):
    type: Literal["xpbd"] = "xpbd"
    iterations: int = 1
    soft_body_relaxation: float = 0.9
    soft_contact_relaxation: float = 0.9
    joint_linear_relaxation: float = 0.7
    joint_angular_relaxation: float = 0.4
    joint_linear_compliance: float = 0
    joint_angular_compliance: float = 0
    rigid_contact_relaxation: float = 0.8
    rigid_contact_con_weighting: bool = True
    angular_damping: float = 0
    enable_restitution: bool = False


class XPBDSolver(BaseSolver):
    solver_key = "xpbd"
    model_cls = XPBDSolverModel
    solver_type_id: int = 0

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def setup(self, model):
        self.solver = newton.solvers.SolverXPBD(
            model,
            **self.config
        )

    def step(self, state_in, state_out, control, contacts, dt):
        # Newton XPBD uses wp.clone on body_f (joint force staging) and on
        # body_q/body_qd inside _apply_body_deltas when requires_grad. ArticulationBody
        # always creates free joints, so these paths always run. wp.clone/assign are
        # invisible to Warp reverse-mode AD:
        #   - body_f clone zeros torque→force→integrate grads
        #   - body_q clone zeros orientation rewards (FaceToTarget) → actions
        #   - body_qd clone would zero velocity rewards (patched the same way)
        # Replace body-count clones with tape-visible kernel copies; keep iterations.
        if not getattr(state_in, "requires_grad", False):
            self.solver.step(state_in, state_out, control, contacts, dt)
            return

        body_count = self.solver.model.body_count
        real_clone = wp.clone

        def ad_safe_clone(arr, *args, **kwargs):
            dtype = getattr(arr, "dtype", None)
            shape = getattr(arr, "shape", ())
            if len(shape) == 1 and shape[0] == body_count:
                if dtype == wp.spatial_vector:
                    out = wp.zeros_like(arr)
                    wp.launch(
                        _copy_spatial_vector_kernel,
                        dim=body_count,
                        inputs=[arr, out],
                        device=arr.device,
                    )
                    return out
                if dtype == wp.transform:
                    out = wp.zeros_like(arr)
                    wp.launch(
                        _copy_transform_kernel,
                        dim=body_count,
                        inputs=[arr, out],
                        device=arr.device,
                    )
                    return out
            return real_clone(arr, *args, **kwargs)

        wp.clone = ad_safe_clone
        try:
            self.solver.step(state_in, state_out, control, contacts, dt)
        finally:
            wp.clone = real_clone

    def post_teleport_sync(self, state):
        # XPBD 主要依賴 state.body_q_prev，通常已在您的 Kernel 中處理完畢
        pass
