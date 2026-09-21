"""Contact queries for boxing_v1. Returns Torch tensors.

MuJoCo-Warp contact decode stays in this boxing_v1 plugin. Rewards call
``max_force`` only.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import warp as wp

from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5


@wp.kernel
def _accumulate_pair_max_force_kernel(
    max_force: wp.array(dtype=float),
    found: wp.array(dtype=wp.int32),
    bodies_a: wp.array2d(dtype=wp.int32),
    n_a: int,
    bodies_b: wp.array2d(dtype=wp.int32),
    n_b: int,
    contact_worldid: wp.array(dtype=wp.int32),
    contact_geom: wp.array(dtype=wp.vec2i),
    contact_frame: wp.array(dtype=wp.mat33),
    contact_friction: wp.array(dtype=vec5),
    contact_dim: wp.array(dtype=wp.int32),
    contact_efc_address: wp.array(dtype=wp.int32, ndim=2),
    geom_bodyid: wp.array(dtype=wp.int32),
    mjc_body_to_newton: wp.array(dtype=wp.int32, ndim=2),
    efc_force: wp.array2d(dtype=float),
    ngeom: int,
    njmax: int,
    nbody_mj: int,
    nacon: wp.array(dtype=wp.int32),
    opt_cone: int,
):
    c = wp.tid()
    if c >= nacon[0]:
        return
    w = contact_worldid[c]
    g = contact_geom[c]
    if g[0] < 0 or g[0] >= ngeom or g[1] < 0 or g[1] >= ngeom:
        return
    b0 = geom_bodyid[g[0]]
    b1 = geom_bodyid[g[1]]
    if b0 < 0 or b0 >= nbody_mj or b1 < 0 or b1 >= nbody_mj:
        return
    na_id = mjc_body_to_newton[w, b0]
    nb_id = mjc_body_to_newton[w, b1]
    if na_id < 0 or nb_id < 0:
        return
    matched = int(0)
    for i in range(n_a):
        a = bodies_a[w, i]
        if a < 0:
            continue
        for j in range(n_b):
            b = bodies_b[w, j]
            if b < 0:
                continue
            if (na_id == a and nb_id == b) or (na_id == b and nb_id == a):
                matched = int(1)
    if matched == 0:
        return
    force = contact_force_fn(
        opt_cone,
        contact_frame,
        contact_friction,
        contact_dim,
        contact_efc_address,
        efc_force,
        njmax,
        nacon,
        w,
        c,
        True,
    )
    mag = wp.sqrt(force[0] * force[0] + force[1] * force[1] + force[2] * force[2])
    wp.atomic_max(max_force, w, mag)
    wp.atomic_max(found, w, 1)


class ContactQuery:
    """Pairwise max contact force between two Newton-body groups, per world."""

    def __init__(self, physics_manager: Any) -> None:
        self.physics_manager = physics_manager
        self.device = getattr(physics_manager, "device", None)
        self.num_env = int(getattr(getattr(physics_manager, "model", None), "world_count", 0) or 0)
        self._max_force = None
        self._found = None
        self._solver_ok = False
        self._ngeom = 0
        self._njmax = 0
        self._nbody_mj = 0
        self._naconmax = 0
        self._opt_cone = 0
        self._bind_solver()

    def _bind_solver(self) -> None:
        pm = self.physics_manager
        solver = getattr(getattr(pm, "solver_handler", None), "solver", None)
        if solver is None or not hasattr(solver, "mjw_data") or not hasattr(solver, "mjw_model"):
            return
        mj_model = solver.mjw_model
        mj_data = solver.mjw_data
        self._ngeom = int(mj_model.ngeom)
        self._njmax = int(mj_data.njmax)
        self._nbody_mj = int(mj_model.nbody)
        self._naconmax = int(mj_data.naconmax)
        self._opt_cone = int(mj_model.opt.cone)
        if self.num_env <= 0:
            self.num_env = int(getattr(mj_data, "nworld", 0) or 0)
        if self.num_env <= 0:
            return
        self._max_force = wp.zeros(self.num_env, dtype=float, device=self.device)
        self._found = wp.zeros(self.num_env, dtype=wp.int32, device=self.device)
        self._solver_ok = True

    @classmethod
    def try_from_physics(cls, physics_manager: Any) -> Optional["ContactQuery"]:
        if physics_manager is None:
            return None
        query = cls(physics_manager)
        if not query._solver_ok:
            return None
        return query

    def max_force(
        self,
        newton_bodies_a: torch.Tensor,
        newton_bodies_b: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return ``(max_force, found)`` per world, or ``(None, None)`` if unavailable."""
        if not self._solver_ok or self._max_force is None:
            return None, None
        pm = self.physics_manager
        solver = getattr(getattr(pm, "solver_handler", None), "solver", None)
        if solver is None:
            return None, None
        bodies_a = newton_bodies_a
        bodies_b = newton_bodies_b
        if bodies_a.ndim != 2 or bodies_b.ndim != 2:
            return None, None
        if int(bodies_a.shape[0]) != self.num_env or int(bodies_b.shape[0]) != self.num_env:
            return None, None
        a_wp = wp.from_torch(bodies_a, dtype=wp.int32)
        b_wp = wp.from_torch(bodies_b, dtype=wp.int32)
        self._max_force.zero_()
        self._found.zero_()
        contact = solver.mjw_data.contact
        wp.launch(
            _accumulate_pair_max_force_kernel,
            dim=self._naconmax,
            inputs=[
                self._max_force,
                self._found,
                a_wp,
                int(bodies_a.shape[1]),
                b_wp,
                int(bodies_b.shape[1]),
                contact.worldid,
                contact.geom,
                contact.frame,
                contact.friction,
                contact.dim,
                contact.efc_address,
                solver.mjw_model.geom_bodyid,
                solver.mjc_body_to_newton,
                solver.mjw_data.efc.force,
                self._ngeom,
                self._njmax,
                self._nbody_mj,
                solver.mjw_data.nacon,
                self._opt_cone,
            ],
            device=self.device,
        )
        return wp.to_torch(self._max_force), wp.to_torch(self._found)

    @property
    def last_found(self) -> Optional[Any]:
        return self._found

    @property
    def last_max_force(self) -> Optional[Any]:
        return self._max_force