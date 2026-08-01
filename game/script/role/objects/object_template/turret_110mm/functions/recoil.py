"""turret_110mm recoil helper.

Directly imported and called by the turret's own aim action
(``turret_110mm/functions/aim.py``) after a successful muzzle fire, so the
per-pattern recoil physics stays out of the generic ``Shoot`` ability and
lives with the object template that owns it.

The recoil is staged in the turret articulation's control-force buffer
(``control_force_gpus``) instead of the physics ``state.body_f`` array:
``body_f`` is zeroed every substep by ``state.clear_forces()`` inside
``simulate()`` before the solver runs, whereas the control-force buffer is
consumed by ``articulation_body.apply_controls()`` after ``clear_forces()``
and cleared once per frame. This mirrors how the ``Shoot`` ability delivers
the bullet's forward impulse and keeps the recoil warp-kernel based /
differentiable.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def _apply_turret_recoil_kernel(
    control_force: wp.array3d(dtype=wp.vec3),
    world: wp.int32,
    obj_idx: wp.int32,
    recoil: wp.vec3,
):
    tid = wp.tid()
    if tid != 0:
        return
    cur = control_force[world, obj_idx, 0]
    control_force[world, obj_idx, 0] = wp.vec3(
        cur[0] + recoil[0],
        cur[1] + recoil[1],
        cur[2] + recoil[2],
    )


def apply_turret_recoil(
    *,
    articulation_body,
    tool_pattern: str,
    world: int,
    tool_root_body_idx: int,
    barrel_forward_dir,
    recoil_force: float,
) -> None:
    """Add ``-recoil_force * barrel_forward`` to the turret root body force.

    The force is written into the tool pattern's articulation control-force
    buffer (shape ``(world_count, count_per_world, 1)``), which
    ``articulation_body.apply_controls`` folds into ``state.body_f`` during
    the next substep — after ``clear_forces()``. Returns without effect when
    the tool pattern has no control buffers (e.g. not yet configured) or the
    root body index does not map to a view slot.
    """
    if articulation_body is None or not tool_pattern:
        return
    pattern = f"tool_{tool_pattern}"
    force_buf = getattr(articulation_body, "control_force_gpus", {}).get(pattern)
    body_indices = getattr(articulation_body, "view_body_local_indices_gpus", {}).get(pattern)
    if force_buf is None or body_indices is None:
        return
    indices_np = body_indices.numpy()
    view_obj_idx = -1
    for i, body_idx in enumerate(indices_np):
        if int(body_idx) == int(tool_root_body_idx):
            view_obj_idx = i
            break
    if view_obj_idx < 0:
        return
    recoil = -float(recoil_force)
    vec = wp.vec3(
        recoil * float(barrel_forward_dir[0]),
        recoil * float(barrel_forward_dir[1]),
        recoil * float(barrel_forward_dir[2]),
    )
    wp.launch(
        _apply_turret_recoil_kernel,
        dim=1,
        inputs=[force_buf, int(world), view_obj_idx, vec],
        device=force_buf.device,
    )
