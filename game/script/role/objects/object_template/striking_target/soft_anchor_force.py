"""Warp linear spring between a kinematic anchor and a dynamic target body.

Parameters come from the object's YAML (copied into builder metadata) and
body leaf names come from ``control_configs.yaml``. Bound from this template's
``register.setup(environment)`` after physics finalize.

Force is staged in the articulation control-force buffer (same path as
``move_topdown_viewing_angle``) so ``PhysicsManager.simulate`` only applies
controls — it does not host object-specific substep callbacks.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import warp as wp

from script.role.base_role import BaseRole
from script.role.objects.object_template.loader import load_template_control_config


@wp.kernel
def apply_soft_anchor_force_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_inv_mass: wp.array(dtype=float),
    control_force: wp.array3d(dtype=wp.vec3),
    anchor_body: wp.array(dtype=int),
    target_body: wp.array(dtype=int),
    world_idx: wp.array(dtype=int),
    obj_idx: wp.array(dtype=int),
    target_slot: wp.array(dtype=int),
    rest_offset: wp.array(dtype=wp.vec3),
    stiffness: wp.array(dtype=float),
    damping: wp.array(dtype=float),
    max_force: wp.array(dtype=float),
):
    tid = wp.tid()
    target_idx = target_body[tid]
    if target_idx < 0 or target_idx >= body_q.shape[0]:
        return
    if body_inv_mass[target_idx] <= 0.0:
        return

    anchor_idx = anchor_body[tid]
    if anchor_idx < 0 or anchor_idx >= body_q.shape[0]:
        return

    anchor_tf = body_q[anchor_idx]
    target_tf = body_q[target_idx]
    offset_world = wp.quat_rotate(anchor_tf.q, rest_offset[tid])
    rest = wp.vec3(
        anchor_tf.p[0] + offset_world[0],
        anchor_tf.p[1] + offset_world[1],
        anchor_tf.p[2] + offset_world[2],
    )
    error = wp.vec3(
        rest[0] - target_tf.p[0],
        rest[1] - target_tf.p[1],
        rest[2] - target_tf.p[2],
    )

    anchor_qd = body_qd[anchor_idx]
    target_qd = body_qd[target_idx]
    anchor_lin = wp.vec3(anchor_qd[0], anchor_qd[1], anchor_qd[2])
    anchor_ang = wp.vec3(anchor_qd[3], anchor_qd[4], anchor_qd[5])
    target_lin = wp.vec3(target_qd[0], target_qd[1], target_qd[2])
    rest_vel = anchor_lin + wp.cross(anchor_ang, offset_world)
    v_rel = target_lin - rest_vel

    raw = stiffness[tid] * error - damping[tid] * v_rel
    length = wp.length(raw)
    limit = max_force[tid]
    scale = 1.0
    if length > limit and length > 1.0e-8:
        scale = limit / length
    force = raw * scale

    w = world_idx[tid]
    o = obj_idx[tid]
    b = target_slot[tid]
    cur = control_force[w, o, b]
    control_force[w, o, b] = wp.vec3(
        cur[0] + force[0],
        cur[1] + force[1],
        cur[2] + force[2],
    )


def _leaf_index(path_map: Dict[str, Any], leaf: str) -> Optional[int]:
    name = str(leaf or "").strip().strip("/")
    if not name or not path_map:
        return None
    matches = [
        int(idx)
        for path, idx in path_map.items()
        if str(path).rstrip("/").split("/")[-1] == name
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _control_body_names(pattern: str) -> Tuple[str, str]:
    raw = load_template_control_config(str(pattern))
    section = raw.get(str(pattern), raw) if isinstance(raw, dict) else {}
    if not isinstance(section, dict):
        section = {}
    cfg = section.get("rigid_soft_anchor") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    return str(cfg.get("anchor_body") or ""), str(cfg.get("target_body") or "")


def _constraint_params(meta: Dict[str, Any]) -> Dict[str, Any]:
    payload = meta.get("passive_constraint") if isinstance(meta, dict) else None
    return dict(payload) if isinstance(payload, dict) else {}


def _num_bodies_env(environment: Any) -> int:
    articulation_body = getattr(environment, "articulation_body", None)
    n = int(getattr(articulation_body, "num_rigid_bodies_env", 0) or 0)
    if n > 0:
        return n
    physics_manager = getattr(environment, "physics_manager", None)
    model = getattr(physics_manager, "model", None) if physics_manager is not None else None
    num_env = max(int(getattr(environment, "num_env", 1) or 1), 1)
    if model is None:
        return 0
    return int(model.body_count) // num_env


def attach_if_present(environment: Any) -> Optional["SoftAnchorForceConstraint"]:
    """Attach the soft-anchor force when this object type is in the environment."""
    constraints = _create_soft_anchor_constraints(environment)
    if not constraints:
        return None
    physics_manager = getattr(environment, "physics_manager", None)
    if physics_manager is None:
        return None
    hooks = getattr(environment, "_object_template_runtime_hooks", None)
    if hooks is None:
        hooks = []
        setattr(environment, "_object_template_runtime_hooks", hooks)
    attached = None
    for constraint in constraints:
        constraint.attach(physics_manager)
        hooks.append(constraint)
        attached = constraint
    return attached


def _create_soft_anchor_constraints(environment: Any) -> List["SoftAnchorForceConstraint"]:
    physics_manager = getattr(environment, "physics_manager", None)
    articulation_body = getattr(environment, "articulation_body", None)
    if physics_manager is None or getattr(physics_manager, "model", None) is None:
        return []
    if articulation_body is None:
        return []

    params = list(getattr(BaseRole, "_object_game_params", None) or [])
    num_env = max(int(getattr(environment, "num_env", 1) or 1), 1)
    num_objects_env = int(getattr(BaseRole, "_num_objects_env", 0) or 0)
    env0 = params[:num_objects_env] if num_objects_env > 0 else params
    metadata = getattr(physics_manager, "object_metadata_by_role", None) or {}
    num_bodies_env = _num_bodies_env(environment)
    if num_bodies_env <= 0:
        return []

    grouped: Dict[str, Dict[str, List[Any]]] = defaultdict(
        lambda: {
            "anchor_ids": [],
            "target_ids": [],
            "worlds": [],
            "obj_idxs": [],
            "target_slots": [],
            "offsets": [],
            "stiffness": [],
            "damping": [],
            "max_force": [],
            "material_jobs": [],
        }
    )
    template_shape_count = int(getattr(physics_manager, "_template_shape_count", 0) or 0)

    for role_id, item in enumerate(env0):
        if not isinstance(item, dict):
            continue
        if str(item.get("shape_key") or "") != "rigid_soft_anchor":
            continue
        runtime_pattern = str(item.get("runtime_pattern") or "")
        if not runtime_pattern or runtime_pattern not in getattr(articulation_body, "control_force_gpus", {}):
            continue
        pattern_roles = getattr(articulation_body, "patterns", {}).get(runtime_pattern) or []
        try:
            view_obj_idx = list(pattern_roles).index(role_id)
        except ValueError:
            continue
        meta = metadata.get(role_id) or {}
        path_body_map = meta.get("path_body_map") or {}
        path_shape_map = meta.get("path_shape_map") or {}
        template_pattern = str(item.get("pattern") or "")
        anchor_leaf, target_leaf = _control_body_names(template_pattern)
        local_anchor = _leaf_index(path_body_map, anchor_leaf)
        local_target = _leaf_index(path_body_map, target_leaf)
        if local_anchor is None or local_target is None:
            continue
        cfg = _constraint_params(meta)
        offset = cfg.get("target_offset")
        if not isinstance(offset, (list, tuple)) or len(offset) < 3:
            continue
        if "linear_stiffness" not in cfg or "linear_damping" not in cfg or "max_linear_force" not in cfg:
            continue
        target_slot = articulation_body.cartesian_body_slot(
            runtime_pattern, view_obj_idx, int(local_target)
        )
        cart = getattr(articulation_body, "cartesian_body_local_indices", {}).get(runtime_pattern) or []
        bpo = int(getattr(articulation_body, "bodies_per_object", {}).get(runtime_pattern, 1) or 1)
        mapped_idx = int(view_obj_idx) * bpo + int(target_slot)
        if mapped_idx >= len(cart) or int(cart[mapped_idx]) != int(local_target):
            continue
        group = grouped[runtime_pattern]
        for world in range(num_env):
            group["anchor_ids"].append(int(local_anchor) + world * num_bodies_env)
            group["target_ids"].append(int(local_target) + world * num_bodies_env)
            group["worlds"].append(int(world))
            group["obj_idxs"].append(int(view_obj_idx))
            group["target_slots"].append(int(target_slot))
            group["offsets"].append((float(offset[0]), float(offset[1]), float(offset[2])))
            group["stiffness"].append(float(cfg["linear_stiffness"]))
            group["damping"].append(float(cfg["linear_damping"]))
            group["max_force"].append(float(cfg["max_linear_force"]))

        if template_shape_count > 0:
            local_anchor_shape = _leaf_index(path_shape_map, anchor_leaf)
            local_target_shape = _leaf_index(path_shape_map, target_leaf)
            for world in range(num_env):
                shape_offset = 1 + world * template_shape_count
                if local_anchor_shape is not None and "anchor_friction" in cfg:
                    group["material_jobs"].append(
                        (
                            shape_offset + int(local_anchor_shape),
                            float(cfg["anchor_friction"]),
                            float(cfg.get("anchor_elasticity", 0.0)),
                        )
                    )
                if local_target_shape is not None and "target_friction" in cfg:
                    group["material_jobs"].append(
                        (
                            shape_offset + int(local_target_shape),
                            float(cfg["target_friction"]),
                            float(cfg.get("target_elasticity", 0.0)),
                        )
                    )

    constraints: List[SoftAnchorForceConstraint] = []
    for runtime_pattern, group in grouped.items():
        if not group["anchor_ids"]:
            continue
        constraints.append(
            SoftAnchorForceConstraint(
                physics_manager,
                articulation_body=articulation_body,
                pattern=runtime_pattern,
                anchor_ids=group["anchor_ids"],
                target_ids=group["target_ids"],
                worlds=group["worlds"],
                obj_idxs=group["obj_idxs"],
                target_slots=group["target_slots"],
                offsets=group["offsets"],
                stiffness=group["stiffness"],
                damping=group["damping"],
                max_force=group["max_force"],
                material_jobs=group["material_jobs"],
            )
        )
    return constraints


class SoftAnchorForceConstraint:
    def __init__(
        self,
        physics_manager: Any,
        *,
        articulation_body: Any,
        pattern: str,
        anchor_ids: Sequence[int],
        target_ids: Sequence[int],
        worlds: Sequence[int],
        obj_idxs: Sequence[int],
        target_slots: Sequence[int],
        offsets: Sequence[Tuple[float, float, float]],
        stiffness: Sequence[float],
        damping: Sequence[float],
        max_force: Sequence[float],
        material_jobs: Sequence[Tuple[int, float, float]],
    ) -> None:
        device = physics_manager.device
        self._pm = physics_manager
        self._ab = articulation_body
        self._pattern = str(pattern)
        self._count = len(anchor_ids)
        self._anchor_body = wp.array(list(anchor_ids), dtype=wp.int32, device=device)
        self._target_body = wp.array(list(target_ids), dtype=wp.int32, device=device)
        self._world_idx = wp.array(list(worlds), dtype=wp.int32, device=device)
        self._obj_idx = wp.array(list(obj_idxs), dtype=wp.int32, device=device)
        self._target_slot = wp.array(list(target_slots), dtype=wp.int32, device=device)
        self._rest_offset = wp.array(list(offsets), dtype=wp.vec3, device=device)
        self._stiffness = wp.array(list(stiffness), dtype=wp.float32, device=device)
        self._damping = wp.array(list(damping), dtype=wp.float32, device=device)
        self._max_force = wp.array(list(max_force), dtype=wp.float32, device=device)
        self._material_jobs = list(material_jobs)
        self._attached = False

    def attach(self, physics_manager: Any) -> None:
        if self._attached or self._count <= 0:
            return
        self._pm = physics_manager
        self._apply_shape_materials(physics_manager)
        self._attached = True

    def _apply_shape_materials(self, physics_manager: Any) -> None:
        model = getattr(physics_manager, "model", None)
        if model is None or not self._material_jobs:
            return
        mu_attr = getattr(model, "shape_material_mu", None)
        if mu_attr is None:
            return
        mu_np = np.asarray(mu_attr.numpy()).copy()
        rest_attr = getattr(model, "shape_material_restitution", None)
        rest_np = np.asarray(rest_attr.numpy()).copy() if rest_attr is not None else None
        for shape_idx, friction, elasticity in self._material_jobs:
            if 0 <= shape_idx < mu_np.shape[0]:
                mu_np[shape_idx] = friction
            if rest_np is not None and 0 <= shape_idx < rest_np.shape[0]:
                rest_np[shape_idx] = elasticity
        mu_attr.assign(mu_np)
        if rest_attr is not None and rest_np is not None:
            rest_attr.assign(rest_np)

    def apply(self) -> None:
        pm = self._pm
        ab = self._ab
        if pm is None or ab is None or self._count <= 0:
            return
        force_buf = getattr(ab, "control_force_gpus", {}).get(self._pattern)
        if force_buf is None:
            return
        state = pm.state_0
        wp.launch(
            apply_soft_anchor_force_kernel,
            dim=self._count,
            inputs=[
                state.body_q,
                state.body_qd,
                pm.model.body_inv_mass,
                force_buf,
                self._anchor_body,
                self._target_body,
                self._world_idx,
                self._obj_idx,
                self._target_slot,
                self._rest_offset,
                self._stiffness,
                self._damping,
                self._max_force,
            ],
            device=pm.device,
        )
