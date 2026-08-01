import newton
import warp as wp

from typing import List, Literal, Optional
from script.simulate.mesh_builder import MeshBuilder
from script.role.objects.base_object import BaseObjectModel, BaseObject
from script.role.objects.collision_shape_override import apply_body_collision_shape_overrides
from script.role.abilities.articulation_control_config.joint_config_registry import (
    apply_physics_init_for_pattern,
)


class MjcfModel(BaseObjectModel):
    type: Literal["mjcf"] = "mjcf"

    file_name: str = "g1.xml"
    file_path_or_source: str = "Action_Game_RL_Assets/assets/external_sources/mjlab_unitree_g1"
    use_mujoco_policy_init: bool = False
    control_task: str | None = None
    control_policy_version: str | None = None
    policy_checkpoint: str | None = None
    policy_device: str | None = None

    floating: bool | None = None
    base_joint: dict | None = None
    parent_body: int = -1
    armature_scale: float = 1.0
    scale: float = 1.0
    hide_visuals: bool = False
    parse_visuals_as_colliders: bool = False
    parse_meshes: bool = True
    parse_sites: bool = True
    parse_visuals: bool = True
    parse_mujoco_options: bool = True
    ignore_names: list[str] | None = None
    ignore_classes: list[str] | None = None
    visual_classes: list[str] | None = None
    collider_classes: list[str] | None = None
    no_class_as_colliders: bool = True
    force_show_colliders: bool = False
    enable_self_collisions: bool = True
    ignore_inertial_definitions: bool = False
    collapse_fixed_joints: bool = False
    verbose: bool = False
    skip_equality_constraints: bool = False
    convert_3d_hinge_to_ball_joints: bool = False
    mesh_maxhullvert: int | None = None
    ctrl_direct: bool = False
    override_root_xform: bool = False
    approximate_meshes: bool = True
    body_collision_shape_overrides: dict[str, str] | None = None


def _join_asset_path(base: str, name: str) -> str:
    base = base.rstrip("/\\")
    name = name.lstrip("/\\")
    return f"{base}/{name}"


class MjcfObject(BaseObject):
    object_key = "mjcf"
    model_cls = MjcfModel
    object_type_id: int = 7

    @staticmethod
    def add_physics(builder_env: newton.ModelBuilder, label: str, data: MjcfModel, pos, rot, vel, **kwargs):
        use_policy_init = bool(data.get("use_mujoco_policy_init"))
        start_q_idx = len(builder_env.joint_q)
        joint_start = builder_env.joint_count

        if use_policy_init:
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder_env)
            builder_env.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
                limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
            )
            builder_env.default_shape_cfg.ke = 1.0e3
            builder_env.default_shape_cfg.kd = 2.0e2
            builder_env.default_shape_cfg.kf = 1.0e3
            builder_env.default_shape_cfg.mu = 0.75

        path = _join_asset_path(data["file_path_or_source"], data["file_name"])
        shape_start = builder_env.shape_count
        visual_classes = data.get("visual_classes")
        if visual_classes is None:
            visual_classes = ("visual",)
        collider_classes = data.get("collider_classes")
        if collider_classes is None:
            collider_classes = ("collision",)

        add_mjcf_kwargs = dict(
            xform=wp.transform(wp.vec3(pos[0], pos[1], pos[2])),
            floating=data.get("floating"),
            base_joint=data.get("base_joint"),
            parent_body=data.get("parent_body", -1),
            armature_scale=data.get("armature_scale", 1.0),
            scale=data.get("scale", 1.0),
            hide_visuals=data.get("hide_visuals", False),
            parse_visuals_as_colliders=data.get("parse_visuals_as_colliders", False),
            parse_meshes=data.get("parse_meshes", True),
            parse_sites=data.get("parse_sites", True),
            parse_visuals=data.get("parse_visuals", True),
            parse_mujoco_options=data.get("parse_mujoco_options", True),
            ignore_names=tuple(data.get("ignore_names") or ()),
            ignore_classes=tuple(data.get("ignore_classes") or ()),
            visual_classes=tuple(visual_classes),
            collider_classes=tuple(collider_classes),
            no_class_as_colliders=data.get("no_class_as_colliders", True),
            force_show_colliders=data.get("force_show_colliders", False),
            enable_self_collisions=data.get("enable_self_collisions", True),
            ignore_inertial_definitions=data.get("ignore_inertial_definitions", False),
            collapse_fixed_joints=data.get("collapse_fixed_joints", False),
            verbose=data.get("verbose", False),
            skip_equality_constraints=data.get("skip_equality_constraints", False),
            convert_3d_hinge_to_ball_joints=data.get("convert_3d_hinge_to_ball_joints", False),
            mesh_maxhullvert=data.get("mesh_maxhullvert"),
            ctrl_direct=data.get("ctrl_direct", False),
            override_root_xform=data.get("override_root_xform", False),
        )
        object_data = builder_env.add_mjcf(path, **add_mjcf_kwargs)
        builder_env.articulation_label[-1] = f"{label}_articulation"

        if use_policy_init:
            apply_physics_init_for_pattern(
                data.get("pattern", "default"),
                builder_env,
                start_q_idx,
                joint_start,
                builder_env.joint_count,
            )
            if data.get("approximate_meshes", True):
                # Scope the approximation to shapes added by this MJCF import only,
                # so a single object's setting never affects other objects that
                # share the same builder.
                shape_indices = [
                    i
                    for i in range(shape_start, builder_env.shape_count)
                    if builder_env.shape_type[i] == newton.GeoType.MESH
                    and builder_env.shape_flags[i] & newton.ShapeFlags.COLLIDE_SHAPES
                ]
                if shape_indices:
                    builder_env.approximate_meshes(
                        "bounding_box", shape_indices=shape_indices
                    )

        # Per-body collision shape type overrides (e.g. wheels -> cylinder).
        # Applied last so they win over any generic approximation above.
        overrides = data.get("body_collision_shape_overrides")
        if overrides:
            try:
                replaced = apply_body_collision_shape_overrides(
                    builder_env,
                    shape_start=shape_start,
                    overrides=dict(overrides),
                )
                if replaced > 0:
                    print(
                        f"[MjcfObject] Applied {replaced} body collision shape "
                        f"override(s) for '{label}'"
                    )
            except Exception as exc:
                print(
                    f"[MjcfObject] body_collision_shape_overrides skipped for "
                    f"'{label}': {exc}"
                )

        return object_data

    @staticmethod
    def add_visual(mesh_builder: MeshBuilder, data: MjcfModel, pos):
        raise NotImplementedError()

    @staticmethod
    def get_size(data: MjcfModel) -> List[float]:
        print(f"{__class__.__name__}.get_size not implemented")
        return [0.0, 0.0, 0.0]
