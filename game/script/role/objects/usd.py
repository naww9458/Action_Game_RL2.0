import newton
import warp as wp

from typing import List, Literal
from script.simulate.mesh_builder import MeshBuilder
from script.role.objects.base_object import BaseObjectModel, BaseObject
from script.role.abilities.articulation_control_config.joint_config_registry import (
    apply_physics_init_for_pattern,
)


class UsdModel(BaseObjectModel):
    type: Literal["usd"] = "usd"

    file_name: str = ""
    file_path_or_source: str = ""
    use_mujoco_policy_init: bool = False
    control_task: str | None = None
    control_policy_version: str | None = None
    policy_checkpoint: str | None = None
    policy_device: str | None = None

    floating: bool | None = None
    base_joint: dict | None = None
    parent_body: int = -1
    only_load_enabled_rigid_bodies: bool = False
    only_load_enabled_joints: bool = True
    joint_drive_gains_scaling: float = 1
    verbose: bool = False
    ignore_paths: list[str] | None = None
    collapse_fixed_joints: bool = False
    enable_self_collisions: bool = True
    apply_up_axis_from_stage: bool = False
    root_path: str = "/"
    joint_ordering: Literal['bfs', 'dfs'] | None = "dfs"
    bodies_follow_joint_ordering: bool = True
    skip_mesh_approximation: bool = False
    load_authored_normals: bool = False
    load_sites: bool = True
    load_visual_shapes: bool = True
    hide_collision_shapes: bool = False
    force_show_colliders: bool = False
    parse_mujoco_options: bool = True
    mesh_maxhullvert: int | None = None
    schema_resolvers: list | None = None
    force_position_velocity_actuation: bool = False
    override_root_xform: bool = False


def _join_asset_path(base: str, name: str) -> str:
    base = base.rstrip("/\\")
    name = name.lstrip("/\\")
    return f"{base}/{name}"


def _load_authored_mesh_normals(asset_path: str, object_data: dict, builder_env) -> None:
    """Load authored faceVarying normals from the USD file and set them on
    matching Newton shape sources.

    Newton's USD importer defaults to ``load_normals=False``, so authored
    normals are dropped during ``add_usd``. This re-opens the stage, loads
    meshes with ``load_normals=True`` (which uses vertex-splitting to convert
    faceVarying normals to per-vertex), and copies the normals onto the
    existing shape sources so the viewer renders hard edges correctly.
    """
    path_shape_map = object_data.get("path_shape_map") if object_data else None
    if not path_shape_map:
        return

    try:
        from pxr import Usd
        import newton.usd as newton_usd
    except ImportError:
        return

    try:
        stage = Usd.Stage.Open(asset_path)
    except Exception:
        return
    if stage is None:
        return

    loaded_count = 0
    for prim_path, shape_idx in path_shape_map.items():
        if shape_idx < 0 or shape_idx >= builder_env.shape_count:
            continue
        mesh = builder_env.shape_source[shape_idx]
        if mesh is None:
            continue
        if getattr(mesh, "_normals", None) is not None:
            continue

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue

        try:
            loaded = newton_usd.get_mesh(prim, load_normals=True)
        except Exception:
            continue

        if getattr(loaded, "_normals", None) is None:
            continue

        device = mesh.vertices.device
        mesh.vertices = wp.array(loaded.vertices, dtype=wp.vec3, device=device)
        mesh.indices = wp.array(loaded.indices, dtype=wp.int32, device=device)
        mesh._normals = wp.array(loaded._normals, dtype=wp.vec3, device=device)
        if getattr(loaded, "_uvs", None) is not None:
            mesh._uvs = wp.array(loaded._uvs, dtype=wp.vec2, device=device)
        loaded_count += 1

    if loaded_count > 0:
        print(f"[UsdObject] Loaded authored normals for {loaded_count} mesh shape(s) from {asset_path}")


class UsdObject(BaseObject):
    object_key = "usd"
    model_cls = UsdModel
    object_type_id: int = 6

    @staticmethod
    def add_physics(builder_env: newton.ModelBuilder, label: str, data: UsdModel, pos, rot, vel, **kwargs):
        use_policy_init = bool(data.get("use_mujoco_policy_init"))
        start_q_idx = len(builder_env.joint_q)
        joint_start = builder_env.joint_count

        if use_policy_init:
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder_env)
            builder_env.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
                limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
            )
            builder_env.default_shape_cfg.ke = 1.0e3 # TODO Hardcode
            builder_env.default_shape_cfg.kd = 2.0e2
            builder_env.default_shape_cfg.kf = 1.0e3
            builder_env.default_shape_cfg.mu = 0.75

        path = _join_asset_path(data["file_path_or_source"], data["file_name"])
        add_usd_kwargs = dict(
            xform=wp.transform(wp.vec3(pos[0], pos[1], pos[2])),
            floating=data["floating"],
            base_joint=data["base_joint"],
            parent_body=data["parent_body"],
            only_load_enabled_rigid_bodies=data["only_load_enabled_rigid_bodies"],
            only_load_enabled_joints=data["only_load_enabled_joints"],
            joint_drive_gains_scaling=data["joint_drive_gains_scaling"],
            verbose=data["verbose"],
            ignore_paths=data["ignore_paths"],
            collapse_fixed_joints=data["collapse_fixed_joints"],
            enable_self_collisions=data["enable_self_collisions"],
            apply_up_axis_from_stage=data["apply_up_axis_from_stage"],
            root_path=data["root_path"],
            joint_ordering=data["joint_ordering"],
            bodies_follow_joint_ordering=data["bodies_follow_joint_ordering"],
            skip_mesh_approximation=data["skip_mesh_approximation"],
            load_sites=data["load_sites"],
            load_visual_shapes=data["load_visual_shapes"],
            hide_collision_shapes=data["hide_collision_shapes"],
            force_show_colliders=data["force_show_colliders"],
            parse_mujoco_options=data["parse_mujoco_options"],
            mesh_maxhullvert=data["mesh_maxhullvert"],
            schema_resolvers=data["schema_resolvers"],
            force_position_velocity_actuation=data["force_position_velocity_actuation"],
            override_root_xform=data["override_root_xform"],
        )
        object_data = builder_env.add_usd(path, **add_usd_kwargs)
        if isinstance(object_data, dict):
            object_data = dict(object_data)
            object_data["joint_start"] = joint_start
            object_data["joint_end"] = builder_env.joint_count
        builder_env.articulation_label[-1] = f"{label}_articulation"

        if use_policy_init:
            load_authored_normals = data.get("load_authored_normals", False)
            if load_authored_normals:
                _load_authored_mesh_normals(path, object_data, builder_env)
            apply_physics_init_for_pattern(
                label,
                builder_env,
                start_q_idx,
                joint_start,
                builder_env.joint_count,
                task_name=data.get("control_task"),
            )
            if not data.get("skip_mesh_approximation", False):
                try:
                    # Keep the authored visual mesh (e.g. wheel-well recesses) and only
                    # approximate hidden collision copies for shapes that still collide.
                    builder_env.approximate_meshes(
                        "bounding_box", keep_visual_shapes=True
                    )
                except Exception as exc:
                    print(
                        f"[UsdObject] approximate_meshes skipped for '{label}': {exc}"
                    )

        return object_data

    @staticmethod
    def add_visual(mesh_builder: MeshBuilder, data: UsdModel, pos):
        raise NotImplementedError()

    @staticmethod
    def get_size(data: UsdModel) -> List[float]:
        print(f"{__class__.__name__}.get_size not implemented")
        return [0.0, 0.0, 0.0]
