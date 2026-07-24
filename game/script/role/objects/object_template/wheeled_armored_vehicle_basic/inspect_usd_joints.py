"""Load wheeled_armored_vehicle_basic.usdc and dump joint/body/DOF mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import warp as wp

# Resolve project root (assets/) and game/ on sys.path
_game_dir = Path(__file__).resolve().parents[5]
_project_root = _game_dir.parent
if str(_game_dir) not in sys.path:
    sys.path.insert(0, str(_game_dir))

import newton  # noqa: E402


def _joint_type_name(joint_type: int) -> str:
    names = {
        int(newton.JointType.FREE): "FREE",
        int(newton.JointType.BALL): "BALL",
        int(newton.JointType.REVOLUTE): "REVOLUTE",
        int(newton.JointType.PRISMATIC): "PRISMATIC",
        int(newton.JointType.FIXED): "FIXED",
        int(newton.JointType.D6): "D6",
    }
    return names.get(int(joint_type), f"UNKNOWN({joint_type})")


def main() -> None:
    asset_path = _project_root / "Action_Game_RL_Assets/assets" / "wheeled_armored_vehicle_basic.usdc"
    if not asset_path.exists():
        raise FileNotFoundError(f"Asset not found: {asset_path}")

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
    )
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 2.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.75

    builder.add_usd(
        str(asset_path),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)),
        floating=None,
        base_joint=None,
        parent_body=-1,
        only_load_enabled_rigid_bodies=False,
        only_load_enabled_joints=True,
        joint_drive_gains_scaling=1.0,
        verbose=True,
        ignore_paths=["/root/GroundPlane"],
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        apply_up_axis_from_stage=False,
        root_path="/",
        joint_ordering="dfs",
        bodies_follow_joint_ordering=True,
        skip_mesh_approximation=False,
        load_sites=True,
        load_visual_shapes=True,
        hide_collision_shapes=True,
        force_show_colliders=False,
        parse_mujoco_options=True,
        mesh_maxhullvert=None,
        schema_resolvers=None,
        force_position_velocity_actuation=True,
        override_root_xform=False,
    )

    joint_qd_start = builder.joint_qd_start
    joint_type = builder.joint_type
    joint_label = builder.joint_label

    print("\n=== Bodies ===")
    print(f"body_count: {builder.body_count}")
    for i in range(builder.body_count):
        label = builder.body_label[i] if i < len(builder.body_label) else "?"
        print(f"  body[{i}]: {label}")

    print("\n=== Joints ===")
    print(f"joint_count: {builder.joint_count}, joint_dof_count: {builder.joint_dof_count}")
    dof_rows: list[tuple[int, int, str, str, int]] = []
    for j in range(builder.joint_count):
        label = str(joint_label[j]) if j < len(joint_label) else "?"
        jtype = _joint_type_name(joint_type[j])
        qd_start = int(joint_qd_start[j])
        qd_end = (
            int(joint_qd_start[j + 1])
            if j + 1 < builder.joint_count
            else builder.joint_dof_count
        )
        dof_count = qd_end - qd_start
        print(f"  joint[{j}]: {label}  type={jtype}  qd=[{qd_start}:{qd_end}]  dofs={dof_count}")
        for dof in range(qd_start, qd_end):
            dof_rows.append((dof, j, label, jtype, dof - qd_start))

    print("\n=== Per-DOF mapping (articulation view order, excluding FREE) ===")
    # Simulate ArticulationView excluding FREE joint
    for dof, j, label, jtype, local_dof in dof_rows:
        if jtype == "FREE":
            continue
        print(f"  dof[{dof}]: joint={label} type={jtype} local_dof={local_dof}")

    left_wheel_dofs = [r for r in dof_rows if "wheels_l" in r[2].lower() or "wheels/l" in r[2].lower()]
    right_wheel_dofs = [r for r in dof_rows if "wheels_r" in r[2].lower() or "wheels/r" in r[2].lower()]
    print(f"\n=== Wheel DOF summary ===")
    print(f"  Left wheel DOFs: {len(left_wheel_dofs)}")
    print(f"  Right wheel DOFs: {len(right_wheel_dofs)}")

    out_path = Path(__file__).parent / "joint_mapping.md"
    lines = [
        "# Wheeled Armored Vehicle Basic — Joint Mapping",
        "",
        f"Asset: `Action_Game_RL_Assets/assets/wheeled_armored_vehicle_basic.usdc`",
        "",
        f"- Bodies: {builder.body_count}",
        f"- Joints: {builder.joint_count}",
        f"- Total DOFs: {builder.joint_dof_count}",
        f"- Left wheel DOFs: {len(left_wheel_dofs)}",
        f"- Right wheel DOFs: {len(right_wheel_dofs)}",
        "",
        "## Bodies",
        "",
    ]
    for i in range(builder.body_count):
        label = builder.body_label[i] if i < len(builder.body_label) else "?"
        lines.append(f"- `{i}`: `{label}`")

    lines.extend(["", "## Joints", ""])
    for j in range(builder.joint_count):
        label = str(joint_label[j]) if j < len(joint_label) else "?"
        jtype = _joint_type_name(joint_type[j])
        qd_start = int(joint_qd_start[j])
        qd_end = (
            int(joint_qd_start[j + 1])
            if j + 1 < builder.joint_count
            else builder.joint_dof_count
        )
        lines.append(
            f"- `{label}` — {jtype}, qd=[{qd_start}:{qd_end}], dofs={qd_end - qd_start}"
        )

    lines.extend(["", "## Per-DOF (non-FREE)", ""])
    lines.append("| DOF | Joint | Type | Local DOF | Role | Target Mode |")
    lines.append("|-----|-------|------|-----------|------|-------------|")
    for dof, j, label, jtype, local_dof in dof_rows:
        if jtype == "FREE":
            continue
        basename = label.split("/")[-1].lower()
        if "vb_susp" in basename and "revolute" in basename:
            role, mode = "suspension", "POSITION"
        elif "d6joint" in basename or "wheels_" in basename:
            if local_dof == 0:
                role, mode = "wheel_spin_primary", "VELOCITY"
            else:
                role, mode = "wheel_spin_extra", "POSITION (locked)"
        else:
            role, mode = "unknown", "POSITION"
        lines.append(f"| {dof} | `{label}` | {jtype} | {local_dof} | {role} | {mode} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
