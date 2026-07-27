"""Unit tests for wheeled_armored_vehicle_basic differential drive."""

from __future__ import annotations

import unittest

from script.role.objects.object_template.wheeled_armored_vehicle_basic.vehicle_drive import (
    DriveSpec,
    VehicleCommandInterface,
    _mix_tank_drive,
    build_vehicle_command_interface,
)
from script.role.objects.object_template.wheeled_armored_vehicle_basic.vehicle_joint_model import (
    JointRole,
    classify_joint_dof,
    resolve_dof_param_for_view,
)
from script.role.objects.object_template.wheeled_armored_vehicle_basic.vehicle_control_config import (
    get_vehicle_task_config,
)


class TankDriveMixTests(unittest.TestCase):
    def test_eight_keyboard_combinations(self) -> None:
        cases = [
            ((1.0, 0.0), (1.0, 1.0)),
            ((-1.0, 0.0), (-1.0, -1.0)),
            ((0.0, 1.0), (-1.0, 1.0)),
            ((0.0, -1.0), (1.0, -1.0)),
            ((1.0, 1.0), (0.0, 1.0)),
            ((1.0, -1.0), (1.0, 0.0)),
            ((-1.0, 1.0), (0.0, -1.0)),
            ((-1.0, -1.0), (-1.0, 0.0)),
        ]
        for (throttle, steer), expected in cases:
            with self.subTest(throttle=throttle, steer=steer):
                self.assertEqual(_mix_tank_drive(throttle, steer), expected)


class ExpandCommandsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.iface = build_vehicle_command_interface(DriveSpec(), human_control={})
        self.joint_labels = [
            "/root/vehicle_body/Susp_L1/Wheels_L1/Wheels_L1/D6Joint",
            "/root/vehicle_body/Susp_R1/Wheels_R1/Wheels_R1/D6Joint",
        ]
        self.rl_mask = [1, 1]

    def _side_targets(self, throttle: float, steer: float) -> tuple[float, float]:
        expanded = self.iface.expand_commands(
            throttle,
            steer,
            joint_labels=self.joint_labels,
            rl_mask=self.rl_mask,
        )
        return expanded[0], expanded[1]

    def test_forward_sets_both_sides_to_max_speed(self) -> None:
        left, right = self._side_targets(1.0, 0.0)
        self.assertAlmostEqual(left, 40.0)
        self.assertAlmostEqual(right, 40.0)

    def test_reverse_sets_both_sides_to_negative_max_speed(self) -> None:
        left, right = self._side_targets(-1.0, 0.0)
        self.assertAlmostEqual(left, -40.0)
        self.assertAlmostEqual(right, -40.0)

    def test_wa_sets_left_zero_right_forward(self) -> None:
        left, right = self._side_targets(1.0, 1.0)
        self.assertAlmostEqual(left, 0.0)
        self.assertAlmostEqual(right, 40.0)


class JointModelTests(unittest.TestCase):
    def test_d6_joint_is_wheel_spin_only(self) -> None:
        label = "/root/vehicle_body/Susp_L1/Wheels_L1/Wheels_L1/D6Joint"
        role = classify_joint_dof(label, local_dof=0, dof_count=1)
        self.assertEqual(role, JointRole.WHEEL_SPIN)

    def test_suspension_joint_role(self) -> None:
        label = "/root/vehicle_body/Susp_L1/Susp_L1/VB_Susp_L1_RevoluteJoint"
        role, spec = resolve_dof_param_for_view(label, 0, {})
        self.assertEqual(role, JointRole.SUSPENSION)
        self.assertIsNotNone(spec)


class VehicleConfigTests(unittest.TestCase):
    def test_command_interface_is_two_dimensional(self) -> None:
        cfg = get_vehicle_task_config()
        iface = cfg.get_command_interface()
        self.assertEqual(iface.command_dim, 2)
        self.assertTrue(iface.uses_direct_joint_torque)
        self.assertEqual(iface.binding_names, ("throttle", "steer"))

    def test_contact_spec_disables_collision_proxies_by_default(self) -> None:
        cfg = get_vehicle_task_config()
        self.assertFalse(cfg.contact_spec.use_collision_proxies)


class VehicleCollisionProxyTests(unittest.TestCase):
    def test_proxies_can_be_disabled(self) -> None:
        from pathlib import Path

        import newton
        import warp as wp
        from newton import GeoType

        from script.role.objects.object_template.wheeled_armored_vehicle_basic.vehicle_body_model import (
            VehicleContactSpec,
            apply_vehicle_collision_proxies,
        )

        asset = (
            Path(__file__).resolve().parents[6]
            / "Action_Game_RL_Assets/assets/wheeled_armored_vehicle_basic.usdc"
        )
        if not asset.exists():
            self.skipTest(f"asset not found: {asset}")

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_usd(
            str(asset),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)),
            hide_collision_shapes=True,
            load_visual_shapes=True,
            skip_mesh_approximation=True,
            only_load_enabled_joints=True,
            ignore_paths=["/root/GroundPlane"],
            force_position_velocity_actuation=True,
            parse_mujoco_options=True,
            joint_ordering="dfs",
            bodies_follow_joint_ordering=True,
            enable_self_collisions=False,
        )
        base_count = builder.shape_count

        disabled = apply_vehicle_collision_proxies(
            builder, VehicleContactSpec(use_collision_proxies=False)
        )
        self.assertEqual(disabled, (0, 0, 0, 0))
        self.assertEqual(builder.shape_count, base_count)
        self.assertEqual(int(builder.shape_type[2]), int(GeoType.MESH))

        enabled_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        enabled_builder.add_usd(
            str(asset),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)),
            hide_collision_shapes=True,
            load_visual_shapes=True,
            skip_mesh_approximation=True,
            only_load_enabled_joints=True,
            ignore_paths=["/root/GroundPlane"],
            force_position_velocity_actuation=True,
            parse_mujoco_options=True,
            joint_ordering="dfs",
            bodies_follow_joint_ordering=True,
            enable_self_collisions=False,
        )
        enabled = apply_vehicle_collision_proxies(
            enabled_builder, VehicleContactSpec(use_collision_proxies=True)
        )
        self.assertEqual(enabled[1], 1)
        self.assertEqual(enabled[2], 6)
        self.assertEqual(int(enabled_builder.shape_type[2]), int(GeoType.SPHERE))


if __name__ == "__main__":
    unittest.main()
