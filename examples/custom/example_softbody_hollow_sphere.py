# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody Hanging
#
# This simulation demonstrates volumetric soft bodies (tetrahedral grids) hanging
# from fixed particles on the left side. Four grids with different damping values
# (1e-1 to 1e-4) showcase the effect of damping on Neo-Hookean elastic behavior.
#
# Command: uv run -m newton.examples softbody.example_softbody_hanging
#
###########################################################################

# Modify from example_softbody_hanging.py

import os

import warp as wp

import newton
import newton.examples
from pxr import Usd


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.solver_type = args.solver
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 5
        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.step_current = 0
        self.step_reset = 10

        if self.solver_type != "vbd":
            raise ValueError("The hanging softbody example only supports the VBD solver.")

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        usd_stage = Usd.Stage.Open("./assets/hollow_sphere2_TetMesh.usdc")
        prim = usd_stage.GetPrimAtPath("/root/sphere/TetMesh")
        tetmesh = newton.TetMesh.create_from_usd(prim)

        # Duck USDA is in meters (metersPerUnit=1.0).
        # Table top is at z=0.2m. Duck center offset ~0.03m above table.
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, -1.5, 1.0),
            rot=wp.quat(0.0, 0.0, 0.0, 1.0),
            scale=0.5,  # already in meters
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=tetmesh,
            particle_radius=0.15,
        )


        # usd_stage = Usd.Stage.Open("./assets/hollow_sphere2.usdc")
        # prim = usd_stage.GetPrimAtPath("/root/sphere/Mesh")
        # mesh = newton.Mesh.create_from_usd(prim)
        # body = builder.add_body()
        # pos = wp.vec3(2.0, 0.0, 1.0)
        # xform = wp.transform(pos, wp.quat_identity())
        # builder.add_shape_mesh(
        #     body=body,
        #     mesh=mesh,
        #     xform=xform,
        # )

        # body = builder.add_body()
        # pos=wp.vec3(0.0, -1.5, 1.0)
        # # pos = wp.vec3(3.0, 0.0, 1.0)
        # xform = wp.transform(pos, wp.quat_identity())
        # builder.add_shape_sphere(
        #     body=body,
        #     radius=0.1,
        #     xform=xform,
        # )

        # for i in range(4):
        #     body = builder.add_body()
        #     pos=wp.vec3(-0.2, -1.6, 1.2)
        #     xform = wp.transform(pos, wp.quat_identity())
        #     builder.add_shape_sphere(
        #         body=body,
        #         radius=0.1,
        #         xform=xform,
        #     )
        #     body = builder.add_body()
        #     pos=wp.vec3(-0.1, -1.7, 0.8)
        #     xform = wp.transform(pos, wp.quat_identity())
        #     builder.add_shape_sphere(
        #         body=body,
        #         radius=0.1,
        #         xform=xform,
        #     )
        #     body = builder.add_body()
        #     pos=wp.vec3(0.0, -1.3, 1.0)
        #     xform = wp.transform(pos, wp.quat_identity())
        #     builder.add_shape_sphere(
        #         body=body,
        #         radius=0.1,
        #         xform=xform,
        #     )
        #     body = builder.add_body()
        #     pos=wp.vec3(0.1, -1.5, 1.0)
        #     xform = wp.transform(pos, wp.quat_identity())
        #     builder.add_shape_sphere(
        #         body=body,
        #         radius=0.1,
        #         xform=xform,
        #     )


        # Color the mesh for VBD solver
        builder.color()

        self.model = builder.finalize()
        # self.model.soft_contact_ke = 1.0e1
        # self.model.soft_contact_kd = 0
        self.model.soft_contact_ke = 10000  # 增加勁度可以讓碰撞更精確（不穿透）
        self.model.soft_contact_kd = -100.0  # 必須增加阻尼，防止能量爆炸

        self.model.soft_contact_mu = 0.5

        print("self.model.body_mass: ", self.model.body_mass)
        print("self.model.particle_mass: ", self.model.particle_mass)
        print("self.model.particle_mass.shape: ", self.model.particle_mass.shape)
        
        # mass_list = [1.0, 1.0]
        # self.model.body_mass = wp.array(mass_list, dtype=wp.float32, device=self.model.device)
        # print("self.model.body_mass: ", self.model.body_mass)
        
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.control = self.model.control()

        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        # Test that particles are in a reasonable range (soft body may settle or deform)
        # We check that they haven't exploded or collapsed completely
        # 4 grids, each roughly 1.2 x 0.4 x 0.4 in size, positioned along Y-axis
        # Initial positions: Y from 1.0 to ~3.2, X from 0 to 1.2, Z around 1.0 to 1.4
        # With fix_left=True, grids hang and sag significantly towards the ground
        p_lower = wp.vec3(-1.0, -0.5, 0.0)
        p_upper = wp.vec3(3.0, 4.0, 3.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--solver",
            help="Type of solver (only 'vbd' supports volumetric soft bodies)",
            type=str,
            choices=["vbd"],
            default="vbd",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
