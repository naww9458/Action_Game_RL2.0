from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import warp as wp
import newton
from newton._src.geometry import raycast


@dataclass
class PickQueryResult:
    shape_idx: int = -1
    body_idx: int = -1
    world_idx: int = 0
    global_role_idx: int = -1
    local_role_idx: int = -1
    label: str = ""


class RaycastSelector:
    def __init__(self, model: newton.Model, device):
        self.model = model
        self.device = device
        self.min_dist = wp.array([1.0e10], dtype=float, device=device)
        self.min_index = wp.array([-1], dtype=int, device=device)
        self.min_body_index = wp.array([-1], dtype=int, device=device)
        self.lock = wp.array([0], dtype=wp.int32, device=device)

    def query(
        self,
        state: newton.State,
        ray_start,
        ray_dir,
        shape_to_role: np.ndarray,
        world_offsets: Optional[wp.array],
        num_objects_env: int,
        role_labels: dict[int, str],
        visible_worlds_mask: Optional[wp.array] = None,
    ) -> PickQueryResult:
        result = PickQueryResult()
        if self.model is None or self.model.shape_count == 0:
            return result

        self.min_dist.fill_(1.0e10)
        self.min_index.fill_(-1)
        self.min_body_index.fill_(-1)
        self.lock.zero_()

        p = wp.vec3f(float(ray_start[0]), float(ray_start[1]), float(ray_start[2]))
        d = wp.vec3f(float(ray_dir[0]), float(ray_dir[1]), float(ray_dir[2]))

        shape_world = (
            self.model.shape_world
            if self.model.shape_world is not None
            else wp.array([], dtype=int, device=self.device)
        )
        if world_offsets is not None:
            offsets = world_offsets
        else:
            offsets = wp.array([], dtype=wp.vec3, device=self.device)

        visible_worlds_mask = visible_worlds_mask if visible_worlds_mask is not None else wp.array(
            [], dtype=int, device=self.device
        )

        # Newton 1.4: 9 inputs + 6 outputs (heightfields use mesh BVH internally).
        wp.launch(
            kernel=raycast.raycast_kernel,
            dim=self.model.shape_count,
            inputs=[
                state.body_q,
                self.model.shape_body,
                self.model.shape_transform,
                self.model.shape_type,
                self.model.shape_scale,
                self.model.shape_source_ptr,
                p,
                d,
                self.lock,
            ],
            outputs=[
                self.min_dist,
                self.min_index,
                self.min_body_index,
                shape_world,
                offsets,
                visible_worlds_mask,
            ],
            device=self.device,
        )
        wp.synchronize()

        dist = float(self.min_dist.numpy()[0])
        shape_idx = int(self.min_index.numpy()[0])
        body_idx = int(self.min_body_index.numpy()[0])
        if dist >= 1.0e10 or shape_idx < 0:
            return result

        result.shape_idx = shape_idx
        result.body_idx = body_idx

        if shape_idx < len(shape_to_role):
            global_role = int(shape_to_role[shape_idx])
            if global_role >= 0:
                result.global_role_idx = global_role
                result.local_role_idx = global_role % num_objects_env
                result.world_idx = global_role // num_objects_env
                result.label = role_labels.get(result.local_role_idx, "")

        if shape_world.shape[0] > 0 and shape_idx < shape_world.shape[0]:
            world_idx_np = int(shape_world.numpy()[shape_idx])
            if world_idx_np >= 0:
                result.world_idx = world_idx_np

        return result
