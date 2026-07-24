import numpy as np
import warp as wp

# Contact kind tags for raw buffer extensibility (soft/fluid future use)
CONTACT_KIND_RIGID = 0
CONTACT_KIND_SOFT = 1
CONTACT_KIND_FLUID = 2

INT32_MAX = 2**31 - 1
GROUND_SHAPE_INDEX = 0


@wp.kernel
def record_contacts_kernel(
    contact_count: wp.array(dtype=int),
    shape0_array: wp.array(dtype=int),
    shape1_array: wp.array(dtype=int),
    shape_to_role: wp.array(dtype=int),
    num_roles: int,
    role_contact_matrix: wp.array(dtype=int),
    ground_contact_flags: wp.array(dtype=int),
    raw_shape0: wp.array(dtype=int),
    raw_shape1: wp.array(dtype=int),
    raw_contact_kind: wp.array(dtype=int),
    raw_count: wp.array(dtype=int),
    raw_capacity: int,
    contact_kind: int,
):
    tid = wp.tid()
    limit = contact_count[0]

    if tid >= limit:
        return

    s0 = shape0_array[tid]
    s1 = shape1_array[tid]

    slot = wp.atomic_add(raw_count, 0, 1)
    if slot < raw_capacity:
        raw_shape0[slot] = s0
        raw_shape1[slot] = s1
        raw_contact_kind[slot] = contact_kind

    if s0 == GROUND_SHAPE_INDEX or s1 == GROUND_SHAPE_INDEX:
        other = s1
        if s1 == GROUND_SHAPE_INDEX:
            other = s0
        if other > 0 and other < shape_to_role.shape[0]:
            role = shape_to_role[other]
            if role >= 0 and role < num_roles:
                wp.atomic_max(ground_contact_flags, role, 1)

    if s0 <= 0 or s1 <= 0:
        return
    if s0 >= shape_to_role.shape[0] or s1 >= shape_to_role.shape[0]:
        return

    r0 = shape_to_role[s0]
    r1 = shape_to_role[s1]

    if r0 < 0 or r1 < 0:
        return
    if r0 >= num_roles or r1 >= num_roles:
        return

    idx1 = r0 * num_roles + r1
    idx2 = r1 * num_roles + r0
    wp.atomic_max(role_contact_matrix, idx1, 1)
    wp.atomic_max(role_contact_matrix, idx2, 1)


def build_shape_to_role_map(
    shape_count: int,
    role_shape_ranges: list[tuple[int, int, int]],
    template_shape_count: int,
    num_env: int,
    num_objects_env: int,
) -> np.ndarray:
    """Map Newton shape indices to logical role object indices."""
    shape_to_role = np.full(shape_count, -1, dtype=np.int32)

    for world in range(num_env):
        role_offset = world * num_objects_env
        shape_offset = 1 + world * template_shape_count

        for shape_begin, shape_end, role_index in role_shape_ranges:
            global_role = role_index + role_offset
            for local_s in range(shape_begin, shape_end):
                global_shape = shape_offset + local_s
                if 0 <= global_shape < shape_count:
                    shape_to_role[global_shape] = global_role

    return shape_to_role


class ContactSensor:
    """Scalable contact tracking at role/object level with bounded raw contact buffer."""

    def __init__(
        self,
        num_roles: int,
        shape_count: int,
        shape_to_role_np: np.ndarray,
        raw_capacity: int,
        device: str,
    ):
        self.num_roles = num_roles
        self.shape_count = shape_count
        self.device = device
        self.raw_capacity = raw_capacity

        matrix_size = num_roles * num_roles
        if matrix_size > INT32_MAX:
            raise ValueError(
                f"Role contact matrix size {matrix_size} exceeds int32 limit. "
                f"Reduce num_objects_total ({num_roles})."
            )

        self.role_contact_matrix = wp.zeros(matrix_size, dtype=wp.int32, device=device)
        self.ground_contact_flags = wp.zeros(num_roles, dtype=wp.int32, device=device)
        self.shape_to_role_gpu = wp.array(shape_to_role_np, dtype=wp.int32, device=device)

        self.raw_shape0 = wp.zeros(raw_capacity, dtype=wp.int32, device=device)
        self.raw_shape1 = wp.zeros(raw_capacity, dtype=wp.int32, device=device)
        self.raw_contact_kind = wp.zeros(raw_capacity, dtype=wp.int32, device=device)
        self.raw_count = wp.zeros(1, dtype=wp.int32, device=device)

        matrix_bytes = matrix_size * 4
        map_bytes = shape_count * 4
        raw_bytes = raw_capacity * 12
        print(
            f"ContactSensor: num_roles={num_roles}, shape_count={shape_count}, "
            f"role_matrix={matrix_bytes / (1024 * 1024):.2f} MB, "
            f"shape_map={map_bytes / (1024 * 1024):.2f} MB, "
            f"raw_buffer_cap={raw_capacity} ({raw_bytes / 1024:.1f} KB)"
        )

    @property
    def collision_matrix(self):
        """Backward-compat alias for role-level contact matrix."""
        return self.role_contact_matrix

    def reset_frame(self):
        self.role_contact_matrix.zero_()
        self.ground_contact_flags.zero_()
        self.raw_count.zero_()

    def record_rigid_contacts(self, contacts):
        wp.launch(
            kernel=record_contacts_kernel,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                self.shape_to_role_gpu,
                self.num_roles,
                self.role_contact_matrix,
                self.ground_contact_flags,
                self.raw_shape0,
                self.raw_shape1,
                self.raw_contact_kind,
                self.raw_count,
                self.raw_capacity,
                CONTACT_KIND_RIGID,
            ],
            device=self.device,
        )

    def check_role_collision(self, role_a: int, role_b: int) -> bool:
        matrix_np = self.role_contact_matrix.numpy()
        idx = role_a * self.num_roles + role_b
        return matrix_np[idx] == 1
