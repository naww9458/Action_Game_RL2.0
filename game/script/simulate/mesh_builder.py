import numpy as np
import warp as wp

class MeshBuilder:
    def __init__(self):
        self.vertices = []
        self.indices = []
        self.vertex_count = 0

        # 預定義標準單位的立方體頂點 (範圍 -1 到 1)
        # 順序：0-3 底面, 4-7 頂面
        self._unit_box_verts = np.array([
            [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
            [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
        ], dtype=np.float32)

        # 預定義立方體 12 個三角形的索引 (逆時針 CCW)
        self._box_indices_template = np.array([
            [0, 2, 1], [0, 3, 2], # 底面 (-Z)
            [4, 5, 6], [4, 6, 7], # 頂面 (+Z)
            [0, 1, 5], [0, 5, 4], # 前面 (-Y)
            [1, 2, 6], [1, 6, 5], # 右面 (+X)
            [2, 3, 7], [2, 7, 6], # 後面 (+Y)
            [3, 0, 4], [3, 4, 7]  # 左面 (-X)
        ], dtype=np.int32)

    def _quat_to_matrix(self, q):
        """將四元數 (x, y, z, w) 轉換為 3x3 旋轉矩陣 (純 NumPy 實現)"""
        x, y, z, w = q
        return np.array([
            [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
            [2*x*y + 2*z*w,           1 - 2*x**2 - 2*z**2,   2*y*z - 2*x*w],
            [2*x*z - 2*y*w,           2*y*z + 2*x*w,         1 - 2*x**2 - 2*y**2]
        ], dtype=np.float32)

    def add_box(self, pos, size, rot_quat=(0, 0, 0, 1)):
        """
        pos: [x, y, z]
        size: [hx, hy, hz] 
        rot_quat: [x, y, z, w]
        """
        # 1. 縮放頂點
        scaled_verts = self._unit_box_verts * np.array(size)

        # 2. 旋轉頂點
        rotation_matrix = self._quat_to_matrix(rot_quat)
        # 使用矩陣乘法: (N, 3) @ (3, 3).T
        rotated_verts = np.dot(scaled_verts, rotation_matrix.T)

        # 3. 平移頂點
        world_verts = rotated_verts + np.array(pos)

        # 4. 加入頂點列表
        self.vertices.extend(world_verts.tolist())

        # 5. 加入索引 (偏移當前的頂點數)
        shifted_indices = self._box_indices_template + self.vertex_count
        self.indices.extend(shifted_indices.flatten().tolist())

        # 6. 更新計數器
        self.vertex_count += 8

    def add_sphere(self, pos, radius, slices=8, stacks=8):
        """簡單的球體經緯線剖分"""
        for i in range(stacks + 1):
            phi = np.pi * i / stacks
            for j in range(slices + 1):
                theta = 2 * np.pi * j / slices
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                self.vertices.append([x + pos[0], y + pos[1], z + pos[2]])

        for i in range(stacks):
            for j in range(slices):
                p1 = i * (slices + 1) + j
                p2 = p1 + (slices + 1)
                self.indices.extend([self.vertex_count + p1, self.vertex_count + p2, self.vertex_count + p1 + 1])
                self.indices.extend([self.vertex_count + p2, self.vertex_count + p2 + 1, self.vertex_count + p1 + 1])
        self.vertex_count += (stacks + 1) * (slices + 1)

    def finalize(self, device):
        """生成 Warp Mesh 加速結構"""
        if not self.vertices:
            return None
            
        mesh = wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3, device=device),
            indices=wp.array(self.indices, dtype=wp.int32, device=device)
        )
        return mesh