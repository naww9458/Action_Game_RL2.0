"""
Coupling Index Builder — 為耦合求解器 (SolverCoupledProxy) 建立 Entries 與 Proxies。

在 add_shape 階段累積物件的 body／joint／particle 區間資訊，
於 setup 階段擴展為多世界的全域索引並輸出給 CoupledSolver。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import newton


@dataclass
class CouplingObjectRecord:
    """記錄單一物件在 builder_env 中的範圍索引（尚未擴展至多世界）。"""

    label: str
    domain: str  # "rigid" | "soft"
    body_indices: List[int] = field(default_factory=list)
    joint_indices: List[int] = field(default_factory=list)
    particle_indices: List[int] = field(default_factory=list)


class CouplingIndexBuilder:
    """收集 builder_env 範圍內的 domain 劃分，並在模型 finalize 後生成耦合配置。"""

    def __init__(self) -> None:
        self._objects: List[CouplingObjectRecord] = []
        self.env_body_count: int = 0
        self.env_joint_count: int = 0
        self.env_particle_count: int = 0
        self.num_env: int = 1
        self._structure_finalized: bool = False

    def finalize_structure(
        self,
        env_body_count: int,
        env_joint_count: int,
        env_particle_count: int,
        num_env: int,
    ) -> None:
        """在 builder.add_world 之後、solver 創建之前鎖定環境結構參數。"""
        self.env_body_count = env_body_count
        self.env_joint_count = env_joint_count
        self.env_particle_count = env_particle_count
        self.num_env = num_env
        self._structure_finalized = True

    def record_object(
        self,
        label: str,
        body_start: int,
        body_end: int,
        joint_start: int,
        joint_end: int,
        particle_start: int,
        particle_end: int,
    ) -> None:
        """記錄一個物件在 builder_env 中的索引範圍。domain 由是否新增粒子自動判定。"""
        body_indices = list(range(body_start, body_end))
        joint_indices = list(range(joint_start, joint_end))
        particle_indices = list(range(particle_start, particle_end))
        domain = "soft" if particle_indices else "rigid"
        self._objects.append(
            CouplingObjectRecord(
                label=label,
                domain=domain,
                body_indices=body_indices,
                joint_indices=joint_indices,
                particle_indices=particle_indices,
            )
        )

    # ------------------------------------------------------------------
    # 索引擴展（per-world → model-global）
    # ------------------------------------------------------------------
    @staticmethod
    def _expand(indices: List[int], stride: int, num_env: int) -> List[int]:
        if not indices:
            return []
        result: List[int] = []
        for w in range(num_env):
            offset = w * stride
            for idx in indices:
                result.append(offset + idx)
        return result

    def _expand_body(self, indices: List[int], env_body_count: int, num_env: int) -> List[int]:
        return self._expand(indices, env_body_count, num_env)

    def _expand_joint(self, indices: List[int], env_joint_count: int, num_env: int) -> List[int]:
        return self._expand(indices, env_joint_count, num_env)

    def _expand_particle(self, indices: List[int], env_particle_count: int, num_env: int) -> List[int]:
        return self._expand(indices, env_particle_count, num_env)

    # ------------------------------------------------------------------
    # 聚合查詢
    # ------------------------------------------------------------------
    def _collect(self, domain: str, attr: str) -> List[int]:
        result: List[int] = []
        for obj in self._objects:
            if obj.domain == domain:
                result.extend(getattr(obj, attr))
        return result

    def rigid_body_indices(
        self, env_body_count: int, num_env: int
    ) -> List[int]:
        return self._expand_body(self._collect("rigid", "body_indices"), env_body_count, num_env)

    def rigid_joint_indices(
        self, env_joint_count: int, num_env: int
    ) -> List[int]:
        return self._expand_joint(self._collect("rigid", "joint_indices"), env_joint_count, num_env)

    def soft_particle_indices(
        self, env_particle_count: int, num_env: int
    ) -> List[int]:
        return self._expand_particle(self._collect("soft", "particle_indices"), env_particle_count, num_env)

    # ------------------------------------------------------------------
    # 建立 SolverCoupledProxy 的 Entry 與 Config
    # ------------------------------------------------------------------
    def build_entries(
        self,
        rigid_solver_factory: Callable,
        soft_solver_factory: Callable,
        env_body_count: int,
        env_joint_count: int,
        env_particle_count: int,
        num_env: int,
    ):
        from newton.solvers.experimental.coupled import SolverCoupledProxy  # type: ignore[import-untyped]

        rigid_bodies = self.rigid_body_indices(env_body_count, num_env)
        rigid_joints = self.rigid_joint_indices(env_joint_count, num_env)
        soft_particles = self.soft_particle_indices(env_particle_count, num_env)

        entries = []
        if rigid_bodies or rigid_joints:
            entries.append(
                SolverCoupledProxy.Entry(
                    name="rigid",
                    solver=rigid_solver_factory,
                    bodies=rigid_bodies,
                    joints=rigid_joints if rigid_joints else None,
                )
            )
        if soft_particles:
            entries.append(
                SolverCoupledProxy.Entry(
                    name="soft",
                    solver=soft_solver_factory,
                    particles=soft_particles,
                )
            )
        return entries

    def build_proxy_config(
        self,
        env_body_count: int,
        env_joint_count: int,
        env_particle_count: int,
        num_env: int,
        coupling_mode: str = "lagged",
        mass_scale: float = 1.0,
        proxy_iterations: int = 1,
    ):
        from newton.solvers.experimental.coupled import SolverCoupledProxy  # type: ignore[import-untyped]

        rigid_bodies = self.rigid_body_indices(env_body_count, num_env)
        soft_particles = self.soft_particle_indices(env_particle_count, num_env)

        proxies = []
        if rigid_bodies and soft_particles:
            proxies.append(
                SolverCoupledProxy.Proxy(
                    source="rigid",
                    destination="soft",
                    bodies=rigid_bodies,
                    mass_scale=mass_scale,
                    mode=coupling_mode,
                    collision_pipeline=lambda model: newton.CollisionPipeline(model),
                    collide_interval=1,
                )
            )

        return SolverCoupledProxy.Config(
            proxies=proxies,
            iterations=proxy_iterations,
        )
