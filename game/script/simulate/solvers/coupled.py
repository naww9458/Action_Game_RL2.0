"""
CoupledSolver — 基於 newton.solvers.SolverCoupledProxy 的多域耦合求解器。

支援多求解器組合選擇，當選中多個求解器時自動啟用耦合模式：
- rigid domain: MuJoCo / VBD / XPBD
- soft domain:  VBD / XPBD

兩個 domain 透過 SolverCoupledProxy 耦合執行。
"""

from __future__ import annotations

import sys
from typing import Literal, Optional, TYPE_CHECKING

import newton
import warp as wp

from script.simulate.solvers.base_solver import BaseSolverModel, BaseSolver
from script.simulate.coupling_index_builder import CouplingIndexBuilder

if TYPE_CHECKING:
    pass

# ------------------------------------------------------------------
# 環境相容層：修復第三方套件之間的版本不相容，讓 MuJoCo 後端可執行。
#   - mujoco_warp 0.0.1 依賴 ``warp.context``，warp 1.16 改為 ``warp._src.context``。
#   - newton 1.4.0 引用 ``mujoco.mjtDisableBit.mjDSBL_MULTICCD``，mujoco 3.4 更名為
#     ``mjDSBL_NATIVECCD``；且 mujoco_warp 0.0.1 不認識該 disable bit，
#     因此在 mujoco 3.4+ 環境需改以 ``enable_multiccd=True`` 略過設定 disable bit。
# 這些都只影響匯入層，執行時不會再變動，因此在此一次性打補丁即可。
# ------------------------------------------------------------------
_NEEDS_MULTICCD_COMPAT: bool = False


def _apply_mujoco_compat_shims() -> bool:
    """套用 MuJoCo 相容補丁，回傳是否需要以 ``enable_multiccd=True`` 規避。"""
    global _NEEDS_MULTICCD_COMPAT

    # mujoco_warp 0.0.1 同時使用 ``import warp.context`` 與 ``wp.context.runtime``，
    # warp 1.16 移除了頂層 ``warp.context``，因此從 ``warp._src.context`` 補上。
    if "warp.context" not in sys.modules:
        src_ctx = getattr(wp, "_src", None)
        if src_ctx is not None and hasattr(src_ctx, "context"):
            sys.modules["warp.context"] = src_ctx.context
    if not hasattr(wp, "context"):
        src_ctx = getattr(wp, "_src", None)
        if src_ctx is not None and hasattr(src_ctx, "context"):
            wp.context = src_ctx.context

    try:
        import mujoco
    except ImportError:
        return False
    disable_bit = getattr(mujoco.mjtDisableBit, "mjDSBL_MULTICCD", None)
    if disable_bit is None:
        native = getattr(mujoco.mjtDisableBit, "mjDSBL_NATIVECCD", None)
        if native is not None:
            mujoco.mjtDisableBit.mjDSBL_MULTICCD = native
            _NEEDS_MULTICCD_COMPAT = True
    return _NEEDS_MULTICCD_COMPAT

# ------------------------------------------------------------------
# solver capability map — 定義每個求解器支援的 domain
# ------------------------------------------------------------------
_RIGID_CAPABLE: set[str] = {"mujoco", "mjc", "vbd", "xpbd"}
_SOFT_CAPABLE: set[str] = {"vbd", "xpbd", "mpm"}
_ALL_SOLVERS: set[str] = _RIGID_CAPABLE | _SOFT_CAPABLE

# UI / 配置層使用的顯示順序（不含 mjc 別名）
RIGID_CAPABLE_SOLVERS: tuple[str, ...] = ("mujoco", "vbd", "xpbd")
SOFT_CAPABLE_SOLVERS: tuple[str, ...] = ("vbd", "xpbd", "mpm")


def normalize_solver_name(name: str) -> str:
    """標準化求解器名稱（mjc → mujoco）。"""
    n = (name or "").lower()
    return "mujoco" if n in ("mjc", "mujoco") else n


def resolve_coupled_domains(config: dict) -> tuple[str, str]:
    """從 coupled 配置推導 (rigid_solver, soft_solver)。"""
    solvers: list[str] = config.get("solvers", ["mujoco", "vbd"])
    if not isinstance(solvers, list) or len(solvers) == 0:
        solvers = ["mujoco", "vbd"]

    normalized = [normalize_solver_name(s) for s in solvers]

    rigid = normalize_solver_name(str(config.get("rigid_solver") or ""))
    soft = normalize_solver_name(str(config.get("soft_solver") or ""))

    if not rigid or rigid not in _RIGID_CAPABLE:
        rigid = ""
        for s in normalized:
            if s in _RIGID_CAPABLE:
                rigid = s
                break
    if not soft or soft not in _SOFT_CAPABLE:
        soft = ""
        for s in normalized:
            if s in _SOFT_CAPABLE:
                soft = s
                break

    return rigid or "mujoco", soft or "vbd"


def coupled_solvers_list(rigid: str, soft: str) -> list[str]:
    """組裝 solvers 列表（去重、保留順序）。"""
    result: list[str] = []
    for name in (normalize_solver_name(rigid), normalize_solver_name(soft)):
        if name and name not in result:
            result.append(name)
    return result


def prune_coupled_solver_configs(config: dict) -> None:
    """移除 solver_configs 中未使用的求解器條目，僅保留當前 rigid / soft 求解器。"""
    if str(config.get("type", "")).lower() != "coupled":
        return
    rigid, soft = resolve_coupled_domains(config)
    allowed = {rigid, soft}
    solver_configs = config.get("solver_configs")
    if not isinstance(solver_configs, dict):
        return
    config["solver_configs"] = {
        normalize_solver_name(key): value
        for key, value in solver_configs.items()
        if normalize_solver_name(key) in allowed
    }


class CoupledSolverModel(BaseSolverModel):
    type: Literal["coupled"] = "coupled"

    # 選中的求解器列表 (多選)，若只有一項則自動切換為單求解器模式
    solvers: list[str] = ["mujoco", "vbd"]

    # domain 指派（可手動覆寫，留空則從 solvers 自動推導）
    # 自動推導規則：rigid → solvers 中第一個 rigid-capable；soft → solvers 中第一個 soft-capable
    rigid_solver: str | None = None
    soft_solver: str | None = None

    # 耦合參數
    coupling_mode: str = "lagged"
    mass_scale: float = 1.0
    proxy_iterations: int = 1

    # 子求解器自定義配置 (依求解器名稱索引)
    # 範例: {"mujoco": {"njmax": 64}, "vbd": {"iterations": 10}}
    solver_configs: dict[str, dict] | None = None


class CoupledSolver(BaseSolver):
    solver_key = "coupled"
    model_cls = CoupledSolverModel
    solver_type_id: int = 3

    def __init__(
        self,
        config: dict,
        builder: "newton.ModelBuilder",
        coupling_builder: Optional[CouplingIndexBuilder] = None,
        **kwargs,
    ):
        super().__init__(config)
        self._coupling_builder = coupling_builder
        self._rigid_solver_name = ""
        self._soft_solver_name = ""

        # VBD 需要 color 拓樸
        resolved = self._resolve_domains()
        if resolved[1] == "vbd" or resolved[0] == "vbd":
            builder.color()

    # ------------------------------------------------------------------
    # domain 解析：從 solvers 列表自動推導 rigid / soft 求解器
    # ------------------------------------------------------------------
    def _resolve_domains(self) -> tuple[str, str]:
        """返回 (rigid_solver, soft_solver)。"""
        rigid, soft = resolve_coupled_domains(self.config)
        self._rigid_solver_name = rigid
        self._soft_solver_name = soft
        return rigid, soft

    # ------------------------------------------------------------------
    # 子求解器 config 合併（新舊格式兼容）
    # ------------------------------------------------------------------
    def _solver_cfg(self, solver_name: str) -> dict:
        """取得指定求解器的配置字典。

        新格式: solver_configs: {mujoco: {...}, vbd: {...}}
        舊格式: rigid_solver_cfg / soft_solver_cfg (依 domain 自動映射)
        """
        # 新格式優先
        sc = self.config.get("solver_configs") or {}
        cfg = dict(sc.get(solver_name, {}))

        # 舊格式向後兼容（僅在新格式無此求解器 config 時生效）
        rigid_name = self._rigid_solver_name
        soft_name = self._soft_solver_name

        if not cfg:
            if solver_name == rigid_name and "rigid_solver_cfg" in self.config:
                cfg = dict(self.config.get("rigid_solver_cfg") or {})
            elif solver_name == soft_name and "soft_solver_cfg" in self.config:
                cfg = dict(self.config.get("soft_solver_cfg") or {})

        return cfg

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def setup(self, model):
        from newton.solvers.experimental.coupled import SolverCoupledProxy  # type: ignore[import-untyped]
        from newton.solvers import SolverMuJoCo, SolverVBD, SolverXPBD, SolverImplicitMPM

        _apply_mujoco_compat_shims()

        rigid_name = self._rigid_solver_name or self._resolve_domains()[0]
        soft_name = self._soft_solver_name or self._resolve_domains()[1]
        rigid_cfg = self._solver_cfg(rigid_name)
        soft_cfg = self._solver_cfg(soft_name)

        # ----------------------------------------------------------------
        # sub-solver factories
        # ----------------------------------------------------------------
        def _make_rigid_solver(m: newton.Model):
            n = rigid_name.lower()
            if n in ("mjc", "mujoco"):
                kwargs = dict(rigid_cfg)
                if _NEEDS_MULTICCD_COMPAT and not kwargs.get("enable_multiccd"):
                    kwargs["enable_multiccd"] = True
                return SolverMuJoCo(model=m, **kwargs)
            if n == "vbd":
                return SolverVBD(model=m, **rigid_cfg)
            if n == "xpbd":
                return SolverXPBD(model=m, **rigid_cfg)
            raise ValueError(f"不支援的剛體求解器: {rigid_name}")

        def _make_mpm_solver(m: newton.Model):
            from script.simulate.solvers.mpm import build_mpm_config

            # SolverImplicitMPM 的建構簽名為 (model, config=Config)，不接受 **kwargs。
            return SolverImplicitMPM(model=m, config=build_mpm_config(soft_cfg))

        def _make_soft_solver(m: newton.Model):
            n = soft_name.lower()
            if n == "vbd":
                return SolverVBD(model=m, **soft_cfg)
            if n == "xpbd":
                return SolverXPBD(model=m, **soft_cfg)
            if n == "mpm":
                return _make_mpm_solver(m)
            raise ValueError(f"不支援的軟體求解器: {soft_name}")

        # ----------------------------------------------------------------
        # 從 builder 取得多世界結構資訊
        # ----------------------------------------------------------------
        if self._coupling_builder is None:
            raise RuntimeError(
                "CoupledSolver 需要 coupling_builder，請在 PhysicsManager.setup 中傳入"
            )

        bb = self._coupling_builder
        num_env = bb.num_env
        env_body_count = bb.env_body_count
        env_joint_count = bb.env_joint_count
        env_particle_count = bb.env_particle_count

        # ----------------------------------------------------------------
        # 建立耦合 entries 與 proxies
        # ----------------------------------------------------------------
        entries = bb.build_entries(
            rigid_solver_factory=_make_rigid_solver,
            soft_solver_factory=_make_soft_solver,
            env_body_count=env_body_count,
            env_joint_count=env_joint_count,
            env_particle_count=env_particle_count,
            num_env=num_env,
        )

        proxy_config = bb.build_proxy_config(
            env_body_count=env_body_count,
            env_joint_count=env_joint_count,
            env_particle_count=env_particle_count,
            num_env=num_env,
            coupling_mode=self.config.get("coupling_mode", "lagged"),
            mass_scale=self.config.get("mass_scale", 1.0),
            proxy_iterations=self.config.get("proxy_iterations", 1),
        )

        self.solver = SolverCoupledProxy(
            model=model,
            entries=entries,
            coupling=proxy_config,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, state_in, state_out, control, contacts, dt):
        self.solver.step(state_in, state_out, control, contacts, dt)

    # ------------------------------------------------------------------
    # post_teleport_sync — 防止 teleport 後求解器內部 prev 緩衝不同步
    #   重置後剛體位置被修改但求解器內部 body_q_prev 未更新，
    #   下一幀會把位姿差誤當速度使剛體瞬間歸位（速度爆炸）。
    #
    #   SolverCoupledProxy 本身不維護 body_q_prev / particle_q_prev，
    #   這些歷史緩衝分別屬於各子求解器（VBD 等），因此必須逐一同步。
    # ------------------------------------------------------------------
    def _iter_sub_solvers(self):
        """依序回傳耦合代理內部的子求解器實例（不存在則空迭代）。"""
        solver = getattr(self, "solver", None)
        if solver is None or not hasattr(solver, "entry_names"):
            return
        for name in solver.entry_names():
            try:
                sub = solver.solver(name)
            except Exception:
                continue
            if sub is not None:
                yield sub

    def _sync_sub_solver_prev(self, sub, attr: str, src) -> None:
        """將 src 複製到子求解器的歷史緩衝（長度吻合時才執行）。"""
        buf = getattr(sub, attr, None)
        if buf is None or getattr(buf, "shape", None) is None:
            return
        if src is None or len(src) != len(buf):
            return
        wp.copy(buf, src)

    def post_teleport_sync(self, state):
        for sub in self._iter_sub_solvers():
            # 剛體歷史位置
            if state.body_q is not None:
                self._sync_sub_solver_prev(sub, "body_q_prev", state.body_q)
                self._sync_sub_solver_prev(sub, "solver_body_q_prev", state.body_q)

            # 軟體粒子歷史位置
            if state.particle_q is not None:
                self._sync_sub_solver_prev(sub, "particle_q_prev", state.particle_q)
                self._sync_sub_solver_prev(
                    sub, "pos_prev_collision_detection", state.particle_q
                )

            disp = getattr(sub, "particle_displacements", None)
            if disp is not None:
                disp.zero_()

            # 子求解器自身的歷史重置（VBD 會設置 pose rebaseline mask，
            # MuJoCo 會清空 warm-start 緩衝），flags=0 表示不改動已重置的狀態。
            sub_reset = getattr(sub, "reset", None)
            if sub_reset is not None:
                try:
                    sub_reset(state, flags=0)
                except TypeError:
                    sub_reset(state)

    def reset_history(self):
        for sub in self._iter_sub_solvers():
            set_rigid = getattr(sub, "set_rigid_history_update", None)
            if set_rigid is not None:
                set_rigid(True)

    @property
    def body_q_prev(self):
        return None
