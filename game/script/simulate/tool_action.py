"""Generic attached-tool action interface.

A tool action is the runtime behavior of a mounted tool (e.g. the
turret_110mm "aim" action). Common mount/simulation modules only know about
``ToolAction`` hooks; the concrete action implementation lives inside the
corresponding ``object_template/<pattern>/functions/`` folder and is loaded
lazily only when a level actually uses that tool pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from script.simulate.mount_joint_registry import MountJointRegistry, ToolMountRecord


class ToolAction:
    """Base class for per-tool mounted behavior.

    Lifecycle:
      ``configure``          → level-setup time (reads level tool_configs + template)
      ``resolve_mount_refs`` → level-setup time (builder stage; resolve joint/body refs)
      ``bind_model``         → after the physics model is built (resolve model indices)
      ``on_attach`` / ``on_detach`` → tool attach/detach (U key)
      ``step``               → every frame while attached (drive joints / write forces)
      ``view_overlay`` / ``draw_view_overlay`` → optional viewer HUD geometry
    """

    #: Human-readable action name (e.g. "aim").
    name: str = ""

    def configure(
        self,
        tool_cfg: dict,
        tool_template: Optional[dict],
    ) -> None:
        """Read action-specific config from the level tool entry + object template."""

    def resolve_mount_refs(
        self,
        *,
        builder,
        path_joint_map: dict,
        path_body_map: dict,
        tool_joint_start: int,
        tool_joint_end: int,
        tool_body_idx: int,
    ) -> None:
        """Resolve tool-local joint/body indices at level-setup time (builder stage)."""

    def bind_model(self, registry: "MountJointRegistry", record: "ToolMountRecord") -> None:
        """Resolve model indices / DOF specs after the physics model is built."""

    def on_attach(
        self,
        registry: "MountJointRegistry",
        record: "ToolMountRecord",
        *,
        world: int,
    ) -> None:
        """Called when the tool is attached."""

    def on_detach(
        self,
        registry: "MountJointRegistry",
        record: "ToolMountRecord",
        *,
        world: int,
    ) -> None:
        """Called when the tool is detached."""

    def step(
        self,
        registry: "MountJointRegistry",
        record: "ToolMountRecord",
        *,
        world: int,
        dt: float,
        camera_yaw: float,
        camera_pitch: float,
        host_role_object_id: int | None,
        body_q,
        body_qd,
        control,
        body_q_np=None,
        joint_q=None,
        joint_qd=None,
        mouse_buttons=None,
        body_f=None,
    ) -> None:
        """Drive the attached tool each frame (called before physics substeps)."""

    def view_overlay(
        self,
        registry: "MountJointRegistry",
        record: "ToolMountRecord",
        *,
        host_role_object_id: int | None,
        window_w: float,
        window_h: float,
        camera_pitch_deg: float,
        camera_fov_deg: float,
    ) -> object:
        """Return optional screen-space overlay data; None = nothing to draw."""
        return None

    def draw_view_overlay(self, overlay: object, shapes_module, batch=None) -> Optional[tuple]:
        """Draw the overlay produced by :meth:`view_overlay`.

        Returns ``(batch, shapes)`` so the caller can keep references alive
        until the frame is drawn, or None.
        """
        return None

    def forward_local(self) -> tuple[float, float, float]:
        """Tool-local forward direction used for debug geometry (default +X)."""
        return (1.0, 0.0, 0.0)

    def set_rl_control(self, values: Sequence[float]) -> None:
        """Optional hook: apply one RL / inspector action slice for this tool."""

    def clear_rl_control(self) -> None:
        """Optional hook: drop RL / inspector control before the next frame."""

    def rl_control_active(self) -> bool:
        """Whether :meth:`set_rl_control` is driving this frame (default false)."""
        return False
