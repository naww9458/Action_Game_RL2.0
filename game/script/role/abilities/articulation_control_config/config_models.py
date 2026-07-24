from pydantic import BaseModel, Field
from typing import Optional, Tuple


class JointParam(BaseModel):
    scale: float = Field(..., description="Action offset scale in radians")
    nominal: Optional[float] = Field(None, description="Default posture angle")
    rl_controllable: bool = Field(True, description="Whether RL controls this joint")
    effort_limit: Optional[float] = None
    stiffness: Optional[float] = None
    kp: Optional[float] = None
    kd: Optional[float] = None

    def resolved_scale(self) -> float:
        if self.effort_limit is not None and self.stiffness is not None and self.stiffness > 0.0:
            return 0.25 * self.effort_limit / self.stiffness
        return self.scale


class TaskParam(BaseModel):
    soft_limit_factor: float = 0.9
    default: JointParam = Field(default_factory=lambda: JointParam(scale=0.35, nominal=0.0))


def apply_soft_limits(lim_min: float, lim_max: float, nominal: float, soft_factor: float) -> Tuple[float, float]:
    if abs(lim_max) > 1e5 or abs(lim_min) > 1e5:
        return lim_min, lim_max
    half_lo = nominal - lim_min
    half_hi = lim_max - nominal
    half_range = half_lo if half_lo < half_hi else half_hi
    if half_range <= 0.0:
        return lim_min, lim_max
    shrunk = half_range * soft_factor
    return nominal - shrunk, nominal + shrunk
