"""Resolve Unitree G1 foot Newton body ids and load foot_sensor YAML config.

Lives in the G1 object template so robot-specific body names stay out of core
sensor / level code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import warp as wp
import yaml

_VERSION_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class G1FootSensorConfig:
    bodies: Tuple[str, ...]
    ground_geom_id: int
    ground_height: float
    foot_height_offset: float
    foot_height_site_offset: Tuple[float, float, float]

    @property
    def num_feet(self) -> int:
        return len(self.bodies)


_FOOT_SENSOR_CFG_CACHE: dict[str, G1FootSensorConfig] = {}


def load_g1_foot_sensor_config(
    config_path: Optional[Path] = None,
) -> G1FootSensorConfig:
    path = config_path or (_VERSION_DIR / "control_configs.yaml")
    cache_key = str(path)
    cached = _FOOT_SENSOR_CFG_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with path.open("r", encoding="utf-8") as fh:
        raw_data = yaml.safe_load(fh) or {}
    if not isinstance(raw_data, dict):
        raw_data = {}
    foot_cfg = raw_data.get("unitree_g1") or {}
    if isinstance(foot_cfg, dict):
        nested = foot_cfg.get("foot_sensor")
        foot_cfg = nested if isinstance(nested, dict) else {}
    else:
        foot_cfg = {}

    bodies = foot_cfg.get("bodies")
    if not isinstance(bodies, (list, tuple)) or len(bodies) == 0:
        raise ValueError(f"foot_sensor.bodies must be a non-empty list in {path}")

    if "ground_geom_id" not in foot_cfg:
        raise KeyError(f"foot_sensor.ground_geom_id is required in {path}")
    if "ground_height" not in foot_cfg:
        raise KeyError(f"foot_sensor.ground_height is required in {path}")

    site = foot_cfg.get("foot_height_site_offset")
    if not isinstance(site, (list, tuple)) or len(site) != 3:
        raise KeyError(f"foot_sensor.foot_height_site_offset must be a 3-element list in {path}")
    site_offset = (float(site[0]), float(site[1]), float(site[2]))

    instance = G1FootSensorConfig(
        bodies=tuple(str(b) for b in bodies),
        ground_geom_id=int(foot_cfg["ground_geom_id"]),
        ground_height=float(foot_cfg["ground_height"]),
        foot_height_offset=float(site_offset[2]),
        foot_height_site_offset=site_offset,
    )
    _FOOT_SENSOR_CFG_CACHE[cache_key] = instance
    return instance


@dataclass
class G1FootBodyMapping:
    """Per-world Newton body ids for G1 feet plus mujoco-warp solver constants."""

    ankle_newton_bodies_wp: Any  # wp.array [nworld, num_feet]
    body_suffixes: Tuple[str, ...]
    ngeom: int
    njmax: int
    nbody_mj: int
    naconmax: int
    opt_cone: int


def resolve_g1_foot_body_mapping(
    physics_manager,
    device: str,
    body_suffixes: Optional[Sequence[str]] = None,
) -> G1FootBodyMapping:
    """Resolve per-world Newton body ids of the configured foot bodies.

    Matches ``body_suffixes`` against Newton ``body_label`` (endswith), then maps
    each local Newton body to the corresponding mujoco template body via
    ``mjc_body_to_newton``.
    """
    cfg = load_g1_foot_sensor_config()
    suffixes = tuple(body_suffixes) if body_suffixes is not None else cfg.bodies
    num_feet = len(suffixes)

    pm = physics_manager
    solver = getattr(getattr(pm, "solver_handler", None), "solver", None)
    if solver is None or not hasattr(solver, "mjc_body_to_newton"):
        raise RuntimeError(
            "resolve_g1_foot_body_mapping needs SolverMuJoCo (mjc_body_to_newton)."
        )

    mj_model = solver.mjw_model
    mj_data = solver.mjw_data
    mt = solver.mjc_body_to_newton.numpy()
    nworld, nbody = mt.shape

    nm = pm.model
    bodies_per_world = nm.body_count // nm.world_count
    body_labels = nm.body_label
    if hasattr(body_labels, "numpy"):
        body_labels = body_labels.numpy()
    labels = [str(x) for x in body_labels]

    # Discover local (within-world) Newton body indices in suffix order.
    ankle_local: list[int] = []
    for suffix in suffixes:
        found_local = None
        for i, name in enumerate(labels):
            if name.endswith(suffix):
                local = i % bodies_per_world
                found_local = local
                break
        if found_local is None:
            raise RuntimeError(
                f"Foot body suffix '{suffix}' not found in Newton body labels."
            )
        ankle_local.append(found_local)

    if len(ankle_local) != num_feet:
        raise RuntimeError(
            f"Expected {num_feet} foot Newton bodies, found {ankle_local} "
            f"(suffixes={suffixes})"
        )

    row0 = mt[0]
    ankle_mj: list[int] = []
    for local in ankle_local:
        matches = [j for j in range(nbody) if int(row0[j]) == local]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one mj template body for newton body {local}, "
                f"got {matches}"
            )
        ankle_mj.append(matches[0])

    # Keep suffix order (do not sort) so left/right mapping stays stable.
    geoms = [
        int(g)
        for g in range(mj_model.ngeom)
        if int(mj_model.geom_bodyid.numpy()[g]) in ankle_mj
    ]
    if not geoms:
        raise RuntimeError(
            f"No geoms attached to ankle mj bodies {ankle_mj}; cannot filter foot contacts."
        )

    ankle_newton_np = np.zeros((nworld, num_feet), dtype=np.int32)
    for w in range(nworld):
        for f, mj_b in enumerate(ankle_mj):
            ankle_newton_np[w, f] = mt[w, mj_b]

    return G1FootBodyMapping(
        ankle_newton_bodies_wp=wp.array(ankle_newton_np, dtype=wp.int32, device=device),
        body_suffixes=suffixes,
        ngeom=int(mj_model.ngeom),
        njmax=int(mj_data.njmax),
        nbody_mj=int(mj_model.nbody),
        naconmax=int(mj_data.naconmax),
        opt_cone=int(mj_model.opt.cone),
    )


def resolve_g1_body_mj_id(physics_manager, body_suffix: str) -> int:
    """Resolve the mujoco template body id for a G1 body by label suffix.

    Matches ``body_suffix`` against Newton ``body_label`` (endswith), takes the
    local (within-world) Newton body id, then maps it back to the mujoco
    template body id through ``mjc_body_to_newton``. Used e.g. to locate the
    ``pelvis`` body for the whole-robot angular momentum (mjlab
    ``subtreeangmom``).
    """
    pm = physics_manager
    solver = getattr(getattr(pm, "solver_handler", None), "solver", None)
    if solver is None or not hasattr(solver, "mjc_body_to_newton"):
        raise RuntimeError("resolve_g1_body_mj_id needs SolverMuJoCo (mjc_body_to_newton).")

    mj_model = solver.mjw_model
    mt = solver.mjc_body_to_newton.numpy()
    nworld, nbody = mt.shape

    nm = pm.model
    bodies_per_world = nm.body_count // nm.world_count
    body_labels = nm.body_label
    if hasattr(body_labels, "numpy"):
        body_labels = body_labels.numpy()
    labels = [str(x) for x in body_labels]

    local = None
    for i, name in enumerate(labels):
        if name.endswith(body_suffix):
            local = i % bodies_per_world
            break
    if local is None:
        raise RuntimeError(
            f"Body suffix '{body_suffix}' not found in Newton body labels."
        )

    row0 = mt[0]
    matches = [j for j in range(nbody) if int(row0[j]) == local]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one mj template body for newton body {local}, "
            f"got {matches}"
        )
    return int(matches[0])
