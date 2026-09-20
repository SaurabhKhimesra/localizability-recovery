"""The demonstration's tunnels and their plans, with no ROS import.

Each world is the one the study measured on, built by the same WorldSpec: ``mixed`` is
``locrec/experiments/calibrate_thresholds.py``'s, ``junction`` is ``locrec/experiments/drone_gaze.py``'s.
Both simulators, MuJoCo's publisher and the Gazebo driver, build from here.
"""
from __future__ import annotations

import numpy as np

from locrec import WorldSpec

__all__ = ["WORLDS", "blind_world", "mixed_world", "junction_world", "wall_edges"]


def blind_world(length: float) -> WorldSpec:
    return WorldSpec(
        length=length, n_curves=0, n_junctions=0, n_niches=0,
        straight_lead_in=length, width_min=3.2, width_max=3.2, width_mean=3.2,
    )


def mixed_world(length: float) -> WorldSpec:
    return WorldSpec(
        length=length, n_curves=5, n_junctions=8, n_niches=24,
        structured_stretches=True, width_min=3.2, width_max=3.2, width_mean=3.2,
        n_marker_slots=120,
    )


def junction_world(length: float) -> WorldSpec:
    return WorldSpec(length=length, n_curves=3, n_junctions=4, n_niches=8)


WORLDS = {"blind": blind_world, "mixed": mixed_world, "junction": junction_world}


def wall_edges(world) -> list[np.ndarray]:
    """Floor-level wall lines of the main tunnel, as (N, 2) polylines.

    Drawn from the world's own description: the centreline offset by half the width,
    pushed out by a niche's depth where there is one, and broken where a junction
    opens the wall. It is a drawing of the plan, for display. Nothing estimates
    from it.
    """
    n = len(world.s)
    normal = np.stack([-np.sin(world.heading), np.cos(world.heading)], axis=1)
    lines = []
    for side, sign in (("left", 1.0), ("right", -1.0)):
        # the generator's width array can run longer than the centreline on short worlds
        offset = 0.5 * np.asarray(world.width, dtype=float)[:n].copy()
        is_open = np.zeros(n, dtype=bool)
        for nc in world.niches:
            if nc["side"] == side:
                offset[nc["index"] : nc["index"] + nc["n_samples"]] += nc["depth"]
        for j in world.junctions:
            if j["side"] == side:
                is_open[j["index"] : j["index"] + j["n_samples"]] = True
        edge = world.centreline + sign * offset[:, None] * normal
        start = None
        for i in range(n + 1):
            closed = i < n and not is_open[i]
            if closed and start is None:
                start = i
            elif not closed and start is not None:
                if i - start >= 2:
                    lines.append(edge[start:i])
                start = None
    return lines
