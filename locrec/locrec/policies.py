"""Marker-drop policies for the UGV.

Every policy sees the same thing the estimator sees and nothing else, with one
deliberate exception: ``OraclePlacement`` is handed the true world so it can act
as the upper bound the online policies are measured against. It is a bound, not a
competitor.

Markers are a budgeted resource. A policy that drops one every step would bound
drift beautifully and be useless underground, so every policy reports how many it
spent and the Pareto curve in milestone 4 is drift against that count.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .runner import StepContext
from .worlds import TunnelWorld

__all__ = [
    "NoMarkers",
    "UniformSpacing",
    "DropOnFailure",
    "LocalizabilityScheduler",
    "OraclePlacement",
    "POLICY_NAMES",
]


class _Base:
    name = "base"

    def __call__(self, ctx: StepContext) -> dict:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"policy": self.name}


class NoMarkers(_Base):
    """The milestone 1 baseline, restated as a policy so the harness is identical."""

    name = "none"

    def __call__(self, ctx: StepContext) -> dict:
        return {}


@dataclasses.dataclass
class UniformSpacing(_Base):
    """Drop every ``spacing`` metres of estimated travel, ignoring the geometry.

    The obvious thing to do, and the one a scheduler has to beat: it spends markers
    in the easy stretches where the walls already give the odometry everything it
    needs.
    """

    spacing: float = 10.0
    name: str = "uniform"

    def __post_init__(self) -> None:
        self._next = 0.0

    def __call__(self, ctx: StepContext) -> dict:
        if ctx.estimated_distance >= self._next:
            self._next = ctx.estimated_distance + self.spacing
            return {"drop_marker": True}
        return {}

    def describe(self) -> dict:
        return {"policy": self.name, "spacing_m": self.spacing}


@dataclasses.dataclass
class DropOnFailure(_Base):
    """React to the symptom rather than to the Hessian.

    Fires when the registration itself looks unhealthy: it failed to converge, or
    the inlier count fell below a threshold calibrated on separate seeds. This is
    what a team without a localizability signal would build, and it is late by
    construction, because a degenerate tunnel produces a confident, inlier-rich,
    perfectly converged registration that happens to be sliding.
    """

    min_inliers: float
    name: str = "on_failure"
    cooldown_m: float = 2.0

    def __post_init__(self) -> None:
        self._last_drop = -1e9

    def __call__(self, ctx: StepContext) -> dict:
        reg = ctx.registration
        unhealthy = (not reg.registered) or (not reg.converged) or (reg.num_inliers < self.min_inliers)
        if unhealthy and ctx.estimated_distance - self._last_drop >= self.cooldown_m:
            self._last_drop = ctx.estimated_distance
            return {"drop_marker": True}
        return {}

    def describe(self) -> dict:
        return {"policy": self.name, "min_inliers": self.min_inliers}


@dataclasses.dataclass
class LocalizabilityScheduler(_Base):
    """Drop on the falling edge, then space by range. Three causal rules.

    1. This step degenerate and the previous step not: drop immediately, one step
       into the blind stretch.
    2. Still inside a blind stretch: drop when the robot has travelled twice the
       detection range since the last anchor, less a margin. Never on the ratio
       alone.
    3. Above threshold: never drop.

    Rule 1 is the correction milestone 2b's version needed. That version dropped
    once the ratio was already down, so every anchor it placed was registered in
    the middle of a blind stretch and inherited the error accumulated up to that
    point. Uniform spacing beat it by accident, because some of its markers landed
    where the geometry was still good and were therefore registered accurately. The
    valuable place to leave a marker is the last place you still know where you
    are, which is one step into the blind stretch, not fifty metres in.

    Rule 2 is the duty cycle. The ratio says whether a stretch needs anchors, the
    detection range says how far apart they go inside one. Conflating the two made
    the policy fire on nearly every step of a tunnel that is degenerate throughout.

    The spacing is twice the detection range rather than once it. An anchor is
    useful over the whole window in which it can be seen, and the fix that matters
    is the one that ties the new anchor to the old chain, not a fix on every step.
    Spacing at one range instead spent 64 markers on a 300 m blind run, which is
    three times uniform 15 m spacing for no measured gain.
    """

    ratio_threshold: float
    reliable_range_m: float
    margin_m: float = 1.5
    """Range headroom: one step plus the correspondence distance, so the fix never
    depends on a detection that is about to stop arriving."""
    name: str = "scheduler"

    def __post_init__(self) -> None:
        self._last_drop: float | None = None
        self._was_degenerate = False

    def __call__(self, ctx: StepContext) -> dict:
        loc = ctx.localizability
        if loc is None:
            return {}

        degenerate = loc.ratio < self.ratio_threshold
        falling_edge = degenerate and not self._was_degenerate
        self._was_degenerate = degenerate

        if not degenerate:
            # the geometry is carrying the estimate: spend nothing
            return {}

        # The spacing rule applies to the falling edge as well, and the chain is not
        # forgotten when a stretch turns localizable again. Clearing it meant every
        # re-entry dropped a marker, and the detector does not enter a blind stretch
        # once: on a 300 m blind run the ratio crosses the threshold 77 times, so
        # the falling edge fired 77 times and the policy spent 64 markers where the
        # spacing rule wanted 24. That is chatter being paid for in hardware.
        reach = max(2.0 * self.reliable_range_m - self.margin_m, self.margin_m)
        if self._last_drop is None or ctx.estimated_distance - self._last_drop >= reach:
            # On entering a blind stretch this fires at the edge, which is rule 1:
            # the valuable place to leave a marker is the last place you still knew
            # where you were. Deeper in, it is the duty cycle.
            return self._drop(ctx)
        del falling_edge
        return {}

    def _drop(self, ctx: StepContext) -> dict:
        self._last_drop = ctx.estimated_distance
        return {"drop_marker": True}

    def describe(self) -> dict:
        return {
            "policy": self.name,
            "ratio_threshold": self.ratio_threshold,
            "reliable_range_m": self.reliable_range_m,
            "margin_m": self.margin_m,
        }


@dataclasses.dataclass
class OraclePlacement(_Base):
    """Upper bound: knows the true world and plans the drops offline.

    Given a budget, it spreads markers over the stretches the world actually makes
    degenerate, which is exactly the offline placement problem milestone 8 argues
    is submodular. Here it is solved by even spacing within the blind stretches,
    which is optimal for the min-max-gap objective on a single interval and close
    to it across several.

    It still cannot beat its own registration error: a marker placed by an oracle
    is still registered from the robot's drifting estimate at drop time.
    """

    world: TunnelWorld
    budget: int
    name: str = "oracle"

    def __post_init__(self) -> None:
        self._targets = _blind_stretch_targets(self.world, self.budget)
        self._fired = np.zeros(len(self._targets), dtype=bool)

    def __call__(self, ctx: StepContext) -> dict:
        s = ctx.platform.s  # the oracle is allowed true arclength; nothing else is
        for i, target in enumerate(self._targets):
            if not self._fired[i] and s >= target:
                self._fired[i] = True
                return {"drop_marker": True}
        return {}

    def describe(self) -> dict:
        return {"policy": self.name, "budget": self.budget}


def _blind_stretch_targets(world: TunnelWorld, budget: int) -> list[float]:
    """Arclengths at which to drop, spreading a budget over the featureless stretches."""
    if budget <= 0:
        return []
    s = world.s
    blind = ~world.feature_mask & (np.abs(world.curvature) <= 1e-6)
    runs: list[tuple[float, float]] = []
    start = None
    for i, flag in enumerate(blind):
        if flag and start is None:
            start = s[i]
        elif not flag and start is not None:
            runs.append((start, s[i]))
            start = None
    if start is not None:
        runs.append((start, s[-1]))
    runs = [(a, b) for a, b in runs if b - a > 1.0]
    if not runs:
        return list(np.linspace(0.0, float(s[-1]), budget + 2)[1:-1])

    total = sum(b - a for a, b in runs)
    targets: list[float] = []
    for a, b in runs:
        share = max(int(round(budget * (b - a) / total)), 1)
        # markers sit at the centres of equal sub-intervals of the blind run
        edges = np.linspace(a, b, share + 1)
        targets.extend(0.5 * (edges[:-1] + edges[1:]))
    targets.sort()
    return targets[:budget]


POLICY_NAMES = ("none", "uniform", "on_failure", "scheduler", "oracle")
