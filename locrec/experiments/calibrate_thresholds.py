"""Calibrate the policy thresholds on seeds disjoint from the evaluation seeds.

The two quantiles below are declared here, in code, before any evaluation run is
looked at. They are not swept and they are not revisited after seeing a drift
number; if a later milestone changes them, the change belongs in the commit
history where it can be argued with.

    SCHEDULER_QUANTILE = 0.9
        The ratio is now a gate, not a trigger: it decides whether a stretch needs
        anchors at all, and the detection range decides their spacing inside one. A
        gate has to recognise a blind tunnel as blind, so the threshold is a high
        quantile of what a tunnel that is degenerate by construction produces,
        matching the guard's logic. It was the median while the ratio was a trigger;
        the semantics changed, so the quantile changed with them.

    FAILURE_QUANTILE = 0.1
        The failure watcher is meant to represent a team with no localizability
        signal, reacting to a registration that looks unhealthy. "Unhealthy"
        means clearly bad rather than merely below average, so the bottom decile.

    REQUIRED_FIX_INTERVAL_M = 1.25
        The marker spacing a scheduler may use follows from how far a strip is
        usefully seen, which is a measured property of the sensor and the target
        rather than the datasheet range. A strip viewed along a tunnel is nearly
        edge on, so the useful window is far shorter than the sensor's 10 m.

        The criterion is what the estimator needs, not what the sensor manages per
        scan. Dead reckoning drifts by one marker measurement sigma, about 0.025 m
        at a typical range, over 0.025 / 0.02 = 1.25 m of travel, so a fix at least
        that often keeps the drift between fixes inside the measurement itself.
        With a 0.5 m step that is a detection rate of 0.4, and the reliable range is
        the largest distance meeting it.

        This started as a flat 0.9 per-scan detection rate, which gave 3.5 m and
        would have forced the scheduler to spend markers faster than uniform
        spacing. That criterion was wrong rather than inconvenient: a fix every
        scan is not what the filter needs, and per-scan reliability is not the
        quantity the spacing depends on.

    GUARD_QUANTILE = 0.9
REQUIRED_FIX_INTERVAL_M = 1.25
MEASUREMENT_SIGMA_M = 0.025
        The estimator's absolute guard, which decides when the registration's
        translational information is discarded rather than trusted. These
        calibration runs are in a tunnel that is degenerate by construction, and
        the ground-truth-map diagnostic shows the along-track information there is
        an artefact. So anything as degenerate as what this tunnel produces must be
        discarded, which means a high quantile of its ratio distribution rather
        than a low one. It is deliberately not the same number as the scheduler's
        threshold: one decides what the estimator believes, the other decides when
        a policy spends a marker.

    CALIBRATION_ANGLES = 19
        Orientations of the tunnel against the estimator's voxel grids: 0 to 90
        degrees inclusive in 5 degree steps, pooled into one distribution before any
        quantile above is taken. The grids are square, so the ratio is periodic in 90
        degrees and one quadrant covers every orientation. The number was fixed
        before any swept output was looked at.

        Before this, the sweep did not exist. ``blind_world`` has no curvature, so
        its heading is identically zero and every calibration sample came from a
        corridor lying exactly along a grid axis. That orientation is the minimum of
        a curve spanning 19 percent, so the threshold could only ever come out too
        low and the gate could only ever under-fire, never over-fire. The measured
        curve and what it cost a live run are in ``docs/failures.md`` number 21.

        0 and 90 degrees are the same orientation under that symmetry, so that one
        alignment carries double weight. It is also the orientation with the lowest
        ratios, so the bias it leaves is toward the old threshold, not away from it.

Calibration seeds are 100..100+N-1, evaluation seeds are 0..M-1. They never meet.

    python experiments/calibrate_thresholds.py --seeds 4
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from multiprocessing import Pool

import numpy as np

from locrec import NoMarkers, RunConfig, SPINNING_360, TunnelSim, WorldSpec, run_pass
from locrec.se3 import make_T
from locrec.localizability import calibrate_threshold
from locrec.odometry import OdometryConfig

ROOT = pathlib.Path(__file__).resolve().parents[1]
CALIBRATION_SEED_BASE = 100
CALIBRATION_ANGLES = 19
SCHEDULER_MARGIN_M = 0.5 + 1.0  # one step length plus the correspondence distance
SCHEDULER_QUANTILE = 0.9
FAILURE_QUANTILE = 0.1
GUARD_QUANTILE = 0.9
REQUIRED_FIX_INTERVAL_M = 1.25
MEASUREMENT_SIGMA_M = 0.025


def mixed_world(length: float):
    """Alternating blind and structured runs, lengths drawn per seed.

    The world a scheduler can be tested in. Its claim is not lower drift than
    uniform spacing, it is the same drift for fewer markers, and a tunnel that is
    featureless end to end cannot show that because there is nowhere to save one.
    """
    from locrec import WorldSpec

    return WorldSpec(
        length=length,
        n_curves=5,
        n_junctions=8,
        n_niches=24,
        structured_stretches=True,
        width_min=3.2,
        width_max=3.2,
        width_mean=3.2,
        n_marker_slots=120,
    )


def blind_world(length: float, heading_offset_deg: float = 0.0):
    """A tunnel with nothing in it: no curves, no junctions, no niches, constant width.

    This is the world the marker story is about. Width variation is switched off
    on purpose: the ribbed shell turns a varying width into a centimetre-scale
    staircase with a 1 m pitch, which is texture a real tunnel would not have in
    that form, and it would hand the odometry along-track information for free.
    """
    from locrec import WorldSpec

    return WorldSpec(
        length=length,
        n_curves=0,
        n_junctions=0,
        n_niches=0,
        straight_lead_in=length,
        width_min=3.2,
        width_max=3.2,
        width_mean=3.2,
        heading_offset_deg=heading_offset_deg,
    )


def calibration_angles() -> np.ndarray:
    """The declared orientation grid, 0 to 90 degrees inclusive."""
    return np.linspace(0.0, 90.0, CALIBRATION_ANGLES)


def _one(job):
    """One calibration run. Returns its finite ratios and inlier counts.

    The guard is off while it is being calibrated, so the distribution is a property
    of the raw registration rather than of the guard's own effect.
    """
    length, angle, seed, platform = job
    r = run_pass(
        RunConfig(
            seed=seed,
            platform=platform,
            world=blind_world(length, angle),
            odometry=OdometryConfig(gicp_ratio_floor=None),
        ),
        NoMarkers(),
    )
    v = r.array("loc_ratio")
    n = r.array("num_inliers")
    ok = np.isfinite(v)
    return angle, seed, platform, v[ok].tolist(), n[ok].tolist()


def marker_reliable_range(
    n_offsets: int = 9, max_range: float = 12.0, step_length: float = 0.5
) -> float:
    """Largest distance at which the strip is seen often enough to hold the estimate.

    Sampled over sub-step offsets, so the answer is not one lucky beam alignment.
    """
    required_rate = step_length / REQUIRED_FIX_INTERVAL_M
    spec = WorldSpec(
        length=60.0,
        n_curves=0,
        n_junctions=0,
        n_niches=0,
        straight_lead_in=60.0,
        width_min=3.2,
        width_max=3.2,
        width_mean=3.2,
        n_marker_slots=2,
    )
    sim = TunnelSim(0, spec, SPINNING_360)
    sim.drop_marker_on_wall(make_T(np.eye(3), [30.0, 0.0, 0.7]))
    offsets = np.linspace(0.0, 1.0, n_offsets, endpoint=False)
    best = 0.0
    for d in np.arange(1.0, max_range + 0.01, 0.5):
        hits = 0
        for o in offsets:
            T = make_T(np.eye(3), [30.0 - d - o, 0.0, 0.7])
            if sim.marker_detections(sim.scan(T)):
                hits += 1
        rate = hits / len(offsets)
        print(f"  marker detection at {d:4.1f} m: {rate:.2f}", flush=True)
        if rate >= required_rate:
            best = float(d)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--length", type=float, default=200.0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    angles = calibration_angles()
    seeds = [CALIBRATION_SEED_BASE + i for i in range(args.seeds)]
    # the drone sees a different tunnel: 90 degrees at 30 m rather than 360 at 10, so
    # its ratio distribution is its own and a UGV threshold applied to it would never
    # fire. Same rule, same seeds, same angles, separate number.
    jobs = [
        (args.length, float(a), s, plat)
        for a in angles
        for s in seeds
        for plat in ("ugv", "drone")
    ]
    print(
        f"{len(jobs)} runs: {len(angles)} angles, {len(seeds)} seeds, 2 platforms, "
        f"{args.length:.0f} m, {args.workers} workers",
        flush=True,
    )

    collected = []
    t0 = time.time()
    # workers recycle, so a long sweep does not accumulate one world per job
    with Pool(args.workers, maxtasksperchild=8) as pool:
        for res in pool.imap_unordered(_one, jobs):
            collected.append(res)
            n_done = len(collected)
            if n_done % 20 == 0 or n_done == len(jobs):
                el = time.time() - t0
                print(
                    f"  {n_done}/{len(jobs)} runs, {el / 60:.1f} min elapsed, "
                    f"{el / n_done * (len(jobs) - n_done) / 60:.1f} min left",
                    flush=True,
                )

    # pooled in a fixed order, so the result does not depend on who finished first
    collected.sort(key=lambda r: (r[0], r[1], r[2]))
    ratios: list[float] = [x for r in collected if r[2] == "ugv" for x in r[3]]
    inliers: list[float] = [x for r in collected if r[2] == "ugv" for x in r[4]]
    drone_ratios: list[float] = [x for r in collected if r[2] == "drone" for x in r[3]]

    print("\nblind-tunnel ratio by orientation, pooled over seeds:", flush=True)
    print(f"{'angle':>7} {'n':>6} {'median':>11} {'p90 (ugv)':>11} {'median':>11} {'p90 (drone)':>12}")
    for a in angles:
        u = np.array([x for r in collected if r[0] == a and r[2] == "ugv" for x in r[3]])
        d = np.array([x for r in collected if r[0] == a and r[2] == "drone" for x in r[3]])
        print(
            f"{a:>6.1f}d {u.size:>6} {np.median(u):>11.4e} {np.quantile(u, 0.9):>11.4e} "
            f"{np.median(d):>11.4e} {np.quantile(d, 0.9):>12.4e}",
            flush=True,
        )

    out = {
        "calibration_seeds": seeds,
        "calibration_angles_deg": [round(float(a), 6) for a in angles],
        "n_angles": CALIBRATION_ANGLES,
        "length_m": args.length,
        "scheduler_quantile": SCHEDULER_QUANTILE,
        "failure_quantile": FAILURE_QUANTILE,
        "guard_quantile": GUARD_QUANTILE,
        "ratio_threshold": calibrate_threshold(np.array(ratios), SCHEDULER_QUANTILE),
        "gicp_ratio_floor": calibrate_threshold(np.array(ratios), GUARD_QUANTILE),
        "min_inliers": calibrate_threshold(np.array(inliers), FAILURE_QUANTILE),
        "required_fix_interval_m": REQUIRED_FIX_INTERVAL_M,
        "required_detection_rate": 0.5 / REQUIRED_FIX_INTERVAL_M,
        "marker_reliable_range_m": marker_reliable_range(),
        "scheduler_margin_m": SCHEDULER_MARGIN_M,
        "drone_ratio_threshold": calibrate_threshold(
            np.array(drone_ratios), SCHEDULER_QUANTILE
        ),
        "drone_gicp_ratio_floor": calibrate_threshold(
            np.array(drone_ratios), GUARD_QUANTILE
        ),
        "n_samples": len(ratios),
        "n_drone_samples": len(drone_ratios),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    path = ROOT / "results" / "thresholds.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
