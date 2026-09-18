"""What a higher localizability threshold costs where the geometry is fine.

The scheduler's threshold is a gate: below it the stretch is called blind and a
marker gets spent. Raising it so that it covers every orientation of the corridor
(``docs/failures.md`` number 21) buys blind-stretch recall, and it has to cost
false alarms, scans in a structured stretch called degenerate when the geometry
constrains the estimate perfectly well. This measures both sides, at the old
threshold and the new one, so the trade is a number rather than an assumption.

Evaluation seeds 0..N-1, which never meet the calibration seeds. The mixed world,
because that is the world a scheduler is tested in, and at four orientations,
because one orientation is what caused the problem being fixed. No policy runs:
the quantity is a property of the detector, not of what a scheduler does with it.

    python experiments/threshold_false_alarms.py --seeds 5

Distance is to the nearest thing that makes a stretch not blind: a junction
opening, a niche, curved centreline, or an end cap. The headline false alarm rate
is the nearest bin, scans within a metre of structure.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from multiprocessing import Pool

import numpy as np

from locrec import NoMarkers, RunConfig, WorldSpec, run_pass
from locrec.odometry import OdometryConfig
from locrec.runner import write_csv
from locrec.worlds import build_world

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANGLES = tuple(float(a) for a in np.linspace(0.0, 90.0, 8, endpoint=False))
"""Eight orientations over the quadrant. The endpoint is excluded because 90 degrees
is 0 degrees again under the grid's symmetry, so including both would weight that one
alignment twice."""
BINS = ((0.0, 1.0, "on it (<=1 m)"), (1.0, 5.0, "1-5 m"), (5.0, 10.0, "5-10 m"),
        (10.0, float("inf"), "blind (>10 m)"))


def mixed_world(length: float, heading_offset_deg: float = 0.0) -> WorldSpec:
    """The alternating world from the calibration script, at a chosen orientation."""
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
        heading_offset_deg=heading_offset_deg,
    )


def _one(job):
    """One evaluation run, returned as (ratio, distance to structure) per scan."""
    length, angle, seed, platform = job
    spec = mixed_world(length, angle)
    r = run_pass(
        RunConfig(
            seed=seed,
            platform=platform,
            world=spec,
            odometry=OdometryConfig(gicp_ratio_floor=None),
        ),
        NoMarkers(),
    )
    world = build_world(seed, spec)
    feat_s = world.s[world.feature_mask]
    curve_s = world.s[np.abs(world.curvature) > 1e-9]
    struct_s = np.concatenate([feat_s, curve_s]) if feat_s.size or curve_s.size else np.zeros(0)
    total = world.total_length

    s = r.array("s")
    ratio = r.array("loc_ratio")
    ok = np.isfinite(ratio)
    s, ratio = s[ok], ratio[ok]
    if struct_s.size:
        d_struct = np.abs(s[:, None] - struct_s[None, :]).min(axis=1)
    else:
        d_struct = np.full(s.shape, np.inf)
    # the dead-end walls sit one sample beyond each end (worlds.py, end caps)
    d_cap = np.minimum(s + world.spec.ds, total + world.spec.ds - s)
    return angle, seed, platform, ratio.tolist(), np.minimum(d_struct, d_cap).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--length", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--old-ugv", type=float, default=0.0018445884928346022)
    ap.add_argument("--old-drone", type=float, default=0.0028264114279125683)
    args = ap.parse_args()

    thresholds = json.loads((ROOT / "results" / "thresholds.json").read_text())
    new = {"ugv": thresholds["ratio_threshold"], "drone": thresholds["drone_ratio_threshold"]}
    old = {"ugv": args.old_ugv, "drone": args.old_drone}

    jobs = [
        (args.length, a, seed, plat)
        for a in ANGLES
        for seed in range(args.seeds)
        for plat in ("ugv", "drone")
    ]
    print(f"{len(jobs)} runs: {len(ANGLES)} angles, {args.seeds} seeds, 2 platforms, "
          f"{args.length:.0f} m mixed world", flush=True)
    t0 = time.time()
    collected = []
    # workers recycle: a 300 m MuJoCo world per job is not fully released, and a
    # long pool of reused workers walks into the OOM killer part way through
    with Pool(args.workers, maxtasksperchild=4) as pool:
        for res in pool.imap_unordered(_one, jobs):
            collected.append(res)
            if len(collected) % 10 == 0 or len(collected) == len(jobs):
                el = time.time() - t0
                print(f"  {len(collected)}/{len(jobs)} runs, {el / 60:.1f} min", flush=True)
    collected.sort(key=lambda r: (r[0], r[1], r[2]))

    rows = []
    for plat in ("ugv", "drone"):
        ratio = np.array([x for r in collected if r[2] == plat for x in r[3]])
        dist = np.array([x for r in collected if r[2] == plat for x in r[4]])
        print(f"\n{plat}: old threshold {old[plat]:.6e}, new {new[plat]:.6e} "
              f"({new[plat] / old[plat] - 1:+.1%})")
        print(f"{'bin':>16} {'n':>6} {'called degenerate, old':>23} {'new':>8} {'change':>8}")
        for lo, hi, name in BINS:
            sel = (dist > lo) & (dist <= hi) if lo else (dist <= hi)
            if not sel.any():
                continue
            o = float(np.mean(ratio[sel] < old[plat]))
            n = float(np.mean(ratio[sel] < new[plat]))
            print(f"{name:>16} {int(sel.sum()):>6} {o:>23.3f} {n:>8.3f} {n - o:>+8.3f}")
            rows.append({"platform": plat, "group": "distance", "bin": name,
                         "n": int(sel.sum()), "threshold_old": old[plat],
                         "threshold_new": new[plat], "degenerate_old": o,
                         "degenerate_new": n, "blind_hit_old": "", "blind_hit_new": ""})
        for a in ANGLES:
            ra = np.array([x for r in collected if r[2] == plat and r[0] == a for x in r[3]])
            da = np.array([x for r in collected if r[2] == plat and r[0] == a for x in r[4]])
            on, blind = da <= 1.0, da > 10.0
            print(f"    {a:>5.1f} deg: false alarm {np.mean(ra[on] < old[plat]):.3f} -> "
                  f"{np.mean(ra[on] < new[plat]):.3f},  blind hit rate "
                  f"{np.mean(ra[blind] < old[plat]):.3f} -> {np.mean(ra[blind] < new[plat]):.3f}")
            rows.append({"platform": plat, "group": "angle", "bin": f"{a:g} deg",
                         "n": int(ra.size), "threshold_old": old[plat],
                         "threshold_new": new[plat],
                         "degenerate_old": float(np.mean(ra[on] < old[plat])),
                         "degenerate_new": float(np.mean(ra[on] < new[plat])),
                         "blind_hit_old": float(np.mean(ra[blind] < old[plat])),
                         "blind_hit_new": float(np.mean(ra[blind] < new[plat]))})

    (ROOT / "results").mkdir(exist_ok=True)
    path = ROOT / "results" / "threshold_false_alarms.csv"
    write_csv(rows, str(path))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
