"""Milestone 2c: which estimator change removed the milestone 2 marker effect?

Milestone 2 measured markers cutting along-track error in a blind tunnel by a
paired median of 0.73 m on a 2.97 m baseline, 7 of 8 seeds. After the milestone 2b
and 2c estimator changes the same world and the same seeds give a paired
difference that straddles zero on a 2.92 m baseline, so the baseline is unchanged
and something in the estimator is responsible rather than the world.

This turns each change off one at a time, on the blind world with uniform spacing,
and reports the paired difference against no markers on the same seeds.

    python experiments/marker_ablation.py --seeds 8 --workers 2

Nothing here is tuned. The variants are the switches as they already exist in
``OdometryConfig``; the point is attribution, not a better number.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

from locrec import NoMarkers, RunConfig, UniformSpacing, run_pass
from locrec.odometry import OdometryConfig
from locrec.runner import write_csv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from calibrate_thresholds import blind_world  # noqa: E402
from marker_drop import bootstrap_ci  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

VARIANTS = {
    "none": {},                                  # no markers, as-shipped estimator
    "shipped": {},                               # markers, as-shipped estimator
    "absolute_anchors": {"relative_anchors": False},
    "no_yaw_fix": {"yaw_fix": False},
    "no_resurvey": {"resurvey_anchors": False},
}


def _one(job):
    label, seed, length, th = job
    cfg = RunConfig(
        seed=seed,
        platform="ugv",
        world=blind_world(length),
        odometry=OdometryConfig(
            num_threads=1,
            gicp_ratio_floor=th["gicp_ratio_floor"],
            **VARIANTS[label],
        ),
    )
    policy = NoMarkers() if label == "none" else UniformSpacing(spacing=15.0)
    r = run_pass(cfg, policy)
    return {
        "variant": label,
        "seed": seed,
        "final_along_m": r.final_along_track_error,
        "final_lateral_m": r.final_lateral_error,
        "markers_placed": r.markers_placed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--length", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    th = json.loads((ROOT / "results" / "thresholds.json").read_text())
    jobs = [
        (label, seed, args.length, th)
        for label in VARIANTS
        for seed in range(args.seeds)
    ]
    print(f"{len(jobs)} runs, blind world, {args.length:.0f} m")

    rows = []
    if args.workers > 1:
        # workers recycle: a 300 m world per job is not fully released and a long
        # sweep otherwise walks into the OOM killer part way through
        with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
            for i, row in enumerate(pool.imap_unordered(_one, jobs), 1):
                rows.append(row)
                if i % 5 == 0 or i == len(jobs):
                    print(f"[{i}/{len(jobs)}]", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            rows.append(_one(job))
            print(f"[{i}/{len(jobs)}]", flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    write_csv(rows, str(ROOT / "results" / "marker_ablation.csv"))

    base = {r["seed"]: r["final_along_m"] for r in rows if r["variant"] == "none"}
    print(f"\n{'variant':<14}{'along':>8}{'markers':>9}{'paired':>9}{'95% CI':>18}{'wins':>7}")
    for label in VARIANTS:
        sub = [r for r in rows if r["variant"] == label]
        med = float(np.median([r["final_along_m"] for r in sub]))
        mk = float(np.median([r["markers_placed"] for r in sub]))
        if label == "none":
            print(f"{label:<14}{med:>8.2f}{mk:>9.0f}")
            continue
        diffs = [base[r["seed"]] - r["final_along_m"] for r in sub if r["seed"] in base]
        pm, lo, hi = bootstrap_ci(diffs)
        wins = sum(d > 0 for d in diffs)
        print(
            f"{label:<14}{med:>8.2f}{mk:>9.0f}{pm:>9.2f}"
            f"{f'[{lo:.2f}, {hi:.2f}]':>18}{wins:>4}/{len(diffs)}"
        )
    print("\npositive paired means markers helped on that variant")
    print("wrote results/marker_ablation.csv")


if __name__ == "__main__":
    main()
