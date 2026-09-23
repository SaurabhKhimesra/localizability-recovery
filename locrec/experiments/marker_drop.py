"""Milestone 2: drift against markers spent, four baselines and the scheduler.

Common random numbers throughout. A seed fixes the world, the sensor noise and
the motion prior's scale and bias draws, so every policy on seed k walks the same
tunnel with the same odometry errors and the only thing that differs is where the
markers go.

    python experiments/marker_drop.py --seeds 8 --workers 2

Workers are processes, each running ``small_gicp`` with one thread, because the
registration is already the smaller half of the per-step cost and process
parallelism scales further than its internal threading on a laptop.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib

import numpy as np

from locrec import (
    LocalizabilityScheduler,
    NoMarkers,
    OraclePlacement,
    RunConfig,
    UniformSpacing,
    build_world,
    run_pass,
)
from locrec.odometry import OdometryConfig
from locrec.policies import DropOnFailure
from locrec.runner import write_csv

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from calibrate_thresholds import blind_world, mixed_world  # noqa: E402

WORLDS = {"blind": blind_world, "mixed": mixed_world}

ROOT = pathlib.Path(__file__).resolve().parents[1]


def policy_grid(thresholds: dict, uniform_spacings, oracle_budgets):
    """(label, factory) pairs. The factory takes the world and returns a fresh policy."""
    grid = [("none", lambda w: NoMarkers())]
    for sp in uniform_spacings:
        grid.append((f"uniform_{sp:g}m", lambda w, sp=sp: UniformSpacing(spacing=sp)))
    grid.append(
        (
            "on_failure",
            lambda w: DropOnFailure(min_inliers=thresholds["min_inliers"]),
        )
    )
    grid.append(
        (
            "scheduler",
            lambda w: LocalizabilityScheduler(
                ratio_threshold=thresholds["ratio_threshold"],
                reliable_range_m=thresholds["marker_reliable_range_m"],
                margin_m=thresholds["scheduler_margin_m"],
            ),
        )
    )
    for b in oracle_budgets:
        grid.append((f"oracle_{b}", lambda w, b=b: OraclePlacement(world=w, budget=b)))
    return grid


def _one(job):
    label, seed, length, gicp_ratio_floor, thresholds, uniform, oracle, world_name = job
    world = WORLDS[world_name](length)
    factory = dict(policy_grid(thresholds, uniform, oracle))[label]
    cfg = RunConfig(
        seed=seed,
        platform="ugv",
        world=world,
        odometry=OdometryConfig(num_threads=1, gicp_ratio_floor=gicp_ratio_floor),
    )
    r = run_pass(cfg, factory(build_world(seed, world)))
    e = r.array("trans_err_m")
    return {
        "policy": label,
        "seed": seed,
        "world": world_name,
        "gicp_ratio_floor": "none" if gicp_ratio_floor is None else gicp_ratio_floor,
        "markers_placed": r.markers_placed,
        "final_drift_m": r.final_translation_error,
        "final_along_m": r.final_along_track_error,
        "final_lateral_m": r.final_lateral_error,
        "final_rot_rad": r.final_rotation_error,
        "max_drift_m": float(e.max()),
        "drift_per_100m": 100.0 * r.final_translation_error / max(r.path_length, 1e-9),
        "path_length_m": r.path_length,
        "landmark_steps": int(r.array("landmarks_used").sum()),
    }


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap of the median. Reported on every headline number."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n_boot, v.size), replace=True)
    meds = np.median(draws, axis=1)
    return (
        float(np.median(v)),
        float(np.quantile(meds, alpha / 2)),
        float(np.quantile(meds, 1 - alpha / 2)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--length", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--gicp-ratio-floor",
        type=float,
        default=None,
        help="Override the calibrated absolute guard. Defaults to the calibrated value "
        "in results/thresholds.json.",
    )
    ap.add_argument("--uniform", type=float, nargs="*", default=[15.0, 8.0, 5.0])
    ap.add_argument("--oracle", type=int, nargs="*", default=[8, 16, 25])
    ap.add_argument("--world", choices=sorted(WORLDS), default="mixed")
    ap.add_argument("--out", default="milestone2b_marker_drop.csv")
    args = ap.parse_args()

    thresholds = json.loads((ROOT / "results" / "thresholds.json").read_text())
    print(
        "thresholds from calibration seeds "
        f"{thresholds['calibration_seeds']}: ratio<{thresholds['ratio_threshold']:.4g}, "
        f"inliers<{thresholds['min_inliers']:.0f}"
    )

    floor = (
        args.gicp_ratio_floor
        if args.gicp_ratio_floor is not None
        else thresholds.get("gicp_ratio_floor")
    )
    print(f"absolute guard: registration information discarded below ratio {floor:.4g}")
    labels = [label for label, _ in policy_grid(thresholds, args.uniform, args.oracle)]
    jobs = [
        (label, seed, args.length, floor, thresholds, args.uniform, args.oracle, args.world)
        for label in labels
        for seed in range(args.seeds)
    ]

    if args.workers > 1:
        # workers recycle: a 300 m world per job is not fully released and a long
        # sweep otherwise walks into the OOM killer part way through
        with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
            rows = []
            for i, row in enumerate(pool.imap_unordered(_one, jobs), 1):
                rows.append(row)
                print(f"[{i}/{len(jobs)}] {row['policy']} seed {row['seed']} "
                      f"drift {row['final_drift_m']:.2f} markers {row['markers_placed']}",
                      flush=True)
    else:
        rows = []
        for i, job in enumerate(jobs, 1):
            row = _one(job)
            rows.append(row)
            print(f"[{i}/{len(jobs)}] {row['policy']} seed {row['seed']} "
                  f"drift {row['final_drift_m']:.2f} markers {row['markers_placed']}",
                  flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    write_csv(rows, str(ROOT / "results" / args.out))

    header = f"{'policy':<14}{'markers':>8}{'along-track':>13}{'95% CI':>20}{'lateral':>10}"
    print("\n" + header)
    for label in labels:
        sub = [r for r in rows if r["policy"] == label]
        med, lo, hi = bootstrap_ci([r["final_along_m"] for r in sub])
        lat = float(np.median([r["final_lateral_m"] for r in sub]))
        mk = float(np.median([r["markers_placed"] for r in sub]))
        print(
            f"{label:<14}{mk:>8.0f}{med:>13.2f}{f'[{lo:.2f}, {hi:.2f}]':>20}{lat:>10.2f}"
        )
    print(f"\nwrote results/{args.out}")


if __name__ == "__main__":
    main()
