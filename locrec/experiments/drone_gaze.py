"""Milestone 3: where a limited-FOV scanner should look, and what it buys.

Four policies on the same seeds and the same worlds. The drone's path is the
tunnel centreline in every case, so mission time is identical across policies by
construction and the real cost axis is yaw rate. That is reported as measured
rather than modelled: coupling speed to slew rate would be inventing a vehicle.

    python experiments/drone_gaze.py --seeds 8 --workers 2 --world all
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

from locrec import RunConfig, run_pass
from locrec.gaze import ForwardGaze, GlanceGaze, GreedyGaze, OracleGaze, SweepGaze
from locrec.odometry import OdometryConfig
from locrec.runner import write_csv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from calibrate_thresholds import blind_world, mixed_world  # noqa: E402
from marker_drop import bootstrap_ci  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def junction_world(length: float):
    """The milestone 1 world: curves and junctions scattered, not in bands."""
    from locrec import WorldSpec

    return WorldSpec(length=length, n_curves=3, n_junctions=4, n_niches=8)


WORLDS = {"blind": blind_world, "mixed": mixed_world, "junction": junction_world}


def policies(thresholds: dict):
    t = thresholds["drone_ratio_threshold"]
    return {
        "forward": lambda: ForwardGaze(),
        "sweep": lambda: SweepGaze(),
        "greedy": lambda: GreedyGaze(ratio_threshold=t),
        "greedy_forward": lambda: GreedyGaze(
            ratio_threshold=t, cone_deg=60.0, name="greedy_forward"
        ),
        "glance": lambda: GlanceGaze(ratio_threshold=t, cone_deg=60.0),
        "oracle": lambda: OracleGaze(ratio_threshold=t, cone_deg=60.0),
    }


def _one(job):
    label, seed, world_name, length, thresholds = job
    world = WORLDS[world_name](length)
    cfg = RunConfig(
        seed=seed,
        platform="drone",
        world=world,
        odometry=OdometryConfig(
            num_threads=1, gicp_ratio_floor=thresholds["drone_gicp_ratio_floor"]
        ),
    )
    r = run_pass(cfg, policies(thresholds)[label]())
    ratio = r.array("loc_ratio")
    blind_fraction = float(
        np.mean(ratio[np.isfinite(ratio)] < thresholds["drone_ratio_threshold"])
    )
    yaw = r.array("yaw")
    dt = cfg.platform_spec.step_length / cfg.platform_spec.speed_mps
    dyaw = np.abs(np.diff(np.unwrap(yaw[np.isfinite(yaw)])))
    return {
        "policy": label,
        "seed": seed,
        "world": world_name,
        "final_along_m": r.final_along_track_error,
        "final_lateral_m": r.final_lateral_error,
        "final_rot_rad": r.final_rotation_error,
        "path_length_m": r.path_length,
        "mission_time_s": len(r.rows) * dt,
        "blind_fraction": blind_fraction,
        "mean_yaw_rate_dps": float(np.rad2deg(dyaw.mean() / dt)) if dyaw.size else 0.0,
        "yaw_effort_rad": r.yaw_effort,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--length", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--world", default="all")
    ap.add_argument("--out", default="milestone3_drone_gaze.csv")
    args = ap.parse_args()

    thresholds = json.loads((ROOT / "results" / "thresholds.json").read_text())
    worlds = sorted(WORLDS) if args.world == "all" else [args.world]
    labels = list(policies(thresholds))
    jobs = [
        (label, seed, world, args.length, thresholds)
        for world in worlds
        for label in labels
        for seed in range(args.seeds)
    ]
    print(
        f"drone ratio threshold {thresholds['drone_ratio_threshold']:.4g}, "
        f"guard {thresholds['drone_gicp_ratio_floor']:.4g}, {len(jobs)} runs"
    )

    rows = []
    if args.workers > 1:
        # workers recycle: a 300 m world per job is not fully released and a long
        # sweep otherwise walks into the OOM killer part way through
        with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
            for i, row in enumerate(pool.imap_unordered(_one, jobs), 1):
                rows.append(row)
                print(
                    f"[{i}/{len(jobs)}] {row['world']:8s} {row['policy']:8s} "
                    f"seed {row['seed']} along {row['final_along_m']:.2f} "
                    f"yaw {row['mean_yaw_rate_dps']:.1f} dps",
                    flush=True,
                )
    else:
        for i, job in enumerate(jobs, 1):
            rows.append(_one(job))
            print(f"[{i}/{len(jobs)}] done", flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    write_csv(rows, str(ROOT / "results" / args.out))

    for world in worlds:
        print(f"\n=== {world} ===")
        print(f"{'policy':<9}{'along':>8}{'95% CI':>18}{'lateral':>9}{'yaw dps':>9}{'blind':>7}")
        base = [
            r for r in rows if r["world"] == world and r["policy"] == "forward"
        ]
        base_by_seed = {r["seed"]: r["final_along_m"] for r in base}
        for label in labels:
            sub = [r for r in rows if r["world"] == world and r["policy"] == label]
            med, lo, hi = bootstrap_ci([r["final_along_m"] for r in sub])
            lat = float(np.median([r["final_lateral_m"] for r in sub]))
            yr = float(np.median([r["mean_yaw_rate_dps"] for r in sub]))
            bf = float(np.median([r["blind_fraction"] for r in sub]))
            print(
                f"{label:<9}{med:>8.2f}{f'[{lo:.2f}, {hi:.2f}]':>18}"
                f"{lat:>9.2f}{yr:>9.1f}{bf:>7.2f}"
            )
        print("paired against forward, positive means the gaze policy helped:")
        for label in labels:
            if label == "forward":
                continue
            sub = [r for r in rows if r["world"] == world and r["policy"] == label]
            diffs = [
                base_by_seed[r["seed"]] - r["final_along_m"]
                for r in sub
                if r["seed"] in base_by_seed
            ]
            med, lo, hi = bootstrap_ci(diffs)
            wins = sum(d > 0 for d in diffs)
            print(
                f"  {label:<8}{med:>8.2f}  [{lo:.2f}, {hi:.2f}]  {wins}/{len(diffs)} seeds"
            )
    print(f"\nwrote results/{args.out}")


if __name__ == "__main__":
    main()
