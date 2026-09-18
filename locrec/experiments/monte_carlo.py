"""Milestone 4: the full grid, the sensitivities, and the detector's lead time.

One CSV row per run, every parameter recorded in the row, so a result can always
be traced back to the configuration that produced it without consulting anything
outside the file.

    python experiments/monte_carlo.py --platform ugv   --world all --seeds 50 --workers 4 --out results/
    python experiments/monte_carlo.py --platform drone --world all --seeds 50 --workers 4 --out results/
    python experiments/summarise.py results/ --out docs/

Evaluation seeds are 0 to N-1, calibration seeds are 100 to 119, and they never
meet. Sensitivities move one parameter at a time from the default, so a tornado
chart of them means what it looks like it means.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

from locrec import (
    LIMITED_FOV,
    SPINNING_360,
    LocalizabilityScheduler,
    NoMarkers,
    OraclePlacement,
    RunConfig,
    UniformSpacing,
    build_world,
    run_pass,
)
from locrec.gaze import ForwardGaze, GlanceGaze, GreedyGaze, OracleGaze, SweepGaze
from locrec.odometry import MotionPriorSpec, OdometryConfig
from locrec.policies import DropOnFailure
from locrec.runner import write_csv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from calibrate_thresholds import blind_world, mixed_world  # noqa: E402
from drone_gaze import junction_world  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORLDS = {"blind": blind_world, "mixed": mixed_world, "junction": junction_world}

SENSITIVITIES = {
    "range_scale": [0.5, 1.0, 2.0],
    "odom_scale": [0.5, 1.0, 2.0],
    "marker_height_m": [0.5, 1.0, 1.5],  # UGV only
    "fov_deg": [70.0, 90.0, 120.0],  # drone only
}


def ugv_policies(th: dict):
    return {
        "none": lambda w: NoMarkers(),
        "uniform_15m": lambda w: UniformSpacing(spacing=15.0),
        "uniform_8m": lambda w: UniformSpacing(spacing=8.0),
        "on_failure": lambda w: DropOnFailure(min_inliers=th["min_inliers"]),
        "scheduler": lambda w: LocalizabilityScheduler(
            ratio_threshold=th["ratio_threshold"],
            reliable_range_m=th["marker_reliable_range_m"],
            margin_m=th["scheduler_margin_m"],
        ),
        "oracle_12": lambda w: OraclePlacement(world=w, budget=12),
        "oracle_24": lambda w: OraclePlacement(world=w, budget=24),
    }


FORWARD_CONE_DEG = 60.0
"""Half-angle the restricted gaze policies may use, about the direction of travel.

Chosen from the milestone 3 failure rather than by search: the unrestricted policy
maximised map information by looking at the map, which is behind, and stopped
mapping ahead. Sixty degrees is the sweep policy's existing amplitude, so the two
restricted policies are being compared over the same span."""


def drone_policies(th: dict):
    t = th["drone_ratio_threshold"]
    return {
        "forward": lambda w: ForwardGaze(),
        "sweep": lambda w: SweepGaze(),
        "greedy": lambda w: GreedyGaze(ratio_threshold=t),
        "greedy_forward": lambda w: GreedyGaze(
            ratio_threshold=t, cone_deg=FORWARD_CONE_DEG, name="greedy_forward"
        ),
        "glance": lambda w: GlanceGaze(ratio_threshold=t, cone_deg=FORWARD_CONE_DEG),
        "oracle": lambda w: OracleGaze(ratio_threshold=t, cone_deg=FORWARD_CONE_DEG),
    }


def build_config(platform, world_name, seed, length, th, sens):
    """Config plus the policy factory table, with one sensitivity applied."""
    world = WORLDS[world_name](length)
    if platform == "ugv":
        lidar = SPINNING_360
        guard = th["gicp_ratio_floor"]
    else:
        lidar = LIMITED_FOV
        guard = th["drone_gicp_ratio_floor"]

    prior = MotionPriorSpec()
    if sens.get("range_scale", 1.0) != 1.0:
        lidar = dataclasses.replace(lidar, max_range=lidar.max_range * sens["range_scale"])
    if sens.get("odom_scale", 1.0) != 1.0:
        prior = dataclasses.replace(
            prior, odom_scale_sigma=prior.odom_scale_sigma * sens["odom_scale"]
        )
    if platform == "ugv" and sens.get("marker_height_m"):
        world = dataclasses.replace(world, marker_height=sens["marker_height_m"])
    if platform == "drone" and sens.get("fov_deg"):
        lidar = dataclasses.replace(lidar, fov_azimuth_deg=sens["fov_deg"])

    cfg = RunConfig(
        seed=seed,
        platform=platform,
        world=world,
        lidar=lidar,
        prior=prior,
        odometry=OdometryConfig(num_threads=1, gicp_ratio_floor=guard),
    )
    return cfg, world


def detector_lead_times(result, world, threshold: float, dt: float) -> list[float]:
    """Seconds between the detector firing and along-track error taking off.

    For each blind stretch of the true world: find the first step inside it where
    the detector calls the scan degenerate, and the first step where the rolling
    growth rate of along-track error exceeds twice the rate over the ten steps
    before the stretch. Positive means the detector fired first, which is the only
    way the signal is worth anything to a planner.
    """
    s = result.array("s")
    err = np.abs(result.array("along_err_m"))
    ratio = result.array("loc_ratio")
    blind = ~world.feature_mask & (np.abs(world.curvature) <= 1e-6)

    runs = []
    start = None
    for i, flag in enumerate(blind):
        if flag and start is None:
            start = world.s[i]
        elif not flag and start is not None:
            runs.append((start, world.s[i]))
            start = None
    if start is not None:
        runs.append((start, world.s[-1]))

    out = []
    win = 10
    for a, b in runs:
        if b - a < 15.0:
            continue
        inside = np.flatnonzero((s >= a) & (s <= b))
        if inside.size < 2 * win:
            continue
        i0 = inside[0]
        pre = err[max(i0 - win, 0) : i0]
        pre_rate = float(np.polyfit(np.arange(pre.size), pre, 1)[0]) if pre.size > 2 else 0.0
        pre_rate = max(abs(pre_rate), 1e-4)

        fire = None
        grow = None
        for j in inside:
            if fire is None and np.isfinite(ratio[j]) and ratio[j] < threshold:
                fire = j
            if grow is None and j - i0 >= win:
                seg = err[j - win : j]
                rate = float(np.polyfit(np.arange(win), seg, 1)[0])
                if rate > 2.0 * pre_rate:
                    grow = j
        if fire is not None and grow is not None:
            out.append(float((grow - fire) * dt))
    return out


def _one(job):
    platform, label, seed, world_name, length, th, sens = job
    cfg, world_spec = build_config(platform, world_name, seed, length, th, sens)
    world = build_world(seed, world_spec)
    table = ugv_policies(th) if platform == "ugv" else drone_policies(th)
    t0 = time.time()
    policy = table[label](world)
    r = run_pass(cfg, policy)
    dt = cfg.platform_spec.step_length / cfg.platform_spec.speed_mps
    thr = th["ratio_threshold"] if platform == "ugv" else th["drone_ratio_threshold"]
    ratio = r.array("loc_ratio")
    leads = detector_lead_times(r, world, thr, dt)
    yaw = r.array("yaw")
    dyaw = np.abs(np.diff(np.unwrap(yaw[np.isfinite(yaw)]))) if np.isfinite(yaw).any() else np.zeros(0)
    return {
        "platform": platform,
        "policy": label,
        "seed": seed,
        "world": world_name,
        "length_m": length,
        "range_scale": sens.get("range_scale", 1.0),
        "odom_scale": sens.get("odom_scale", 1.0),
        "marker_height_m": sens.get("marker_height_m", cfg.world.marker_height),
        "fov_deg": sens.get("fov_deg", cfg.lidar_spec().fov_azimuth_deg),
        "sensor_range_m": cfg.lidar_spec().max_range,
        "odom_scale_sigma": cfg.prior.odom_scale_sigma,
        "ratio_threshold": thr,
        "final_along_m": r.final_along_track_error,
        "final_lateral_m": r.final_lateral_error,
        "final_rot_rad": r.final_rotation_error,
        "markers_placed": r.markers_placed,
        "path_length_m": r.path_length,
        "mission_time_s": len(r.rows) * dt,
        "blind_fraction": float(np.mean(ratio[np.isfinite(ratio)] < thr)),
        "mean_yaw_rate_dps": float(np.rad2deg(dyaw.mean() / dt)) if dyaw.size else 0.0,
        "glance_fraction": float(getattr(policy, "glance_fraction", float("nan"))),
        "scale_estimate": float(getattr(r.odometry, "scale", float("nan"))),
        "lead_time_median_s": float(np.median(leads)) if leads else float("nan"),
        "lead_time_n": len(leads),
        "wall_time_s": time.time() - t0,
    }


def build_jobs(platform, worlds, seeds, length, th, sensitivities: bool):
    table = ugv_policies(th) if platform == "ugv" else drone_policies(th)
    jobs = [
        (platform, label, seed, world, length, th, {})
        for world in worlds
        for label in table
        for seed in range(seeds)
    ]
    if not sensitivities:
        return jobs
    # one parameter at a time from the default, on the mixed world only, with the
    # two policies that matter for the headline
    keys = ["range_scale", "odom_scale"]
    keys += ["marker_height_m"] if platform == "ugv" else ["fov_deg"]
    headline = (
        ["none", "scheduler"] if platform == "ugv" else ["forward", "greedy_forward"]
    )
    for key in keys:
        for value in SENSITIVITIES[key]:
            if key in ("range_scale", "odom_scale") and value == 1.0:
                continue
            if key == "marker_height_m" and value == 1.0:
                continue
            if key == "fov_deg" and value == 90.0:
                continue
            for label in headline:
                for seed in range(seeds):
                    jobs.append((platform, label, seed, "mixed", length, th, {key: value}))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["ugv", "drone"], required=True)
    ap.add_argument("--world", default="all")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--length", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-sensitivities", action="store_true")
    ap.add_argument("--out", default="results/")
    args = ap.parse_args()

    th = json.loads((ROOT / "results" / "thresholds.json").read_text())
    worlds = sorted(WORLDS) if args.world == "all" else [args.world]
    jobs = build_jobs(
        args.platform, worlds, args.seeds, args.length, th, not args.no_sensitivities
    )
    print(f"{args.platform}: {len(jobs)} runs over {worlds}")

    started = time.time()
    rows = []
    if args.workers > 1:
                # workers recycle: a 300 m world per job is not fully released and a long
        # sweep otherwise walks into the OOM killer part way through
        with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
            for i, row in enumerate(pool.imap_unordered(_one, jobs), 1):
                rows.append(row)
                if i % 5 == 0 or i == len(jobs):
                    rate = (time.time() - started) / i
                    print(
                        f"[{i}/{len(jobs)}] {rate:.1f} s/run, "
                        f"{(len(jobs) - i) * rate / 60:.0f} min left",
                        flush=True,
                    )
    else:
        for i, job in enumerate(jobs, 1):
            rows.append(_one(job))
            if i % 5 == 0 or i == len(jobs):
                rate = (time.time() - started) / i
                print(f"[{i}/{len(jobs)}] {rate:.1f} s/run", flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"monte_carlo_{args.platform}.csv"
    write_csv(rows, str(path))

    elapsed = time.time() - started
    per_run = elapsed / max(len(jobs), 1)
    print(f"\nwrote {path}: {len(rows)} rows in {elapsed / 60:.1f} min")
    print(f"{per_run:.1f} s per run at {args.workers} workers")
    for n in (50,):
        scaled = build_jobs(args.platform, worlds, n, args.length, th, True)
        for w in (4, 8):
            hours = len(scaled) * per_run * args.workers / w / 3600.0
            print(
                f"projected: {len(scaled)} runs at {n} seeds, "
                f"{hours:.1f} h at {w} workers"
            )


if __name__ == "__main__":
    main()
