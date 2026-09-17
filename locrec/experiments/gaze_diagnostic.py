"""Milestone 3: why does the information-greedy gaze lose?

The grid says greedy gaze is four to ten times worse than pointing the sensor
along the path, on every world. That is a large enough negative to be worth an
explanation rather than a shrug, so this dumps the per-step trace of the two
policies on the same seed and prints the quantities that could explain it:

* inlier count and registration success, which fall if the scan stops overlapping
  the part of the map that is built,
* the along-track error increments against the yaw slew in the same step,
* how much of the run the sensor spends pointed away from travel.

No parameter is changed here and nothing is tuned. This is a diagnosis of a
result that has already been recorded.

    python experiments/gaze_diagnostic.py --seed 0 --world blind
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from locrec import RunConfig, run_pass
from locrec.gaze import ForwardGaze, GreedyGaze
from locrec.odometry import OdometryConfig
from locrec.runner import write_csv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from calibrate_thresholds import blind_world, mixed_world  # noqa: E402
from drone_gaze import junction_world  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORLDS = {"blind": blind_world, "mixed": mixed_world, "junction": junction_world}


def trace(label, seed, world_name, length, th):
    cfg = RunConfig(
        seed=seed,
        platform="drone",
        world=WORLDS[world_name](length),
        odometry=OdometryConfig(
            num_threads=1, gicp_ratio_floor=th["drone_gicp_ratio_floor"]
        ),
    )
    policy = (
        ForwardGaze()
        if label == "forward"
        else GreedyGaze(ratio_threshold=th["drone_ratio_threshold"])
    )
    return cfg, run_pass(cfg, policy)


def report(label, cfg, r):
    dt = cfg.platform_spec.step_length / cfg.platform_spec.speed_mps
    along = np.abs(r.array("along_err_m"))
    yaw = r.array("yaw")
    inl = r.array("num_inliers")
    reg = r.array("registered")
    extra = np.abs(r.array("extra_yaw_rad"))
    slew = np.concatenate([[0.0], np.abs(np.diff(np.unwrap(yaw)))])
    d_along = np.concatenate([[0.0], np.diff(along)])

    big = slew > np.quantile(slew[slew > 0], 0.75) if np.any(slew > 0) else slew > 1
    quiet = ~big
    print(f"\n--- {label} ---")
    print(f"final along-track {along[-1]:.2f} m, lateral {abs(r.array('lateral_err_m'))[-1]:.2f} m")
    print(f"registration succeeded on {100 * reg.mean():.1f} percent of steps")
    print(f"inliers median {np.median(inl):.0f}, 10th percentile {np.quantile(inl, 0.1):.0f}")
    print(f"mean yaw rate {np.rad2deg(slew.mean() / dt):.2f} deg/s, "
          f"max {np.rad2deg(slew.max() / dt):.1f} deg/s")
    print(f"sensor off the direction of travel by more than 30 deg on "
          f"{100 * np.mean(extra > np.deg2rad(30)):.1f} percent of steps")
    print(f"along-track error growth per step: {1e3 * d_along[big].mean():+.2f} mm on the "
          f"quarter of steps with the largest slew, {1e3 * d_along[quiet].mean():+.2f} mm "
          f"on the rest")
    lo = inl < np.quantile(inl, 0.25)
    print(f"and {1e3 * d_along[lo].mean():+.2f} mm on the quarter of steps with the "
          f"fewest inliers, {1e3 * d_along[~lo].mean():+.2f} mm on the rest")
    return [
        {"policy": label, "step": i, "s": s, "along_err_m": a, "yaw_rad": y,
         "slew_rad": sl, "num_inliers": n, "registered": g, "extra_yaw_rad": e}
        for i, (s, a, y, sl, n, g, e) in enumerate(
            zip(r.array("s"), along, yaw, slew, inl, reg, extra)
        )
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world", default="blind", choices=sorted(WORLDS))
    ap.add_argument("--length", type=float, default=300.0)
    args = ap.parse_args()

    th = json.loads((ROOT / "results" / "thresholds.json").read_text())
    rows = []
    for label in ("forward", "greedy"):
        cfg, r = trace(label, args.seed, args.world, args.length, th)
        rows += report(label, cfg, r)
    write_csv(rows, str(ROOT / "results" / f"gaze_diagnostic_{args.world}.csv"))
    print(f"\nwrote results/gaze_diagnostic_{args.world}.csv")


if __name__ == "__main__":
    main()
