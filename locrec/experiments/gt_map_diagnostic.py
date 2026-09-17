"""Is the registration's along-track information real, or spurious?

The estimator registers each scan against a local map built from its own
estimated poses. That is a relative measurement: if the estimate slides along the
tunnel, the map slides with it and the registration stays satisfied. The question
this script answers is whether the registration's large along-track Hessian
reflects a real lock onto tunnel geometry, or an artefact of treating correlated
correspondences as independent evidence.

The test is to build the map from ground-truth poses instead. The map is then a
fixed, correct reference. If the along-track information is real, the estimate
should lock onto it and along-track drift should vanish. If drift persists at
roughly the dead-reckoning rate, the information is spurious and must not be
allowed to outvote an absolute measurement.

    python experiments/gt_map_diagnostic.py --seeds 3
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib

import numpy as np

from locrec import RunConfig, TunnelSim, UGV
from locrec.odometry import MotionPrior, Odometry, OdometryConfig
from locrec.runner import write_csv
from locrec.se3 import inv_T, transform_points

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from calibrate_thresholds import blind_world  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(seed: int, length: float, gt_map: bool) -> dict:
    cfg = RunConfig(seed=seed, platform="ugv", world=blind_world(length))
    lspec = cfg.lidar_spec()
    ocfg = dataclasses.replace(
        OdometryConfig(),
        map_radius=2.0 * lspec.max_range,
        registration_range=min(
            lspec.max_range - 1.0 - cfg.platform_spec.step_length,
            lspec.nyquist_range(OdometryConfig().map_resolution),
        ),
    )
    sim = TunnelSim(seed, cfg.world, lspec)
    plat = UGV(sim.world, cfg.platform_spec)
    prior = MotionPrior(cfg.prior, seed=seed, dt=0.5)

    T_true = plat.pose()
    odom = Odometry(T_true.copy(), ocfg, prior_spec=cfg.prior, dt=0.5, lidar_spec=lspec)

    scan = sim.scan(T_true)
    odom.step(scan.points, np.eye(4))
    if gt_map:
        # overwrite whatever the estimator inserted with the truth-referenced version
        odom.map = type(odom.map)(
            radius=ocfg.map_radius,
            resolution=ocfg.map_resolution,
            min_observations=ocfg.map_min_observations,
        )
        odom.map.add(transform_points(T_true, scan.points), centre=T_true[:3, 3])

    rows = []
    while not plat.finished:
        T_prev = T_true
        plat.advance()
        T_true = plat.pose()
        scan = sim.scan(T_true)
        out = odom.step(scan.points, prior.predict(inv_T(T_prev) @ T_true))
        if gt_map:
            # the estimator's own insertion is discarded; the map stays ground truth
            odom.map.add(transform_points(T_true, scan.points), centre=T_true[:3, 3])

        axis = T_true[:3, :3] @ np.array([1.0, 0.0, 0.0])
        err = odom.T[:3, 3] - T_true[:3, 3]
        row = {
            "seed": seed,
            "gt_map": int(gt_map),
            "s": plat.s,
            "along_track_err_m": float(err @ axis),
            "lateral_err_m": float(np.linalg.norm(err - (err @ axis) * axis)),
            "num_inliers": out.num_inliers,
            "residual": out.error,
            "lambda_along_raw": float("nan"),
            "lambda_along_cal": float("nan"),
            "lambda_min_raw": float("nan"),
            "lambda_min_cal": float("nan"),
            "ratio": float("nan"),
        }
        if out.localizability is not None:
            ev = out.localizability.eigenvectors
            lam = out.localizability.eigenvalues
            # information along the tunnel axis, raw and after the residual calibration
            along_raw = float(axis @ (ev @ np.diag(lam) @ ev.T) @ axis)
            dof = max(out.num_inliers - 6, 1)
            scale = dof / out.error if np.isfinite(out.error) and out.error > 1e-12 else 1.0
            row["lambda_along_raw"] = along_raw
            row["lambda_along_cal"] = along_raw * scale
            row["lambda_min_raw"] = float(lam[0])
            row["lambda_min_cal"] = float(lam[0] * scale)
            row["ratio"] = out.localizability.ratio
        rows.append(row)
    return rows


def dead_reckoning(seed: int, length: float) -> float:
    cfg = RunConfig(seed=seed, platform="ugv", world=blind_world(length))
    sim = TunnelSim(seed, cfg.world, cfg.lidar_spec())
    plat = UGV(sim.world, cfg.platform_spec)
    prior = MotionPrior(cfg.prior, seed=seed, dt=0.5)
    T = plat.pose()
    T_est = T.copy()
    while not plat.finished:
        T_prev = T
        plat.advance()
        T = plat.pose()
        T_est = T_est @ prior.predict(inv_T(T_prev) @ T)
    axis = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
    return float((T_est[:3, 3] - T[:3, 3]) @ axis)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--length", type=float, default=120.0)
    args = ap.parse_args()

    all_rows = []
    print(f"{'seed':>5}{'dead reckoning':>16}{'est map':>12}{'gt map':>12}")
    for seed in range(args.seeds):
        dr = dead_reckoning(seed, args.length)
        finals = {}
        for gt in (False, True):
            rows = run(seed, args.length, gt)
            all_rows.extend(rows)
            finals[gt] = rows[-1]["along_track_err_m"]
        print(f"{seed:>5}{dr:>16.2f}{finals[False]:>12.2f}{finals[True]:>12.2f}", flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    write_csv(all_rows, str(ROOT / "results" / "gt_map_diagnostic.csv"))

    for gt in (False, True):
        sub = [
            r
            for r in all_rows
            if r["gt_map"] == int(gt) and np.isfinite(r["lambda_along_cal"])
        ]
        if not sub:
            continue
        cal = np.median([r["lambda_along_cal"] for r in sub])
        raw = np.median([r["lambda_along_raw"] for r in sub])
        print(
            f"{'gt map' if gt else 'est map'}: along-track lambda raw {raw:.3g} "
            f"calibrated {cal:.3g}, implied sigma {1.0 / np.sqrt(cal) * 1000:.1f} mm, "
            f"median ratio {np.median([r['ratio'] for r in sub]):.2e}"
        )
    print("wrote results/gt_map_diagnostic.csv")


if __name__ == "__main__":
    main()
