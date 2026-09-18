"""Is what is left of the marker chain's bias in the strip fit, or in the estimator?

Declared before the first run. Mixed world, seeds 0 to 7, 300 m, one thread, MuJoCo, the
calibrated thresholds, the scheduler. Five variants of the same runs:

* ``none``            no markers, the baseline every other variant is paired against
* ``real``            the pipeline as it stands, the strip fit uncorrected
* ``corrected``       ``OdometryConfig.correct_strip_bias`` on (``landmarks.strip_fit_bias``)
* ``perfect``         the marker's true position instead of the fit, the same detections, the
                      same beam count and the same ``seen_width_m``, so only the point moves
* ``thin``            a 1 mm strip, which has almost no end face. A diagnostic reference only,
                      not a thing to ship: it changes the world.

What each outcome would mean, written down before running. The quantity is the signed final
along-track error, averaged over the seeds, which is -0.86 m for ``real`` and +0.57 m for
``thin``: a lag of about 1.4 m that the correction only took 0.3 m out of.

* ``perfect`` lands near ``thin``: everything left is in the strip fit, and the fit's error is
  not only the mean the correction removes. Then look at its spread and at what the estimator
  is told that spread is.
* ``perfect`` stays near ``real``: the fit is not the whole story and the estimator's own
  handling of these fixes is, so the next thing to read is the weighting.

    python experiments/marker_bias_ablation.py --workers 6
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

from locrec import LocalizabilityScheduler, NoMarkers, RunConfig, run_pass
from locrec.odometry import OdometryConfig
from locrec.sim import MarkerDetection

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from calibrate_thresholds import mixed_world  # noqa: E402
from marker_drop import bootstrap_ci  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
VARIANTS = ("none", "real", "corrected", "perfect", "thin")
THIN_M = 0.001


def perfect_detection(sim, T_true, detections):
    """The marker's true position instead of the fit, everything else left alone.

    The beam count and the seen width are kept, so the covariance the estimator computes for the
    observation is the one it would have computed anyway and only the point moves.
    """
    if not detections:
        return detections
    world = sim.marker_positions()
    R, t = T_true[:3, :3], T_true[:3, 3]
    out = []
    for d in detections:
        if d.slot >= world.shape[0]:
            out.append(d)
            continue
        p = R.T @ (world[d.slot] - t)
        out.append(MarkerDetection(slot=d.slot, point_sensor=p, n_beams=d.n_beams,
                                   range_m=float(np.linalg.norm(p)), seen_width_m=d.seen_width_m,
                                   n_columns=d.n_columns))
    return out


def _one(job):
    variant, seed, length = job
    th = json.loads((ROOT / "results" / "thresholds.json").read_text())
    world = mixed_world(length)
    if variant == "thin":
        world = dataclasses.replace(world, marker_thickness=THIN_M)
    cfg = RunConfig(seed=seed, platform="ugv", world=world,
                    odometry=OdometryConfig(num_threads=1, gicp_ratio_floor=th["gicp_ratio_floor"],
                                            correct_strip_bias=variant == "corrected"))
    policy = NoMarkers() if variant == "none" else LocalizabilityScheduler(
        ratio_threshold=th["ratio_threshold"], reliable_range_m=th["marker_reliable_range_m"],
        margin_m=th["scheduler_margin_m"])
    r = run_pass(cfg, policy, detection_hook=perfect_detection if variant == "perfect" else None)
    rows = r.rows
    fixes = list(getattr(r.odometry, "fix_log", []))
    far = [f for f in fixes if f["range_m"] > 5.0]
    return {
        "variant": variant, "seed": seed,
        "final_along_m": r.final_along_track_error * np.sign(rows[-1]["along_err_m"] or 1.0),
        "signed_final_along_m": rows[-1]["along_err_m"],
        "final_total_m": r.final_translation_error,
        "markers": r.markers_placed, "fixes": len(fixes), "fixes_beyond_5m": len(far),
        "mean_residual_cm": 100.0 * float(np.mean([f["residual_along"] for f in fixes])) if fixes else 0.0,
        "mean_residual_far_cm": 100.0 * float(np.mean([f["residual_along"] for f in far])) if far else 0.0,
        "sd_residual_far_cm": 100.0 * float(np.std([f["residual_along"] for f in far])) if far else 0.0,
        "mean_sigma_far_cm": 100.0 * float(np.mean([np.sqrt(f["r_along"]) for f in far])) if far else 0.0,
        "mean_gain_far": float(np.mean([f["gain_along"] for f in far])) if far else 0.0,
        "scale_est": float(getattr(r.odometry, "scale", np.nan)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--length", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    jobs = [(v, s, args.length) for v in VARIANTS for s in range(args.seeds)]
    with Pool(args.workers, maxtasksperchild=4) as pool:
        rows = sorted(pool.imap_unordered(_one, jobs), key=lambda r: (VARIANTS.index(r["variant"]), r["seed"]))
    out = ROOT / "results" / "marker_bias_ablation.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    base = {r["seed"]: abs(r["signed_final_along_m"]) for r in rows if r["variant"] == "none"}
    print(f"{'variant':>10} {'signed mean':>12} {'median |e|':>11} {'worse than none':>16} {'fixes>5m':>9} "
          f"{'far resid':>10} {'far sd':>8} {'far sigma':>10} {'far gain':>9}")
    for v in VARIANTS:
        rs = [r for r in rows if r["variant"] == v]
        signed = [r["signed_final_along_m"] for r in rs]
        worse = sum(1 for r in rs if abs(r["signed_final_along_m"]) > base[r["seed"]])
        print(f"{v:>10} {np.mean(signed):>+12.2f} {np.median([abs(x) for x in signed]):>11.2f} "
              f"{worse:>16d} {np.mean([r['fixes_beyond_5m'] for r in rs]):>9.1f} "
              f"{np.mean([r['mean_residual_far_cm'] for r in rs]):>+10.2f} "
              f"{np.mean([r['sd_residual_far_cm'] for r in rs]):>8.2f} "
              f"{np.mean([r['mean_sigma_far_cm'] for r in rs]):>10.2f} "
              f"{np.mean([r['mean_gain_far'] for r in rs]):>9.2f}")
        if v != "none":
            d = [base[r["seed"]] - abs(r["signed_final_along_m"]) for r in rs]
            med, lo, hi = bootstrap_ci(d)
            print(f"{'':>10} paired against none: {med:+.2f} m [{lo:+.2f}, {hi:+.2f}], "
                  f"per seed {[round(x, 2) for x in signed]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
