"""Does correcting the strip fit's end-face bias stop markers hurting the seeds that were good?

Declared before the first run, the same test ``credit_registration.py`` failed. Mixed world,
seeds 0 to 7, 300 m, one thread, MuJoCo, the calibrated thresholds, the 2 cm strip unchanged.
Statistic: final along-track error, scheduler against no markers, paired by seed, with
``OdometryConfig.correct_strip_bias`` off and on. It counts as a fix only if fewer seeds are made
worse by markers with the flag on AND the median paired benefit does not fall by more than a
third. Expectation written down beforehand, from the 1 mm strip counterfactual: median near
0.5 m, one seed worse. Anything else is reported as it is.

    python experiments/strip_bias_grid.py --workers 6
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

from locrec import LocalizabilityScheduler, NoMarkers, RunConfig, run_pass
from locrec.odometry import OdometryConfig

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from calibrate_thresholds import mixed_world  # noqa: E402
from marker_drop import bootstrap_ci  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _one(job):
    seed, policy, credit, length = job
    th = json.loads((ROOT / "results" / "thresholds.json").read_text())
    cfg = RunConfig(seed=seed, platform="ugv", world=mixed_world(length),
                    odometry=OdometryConfig(num_threads=1, gicp_ratio_floor=th["gicp_ratio_floor"],
                                            correct_strip_bias=credit))
    pol = NoMarkers() if policy == "none" else LocalizabilityScheduler(
        ratio_threshold=th["ratio_threshold"], reliable_range_m=th["marker_reliable_range_m"],
        margin_m=th["scheduler_margin_m"])
    r = run_pass(cfg, pol)
    return {"seed": seed, "policy": policy, "correct_strip_bias": int(credit),
            "final_along_m": r.final_along_track_error, "final_total_m": r.final_translation_error,
            "mean_abs_along_m": float(np.mean(np.abs(r.array("along_err_m")))), "markers": r.markers_placed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--length", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    jobs = [(s, p, c, args.length) for s in range(args.seeds) for p in ("none", "scheduler") for c in (False, True)]
    with Pool(args.workers, maxtasksperchild=4) as pool:
        rows = sorted(pool.imap_unordered(_one, jobs), key=lambda r: (r["correct_strip_bias"], r["seed"], r["policy"]))
    out = ROOT / "results" / "strip_bias_grid.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for credit in (0, 1):
        by = {(r["policy"], r["seed"]): r["final_along_m"] for r in rows if r["correct_strip_bias"] == credit}
        d = [by[("none", s)] - by[("scheduler", s)] for s in range(args.seeds)]
        med, lo, hi = bootstrap_ci(d)
        none_median = np.median([by[("none", s)] for s in range(args.seeds)])
        print(f"correct_strip_bias={credit}: median none {none_median:.2f} m, "
              f"scheduler {np.median([by[('scheduler', s)] for s in range(args.seeds)]):.2f} m; paired {med:+.2f} m "
              f"[{lo:+.2f}, {hi:+.2f}]; worse with markers on seeds {[s for s in range(args.seeds) if d[s] < 0]}")
        print("   per seed (none, scheduler):",
              [(s, round(by[("none", s)], 2), round(by[("scheduler", s)], 2)) for s in range(args.seeds)])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
