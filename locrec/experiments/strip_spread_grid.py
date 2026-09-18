"""Does taking a marker observation's spread from the measured fit stop markers hurting seeds?

Declared before the first run, the same test ``credit_registration.py`` and ``strip_bias_grid.py``
were held to. Mixed world, seeds 0 to 7, 300 m, one thread, MuJoCo, the calibrated thresholds, the
2 cm strip unchanged, and ``correct_strip_bias`` at its default. That default is now **off**: the
correction was measured to be sensor-specific and harmful on Gazebo (``docs/failures.md`` number
34), so the shipped configuration is the spread model alone. The first run of this test was made
with the bias correction on and is superseded. Statistic: final along-track error,
scheduler against no markers, paired by seed, with ``OdometryConfig.measure_strip_spread`` off and
on. It counts as a fix only if fewer seeds are made worse by markers with the flag on AND the
median paired benefit does not fall by more than a third.

What the flag changes, and why it should matter. Off, an observation's covariance is modelled by
hand: range noise over the returns, beam quantisation and the strip's width over root twelve,
shared out over the number of distinct beam columns. On, it is the spread of the real fit run at
that geometry (``landmarks.strip_fit_sigma``), which is the same Monte Carlo the bias correction
already comes from. Measured against six real runs, the hand-written term charges 4.87 cm along
the tunnel at 4.4 m where the fit's real error is 4.06 cm rms, and 2.22 cm at 2 m where it is
1.17 cm; the Monte Carlo says 4.66 cm and 0.62 cm.

Expectation written down beforehand: the far single-column fixes are the ones that carry the
chain, and they were over-trusted by a factor of six in variance before the column count was
fixed and are now charged about right, so the change here is mostly at close range and should be
small. Median near 0.9 m, one or two seeds worse. A null is reported as a null.

    python experiments/strip_spread_grid.py --workers 6
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
    seed, policy, spread, length = job
    th = json.loads((ROOT / "results" / "thresholds.json").read_text())
    cfg = RunConfig(seed=seed, platform="ugv", world=mixed_world(length),
                    odometry=OdometryConfig(num_threads=1, gicp_ratio_floor=th["gicp_ratio_floor"],
                                            measure_strip_spread=spread))
    pol = NoMarkers() if policy == "none" else LocalizabilityScheduler(
        ratio_threshold=th["ratio_threshold"], reliable_range_m=th["marker_reliable_range_m"],
        margin_m=th["scheduler_margin_m"])
    r = run_pass(cfg, pol)
    return {"seed": seed, "policy": policy, "measure_strip_spread": int(spread),
            "final_along_m": r.final_along_track_error, "final_total_m": r.final_translation_error,
            "mean_abs_along_m": float(np.mean(np.abs(r.array("along_err_m")))), "markers": r.markers_placed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--length", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    jobs = [(s, p, m, args.length) for s in range(args.seeds) for p in ("none", "scheduler") for m in (False, True)]
    with Pool(args.workers, maxtasksperchild=4) as pool:
        rows = sorted(pool.imap_unordered(_one, jobs),
                      key=lambda r: (r["measure_strip_spread"], r["seed"], r["policy"]))
    out = ROOT / "results" / "strip_spread_grid.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for spread in (0, 1):
        by = {(r["policy"], r["seed"]): r["final_along_m"] for r in rows if r["measure_strip_spread"] == spread}
        d = [by[("none", s)] - by[("scheduler", s)] for s in range(args.seeds)]
        med, lo, hi = bootstrap_ci(d)
        print(f"measure_strip_spread={spread}: median none {np.median([by[('none', s)] for s in range(args.seeds)]):.2f} m, "
              f"scheduler {np.median([by[('scheduler', s)] for s in range(args.seeds)]):.2f} m; paired {med:+.2f} m "
              f"[{lo:+.2f}, {hi:+.2f}]; worse with markers on seeds {[s for s in range(args.seeds) if d[s] < 0]}")
        print("   per seed (none, scheduler):",
              [(s, round(by[("none", s)], 2), round(by[("scheduler", s)], 2)) for s in range(args.seeds)])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
