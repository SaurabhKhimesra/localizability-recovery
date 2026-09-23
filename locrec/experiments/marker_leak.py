"""Milestone 9, part 1: where does the marker benefit leak out of the pipeline?

The bench says a chain of markers turns 4.01 m of dead reckoning into 0.26 m over
200 m. The pipeline says 2.92 m becomes 1.77 m, and some seeds get worse. One of
those two is wrong about something, and a correct estimator cannot make a seed two
metres worse, so this finds the leak before anything is changed.

    python experiments/marker_leak.py --seeds 8 --workers 2 --out docs/

D1 is a per-seed table. D2 plots the worst seed with every fix marked. D3 is four
ablations that each remove one candidate cause:

* registration off, so the estimate is the prior plus the markers and nothing
  else. If the benefit appears here, the frontend is fighting the correction.
* ground-truth association. Detection is already by MuJoCo geom id, so this is
  the same pipeline; the row is kept because ruling association out is worth a row.
* perfect detection: the marker's true position instead of the centroid of its
  returns, with the same beam count so the weighting is unchanged. If the benefit
  appears here, the strip fit is the fault.
* the full pipeline.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from locrec import LocalizabilityScheduler, NoMarkers, RunConfig, run_pass
from locrec.odometry import OdometryConfig
from locrec.sim import MarkerDetection

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from calibrate_thresholds import blind_world  # noqa: E402
from marker_drop import bootstrap_ci  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
INK = "#1c1c1a"
MUTED = "#6b6b66"
GRID = "#e3e3df"
SURFACE = "#fcfcfb"
SERIES = ["#1f6feb", "#d1610a", "#2f8f5b"]


def scheduler(th):
    return LocalizabilityScheduler(
        ratio_threshold=th["ratio_threshold"],
        reliable_range_m=th["marker_reliable_range_m"],
        margin_m=th["scheduler_margin_m"],
    )


def perfect_detection(sim, T_true, detections):
    """Replace each detection's centroid with the marker's true position.

    The beam count is kept, so the measurement covariance the estimator computes is
    identical and only the point moves. This isolates the strip fit.
    """
    if not detections:
        return detections
    world = sim.marker_positions()
    R = T_true[:3, :3]
    t = T_true[:3, 3]
    out = []
    for d in detections:
        if d.slot >= world.shape[0]:
            out.append(d)
            continue
        p = R.T @ (world[d.slot] - t)
        out.append(
            MarkerDetection(
                slot=d.slot, point_sensor=p, n_beams=d.n_beams,
                range_m=float(np.linalg.norm(p)),
            )
        )
    return out


def one(job):
    label, seed, length, th = job
    cfg_kw = dict(
        seed=seed,
        platform="ugv",
        world=blind_world(length),
    )
    guard = th["gicp_ratio_floor"]
    odom = OdometryConfig(num_threads=1, gicp_ratio_floor=guard)
    hook = None
    policy = scheduler(th)

    if label == "none":
        policy = NoMarkers()
    elif label == "no_registration":
        # a scan can never be registered, so the estimate is prior plus markers
        odom = OdometryConfig(
            num_threads=1, gicp_ratio_floor=guard, min_scan_points=10**9
        )
    elif label == "perfect_detection":
        hook = perfect_detection

    r = run_pass(RunConfig(odometry=odom, **cfg_kw), policy, detection_hook=hook)
    fixes = list(getattr(r.odometry, "fix_log", []))
    slots = {s for f in fixes for s in f["slots"]}
    return {
        "variant": label,
        "seed": seed,
        "final_along_m": r.final_along_track_error,
        "final_lateral_m": r.final_lateral_error,
        "markers_placed": r.markers_placed,
        "markers_reobserved": len(slots),
        "n_fixes": len(fixes),
        "fixes_per_marker": len(fixes) / max(len(slots), 1),
        "largest_fix_m": max((abs(f["along"]) for f in fixes), default=0.0),
        "median_gain_along": float(
            np.median([f["gain_along"] for f in fixes]) if fixes else np.nan
        ),
        "median_q_since_along": float(
            np.median([f["q_since_along"] for f in fixes]) if fixes else np.nan
        ),
        "median_r_along": float(
            np.median([f["r_along"] for f in fixes]) if fixes else np.nan
        ),
    }


def run(labels, seeds, length, th, workers):
    jobs = [(label, seed, length, th) for label in labels for seed in range(seeds)]
    rows = []
    if workers > 1:
        # workers recycle: a 300 m world per job is not fully released and a long
        # sweep otherwise walks into the OOM killer part way through
        with mp.get_context("spawn").Pool(workers, maxtasksperchild=4) as pool:
            for i, row in enumerate(pool.imap_unordered(one, jobs), 1):
                rows.append(row)
                print(f"[{i}/{len(jobs)}] {row['variant']} seed {row['seed']} "
                      f"along {row['final_along_m']:.2f}", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            rows.append(one(job))
            print(f"[{i}/{len(jobs)}]", flush=True)
    return rows


def d1(rows):
    base = {r["seed"]: r for r in rows if r["variant"] == "none"}
    sched = sorted(
        (r for r in rows if r["variant"] == "full"), key=lambda r: r["seed"]
    )
    print("\nD1: per seed, blind world, scheduler against no markers")
    print(f"{'seed':>4}{'none':>8}{'sched':>8}{'delta':>8}{'drops':>7}{'seen':>6}"
          f"{'fixes/mk':>10}{'max fix':>9}{'gain':>7}")
    for r in sched:
        b = base[r["seed"]]["final_along_m"]
        print(
            f"{r['seed']:>4}{b:>8.2f}{r['final_along_m']:>8.2f}"
            f"{b - r['final_along_m']:>+8.2f}{r['markers_placed']:>7}"
            f"{r['markers_reobserved']:>6}{r['fixes_per_marker']:>10.1f}"
            f"{r['largest_fix_m']:>9.3f}{r['median_gain_along']:>7.2f}"
        )
    worst = min(sched, key=lambda r: base[r["seed"]]["final_along_m"] - r["final_along_m"])
    print(f"worst seed: {worst['seed']}")
    return worst["seed"]


def d2(seed, length, th, out: pathlib.Path):
    """Error against distance for both policies, with every fix ticked."""
    cfg = dict(seed=seed, platform="ugv", world=blind_world(length),
               odometry=OdometryConfig(num_threads=1, gicp_ratio_floor=th["gicp_ratio_floor"]))
    plain = run_pass(RunConfig(**cfg), NoMarkers())
    sched = run_pass(RunConfig(**cfg), scheduler(th))
    fixes = list(getattr(sched.odometry, "fix_log", []))

    fig, ax = plt.subplots(figsize=(11.0, 5.2), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)

    ax.plot(plain.array("s"), plain.array("along_err_m"), color=MUTED, lw=1.8,
            label="no markers")
    s = sched.array("s")
    ax.plot(s, sched.array("along_err_m"), color=SERIES[0], lw=1.8, label="scheduler")

    drops = np.flatnonzero(sched.array("marker_dropped") > 0)
    for i in drops:
        ax.axvline(s[i], color=SERIES[2], lw=0.8, alpha=0.5)
    for f in fixes:
        k = min(f["step"], s.size - 1)
        ax.plot([s[k]], [sched.array("along_err_m")[k]], marker="|", color=SERIES[1],
                markersize=9, alpha=0.7)
    big = sorted(fixes, key=lambda f: -abs(f["along"]))[:6]
    for f in big:
        k = min(f["step"], s.size - 1)
        ax.annotate(f"{f['along']:+.2f} m", xy=(s[k], sched.array("along_err_m")[k]),
                    xytext=(0, 12), textcoords="offset points", fontsize=7,
                    color=SERIES[1], ha="center")
    ax.set_xlabel("distance travelled (m)", fontsize=10, color=INK)
    ax.set_ylabel("along-track error (m)", fontsize=10, color=INK)
    ax.set_title(
        f"seed {seed}: green lines are drops, orange ticks are fixes, "
        f"{len(fixes)} fixes off {sched.markers_placed} markers",
        fontsize=9, color=MUTED, loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "diag_worst_seed.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    along = sched.array("along_err_m")
    plain_along = plain.array("along_err_m")
    wrong_way = [f for f in fixes if f["along"] * np.sign(along[min(f["step"], along.size - 1)]) > 0]
    print(f"\nD2: {len(fixes)} fixes, {len(wrong_way)} of them pushed the error further "
          f"from zero")
    print(f"    largest fix {max((abs(f['along']) for f in fixes), default=0):.3f} m, "
          f"median |fix| {np.median([abs(f['along']) for f in fixes]) if fixes else float('nan'):.4f} m")
    print(f"    median gain along track {np.median([f['gain_along'] for f in fixes]):.3f}, "
          f"median Q_since {np.median([f['q_since_along'] for f in fixes]):.2e}, "
          f"median R {np.median([f['r_along'] for f in fixes]):.2e}")
    growth_sched = np.diff(np.abs(along))
    growth_plain = np.diff(np.abs(plain_along))
    print(f"    mean |error| growth per step: scheduler {1e3*growth_sched.mean():+.2f} mm, "
          f"no markers {1e3*growth_plain.mean():+.2f} mm")
    return fixes


def d3(rows, labels):
    base = {r["seed"]: r["final_along_m"] for r in rows if r["variant"] == "none"}
    print("\nD3: ablations, blind world, scheduler policy")
    print(f"{'variant':<20}{'along mean':>12}{'median':>9}{'paired':>9}{'95% CI':>20}{'seeds':>7}")
    for label in labels:
        sub = [r for r in rows if r["variant"] == label]
        if not sub:
            continue
        vals = [r["final_along_m"] for r in sub]
        if label == "none":
            print(f"{label:<20}{np.mean(vals):>12.2f}{np.median(vals):>9.2f}")
            continue
        diffs = [base[r["seed"]] - r["final_along_m"] for r in sub if r["seed"] in base]
        m, lo, hi = bootstrap_ci(diffs)
        print(
            f"{label:<20}{np.mean(vals):>12.2f}{np.median(vals):>9.2f}{m:>9.2f}"
            f"{f'[{lo:.2f}, {hi:.2f}]':>20}{sum(d > 0 for d in diffs):>4}/{len(diffs)}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--length", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "docs")
    args = ap.parse_args()

    th = json.loads((ROOT / "results" / "thresholds.json").read_text())
    labels = ["none", "no_registration", "perfect_detection", "full"]
    rows = run(labels, args.seeds, args.length, th, args.workers)
    import csv

    with open(ROOT / "results" / "marker_leak.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    worst = d1(rows)
    d2(worst, args.length, th, args.out)
    d3(rows, labels)
    print("\nwrote results/marker_leak.csv and docs/diag_worst_seed.png")


if __name__ == "__main__":
    main()
