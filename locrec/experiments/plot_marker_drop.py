"""Drift against markers spent, with bootstrap confidence intervals.

    python experiments/plot_marker_drop.py [--csv milestone2_marker_drop.csv]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from marker_drop import bootstrap_ci  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
INK = "#1c1c1a"
MUTED = "#6b6b66"
GRID = "#e3e3df"
FAMILY_COLOR = {
    "none": INK,
    "uniform": "#1f6feb",
    "scheduler": "#d1610a",
    "on_failure": "#2f8f5b",
    "oracle": "#8257d1",
}


def family(label: str) -> str:
    for key in FAMILY_COLOR:
        if label.startswith(key):
            return key
    return "none"


def paired_vs_baseline(rows, baseline="none", key="final_along_m"):
    """Per-seed paired differences against the no-marker run.

    Common random numbers make this the right comparison: the same seed gives the
    same tunnel and the same odometry errors, so the difference isolates what the
    markers did. Comparing two marginal medians instead throws that away and
    reports the spread between seeds as though it were uncertainty about the effect.
    """
    by = defaultdict(dict)
    for r in rows:
        by[r["policy"]][int(r["seed"])] = float(r[key])
    base = by[baseline]
    out = {}
    for policy, per_seed in by.items():
        if policy == baseline:
            continue
        seeds = sorted(set(per_seed) & set(base))
        diffs = [base[s] - per_seed[s] for s in seeds]
        med, lo, hi = bootstrap_ci(diffs)
        out[policy] = {
            "median_reduction_m": med,
            "lo": lo,
            "hi": hi,
            "n": len(seeds),
            "wins": sum(d > 0 for d in diffs),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="milestone2_marker_drop.csv")
    ap.add_argument("--out", default="milestone2_drift_vs_markers.png")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(ROOT / "results" / args.csv)))
    by_policy = defaultdict(list)
    for r in rows:
        by_policy[r["policy"]].append(r)

    points = []
    for label, sub in by_policy.items():
        med, lo, hi = bootstrap_ci([float(r["final_along_m"]) for r in sub])
        points.append(
            {
                "label": label,
                "family": family(label),
                "markers": float(np.median([float(r["markers_placed"]) for r in sub])),
                "median": med,
                "lo": lo,
                "hi": hi,
                "n": len(sub),
            }
        )
    points.sort(key=lambda p: p["markers"])

    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)

    for fam in FAMILY_COLOR:
        fam_pts = [p for p in points if p["family"] == fam]
        if not fam_pts:
            continue
        c = FAMILY_COLOR[fam]
        x = [p["markers"] for p in fam_pts]
        y = [p["median"] for p in fam_pts]
        err = np.array([[p["median"] - p["lo"] for p in fam_pts], [p["hi"] - p["median"] for p in fam_pts]])
        if len(fam_pts) > 1:
            ax.plot(x, y, color=c, lw=2.0, zorder=2)
        ax.errorbar(
            x, y, yerr=err, fmt="o", color=c, ecolor=c, elinewidth=2.0,
            capsize=4, markersize=8, markeredgecolor="#fcfcfb", markeredgewidth=2.0, zorder=3,
        )
        for p in fam_pts:
            ax.annotate(
                p["label"],
                xy=(p["markers"], p["median"]),
                xytext=(0, -20 if fam == "oracle" else 11),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=c,
            )

    n_seeds = points[0]["n"] if points else 0
    ax.set_xlabel("markers spent", fontsize=10, color=INK)
    ax.set_ylabel("final along-track error (m), median of seeds", fontsize=10, color=INK)
    ax.set_title(
        f"300 m of featureless tunnel, {n_seeds} seeds, common random numbers.\n"
        "Bars are 95 percent bootstrap intervals on the median, so they carry the "
        "spread between tunnels.\nThe paired table in the README is the comparison "
        "that isolates what the markers did.",
        fontsize=9,
        color=MUTED,
        loc="left",
    )
    print("\npaired against the no-marker run, same seed (positive means markers helped)")
    print(f"{'policy':<14}{'median reduction':>18}{'95% CI':>22}{'wins':>8}")
    for policy, d in sorted(
        paired_vs_baseline(rows).items(), key=lambda kv: -kv[1]["median_reduction_m"]
    ):
        ci = "[%.2f, %.2f]" % (d["lo"], d["hi"])
        wins = "%d/%d" % (d["wins"], d["n"])
        print(f"{policy:<14}{d['median_reduction_m']:>18.2f}{ci:>22}{wins:>8}")

    out = ROOT / "docs" / args.out
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
