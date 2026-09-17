"""Milestone 2c: drift against markers spent, and the matched-drift table.

    python experiments/pareto_markers.py results/monte_carlo_ugv.csv --out docs/

The scheduler's claim is horizontal on this plot, not vertical. It does not claim
lower drift than uniform spacing; it claims the same drift for fewer markers. So
the number reported is the saving: take a policy's drift, read off how many
markers uniform spacing needs to reach that drift, and subtract what the policy
actually spent.

Only default-parameter rows are used, so the sensitivity sweeps in the same file
do not leak into the headline. Bootstrap intervals are over 1000 resamples of the
per-seed medians.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#1c1c1a"
MUTED = "#6b6b66"
GRID = "#e3e3df"
SURFACE = "#fcfcfb"
SERIES = ["#1f6feb", "#d1610a", "#2f8f5b", "#8257d1", "#b3123f", "#0f766e", INK]


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(n_boot, v.size), replace=True), axis=1)
    return (
        float(np.median(v)),
        float(np.quantile(meds, alpha / 2)),
        float(np.quantile(meds, 1 - alpha / 2)),
    )


def load(path: pathlib.Path):
    rows = []
    for r in csv.DictReader(open(path)):
        for k, v in list(r.items()):
            if k in ("platform", "policy", "world"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = float("nan")
        if r.get("range_scale", 1.0) != 1.0 or r.get("odom_scale", 1.0) != 1.0:
            continue
        if r.get("marker_height_m", 1.0) not in (1.0, float("nan")) and np.isfinite(
            r.get("marker_height_m", float("nan"))
        ):
            continue
        rows.append(r)
    return rows


def per_policy(rows, world):
    by = defaultdict(list)
    for r in rows:
        if r["world"] == world:
            by[r["policy"]].append(r)
    out = {}
    for policy, sub in by.items():
        med, lo, hi = bootstrap_ci([r["final_along_m"] for r in sub])
        out[policy] = {
            "markers": float(np.median([r["markers_placed"] for r in sub])),
            "along": med,
            "lo": lo,
            "hi": hi,
            "lateral": float(np.median([r["final_lateral_m"] for r in sub])),
            "n": len(sub),
        }
    return out


def paired_vs_none(rows, world, policy, baseline="none"):
    """Per-seed difference against the no-marker run on the same sampled world.

    Common random numbers make this the meaningful comparison: the spread between
    tunnels is far larger than the effect, so a marginal interval mostly measures
    which tunnels were drawn rather than what the policy did.
    """
    by = {}
    for r in rows:
        if r["world"] == world and r["policy"] in (policy, baseline):
            by.setdefault(r["policy"], {})[r["seed"]] = r["final_along_m"]
    base, sub = by.get(baseline, {}), by.get(policy, {})
    diffs = [base[s] - sub[s] for s in sub if s in base]
    med, lo, hi = bootstrap_ci(diffs)
    return med, lo, hi, sum(d > 0 for d in diffs), len(diffs)


def matched(stats):
    """How many markers uniform spacing needs to reach each policy's drift."""
    uni = sorted(
        ((v["markers"], v["along"]) for k, v in stats.items() if k.startswith("uniform")),
        key=lambda p: p[0],
    )
    if "none" in stats:
        uni = [(0.0, stats["none"]["along"])] + uni
    if len(uni) < 2:
        return uni, {}
    xs = np.array([p[0] for p in uni])
    ys = np.array([p[1] for p in uni])
    order = np.argsort(ys)[::-1]  # drift falling as markers rise
    res = {}
    for k, v in stats.items():
        if k.startswith("uniform") or k == "none":
            continue
        need = float(np.interp(v["along"], ys[order], xs[order]))
        res[k] = (v["markers"], v["along"], need, need - v["markers"])
    return uni, res


def plot(rows, out: pathlib.Path):
    worlds = sorted({r["world"] for r in rows})
    fig, axes = plt.subplots(
        1, len(worlds), figsize=(5.0 * len(worlds), 4.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    for ax, world in zip(axes, worlds):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_title(f"{world} world", fontsize=9, color=MUTED, loc="left")
        stats = per_policy(rows, world)
        uni = sorted(
            ((v["markers"], v["along"]) for k, v in stats.items() if k.startswith("uniform")),
            key=lambda p: p[0],
        )
        if "none" in stats:
            uni = [(0.0, stats["none"]["along"])] + uni
        if len(uni) >= 2:
            ax.plot(
                [p[0] for p in uni], [p[1] for p in uni], color=MUTED, lw=1.2, ls="--",
                zorder=1, label="uniform spacing",
            )
        for i, (policy, v) in enumerate(sorted(stats.items())):
            c = SERIES[i % len(SERIES)]
            ax.errorbar(
                [v["markers"]], [v["along"]],
                yerr=[[v["along"] - v["lo"]], [v["hi"] - v["along"]]],
                fmt="o", color=c, ecolor=c, elinewidth=2.0, capsize=4, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=3,
            )
            ax.annotate(
                policy, xy=(v["markers"], v["along"]), xytext=(0, 11),
                textcoords="offset points", ha="center", fontsize=8, color=c,
            )
        ax.set_xlabel("markers spent", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    axes[0].set_ylabel("final along-track error (m)", fontsize=10, color=INK)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs"))
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        raise SystemExit(f"no default-parameter rows in {args.csv}")
    args.out.mkdir(parents=True, exist_ok=True)
    plot(rows, args.out / "pareto_markers.png")

    for world in sorted({r["world"] for r in rows}):
        stats = per_policy(rows, world)
        uni, res = matched(stats)
        n = max(v["n"] for v in stats.values())
        print(f"\n=== {world} world, {n} runs per policy ===")
        print(
            "uniform curve (markers, along-track):",
            [(round(a), round(b, 2)) for a, b in uni],
        )
        monotone = all(b >= a for a, b in zip([p[1] for p in uni][1:], [p[1] for p in uni]))
        if not monotone:
            print(
                "uniform curve is not monotone at this seed count, so the matched-drift\n"
                "saving below is interpolation across noise and means nothing. The paired\n"
                "column is the comparison that survives."
            )
        print(
            f"{'policy':<12}{'markers':>8}{'along':>8}{'95% CI':>16}{'lateral':>9}"
            f"{'paired':>8}{'paired 95% CI':>18}{'wins':>7}{'saving':>9}"
        )
        for k, v in sorted(stats.items()):
            ci = f"[{v['lo']:.2f}, {v['hi']:.2f}]"
            save = res.get(k, (None, None, float("nan"), float("nan")))[3]
            save_s = "" if not np.isfinite(save) else f"{save:+.1f}"
            pm, plo, phi, wins, n = paired_vs_none(rows, world, k)
            pair_s = "" if k == "none" else f"{pm:+.2f}"
            pci = "" if k == "none" else f"[{plo:+.2f}, {phi:+.2f}]"
            wins_s = "" if k == "none" else f"{wins}/{n}"
            print(
                f"{k:<12}{v['markers']:>8.0f}{v['along']:>8.2f}{ci:>16}{v['lateral']:>9.2f}"
                f"{pair_s:>8}{pci:>18}{wins_s:>7}{save_s:>9}"
            )
    print(f"\nwrote {args.out / 'pareto_markers.png'}")


if __name__ == "__main__":
    main()
