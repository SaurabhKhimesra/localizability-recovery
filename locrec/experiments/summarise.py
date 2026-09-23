"""Turn the Monte Carlo CSVs into the four plots and one summary table.

    python experiments/summarise.py results/ --out docs/

Every headline carries a bootstrap interval over 1000 resamples. Where common
random numbers make a paired comparison possible, the paired number is the one
reported, because the spread between tunnels is much larger than the effect and a
marginal interval mostly measures the former.
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

INK = "#1c1c1a"
MUTED = "#6b6b66"
GRID = "#e3e3df"
SURFACE = "#fcfcfb"
SERIES = ["#1f6feb", "#d1610a", "#2f8f5b", "#8257d1", INK]


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


def style(ax, title=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    if title:
        ax.set_title(title, fontsize=9, color=MUTED, loc="left")


def load(path: pathlib.Path):
    rows = []
    for f in sorted(path.glob("monte_carlo_*.csv")):
        for r in csv.DictReader(open(f)):
            for k, v in list(r.items()):
                if k in ("platform", "policy", "world"):
                    continue
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = float("nan")
            rows.append(r)
    return rows


def baseline(rows, platform):
    return "none" if platform == "ugv" else "forward"


DEFAULTS = {"ugv": ("marker_height_m", 1.0), "drone": ("fov_deg", 90.0)}


def is_default(r, platform) -> bool:
    """A run with every swept parameter at its default value.

    The sensitivity runs live in the same file and are on the mixed world, so
    forgetting the per-platform parameter here silently pools three fields of view
    into the headline. It did: the drone's mixed-world greedy median read 10.93 m
    pooled against 23.01 m at the default 90 degrees.
    """
    if r["platform"] != platform or r["range_scale"] != 1.0 or r["odom_scale"] != 1.0:
        return False
    key, default = DEFAULTS[platform]
    value = r.get(key, default)
    return not np.isfinite(value) or value == default


def is_default_in(r, platform) -> bool:
    """The per-platform swept parameter is at its default, ignoring the others."""
    key, default = DEFAULTS[platform]
    value = r.get(key, default)
    return not np.isfinite(value) or value == default


def default_rows(rows, platform):
    return [r for r in rows if is_default(r, platform)]


def pareto_ugv(rows, out: pathlib.Path):
    sub = default_rows(rows, "ugv")
    worlds = sorted({r["world"] for r in sub})
    fig, axes = plt.subplots(1, len(worlds), figsize=(4.6 * len(worlds), 4.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    for ax, world in zip(axes, worlds):
        style(ax, world)
        by = defaultdict(list)
        for r in sub:
            if r["world"] == world:
                by[r["policy"]].append(r)
        for i, (policy, rs) in enumerate(sorted(by.items())):
            med, lo, hi = bootstrap_ci([r["final_along_m"] for r in rs])
            mk = float(np.median([r["markers_placed"] for r in rs]))
            c = SERIES[i % len(SERIES)]
            ax.errorbar(
                [mk], [med], yerr=[[med - lo], [hi - med]], fmt="o", color=c,
                ecolor=c, elinewidth=2.0, capsize=4, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2.0,
            )
            ax.annotate(
                policy, xy=(mk, med), xytext=(0, 11), textcoords="offset points",
                ha="center", fontsize=8, color=c,
            )
        ax.set_xlabel("markers spent", fontsize=10, color=INK)
    axes[0].set_ylabel("final along-track error (m)", fontsize=10, color=INK)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def drone_cost(rows, out: pathlib.Path):
    sub = default_rows(rows, "drone")
    if not sub:
        return
    worlds = sorted({r["world"] for r in sub})
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, xkey, xlabel in zip(
        axes,
        ("mean_yaw_rate_dps", "mission_time_s"),
        ("mean yaw rate (deg/s)", "mission time (s)"),
    ):
        style(ax)
        for i, world in enumerate(worlds):
            by = defaultdict(list)
            for r in sub:
                if r["world"] == world:
                    by[r["policy"]].append(r)
            xs, ys, labels = [], [], []
            for policy, rs in sorted(by.items()):
                xs.append(float(np.median([r[xkey] for r in rs])))
                ys.append(float(np.median([r["final_along_m"] for r in rs])))
                labels.append(f"{policy}")
            c = SERIES[i % len(SERIES)]
            ax.scatter(xs, ys, color=c, s=55, edgecolor=SURFACE, linewidth=2.0, zorder=3)
            for x, y, lab in zip(xs, ys, labels):
                ax.annotate(
                    lab, xy=(x, y), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8, color=c,
                )
            ax.plot([], [], "o", color=c, label=world)
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
    axes[0].set_ylabel("final along-track error (m)", fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    axes[1].set_title(
        "Mission time is identical by construction: every gaze policy flies the\n"
        "same centreline at the same speed, so yaw rate is the only cost axis.",
        fontsize=8, color=MUTED, loc="left",
    )
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def sensitivity_tornado(rows, out: pathlib.Path):
    bars = []
    for platform in ("ugv", "drone"):
        base = [r for r in rows if r["world"] == "mixed" and is_default(r, platform)]
        if not base:
            continue
        base_med = {}
        for policy in {r["policy"] for r in base}:
            base_med[policy] = float(
                np.median([r["final_along_m"] for r in base if r["policy"] == policy])
            )
        for key, label in (
            ("range_scale", "sensor range"),
            ("odom_scale", "odometry scale error"),
            ("marker_height_m", "strip height"),
            ("fov_deg", "field of view"),
        ):
            values = sorted(
                {r[key] for r in rows if r["platform"] == platform and key in r}
            )
            for value in values:
                sub = [
                    r
                    for r in rows
                    if r["platform"] == platform
                    and r["world"] == "mixed"
                    and r[key] == value
                    and (key == "range_scale" or r["range_scale"] == 1.0)
                    and (key == "odom_scale" or r["odom_scale"] == 1.0)
                    and (key == DEFAULTS[platform][0] or is_default_in(r, platform))
                ]
                sub = [r for r in sub if r["policy"] in base_med]
                if len(sub) < 3:
                    continue
                med = float(np.median([r["final_along_m"] for r in sub]))
                ref = float(np.median([base_med[r["policy"]] for r in sub]))
                if not np.isfinite(ref) or ref <= 0 or abs(med - ref) < 1e-9:
                    continue
                bars.append((f"{platform} {label} {value:g}", med - ref))
    if not bars:
        return
    bars = sorted(bars, key=lambda b: abs(b[1]))
    fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(bars) + 1.6), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    style(ax, "Change in median along-track error from the default, one parameter at a time")
    colors = [SERIES[0] if v < 0 else SERIES[1] for _, v in bars]
    ax.barh([b[0] for b in bars], [b[1] for b in bars], color=colors, height=0.6)
    ax.axvline(0.0, color=MUTED, linewidth=1.0)
    ax.set_xlabel("change in along-track error (m), negative is better", fontsize=10, color=INK)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def lead_time(rows, out: pathlib.Path):
    vals = defaultdict(list)
    for r in rows:
        v = r.get("lead_time_median_s", float("nan"))
        if np.isfinite(v):
            vals[r["platform"]].append(v)
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    style(
        ax,
        "Seconds between the detector calling a stretch degenerate and along-track\n"
        "error growth passing twice its pre-stretch rate. Positive means warning.",
    )
    for i, (platform, v) in enumerate(sorted(vals.items())):
        ax.hist(
            v, bins=20, alpha=0.65, color=SERIES[i % len(SERIES)], label=platform,
            edgecolor=SURFACE,
        )
        ax.axvline(np.median(v), color=SERIES[i % len(SERIES)], linewidth=2.0)
    ax.axvline(0.0, color=MUTED, linewidth=1.0)
    ax.set_xlabel("lead time (s)", fontsize=10, color=INK)
    ax.set_ylabel("runs", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def summary_table(rows, out: pathlib.Path):
    lines = []
    for platform in ("ugv", "drone"):
        sub = default_rows(rows, platform)
        base = baseline(rows, platform)
        for world in sorted({r["world"] for r in sub}):
            per_seed = defaultdict(dict)
            for r in sub:
                if r["world"] == world:
                    per_seed[r["policy"]][r["seed"]] = r
            if base not in per_seed:
                continue
            for policy, by_seed in sorted(per_seed.items()):
                along = [r["final_along_m"] for r in by_seed.values()]
                lat = [r["final_lateral_m"] for r in by_seed.values()]
                med, lo, hi = bootstrap_ci(along)
                diffs = [
                    per_seed[base][s]["final_along_m"] - by_seed[s]["final_along_m"]
                    for s in by_seed
                    if s in per_seed[base]
                ]
                pm, plo, phi = bootstrap_ci(diffs)
                lines.append(
                    {
                        "platform": platform,
                        "world": world,
                        "policy": policy,
                        "n_seeds": len(by_seed),
                        "along_median_m": round(med, 3),
                        "along_lo": round(lo, 3),
                        "along_hi": round(hi, 3),
                        "lateral_median_m": round(float(np.median(lat)), 3),
                        "markers_median": round(
                            float(np.median([r["markers_placed"] for r in by_seed.values()])), 1
                        ),
                        "yaw_rate_dps": round(
                            float(np.median([r["mean_yaw_rate_dps"] for r in by_seed.values()])), 2
                        ),
                        "paired_vs_baseline_m": round(pm, 3),
                        "paired_lo": round(plo, 3),
                        "paired_hi": round(phi, 3),
                        "seeds_improved": sum(d > 0 for d in diffs),
                    }
                )
    if not lines:
        return
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lines[0]))
        w.writeheader()
        w.writerows(lines)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs"))
    args = ap.parse_args()

    rows = load(args.results)
    if not rows:
        raise SystemExit(f"no monte_carlo_*.csv under {args.results}")
    args.out.mkdir(parents=True, exist_ok=True)

    pareto_ugv(rows, args.out / "pareto_ugv.png")
    drone_cost(rows, args.out / "drone_cost.png")
    sensitivity_tornado(rows, args.out / "sensitivity_tornado.png")
    lead_time(rows, args.out / "lead_time.png")
    lines = summary_table(rows, args.results / "summary.csv")

    print(f"{len(rows)} runs")
    if lines:
        print(
            f"{'platform':<8}{'world':<10}{'policy':<12}{'along':>8}"
            f"{'paired':>9}{'95% CI':>18}{'seeds':>7}"
        )
        for r in lines:
            ci = f"[{r['paired_lo']:.2f}, {r['paired_hi']:.2f}]"
            print(
                f"{r['platform']:<8}{r['world']:<10}{r['policy']:<12}"
                f"{r['along_median_m']:>8.2f}{r['paired_vs_baseline_m']:>9.2f}"
                f"{ci:>18}{r['seeds_improved']:>4}/{r['n_seeds']}"
            )
    print(f"wrote plots to {args.out} and {args.results / 'summary.csv'}")


if __name__ == "__main__":
    main()
