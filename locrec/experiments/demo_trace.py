"""Milestone 1 demo: one open-loop pass per platform through the same tunnel.

Writes one CSV row per step to ``results/`` and a three-panel diagnostic figure to
``docs/``. No recovery policy is active here; this is the plain baseline the later
milestones have to beat.

    python experiments/demo_trace.py --seeds 1
"""
from __future__ import annotations

import argparse
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from locrec import RunConfig, WorldSpec, build_world, run_pass  # noqa: E402
from locrec.runner import write_csv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLATFORM_COLOR = {"ugv": "#1f6feb", "drone": "#d1610a"}
PLATFORM_LABEL = {"ugv": "UGV, 360 deg LiDAR", "drone": "Drone, 90 deg LiDAR"}
INK = "#1c1c1a"
MUTED = "#6b6b66"
GRID = "#e3e3df"
FEATURE = "#e8e3d6"
CURVE = "#f2f0ea"


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def _bands(ax, s, mask, color):
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = s[i]
        elif not flag and start is not None:
            ax.axvspan(start, s[i], color=color, lw=0, zorder=0)
            start = None
    if start is not None:
        ax.axvspan(start, s[-1], color=color, lw=0, zorder=0)


def _feature_bands(ax, world):
    """Shade where the world has structure the estimator can use: curves (pale) and
    junctions or niches (darker). The detector is never told where any of it is."""
    _bands(ax, world.s, np.abs(world.curvature) > 1e-6, CURVE)
    _bands(ax, world.s, world.feature_mask, FEATURE)


def make_figure(results: dict, world, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.4), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    ax_loc, ax_err, ax_plan = axes

    for ax in (ax_loc, ax_err):
        _style(ax)
        _feature_bands(ax, world)
    _style(ax_plan)

    for name, r in results.items():
        c = PLATFORM_COLOR[name]
        s = r.array("s")
        ratio = r.array("loc_ratio")
        ax_loc.semilogy(s, ratio, color=c, lw=2.0, label=PLATFORM_LABEL[name])
        ax_err.plot(s, r.array("trans_err_m"), color=c, lw=2.0, label=PLATFORM_LABEL[name])
        ax_err.annotate(
            f"{PLATFORM_LABEL[name].split(',')[0]}  {r.final_translation_error:.2f} m",
            xy=(s[-1], r.array("trans_err_m")[-1]),
            xytext=(-6, 10 if name == "ugv" else -14),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=c,
        )

    ax_loc.set_ylabel("localizability ratio\n$\\lambda_{min}/\\lambda_{max}$", fontsize=9, color=INK)
    ax_loc.set_title(
        "Pale bands are curved sections, darker bands are junctions and niches. "
        "The detector is never told where any of them are.",
        fontsize=9,
        color=MUTED,
        loc="left",
    )
    ax_loc.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")

    ax_err.set_ylabel("translation error (m)", fontsize=9, color=INK)
    ax_err.set_xlabel("arclength along tunnel (m)", fontsize=9, color=INK)

    ref = next(iter(results.values()))
    ax_plan.plot(
        ref.array("x_true"), ref.array("y_true"), color=MUTED, lw=2.0, label="ground truth"
    )
    for name, r in results.items():
        ax_plan.plot(
            r.array("x_est"),
            r.array("y_est"),
            color=PLATFORM_COLOR[name],
            lw=2.0,
            ls="--",
            label=f"{PLATFORM_LABEL[name]}, estimate",
        )
    ax_plan.set_aspect("equal", adjustable="datalim")
    ax_plan.set_xlabel("x (m)", fontsize=9, color=INK)
    ax_plan.set_ylabel("y (m)", fontsize=9, color=INK)
    ax_plan.legend(frameon=False, fontsize=8, labelcolor=INK, loc="best")

    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1, help="run seeds 0..N-1")
    ap.add_argument("--length", type=float, default=160.0)
    args = ap.parse_args()

    world = WorldSpec(length=args.length)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "docs").mkdir(exist_ok=True)

    all_rows: list[dict] = []
    for seed in range(args.seeds):
        results = {}
        for platform in ("ugv", "drone"):
            t0 = time.time()
            r = run_pass(RunConfig(seed=seed, platform=platform, world=world))
            results[platform] = r
            for row in r.rows:
                all_rows.append({"seed": seed, "platform": platform, **row})
            print(
                f"seed {seed:2d}  {platform:5s}  "
                f"final drift {r.final_translation_error:6.2f} m over "
                f"{r.path_length:5.0f} m  ({time.time() - t0:.0f}s)"
            )
        if seed == 0:
            make_figure(results, build_world(seed, world), ROOT / "docs" / "milestone1_trace.png")

    write_csv(all_rows, str(ROOT / "results" / "milestone1_baseline.csv"))
    print(f"wrote results/milestone1_baseline.csv ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
