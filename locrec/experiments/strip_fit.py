"""Milestone 9: the residual distribution of a marker observation, before and after.

The leak diagnostics named detection as the fault: with the marker's true position
substituted, a chain of markers behaves like the bench test, and with the centroid
of its returns it does not. This measures the thing directly. A strip is bolted on
at a known point, the robot drives past, and at every range the observation is
compared with that point using the *true* pose, so no estimator error is involved.

    python experiments/strip_fit.py --out docs/

What is printed is the horizontal residual, split into the two axes the estimator
actually uses, against the standard deviation the estimator's own model claims at
that range and beam count. A bias larger than the modelled sigma is the failure:
the filter treats a systematic offset as independent evidence and follows it.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from locrec import SPINNING_360, TunnelSim, UGV  # noqa: E402
from locrec.landmarks import LandmarkSpec, measurement_information  # noqa: E402
from locrec.runner import RunConfig  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from calibrate_thresholds import blind_world  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
INK = "#1c1c1a"
MUTED = "#6b6b66"
GRID = "#e3e3df"
SURFACE = "#fcfcfb"
SERIES = ["#1f6feb", "#d1610a"]


def sweep(seed: int, length: float, drop_every_m: float = 25.0):
    """Drive the tunnel, drop strips, and record every later observation of each."""
    world = blind_world(length)
    sim = TunnelSim(seed, world, SPINNING_360)
    cfg = RunConfig(seed=seed, platform="ugv", world=world, lidar=SPINNING_360)
    plat = UGV(sim.world, cfg.platform_spec)
    spec = LandmarkSpec()

    anchors: dict[int, np.ndarray] = {}
    rows = []
    next_drop = 5.0
    while not plat.finished:
        T = plat.pose()
        if plat.s >= next_drop and sim.n_markers_placed < sim.marker_capacity:
            slot, offset = sim.drop_marker_on_wall(T)
            anchors[slot] = T[:3, :3] @ offset + T[:3, 3]
            next_drop += drop_every_m
        scan = sim.scan(T)
        for fit in (True, False):
            for d in sim.marker_detections(scan, fit_strip=fit):
                if d.slot not in anchors:
                    continue
                q = T[:3, :3] @ d.point_sensor + T[:3, 3]
                err = (q - anchors[d.slot])[:2]
                p = d.point_sensor
                horiz = np.array([p[0], p[1]])
                r = float(np.linalg.norm(horiz))
                if r < 1e-6:
                    continue
                e_r = horiz / r
                e_h = np.array([-e_r[1], e_r[0]])
                omega = measurement_information(
                    p, d.n_beams, SPINNING_360, spec,
                    world.marker_height, world.marker_width,
                )
                cov = np.linalg.pinv((omega)[:2, :2])
                rows.append(
                    {
                        "fit": fit,
                        "slot": d.slot,
                        "range": d.range_m,
                        "n_beams": d.n_beams,
                        "radial": float(err @ e_r),
                        "tangential": float(err @ e_h),
                        "sigma_radial": float(np.sqrt(max(e_r @ cov @ e_r, 0.0))),
                        "sigma_tangential": float(np.sqrt(max(e_h @ cov @ e_h, 0.0))),
                    }
                )
        plat.advance()
    return rows


def report(rows, label):
    sub = [r for r in rows if r["fit"] is label]
    if not sub:
        return
    name = "geometric fit" if label else "centroid"
    print(f"\n{name}: {len(sub)} observations")
    print(f"{'range band':>12}{'n':>6}{'radial bias':>13}{'std':>8}{'sigma':>8}"
          f"{'tang bias':>11}{'std':>8}{'sigma':>8}")
    edges = [0, 2, 3, 4, 5, 6, 8, 100]
    for a, b in zip(edges, edges[1:]):
        band = [r for r in sub if a <= r["range"] < b]
        if len(band) < 5:
            continue
        rad = np.array([r["radial"] for r in band])
        tan = np.array([r["tangential"] for r in band])
        print(
            f"{f'{a} to {b} m':>12}{len(band):>6}{rad.mean():>+13.4f}{rad.std():>8.4f}"
            f"{np.mean([r['sigma_radial'] for r in band]):>8.4f}"
            f"{tan.mean():>+11.4f}{tan.std():>8.4f}"
            f"{np.mean([r['sigma_tangential'] for r in band]):>8.4f}"
        )
    rad = np.abs([r["radial"] for r in sub])
    sig = np.array([r["sigma_radial"] for r in sub])
    print(f"  median |radial| / modelled sigma: {np.median(rad / np.maximum(sig, 1e-9)):.2f}")


def plot(rows, out: pathlib.Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, key, title in zip(axes, ("radial", "tangential"),
                              ("radial (range direction)", "tangential (bearing direction)")):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9)
        for i, fit in enumerate((False, True)):
            sub = [r for r in rows if r["fit"] is fit]
            ax.scatter([r["range"] for r in sub], [r[key] for r in sub], s=9,
                       color=SERIES[i], alpha=0.45,
                       label="geometric fit" if fit else "centroid")
        sub = [r for r in rows if r["fit"] is False]
        order = np.argsort([r["range"] for r in sub])
        rr = np.array([sub[i]["range"] for i in order])
        ss = np.array([sub[i][f"sigma_{key}"] for i in order])
        ax.plot(rr, ss, color=MUTED, lw=1.0, ls="--", label="modelled sigma")
        ax.plot(rr, -ss, color=MUTED, lw=1.0, ls="--")
        ax.axhline(0.0, color=INK, lw=0.8)
        ax.set_xlabel("range to strip (m)", fontsize=10, color=INK)
        ax.set_title(title, fontsize=9, color=MUTED, loc="left")
    axes[0].set_ylabel("observation minus registered point (m)", fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "strip_fit_residuals.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--length", type=float, default=120.0)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "docs")
    args = ap.parse_args()

    rows = sweep(args.seed, args.length)
    report(rows, False)
    report(rows, True)
    plot(rows, args.out)
    print(f"\nwrote {args.out / 'strip_fit_residuals.png'}")


if __name__ == "__main__":
    main()
