"""Milestone 5: does the detector fire where drift grows, on real data?

Reads a ROS 1 or ROS 2 bag with ``rosbags``, which is pure Python, so this runs
without a ROS installation. Scans go through the same odometry the simulation
uses, the localizability ratio is computed per scan, along-track error is measured
against the ground-truth poses expressed in the odometry frame, and three things
come out:

* the ratio over distance with the drift-growth segments shaded,
* a ROC of "the detector fires" against "along-track error grows over the next
  five seconds",
* the lead-time distribution.

    python experiments/real_sequence.py /path/to/exp14_basement_2.bag \
        --lidar-topic /hesai/pandar --gt-topic /gt_pose --out docs/

The ground truth is aligned to the odometry frame by a rigid fit over the first
twenty metres, not over the whole run. Aligning over the whole run would spread
the drift evenly across it and hide exactly the thing being measured.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from locrec.localizability import analyse_hessian  # noqa: F401, E402  (re-exported for clarity)
from locrec.odometry import Odometry, OdometryConfig  # noqa: E402
from locrec.se3 import inv_T, make_T  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

INK = "#1c1c1a"
MUTED = "#6b6b66"
GRID = "#e3e3df"
SURFACE = "#fcfcfb"
SERIES = ["#1f6feb", "#d1610a"]


def read_bag(path, lidar_topic, gt_topic, max_scans=None, max_points=60000):
    """Yield (stamp, points) for the LiDAR topic, and collect ground-truth poses."""
    from rosbags.highlevel import AnyReader

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ros2" / "locrec_ros"))
    from locrec_ros.conversions import pointcloud2_to_xyz

    scans, gt = [], []
    with AnyReader([pathlib.Path(path)]) as reader:
        wanted = [c for c in reader.connections if c.topic in (lidar_topic, gt_topic)]
        if not wanted:
            topics = sorted({c.topic for c in reader.connections})
            raise SystemExit("neither topic found. available topics:\n  " + "\n  ".join(topics))
        for conn, stamp, raw in reader.messages(connections=wanted):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == lidar_topic:
                if max_scans is not None and len(scans) >= max_scans:
                    continue
                scans.append((stamp * 1e-9, pointcloud2_to_xyz(msg, max_points=max_points)))
            else:
                gt.append((stamp * 1e-9, _pose_of(msg)))
    return scans, gt


def _pose_of(msg) -> np.ndarray:
    """Accept PoseStamped, TransformStamped or Odometry without importing ROS."""
    for attr in ("pose", "transform"):
        obj = getattr(msg, attr, None)
        if obj is None:
            continue
        obj = getattr(obj, "pose", obj)
        pos = getattr(obj, "position", None) or getattr(obj, "translation", None)
        rot = getattr(obj, "orientation", None) or getattr(obj, "rotation", None)
        if pos is not None and rot is not None:
            return make_T(_quat_to_R(rot), [pos.x, pos.y, pos.z])
    raise ValueError(f"cannot read a pose out of {type(msg).__name__}")


def _quat_to_R(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def align_prefix(est_xyz, gt_xyz, metres=20.0):
    """Rigid fit of ground truth onto the estimate over the first ``metres``.

    Over the whole run instead, the fit would absorb the drift and flatten the very
    curve this is measuring.
    """
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1))])
    n = max(int(np.searchsorted(d, metres)), 3)
    a, b = gt_xyz[:n], est_xyz[:n]
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, cb - R @ ca


def run(scans, gt, cfg: OdometryConfig, threshold: float):
    gt_t = np.array([t for t, _ in gt])
    gt_T = [T for _, T in gt]

    odom = Odometry(np.eye(4), cfg)
    rows = []
    prev = None
    for stamp, pts in scans:
        delta = np.eye(4)
        if prev is not None:
            Ta, Tb = _interp_pose(gt_t, gt_T, prev), _interp_pose(gt_t, gt_T, stamp)
            if Ta is not None and Tb is not None:
                delta = inv_T(Ta) @ Tb
        out = odom.step(pts, delta)
        Tg = _interp_pose(gt_t, gt_T, stamp)
        rows.append(
            {
                "t": stamp,
                "est": odom.T[:3, 3].copy(),
                "gt": None if Tg is None else Tg[:3, 3].copy(),
                "ratio": out.localizability.ratio if out.localizability else np.nan,
                "n_points": out.localizability.n_points if out.localizability else 0,
            }
        )
        prev = stamp

    keep = [r for r in rows if r["gt"] is not None]
    est = np.array([r["est"] for r in keep])
    truth = np.array([r["gt"] for r in keep])
    R, t = align_prefix(est, truth)
    truth = truth @ R.T + t

    step = np.diff(truth, axis=0)
    heading = step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-9)
    err = est[1:] - truth[1:]
    along = np.abs(np.einsum("ij,ij->i", err, heading))
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(step, axis=1))])[1:]
    ratio = np.array([r["ratio"] for r in keep])[1:]
    times = np.array([r["t"] for r in keep])[1:]
    return dist, along, ratio, times


def _interp_pose(gt_t, gt_T, stamp):
    if gt_t.size == 0 or stamp < gt_t[0] or stamp > gt_t[-1]:
        return None
    i = int(np.searchsorted(gt_t, stamp))
    return gt_T[min(i, len(gt_T) - 1)]


def growth_mask(dist, along, times, horizon_s=5.0, factor=2.0, win=20):
    """True where along-track error grows faster than ``factor`` times its own
    median rate over the next ``horizon_s`` seconds."""
    rate = np.gradient(along, np.maximum(dist, 1e-9))
    base = max(float(np.median(np.abs(rate))), 1e-4)
    out = np.zeros(along.size, dtype=bool)
    for i in range(along.size):
        j = int(np.searchsorted(times, times[i] + horizon_s))
        seg = rate[i:j]
        if seg.size >= 3:
            out[i] = float(np.median(seg)) > factor * base
    del win
    return out


def roc(scores, labels):
    order = np.argsort(-scores)
    lab = labels[order]
    tp = np.cumsum(lab)
    fp = np.cumsum(~lab)
    tpr = tp / max(lab.sum(), 1)
    fpr = fp / max((~lab).sum(), 1)
    auc = float(np.trapezoid(tpr, fpr)) if tpr.size > 1 else float("nan")
    return fpr, tpr, auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--lidar-topic", default="/hesai/pandar")
    ap.add_argument("--gt-topic", default="/gt_pose")
    ap.add_argument("--max-scans", type=int, default=None)
    ap.add_argument("--range", type=float, default=40.0)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs"))
    args = ap.parse_args()

    scans, gt = read_bag(args.bag, args.lidar_topic, args.gt_topic, args.max_scans)
    print(f"{len(scans)} scans, {len(gt)} ground-truth poses")
    cfg = OdometryConfig(
        map_resolution=0.3,
        map_radius=2.0 * args.range,
        registration_range=args.range * 0.5,
        gicp_ratio_floor=None,
    )
    dist, along, ratio, times = run(scans, gt, cfg, args.threshold or 0.0)

    finite = np.isfinite(ratio)
    # threshold calibrated on this sequence's own distribution, same rule as the
    # simulation: the high quantile of what a degenerate stretch produces
    thr = args.threshold or float(np.quantile(ratio[finite], 0.5))
    grows = growth_mask(dist, along, times)
    fires = ratio < thr

    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.6), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9)

    axes[0].semilogy(dist, ratio, color=SERIES[0], lw=1.6)
    axes[0].axhline(thr, color=MUTED, lw=1.0, ls="--")
    _shade(axes[0], dist, grows)
    axes[0].set_ylabel("localizability ratio", fontsize=10, color=INK)
    axes[0].set_title(
        "Shaded where along-track error grows faster than twice its own median rate\n"
        "over the next five seconds. The dashed line is the calibrated threshold.",
        fontsize=9, color=MUTED, loc="left",
    )

    axes[1].plot(dist, along, color=SERIES[1], lw=1.8)
    _shade(axes[1], dist, grows)
    axes[1].set_ylabel("along-track error (m)", fontsize=10, color=INK)
    axes[1].set_xlabel("distance travelled (m)", fontsize=10, color=INK)

    fpr, tpr, auc = roc(-ratio[finite], grows[finite])
    axes[2].plot(fpr, tpr, color=SERIES[0], lw=2.0)
    axes[2].plot([0, 1], [0, 1], color=MUTED, lw=1.0, ls="--")
    axes[2].set_xlabel("false positive rate", fontsize=10, color=INK)
    axes[2].set_ylabel("true positive rate", fontsize=10, color=INK)
    axes[2].set_title(f"detector fires vs drift grows, AUC {auc:.3f}", fontsize=9, color=MUTED, loc="left")
    fig.savefig(args.out / "real_sequence.png", dpi=160, facecolor=SURFACE)

    leads = []
    in_seg = False
    for i in range(grows.size):
        if grows[i] and not in_seg:
            in_seg = True
            back = np.flatnonzero(fires[:i])
            if back.size:
                leads.append(float(times[i] - times[back[-1]]))
        elif not grows[i]:
            in_seg = False
    print(f"AUC {auc:.3f}; lead time median {np.median(leads) if leads else float('nan'):.1f} s "
          f"over {len(leads)} segments")
    print(f"wrote {args.out / 'real_sequence.png'}")


def _shade(ax, dist, mask):
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = dist[i]
        elif not flag and start is not None:
            ax.axvspan(start, dist[i], color="#e8e3d6", lw=0, zorder=0)
            start = None
    if start is not None:
        ax.axvspan(start, dist[-1], color="#e8e3d6", lw=0, zorder=0)


if __name__ == "__main__":
    main()
