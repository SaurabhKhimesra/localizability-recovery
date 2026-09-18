"""Milestone 7: offscreen MuJoCo render, chase camera plus a top-down inset.

    MUJOCO_GL=egl python experiments/render_video.py --platform ugv --out docs/

Falls back to osmesa if EGL is unavailable, which is the usual situation on a
headless box without a GPU. The inset is drawn with matplotlib and composited over
the rendered frame, because it is showing estimator state rather than geometry and
MuJoCo has no way to draw it.

Thirty seconds per platform at 30 fps, plus a ten second GIF of each for the
README.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

ROOT = pathlib.Path(__file__).resolve().parents[1]
INK = "#1c1c1a"
TRUE_COLOR = "#6b6b66"
EST_COLOR = "#1f6feb"
MARK_COLOR = "#2f8f5b"


def setup_gl() -> str:
    """Pick a backend before mujoco is imported. EGL first, osmesa as the fallback."""
    if "MUJOCO_GL" in os.environ:
        return os.environ["MUJOCO_GL"]
    for backend in ("egl", "osmesa"):
        os.environ["MUJOCO_GL"] = backend
        try:
            import mujoco  # noqa: F401

            mujoco.MjModel.from_xml_string("<mujoco/>")
            return backend
        except Exception:  # noqa: BLE001  any backend failure means try the next
            for mod in [m for m in list(sys.modules) if m.startswith("mujoco")]:
                del sys.modules[mod]
    raise SystemExit("no usable MuJoCo GL backend: set MUJOCO_GL=egl or osmesa")


def chase_camera(model, T_true, distance=10.0, elevation_deg=-20.0):
    """Behind and above the platform, looking down the tunnel.

    The world is a ribbed shell with an open top and no lights, so a camera placed
    inside it sees an unlit tube: the only thing the default headlight picks out is
    the far opening. Looking in from above the shell instead shows the floor, the
    ribs and the marker strips, which is what the video is for. Settings were
    picked by rendering a probe frame at five of them, not by guessing.
    """
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    yaw = float(np.arctan2(T_true[1, 0], T_true[0, 0]))
    cam.lookat[:] = T_true[:3, 3] + np.array([0.0, 0.0, 0.5])
    cam.distance = distance
    cam.azimuth = np.rad2deg(yaw) + 180.0
    cam.elevation = elevation_deg
    return cam


def inset(width, height, true_xy, est_xy, markers, yaw_dir, title):
    """Top-down panel as an RGB array, composited over the render."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_axes([0.12, 0.16, 0.86, 0.74])
    ax.set_facecolor("#fcfcfbee")
    fig.patch.set_facecolor("#fcfcfbcc")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=6, colors="#6b6b66", length=2)
    ax.plot(true_xy[:, 0], true_xy[:, 1], color=TRUE_COLOR, lw=1.6, label="true")
    ax.plot(est_xy[:, 0], est_xy[:, 1], color=EST_COLOR, lw=1.6, ls="--", label="estimate")
    if len(markers):
        m = np.asarray(markers)
        ax.scatter(m[:, 0], m[:, 1], s=14, color=MARK_COLOR, zorder=3, label="markers")
    if yaw_dir is not None and len(true_xy):
        p = true_xy[-1]
        ax.arrow(
            p[0], p[1], yaw_dir[0] * 4.0, yaw_dir[1] * 4.0, color="#d1610a",
            width=0.25, head_width=1.1, length_includes_head=True, zorder=4,
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, fontsize=6, labelcolor=INK, loc="upper left")
    ax.set_title(title, fontsize=7, color=INK, loc="left")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def render(platform: str, seed: int, seconds: float, fps: int, out_dir: pathlib.Path):
    backend = setup_gl()
    import mujoco

    from locrec import (
        LIMITED_FOV,
        SPINNING_360,
        LocalizabilityScheduler,
        RunConfig,
        TunnelSim,
        UGV,
        Drone,
    )
    from locrec.gaze import GreedyGaze
    from locrec.odometry import MotionPrior, Odometry, OdometryConfig
    from locrec.runner import StepContext, compute_normals
    from locrec.se3 import inv_T
    import json

    from calibrate_thresholds import mixed_world

    th = json.loads((ROOT / "results" / "thresholds.json").read_text())
    world_spec = mixed_world(160.0)
    lidar = SPINNING_360 if platform == "ugv" else LIMITED_FOV
    guard = th["gicp_ratio_floor"] if platform == "ugv" else th["drone_gicp_ratio_floor"]
    cfg = RunConfig(
        seed=seed,
        platform=platform,
        world=world_spec,
        lidar=lidar,
        odometry=OdometryConfig(gicp_ratio_floor=guard),
    )
    sim = TunnelSim(seed, world_spec, lidar)
    plat = UGV(sim.world, cfg.platform_spec) if platform == "ugv" else Drone(sim.world, cfg.platform_spec)
    policy = (
        LocalizabilityScheduler(
            ratio_threshold=th["ratio_threshold"],
            reliable_range_m=th["marker_reliable_range_m"],
            margin_m=th["scheduler_margin_m"],
        )
        if platform == "ugv"
        else GreedyGaze(ratio_threshold=th["drone_ratio_threshold"])
    )
    dt = cfg.platform_spec.step_length / cfg.platform_spec.speed_mps
    prior = MotionPrior(cfg.prior, seed=seed, dt=dt)
    odom = Odometry(
        plat.pose(),
        OdometryConfig(
            gicp_ratio_floor=guard,
            map_radius=2.0 * lidar.max_range,
            registration_range=min(
                lidar.max_range - 1.5, lidar.nyquist_range(OdometryConfig().map_resolution)
            ),
        ),
        prior_spec=cfg.prior,
        dt=dt,
        lidar_spec=lidar,
    )

    # A tunnel has no sky and the world carries no lights, so the only thing that
    # illuminates the interior is the camera's own headlight. Left at its default
    # the render is a dark tube with a bright rim.
    sim.model.vis.headlight.active = 1
    sim.model.vis.headlight.ambient[:] = 0.45
    sim.model.vis.headlight.diffuse[:] = 0.85
    sim.model.vis.headlight.specular[:] = 0.1

    renderer = mujoco.Renderer(sim.model, height=720, width=1280)
    frames = []
    true_xy, est_xy = [], []
    T_true = plat.pose()
    scan = sim.scan(T_true)
    det = sim.marker_detections(scan)
    step_out = odom.step(scan.points, np.eye(4), det)
    n_frames = int(seconds * fps)

    for k in range(n_frames):
        if plat.finished:
            break
        if policy is not None:
            ctx = StepContext(
                step=k, sim=sim, platform=plat, T_est=odom.T.copy(),
                localizability=step_out.localizability, detections=det,
                registration=step_out, markers_placed=sim.n_markers_placed,
                markers_remaining=sim.marker_capacity - sim.n_markers_placed,
                estimated_distance=float(np.sum(np.linalg.norm(np.diff(np.array(est_xy or [[0, 0]]), axis=0), axis=1))) if len(est_xy) > 1 else 0.0,
                local_map_points=odom.map.points if getattr(policy, "needs_map", False) else None,
                local_map_normals=compute_normals(odom.map.points) if getattr(policy, "needs_map", False) else None,
                path_length=plat.s,
            )
            action = policy(ctx) or {}
            if "yaw_target" in action and isinstance(plat, Drone):
                plat.command_yaw(float(action["yaw_target"]))
            if action.get("drop_marker") and sim.n_markers_placed < sim.marker_capacity:
                slot, off = sim.drop_marker_on_wall(T_true)
                odom.register_landmark(slot, off)
        elif isinstance(plat, Drone):
            plat.track_heading()

        T_prev = T_true
        plat.advance()
        T_true = plat.pose()
        scan = sim.scan(T_true)
        det = sim.marker_detections(scan)
        step_out = odom.step(scan.points, prior.predict(inv_T(T_prev) @ T_true), det)

        true_xy.append(T_true[:2, 3].copy())
        est_xy.append(odom.T[:2, 3].copy())

        renderer.update_scene(sim.data, camera=chase_camera(sim.model, T_true))
        frame = renderer.render().copy()
        yaw_dir = None
        if isinstance(plat, Drone):
            yaw_dir = np.array([np.cos(plat.yaw), np.sin(plat.yaw)])
        panel = inset(
            420, 300, np.array(true_xy), np.array(est_xy), sim.marker_positions()[:, :2]
            if sim.n_markers_placed else [], yaw_dir,
            f"{platform}  s={plat.s:5.1f} m   markers={sim.n_markers_placed}",
        )
        frame[10 : 10 + panel.shape[0], 10 : 10 + panel.shape[1]] = panel
        frames.append(frame)

    out_dir.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    mp4 = out_dir / f"{platform}.mp4"
    imageio.mimwrite(mp4, frames, fps=fps, quality=7)
    gif = out_dir / f"{platform}.gif"
    stride = max(len(frames) // (10 * 12), 1)
    imageio.mimwrite(gif, [f[::2, ::2] for f in frames[::stride]], fps=12, loop=0)
    print(f"{platform}: {len(frames)} frames via {backend}, wrote {mp4} and {gif}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="all", choices=["all", "ugv", "drone"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "docs")
    args = ap.parse_args()

    platforms = ["ugv", "drone"] if args.platform == "all" else [args.platform]
    for p in platforms:
        render(p, args.seed, args.seconds, args.fps, args.out)


if __name__ == "__main__":
    main()
