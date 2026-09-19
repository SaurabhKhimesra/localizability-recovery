"""Calibrate the thresholds again, on Gazebo's LiDAR, by the procedure calibrate_thresholds.py declares.

Nothing about the procedure is restated here. The quantiles, seeds, orientations, tunnel
length, marker rule and scheduler margin are imported from that script, so this one
cannot drift from it. Only the source of the scans changes: Gazebo's GPU ray sensor
(``locrec.gazebo``) instead of MuJoCo's ray caster. The output goes to its own file,
``results/thresholds_gazebo.json``, because a threshold does not transfer between sensors
(``docs/failures.md`` number 20), and two simulators are two sensors.

Running 760 passes through a live Gazebo server would take hours. Three facts, each
measured before it was relied on, reduce the Gazebo part to one pass per platform:

* ``blind_world`` is the same tunnel for every seed. It has no curves, junctions or
  niches and a constant width, so the seed only changes the sensor noise and the motion
  prior's noise, never the geometry.
* Rotating the tunnel rotates the sensor with it, so the sensor-frame scan does not
  change. The orientation only matters to the estimator's world-aligned voxel grids.
  Checked here on every run: a copy rotated by ``ROTATION_CHECK_DEG`` is scanned at the
  same arclengths and compared beam by beam.
* Gazebo's range noise is additive, zero mean and Gaussian with the configured sigma
  (measured: 2.01 cm against the 2 cm configured, mean +0.006 cm, noiseless frames
  identical to the bit). So one noiseless pass is captured per platform, and each
  seed's noise is drawn from that seed's generator, as ``locrec.lidar.Lidar`` draws it.

Each (orientation, seed, platform) run is then ``run_pass`` itself on those scans, with
the guard off, pooled exactly as in ``calibrate_thresholds.py``.

    python experiments/calibrate_thresholds_gazebo.py --workers 6
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np

from locrec import LIMITED_FOV, SPINNING_360, NoMarkers, RunConfig, TunnelSim, run_pass
from locrec import gazebo as gzl
from locrec.lidar import LidarSpec, Scan
from locrec.localizability import calibrate_threshold
from locrec.odometry import OdometryConfig
from locrec.se3 import euler_zyx, make_T
from locrec.sim import Drone, UGV

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import calibrate_thresholds as base  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OVERSAMPLE = {"ugv": 6, "drone": 3}
"""Horizontal oversampling of each Gazebo sensor, the smallest that reaches Gazebo's largest
depth texture (see ``locrec.gazebo.lidar_sensor``). Measured on noiseless scans against
MuJoCo, per beam: the UGV's 95th percentile range error falls from 13.6 cm to 1.6 cm and
the drone's from 10.2 cm to 2.5 cm."""
LIDARS = {"ugv": SPINNING_360, "drone": LIMITED_FOV}
ROTATION_CHECK_DEG = 45.0
ROTATION_CHECK_POSES = 20
STACK_HEIGHT_M = 10.0
"""The rotated copy sits this far above the original. A tunnel shifted straight up cannot
intersect itself, and both are closed, so neither sensor can see the other copy."""


def _pose(platform: str, world, s: float) -> np.ndarray:
    """The pose ``run_pass`` puts the platform in at arclength ``s``, with no policy acting."""
    plat = UGV(world) if platform == "ugv" else Drone(world)
    plat.s = s
    if platform == "drone":
        # a drone with no gaze policy keeps its initial yaw; in a straight tunnel that is
        # the heading at every arclength
        return make_T(euler_zyx(plat.yaw), plat.world.pose_at(s, plat.spec.sensor_height)[0])
    return plat.pose()


def capture(length: float, cache: pathlib.Path, log_dir: pathlib.Path) -> dict:
    """Noiseless Gazebo scans at every arclength of a blind pass, one set per platform."""
    n_steps = gzl.steps_per_scan()
    rate = 1.0 / (n_steps * gzl.STEP_SIZE)
    world0 = TunnelSim(base.CALIBRATION_SEED_BASE, base.blind_world(length), SPINNING_360).world
    world_rot = TunnelSim(base.CALIBRATION_SEED_BASE, base.blind_world(length, ROTATION_CHECK_DEG), SPINNING_360).world
    clean = {p: dataclasses.replace(LIDARS[p], range_sigma=0.0) for p in LIDARS}
    lift = np.array([0.0, 0.0, STACK_HEIGHT_M])
    vehicles = [gzl.vehicle_model(p, p, clean[p], f"/{p}/lidar", rate, oversample=OVERSAMPLE[p]) for p in LIDARS]
    sdf = gzl.world_sdf([(world0, "tunnel", (0.0, 0.0, 0.0)), (world_rot, "tunnel_rotated", tuple(lift))], vehicles,
                        mesh_dir=log_dir / "meshes")
    path = gzl.write_world(log_dir / "calibration_world.sdf", sdf)
    server = gzl.GazeboServer(path, "tunnel", log_path=log_dir / "calibration_server.log", verbosity=2).start()
    out: dict = {}
    try:
        link = gzl.GazeboLink("tunnel", {f"/{p}/lidar/points": OVERSAMPLE[p] for p in LIDARS})
        link.wait_ready(120, server)
        link.set_poses({p: _pose(p, world0, 0.0) for p in LIDARS})
        link.prime(n_steps)
        arclengths = np.arange(0.0, world0.total_length + 1e-9, 0.5)
        frames = {p: [] for p in LIDARS}
        direction_error = {p: 0.0 for p in LIDARS}
        t0 = time.time()
        for i, s in enumerate(arclengths):
            link.set_poses({p: _pose(p, world0, s) for p in LIDARS})
            link.step(n_steps)
            for p in LIDARS:
                f = link.wait_frame(f"/{p}/lidar/points", link.sim_time)
                r = np.linalg.norm(f.xyz, axis=1)
                frames[p].append(r.astype(np.float32))
                # the replay puts each range on the beam direction locrec would cast it
                # along, so the cloud's beam order has to be that order: checked, not assumed
                ok = np.isfinite(r)
                dots = np.einsum("ij,ij->i", f.xyz[ok] / r[ok, None], _beam_directions(LIDARS[p])[ok])
                direction_error[p] = max(direction_error[p], float(np.arccos(np.clip(dots.min(), -1.0, 1.0))))
            if i % 100 == 0:
                print(f"  captured {i + 1}/{len(arclengths)} poses, {time.time() - t0:.0f} s", flush=True)

        # the rotation check: the same arclengths in the rotated copy
        check = {}
        picks = np.linspace(0, len(arclengths) - 1, ROTATION_CHECK_POSES).round().astype(int)
        for p in LIDARS:
            diffs = []
            for k in picks:
                T = _pose(p, world_rot, arclengths[k])
                T[:3, 3] += lift
                link.set_poses({p: T})
                link.step(n_steps)
                f = link.wait_frame(f"/{p}/lidar/points", link.sim_time)
                r_rot = np.linalg.norm(f.xyz, axis=1)
                r0 = frames[p][k]
                both = np.isfinite(r_rot) & np.isfinite(r0)
                diffs.append(np.abs(r_rot[both] - r0[both]))
                if (np.isfinite(r_rot) != np.isfinite(r0)).any():
                    diffs.append(np.full(int((np.isfinite(r_rot) != np.isfinite(r0)).sum()), np.inf))
            d = np.concatenate(diffs)
            check[p] = {"poses": int(len(picks)), "beams": int(d.size),
                        "p99_abs_m": float(np.percentile(d[np.isfinite(d)], 99)),
                        "max_abs_m": float(d[np.isfinite(d)].max()),
                        "hit_mismatches": int((~np.isfinite(d)).sum())}
            print(f"  rotation check {p}: {check[p]}", flush=True)

        for p in LIDARS:
            if direction_error[p] > 1e-3:
                raise RuntimeError(f"{p}: Gazebo's beams are not in locrec's order "
                                   f"(worst direction error {direction_error[p]:.4f} rad)")
            ranges = np.stack(frames[p])
            dirs = _beam_directions(LIDARS[p])
            np.save(cache / f"gazebo_blind_{p}_ranges.npy", ranges)
            np.save(cache / f"gazebo_blind_{p}_dirs.npy", dirs)
            out[p] = {"poses": int(ranges.shape[0]), "beams": int(ranges.shape[1]),
                      "worst_beam_direction_error_rad": direction_error[p]}
        out["rotation_check"] = check
        out["arclength_step_m"] = 0.5
        link.close()
    finally:
        server.stop()
    return out


def _beam_directions(spec: LidarSpec) -> np.ndarray:
    """Beam directions in the order Gazebo's cloud lists them: rings from the lowest up,
    and within a ring horizontal angles from the minimum up."""
    from locrec.lidar import _ray_directions

    d = _ray_directions(spec).reshape(spec.n_azimuth, spec.n_elevation, 3)
    return np.ascontiguousarray(d.transpose(1, 0, 2).reshape(-1, 3))


class ReplaySim(TunnelSim):
    """A blind-tunnel ``TunnelSim`` whose scans are the captured Gazebo ranges plus noise."""

    def __init__(self, seed, world_spec, lidar_spec, ranges: np.ndarray, dirs: np.ndarray, step: float):
        super().__init__(seed, world_spec, lidar_spec)
        self.ranges, self.dirs, self.step = ranges, dirs, step
        self.rng = np.random.default_rng(seed)

    def scan(self, T_world_sensor: np.ndarray) -> Scan:
        s = float(np.hypot(T_world_sensor[0, 3], T_world_sensor[1, 3]))
        k = int(round(s / self.step))
        r = self.ranges[k].astype(float)
        spec = self.lidar.spec
        hit = np.isfinite(r) & (r >= spec.min_range) & (r <= spec.max_range)
        rv = r[hit]
        if spec.range_sigma > 0.0:
            rv = rv + self.rng.normal(0.0, spec.range_sigma, rv.shape)
        dirs = self.dirs[hit]
        return Scan(points=dirs * rv[:, None], ranges=rv, geom_ids=np.zeros(0, dtype=np.int32),
                    directions=dirs, n_cast=spec.n_rays)


_CACHE: dict = {}


def _one(job):
    length, angle, seed, platform, cache = job
    key = (cache, platform)
    if key not in _CACHE:
        _CACHE[key] = (np.load(pathlib.Path(cache) / f"gazebo_blind_{platform}_ranges.npy", mmap_mode="r"),
                       np.load(pathlib.Path(cache) / f"gazebo_blind_{platform}_dirs.npy"))
    ranges, dirs = _CACHE[key]
    cfg = RunConfig(seed=seed, platform=platform, world=base.blind_world(length, angle),
                    odometry=OdometryConfig(gicp_ratio_floor=None))
    sim = ReplaySim(seed, cfg.world, cfg.lidar_spec(), ranges, dirs, 0.5)
    r = run_pass(cfg, NoMarkers(), sim=sim)
    v = r.array("loc_ratio")
    n = r.array("num_inliers")
    ok = np.isfinite(v)
    return angle, seed, platform, v[ok].tolist(), n[ok].tolist()


def marker_reliable_range_gazebo(log_dir: pathlib.Path, n_offsets: int = 9, max_range: float = 12.0,
                                 step_length: float = 0.5) -> float:
    """``calibrate_thresholds.marker_reliable_range``, with the strip seen by Gazebo's LiDAR."""
    required_rate = step_length / base.REQUIRED_FIX_INTERVAL_M
    spec = dataclasses.replace(base.blind_world(60.0), n_marker_slots=2)
    n_steps = gzl.steps_per_scan()
    probe = TunnelSim(0, spec, SPINNING_360)
    sdf = gzl.world_sdf([(probe.world, "tunnel", (0.0, 0.0, 0.0))],
                        [gzl.vehicle_model("ugv", "ugv", SPINNING_360, "/ugv/lidar",
                                           1.0 / (n_steps * gzl.STEP_SIZE), oversample=OVERSAMPLE["ugv"])],
                        mesh_dir=log_dir / "meshes")
    path = gzl.write_world(log_dir / "marker_world.sdf", sdf)
    server = gzl.GazeboServer(path, "tunnel", log_path=log_dir / "marker_server.log", verbosity=2).start()
    try:
        link = gzl.GazeboLink("tunnel", {"/ugv/lidar/points": OVERSAMPLE["ugv"]})
        link.wait_ready(120, server)
        link.prime(n_steps)
        sim = gzl.GazeboTunnelSim(0, spec, SPINNING_360, link, "ugv", "/ugv/lidar/points", n_steps)
        sim.drop_marker_on_wall(make_T(np.eye(3), [30.0, 0.0, 0.7]))
        offsets = np.linspace(0.0, 1.0, n_offsets, endpoint=False)
        best = 0.0
        for d in np.arange(1.0, max_range + 0.01, 0.5):
            hits = 0
            for o in offsets:
                T = make_T(np.eye(3), [30.0 - d - o, 0.0, 0.7])
                if sim.marker_detections(sim.scan(T)):
                    hits += 1
            rate = hits / len(offsets)
            print(f"  marker detection at {d:4.1f} m: {rate:.2f}", flush=True)
            if rate >= required_rate:
                best = float(d)
        link.close()
        return best
    finally:
        server.stop()


def gz_version() -> str:
    import subprocess

    try:
        return subprocess.run(["gz", "sim", "--versions"], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--length", type=float, default=200.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache", default=str(pathlib.Path.home() / ".cache" / "locrec" / "gazebo_calibration"))
    ap.add_argument("--reuse-capture", action="store_true", help="skip the Gazebo pass if a capture exists")
    ap.add_argument("--out", default=str(ROOT / "results" / "thresholds_gazebo.json"))
    args = ap.parse_args()
    cache = pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    capture_info_path = cache / "capture.json"
    if args.reuse_capture and capture_info_path.is_file():
        capture_info = json.loads(capture_info_path.read_text())
        print(f"reusing the capture in {cache}", flush=True)
    else:
        print("capturing noiseless Gazebo scans of the blind tunnel", flush=True)
        capture_info = capture(args.length, cache, cache)
        capture_info_path.write_text(json.dumps(capture_info, indent=2) + "\n")

    angles = base.calibration_angles()
    seeds = [base.CALIBRATION_SEED_BASE + i for i in range(args.seeds)]
    jobs = [(args.length, float(a), s, plat, str(cache)) for a in angles for s in seeds for plat in ("ugv", "drone")]
    print(f"{len(jobs)} runs: {len(angles)} angles, {len(seeds)} seeds, 2 platforms, "
          f"{args.length:.0f} m, {args.workers} workers", flush=True)
    collected = []
    t0 = time.time()
    with Pool(args.workers, maxtasksperchild=8) as pool:
        for res in pool.imap_unordered(_one, jobs):
            collected.append(res)
            n_done = len(collected)
            if n_done % 20 == 0 or n_done == len(jobs):
                el = time.time() - t0
                print(f"  {n_done}/{len(jobs)} runs, {el / 60:.1f} min elapsed, "
                      f"{el / n_done * (len(jobs) - n_done) / 60:.1f} min left", flush=True)

    collected.sort(key=lambda r: (r[0], r[1], r[2]))
    ratios = [x for r in collected if r[2] == "ugv" for x in r[3]]
    inliers = [x for r in collected if r[2] == "ugv" for x in r[4]]
    drone_ratios = [x for r in collected if r[2] == "drone" for x in r[3]]

    print("\nblind-tunnel ratio by orientation, pooled over seeds:", flush=True)
    print(f"{'angle':>7} {'n':>6} {'median':>11} {'p90 (ugv)':>11} {'median':>11} {'p90 (drone)':>12}")
    for a in angles:
        u = np.array([x for r in collected if r[0] == a and r[2] == "ugv" for x in r[3]])
        d = np.array([x for r in collected if r[0] == a and r[2] == "drone" for x in r[3]])
        print(f"{a:>6.1f}d {u.size:>6} {np.median(u):>11.4e} {np.quantile(u, 0.9):>11.4e} "
              f"{np.median(d):>11.4e} {np.quantile(d, 0.9):>12.4e}", flush=True)

    print("\nmarker reliable range on Gazebo's LiDAR:", flush=True)
    reliable = marker_reliable_range_gazebo(cache)

    out = {
        "sensor": "gazebo gpu_lidar",
        "gazebo_version": gz_version(),
        "oversample": OVERSAMPLE,
        "capture": capture_info,
        "calibration_seeds": seeds,
        "calibration_angles_deg": [round(float(a), 6) for a in angles],
        "n_angles": base.CALIBRATION_ANGLES,
        "length_m": args.length,
        "scheduler_quantile": base.SCHEDULER_QUANTILE,
        "failure_quantile": base.FAILURE_QUANTILE,
        "guard_quantile": base.GUARD_QUANTILE,
        "ratio_threshold": calibrate_threshold(np.array(ratios), base.SCHEDULER_QUANTILE),
        "gicp_ratio_floor": calibrate_threshold(np.array(ratios), base.GUARD_QUANTILE),
        "min_inliers": calibrate_threshold(np.array(inliers), base.FAILURE_QUANTILE),
        "required_fix_interval_m": base.REQUIRED_FIX_INTERVAL_M,
        "required_detection_rate": 0.5 / base.REQUIRED_FIX_INTERVAL_M,
        "marker_reliable_range_m": reliable,
        "scheduler_margin_m": base.SCHEDULER_MARGIN_M,
        "drone_ratio_threshold": calibrate_threshold(np.array(drone_ratios), base.SCHEDULER_QUANTILE),
        "drone_gicp_ratio_floor": calibrate_threshold(np.array(drone_ratios), base.GUARD_QUANTILE),
        "n_samples": len(ratios),
        "n_drone_samples": len(drone_ratios),
    }
    path = pathlib.Path(args.out)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
