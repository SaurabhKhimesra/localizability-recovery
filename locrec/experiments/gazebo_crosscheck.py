"""The study's two comparisons again, with every scan from Gazebo's GPU LiDAR.

    python experiments/gazebo_crosscheck.py --platform ugv   --seeds 8 --jobs 2
    python experiments/gazebo_crosscheck.py --platform drone --seeds 8 --jobs 2

* ``ugv``: no markers against the localizability scheduler, in the mixed world.
* ``drone``: forward gaze against glance gaze, in the junction world.

``--correct-strip-bias off`` runs the same comparison with ``OdometryConfig.correct_strip_bias``
off. Declared before it was first run: the strip fit's expected bias is measured by running the
real fit on synthetic returns from the known strip box, which is a ray-cast, and it was validated
on MuJoCo, which is also a ray-cast. Gazebo's ranges come out of a rendered depth texture. Both
sensors were then measured in situ, on the detections of a real run: at one beam column, where more
than half of all detections are, MuJoCo reads the strip +3.28 cm long with 2.67 cm of scatter and
Gazebo +0.43 cm with 4.74 cm, the same total error split differently. So the correction may be
subtracting an offset the second sensor does not have. The test is whether turning it off
recovers the marker benefit on Gazebo, on the same declared statistics as everything else here;
it is kept for a sensor only if it does not make that sensor's marker result worse.

``--measure-strip-spread off`` does the same for the other half of that model. On, a marker
observation's covariance is the measured spread of the real fit at that geometry
(``landmarks.strip_fit_sigma``); off, it is the hand-written term it replaced, beam quantisation
plus the strip's width over root twelve. Declared before this arm was first run: unlike the mean,
the spread was checked against this sensor before being trusted, the model predicting 4.66 cm of
scatter for a single-column detection at 4.4 m where Gazebo measures 4.74 cm, so it is expected to
hold here; the point of running it is that "expected to hold" is not a measurement of the end-to-end
result, and the flag was only ever A/B'd on MuJoCo. Same criterion: kept only if it does not make
the reference sensor's marker result worse.

Each pass is ``run_pass`` itself on a ``locrec.gazebo.GazeboTunnelSim``, with the thresholds
``experiments/calibrate_thresholds_gazebo.py`` measured on that sensor and the policies built
exactly as ``experiments/marker_drop.py`` and ``experiments/drone_gaze.py`` build them. Every
pass is its own process with its own Gazebo server on its own transport partition, so passes
can run side by side without hearing each other. Not a multiprocessing pool: a pool's workers
are daemonic, and a daemonic process may not start the child that receives LiDAR frames.

Declared before any pass was run, because a demonstration run is chosen from these rows:
the demonstration seed is the one whose paired difference in mean absolute along-track error
over the run (the number the demonstration video reports) is closest to the median of the
paired differences, ties to the lower seed. It is not the seed that looks best.

Along-track here is ``run_pass``'s: the error projected on the sensor's x axis. For a drone
that is the direction it looks, which a glancing drone turns away from the tunnel for a few
seconds at a time; the viewer projects on the track heading instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

OVERSAMPLE = {"ugv": 6, "drone": 3}
WORLD = {"ugv": "mixed", "drone": "junction"}
POLICIES = {"ugv": ("none", "scheduler"), "drone": ("forward", "glance")}


def _world_spec(platform: str, length: float):
    from calibrate_thresholds import mixed_world
    from drone_gaze import junction_world

    return mixed_world(length) if platform == "ugv" else junction_world(length)


def _policy(platform: str, name: str, th: dict):
    from locrec import LocalizabilityScheduler, NoMarkers
    from locrec.gaze import ForwardGaze, GlanceGaze

    if platform == "ugv":
        if name == "none":
            return NoMarkers()
        return LocalizabilityScheduler(ratio_threshold=th["ratio_threshold"],
                                       reliable_range_m=th["marker_reliable_range_m"],
                                       margin_m=th["scheduler_margin_m"])
    if name == "forward":
        return ForwardGaze()
    return GlanceGaze(ratio_threshold=th["drone_ratio_threshold"], cone_deg=60.0)


def one_pass(job) -> dict:
    platform, policy_name, seed, length, thresholds_path, work, correct_bias, spread = job
    # a partition of its own, so parallel servers and links never cross
    os.environ["GZ_PARTITION"] = f"locrec_crosscheck_{os.getpid()}"
    from locrec import LIMITED_FOV, SPINNING_360, RunConfig, TunnelSim, run_pass
    from locrec import gazebo as gzl
    from locrec.odometry import OdometryConfig

    th = json.loads(pathlib.Path(thresholds_path).read_text())
    spec = _world_spec(platform, length)
    lidar = SPINNING_360 if platform == "ugv" else LIMITED_FOV
    floor = th["gicp_ratio_floor"] if platform == "ugv" else th["drone_gicp_ratio_floor"]
    cfg = RunConfig(seed=seed, platform=platform, world=spec, lidar=lidar,
                    odometry=OdometryConfig(num_threads=1, gicp_ratio_floor=floor,
                                            correct_strip_bias=bool(correct_bias),
                                            measure_strip_spread=bool(spread)))
    n_steps = gzl.steps_per_scan(cfg.platform_spec)
    world = TunnelSim(seed, spec, lidar).world
    run_dir = pathlib.Path(work) / f"{platform}_{policy_name}_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sdf = gzl.world_sdf([(world, "tunnel", (0.0, 0.0, 0.0))],
                        [gzl.vehicle_model("vehicle", platform, lidar, "/vehicle/lidar",
                                           1.0 / (n_steps * gzl.STEP_SIZE), oversample=OVERSAMPLE[platform])],
                        mesh_dir=run_dir / "meshes")
    path = gzl.write_world(run_dir / "world.sdf", sdf)
    server = gzl.GazeboServer(path, "tunnel", log_path=run_dir / "server.log", verbosity=1, seed=seed).start()
    link = None
    t0 = time.time()
    try:
        link = gzl.GazeboLink("tunnel", {"/vehicle/lidar/points": OVERSAMPLE[platform]})
        link.wait_ready(120.0, server)
        sim = gzl.GazeboTunnelSim(seed, spec, lidar, link, "vehicle", "/vehicle/lidar/points", n_steps)
        link.set_poses({"vehicle": sim.gazebo_pose(np.eye(4))})
        link.prime(n_steps)
        policy = _policy(platform, policy_name, th)
        r = run_pass(cfg, policy, sim=sim)
    finally:
        if link is not None:
            link.close()
        server.stop()
    along = np.abs(r.array("along_err_m"))
    ratio = r.array("loc_ratio")
    t = th["ratio_threshold"] if platform == "ugv" else th["drone_ratio_threshold"]
    return {
        "platform": platform, "policy": policy_name, "seed": seed, "world": WORLD[platform],
        "path_length_m": r.path_length, "steps": len(r.rows),
        "mean_abs_along_m": float(np.mean(along)), "max_abs_along_m": float(np.max(along)),
        "final_along_m": r.final_along_track_error, "final_lateral_m": r.final_lateral_error,
        "final_total_m": r.final_translation_error,
        "markers": r.markers_placed, "yaw_effort_rad": r.yaw_effort,
        "blind_fraction": float(np.mean(ratio[np.isfinite(ratio)] < t)),
        "glance_fraction": getattr(policy, "glance_fraction", float("nan")),
        "seconds": time.time() - t0,
    }


def paired(rows: list[dict], platform: str, key: str) -> dict[int, float]:
    """Per seed, the first policy's value minus the second's: positive means the second is better."""
    a, b = POLICIES[platform]
    by = {(r["policy"], r["seed"]): r[key] for r in rows}
    return {s: by[(a, s)] - by[(b, s)] for s in sorted({r["seed"] for r in rows}) if (a, s) in by and (b, s) in by}


def demonstration_seed(diffs: dict[int, float]) -> int:
    """The declared rule: closest to the median paired difference, ties to the lower seed.

    With an even number of seeds the median is the midpoint of the two middle values, so
    those two seeds are always exactly as close as each other and the tie rule decides. The
    first version compared the raw floats and chose the higher seed on a difference in the
    17th digit, so distances are compared to a nanometre.
    """
    med = float(np.median(list(diffs.values())))
    return min(diffs, key=lambda s: (round(abs(diffs[s] - med), 9), s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", help=argparse.SUPPRESS)  # internal: run one pass, a JSON job, write a JSON row
    ap.add_argument("--platform", choices=("ugv", "drone"))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--length", type=float, default=300.0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--thresholds", default=str(ROOT / "results" / "thresholds_gazebo.json"))
    ap.add_argument("--result")
    ap.add_argument("--summarise", help="print the summary of an existing results CSV and stop")
    ap.add_argument("--correct-strip-bias", choices=("on", "off"), default="off",
                    help="OdometryConfig.correct_strip_bias for every pass")
    ap.add_argument("--measure-strip-spread", choices=("on", "off"), default="on",
                    help="OdometryConfig.measure_strip_spread for every pass")
    args = ap.parse_args()
    if args.summarise:
        summarise_csv(args.summarise)
        return
    if args.one:
        row = one_pass(tuple(json.loads(args.one)))
        pathlib.Path(args.result).write_text(json.dumps(row))
        return
    if args.platform is None:
        ap.error("--platform is required")

    work = tempfile.mkdtemp(prefix="locrec_crosscheck_")
    correct_bias = args.correct_strip_bias == "on"
    spread = args.measure_strip_spread == "on"
    jobs = [(args.platform, p, s, args.length, args.thresholds, work, correct_bias, spread)
            for s in range(args.seeds) for p in POLICIES[args.platform]]
    print(f"{len(jobs)} passes on Gazebo, {args.jobs} at a time; logs in {work}", flush=True)
    rows = []
    pending = list(jobs)
    running: list[tuple[subprocess.Popen, pathlib.Path, tuple]] = []
    while pending or running:
        while pending and len(running) < args.jobs:
            job = pending.pop(0)
            result = pathlib.Path(work) / f"{job[0]}_{job[1]}_{job[2]}.json"
            log = open(pathlib.Path(work) / f"{job[0]}_{job[1]}_{job[2]}.log", "w")
            proc = subprocess.Popen([sys.executable, __file__, "--one", json.dumps(job), "--result", str(result)],
                                    stdout=log, stderr=subprocess.STDOUT)
            running.append((proc, result, job))
        time.sleep(1.0)
        for item in list(running):
            proc, result, job = item
            if proc.poll() is None:
                continue
            running.remove(item)
            if proc.returncode != 0 or not result.is_file():
                raise RuntimeError(f"pass {job[:3]} failed with exit code {proc.returncode}; log in {work}")
            row = json.loads(result.read_text())
            rows.append(row)
            print(f"  {row['policy']:>9} seed {row['seed']}: mean |along| {row['mean_abs_along_m']:.2f} m, "
                  f"final total {row['final_total_m']:.2f} m, markers {row['markers']}, "
                  f"{row['seconds']:.0f} s", flush=True)
    rows.sort(key=lambda r: (r["seed"], r["policy"]))
    out = ROOT / "results" / f"gazebo_crosscheck_{args.platform}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summarise(rows, args.platform)
    print(f"wrote {out}")


def summarise(rows: list[dict], platform: str) -> None:
    """Medians and paired medians with the study's bootstrap interval, as the README's tables
    report them, and the demonstration seed by the declared rule."""
    from marker_drop import bootstrap_ci

    a, b = POLICIES[platform]
    for key in ("final_along_m", "final_total_m", "mean_abs_along_m"):
        va = [r[key] for r in rows if r["policy"] == a]
        vb = [r[key] for r in rows if r["policy"] == b]
        d = paired(rows, platform, key)
        med, lo, hi = bootstrap_ci(list(d.values()))
        better = sum(1 for v in d.values() if v > 0)
        print(f"{key}: median {a} {np.median(va):.2f} m, {b} {np.median(vb):.2f} m "
              f"(means {np.mean(va):.2f} and {np.mean(vb):.2f}); paired {a} - {b} median {med:+.2f} m "
              f"[{lo:+.2f}, {hi:+.2f}], {b} better in {better} of {len(d)} seeds")
    diffs = paired(rows, platform, "mean_abs_along_m")
    print(f"demonstration seed by the declared rule: {demonstration_seed(diffs)} "
          f"(paired differences {', '.join(f'{s}: {v:+.2f}' for s, v in diffs.items())})")


def summarise_csv(path: str) -> None:
    with open(path) as fh:
        raw = list(csv.DictReader(fh))
    rows = []
    for r in raw:
        row = dict(r)
        row["seed"] = int(r["seed"])
        for k in ("mean_abs_along_m", "final_along_m", "final_total_m"):
            row[k] = float(r[k])
        rows.append(row)
    summarise(rows, rows[0]["platform"])


if __name__ == "__main__":
    main()
