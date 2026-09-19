"""The team pass again, with every scan from Gazebo's GPU LiDAR.

Same question, same declared statistic and same criterion as ``experiments/team_pass.py``, which
should be read first; this only changes the simulator. Mixed world, seeds 0 to 7, 300 m, one
thread, the thresholds ``calibrate_thresholds_gazebo.py`` measured on this sensor, and the drone
trailing 60 steps behind.

Three passes per seed, each with its own Gazebo server on its own transport partition, because
``run_pass`` needs a simulator with no markers mounted yet:

1. the ground robot with the localizability scheduler, which mounts strips and records, per strip,
   what it would tell the team: slot, the position it believes, the 2x2 that placed it, the face
   normal;
2. the drone, solo, with the strips physically on the wall;
3. the drone, team, on the same wall with the same strips, told about them.

The strips are spawned into passes 2 and 3 identically, so those two differ only in whether the
estimator was told. That is the control: the scans are the same modulo Gazebo's own seeded noise,
which is drawn per render and therefore differs between two separate servers. Two passes on two
servers are not scan for scan the same run, and this experiment does not claim they are; the
paired statistic is over seeds, not over scans.

    python experiments/team_pass_gazebo.py --seeds 8 --jobs 2
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
LAG_STEPS = 60


def _sim_for(platform: str, seed: int, spec, work: pathlib.Path):
    """A Gazebo server, link and sim for one pass, built as gazebo_crosscheck builds them."""
    from locrec import LIMITED_FOV, SPINNING_360, RunConfig, TunnelSim
    from locrec import gazebo as gzl

    lidar = SPINNING_360 if platform == "ugv" else LIMITED_FOV
    cfg = RunConfig(seed=seed, platform=platform, world=spec, lidar=lidar)
    n_steps = gzl.steps_per_scan(cfg.platform_spec)
    world = TunnelSim(seed, spec, lidar).world
    work.mkdir(parents=True, exist_ok=True)
    sdf = gzl.world_sdf(
        [(world, "tunnel", (0.0, 0.0, 0.0))],
        [gzl.vehicle_model("vehicle", platform, lidar, "/vehicle/lidar",
                           1.0 / (n_steps * gzl.STEP_SIZE), oversample=OVERSAMPLE[platform])],
        mesh_dir=work / "meshes")
    path = gzl.write_world(work / "world.sdf", sdf)
    server = gzl.GazeboServer(path, "tunnel", log_path=work / "server.log", verbosity=1,
                              seed=seed).start()
    link = gzl.GazeboLink("tunnel", {"/vehicle/lidar/points": OVERSAMPLE[platform]})
    link.wait_ready(120.0, server)
    sim = gzl.GazeboTunnelSim(seed, spec, lidar, link, "vehicle", "/vehicle/lidar/points", n_steps)
    link.set_poses({"vehicle": sim.gazebo_pose(np.eye(4))})
    link.prime(n_steps)
    return sim, link, server


def one_seed(job) -> dict:
    seed, length, lag, work = job
    os.environ["GZ_PARTITION"] = f"locrec_team_{os.getpid()}"
    from calibrate_thresholds import mixed_world
    from team_pass import follower, leader

    th = json.loads((ROOT / "results" / "thresholds_gazebo.json").read_text())
    spec = mixed_world(length)
    base = pathlib.Path(work) / f"seed{seed}"

    sim, link, server = _sim_for("ugv", seed, spec, base / "leader")
    try:
        lead, strips, frame_error = leader(seed, length, th, sim=sim)
    finally:
        link.close()
        server.stop()

    lead_err = lead.array("along_err_m")
    out = {"seed": seed, "strips": len(strips), "leader_along_m": abs(frame_error),
           "leader_markers": lead.markers_placed}
    for team in (False, True):
        name = "team" if team else "solo"
        sim, link, server = _sim_for("drone", seed, spec, base / name)
        try:
            r = follower(seed, length, th, strips, team, lag, sim=sim)
        finally:
            link.close()
            server.stop()
        along = float(r.rows[-1]["along_err_m"])
        out[f"{name}_world_m"] = abs(along)
        out[f"{name}_frame_m"] = abs(along - frame_error)
        out[f"{name}_mean_abs_m"] = float(np.mean(np.abs(r.array("along_err_m"))))
        drone_err = r.array("along_err_m")
        n = min(len(drone_err), len(lead_err) - lag)
        out[f"{name}_consistency_m"] = (float(np.mean(np.abs(drone_err[:n] - lead_err[lag:lag + n])))
                                        if n > 0 else float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--result")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--length", type=float, default=300.0)
    ap.add_argument("--lag", type=int, default=LAG_STEPS)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    if args.one:
        row = one_seed(tuple(json.loads(args.one)))
        pathlib.Path(args.result).write_text(json.dumps(row))
        return

    from marker_drop import bootstrap_ci

    work = tempfile.mkdtemp(prefix="locrec_team_gz_")
    jobs = [(s, args.length, args.lag, work) for s in range(args.seeds)]
    print(f"{len(jobs)} seeds on Gazebo, three passes each, {args.jobs} at a time; logs in {work}",
          flush=True)
    rows, pending, running = [], list(jobs), []
    while pending or running:
        while pending and len(running) < args.jobs:
            job = pending.pop(0)
            result = pathlib.Path(work) / f"row_{job[0]}.json"
            proc = subprocess.Popen(
                [sys.executable, __file__, "--one", json.dumps(job), "--result", str(result)],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            running.append((proc, result, job, time.time()))
        time.sleep(1.0)
        for item in list(running):
            proc, result, job, t0 = item
            if proc.poll() is None:
                continue
            running.remove(item)
            if proc.returncode != 0 or not result.exists():
                print(f"  seed {job[0]} FAILED (exit {proc.returncode})", flush=True)
                continue
            row = json.loads(result.read_text())
            rows.append(row)
            print(f"  seed {row['seed']}: leader {row['leader_along_m']:.2f} m, drone solo "
                  f"{row['solo_frame_m']:.2f} m, team {row['team_frame_m']:.2f} m in the robot's "
                  f"frame, {time.time() - t0:.0f} s", flush=True)

    if not rows:
        print("no seed completed")
        return
    rows.sort(key=lambda r: r["seed"])
    out = ROOT / "results" / "team_pass_gazebo.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'seed':>5} {'strips':>7} {'leader':>8} {'solo world':>11} {'team world':>11} "
          f"{'solo frame':>11} {'team frame':>11}")
    for r in rows:
        print(f"{r['seed']:>5} {r['strips']:>7} {r['leader_along_m']:>8.2f} {r['solo_world_m']:>11.2f} "
              f"{r['team_world_m']:>11.2f} {r['solo_frame_m']:>11.2f} {r['team_frame_m']:>11.2f}")
    for name in ("solo", "team"):
        v = [r[f"{name}_consistency_m"] for r in rows]
        print(f"map-frame consistency, {name}: median {np.median(v):.2f} m, worst {np.max(v):.2f} m")
    for frame in ("frame", "world"):
        d = [r[f"solo_{frame}_m"] - r[f"team_{frame}_m"] for r in rows]
        med, lo, hi = bootstrap_ci(d)
        worse = [r["seed"] for r, x in zip(rows, d) if x < 0]
        print(f"in the robot's {frame}: median solo "
              f"{np.median([r[f'solo_{frame}_m'] for r in rows]):.2f} m, team "
              f"{np.median([r[f'team_{frame}_m'] for r in rows]):.2f} m; paired {med:+.2f} m "
              f"[{lo:+.2f}, {hi:+.2f}]; team worse on seeds {worse}")
    # The demonstration seed, by the rule gazebo_crosscheck.py declares and for the same reason:
    # the seed whose paired difference in mean absolute along-track error over the run, which is
    # the number the video reports, is closest to the median of those differences, ties to the
    # lower seed. It is not the seed that looks best. Written here before the first run.
    diffs = [(r["seed"], r["solo_mean_abs_m"] - r["team_mean_abs_m"]) for r in rows]
    med = float(np.median([d for _, d in diffs]))
    pick = min(diffs, key=lambda sd: (abs(sd[1] - med), sd[0]))[0]
    print(f"demonstration seed by the declared rule: {pick} "
          f"(paired differences {', '.join(f'{s}: {d:+.2f}' for s, d in diffs)})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
