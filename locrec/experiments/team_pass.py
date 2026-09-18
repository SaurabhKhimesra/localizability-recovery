"""Is a drone that mounts nothing better off using strips a ground robot left behind?

Declared before the first run, the same form as ``strip_bias_grid.py`` and ``strip_spread_grid.py``.

Setup. Mixed world, seeds 0 to 7, 300 m, one thread, MuJoCo, the calibrated thresholds. Pass one
is the ground robot with the localizability scheduler, which mounts strips where its own
registration goes degenerate; that pass is unchanged from the study's own. Pass two is a drone
flying the same tunnel with forward gaze, ``lag`` steps behind, that mounts nothing. The strips
robot left are on the wall of the drone's world from the step it mounted them, which because the
drone trails is before the drone reaches them, so they are in both drone passes and the scans are
identical; the two passes differ only in whether the drone's estimator is told about them.

What the robot sends, per strip, and nothing else: the slot, where it believes the strip is, the
2x2 covariance of the observation that placed it, and the face normal. No map, no trajectory, no
covariance over the robot's whole run.

Statistic: final along-track error as ``run_pass`` reports it, team against solo, paired by seed,
median and bootstrap CI as in ``marker_drop.py``. Reported in the ground robot's frame, which is
the frame the drone is localizing in, and separately in the world's, which the drone cannot do
better in than the robot that placed the strips. Secondary: map-frame consistency, the mean over
the run of the gap between the drone's along-track error and the robot's.

Expectation written down beforehand: the drone's solo error in a blind stretch is odometry drift,
metres over 300 m, and the team error should fall to the robot's chain error plus a strip fix,
so a clear paired benefit in the robot's frame and close to none in the world's. The robot's own
chain error is the floor and it is about 1 m (``strip_spread_grid.py``), so a world-frame result
near 1 m either way is the expected null, not a failure. Whatever it shows is what gets written.

    python experiments/team_pass.py --workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

from locrec import LocalizabilityScheduler, RunConfig, run_pass
from locrec.gaze import ForwardGaze
from locrec.odometry import OdometryConfig

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from calibrate_thresholds import mixed_world  # noqa: E402
from marker_drop import bootstrap_ci  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAG_STEPS = 60
"""Steps the drone flies behind the ground robot, 30 m at the 0.5 m step. Far enough that it
never sees the robot and only meets strips that are already on the wall."""


def _thresholds():
    return json.loads((ROOT / "results" / "thresholds.json").read_text())


def _strip_yaw(sim, slot: int) -> float:
    """The yaw the simulator has the strip mounted at, so it can be remounted in another world."""
    qw, qx, qy, qz = sim.data.mocap_quat[sim._marker_mocap_ids[slot]]
    return float(2.0 * np.arctan2(qz, qw))


def leader(seed: int, length: float, th: dict, sim=None):
    """The ground robot's pass, and the message it would send about each strip it mounted."""
    held = {}

    def capture(sim, T_true, detections):
        held.setdefault("sim", sim)
        return detections

    cfg = RunConfig(seed=seed, platform="ugv", world=mixed_world(length),
                    odometry=OdometryConfig(num_threads=1, gicp_ratio_floor=th["gicp_ratio_floor"]))
    pol = LocalizabilityScheduler(ratio_threshold=th["ratio_threshold"],
                                  reliable_range_m=th["marker_reliable_range_m"],
                                  margin_m=th["scheduler_margin_m"])
    result = run_pass(cfg, pol, detection_hook=capture, sim=sim)
    sim = held["sim"]
    truth = sim.marker_positions()

    strips = []
    for slot in result.odometry.landmarks.slots:
        lm = result.odometry.landmarks.get(slot)
        if lm.T_drop is None or lm.offset_drop is None or lm.R_drop is None:
            continue
        normal = -(lm.T_drop[:3, :3] @ lm.offset_drop)[:2]
        n = float(np.linalg.norm(normal))
        if n < 1e-9 or slot >= truth.shape[0]:
            continue
        strips.append({
            "slot": int(slot),
            "reported": lm.position.copy(),        # where the robot believes it is
            "truth": truth[slot].copy(),           # where it actually is, for the drone's world
            "yaw": _strip_yaw(sim, slot),
            "R_drop": np.array(lm.R_drop, dtype=float),
            "normal": normal / n,
            "step": int(lm.registered_at_step),
        })
    strips.sort(key=lambda s: s["step"])
    # the robot's own error at the end, which is the offset between its frame and the world
    frame_error = float(result.rows[-1]["along_err_m"])
    return result, strips, frame_error


def follower(seed: int, length: float, th: dict, strips: list, team: bool, lag: int, sim=None):
    """The drone's pass. The strips are on the wall either way; only the estimator differs."""
    state = {"next": 0}

    def step_hook(k, sim, odom):
        # The drone trails the robot by ``lag`` steps, so its own step k is the robot's step
        # k + lag: a strip the robot mounted at its step i has been on the wall since the drone's
        # step i - lag, which is before the drone reaches it. Adding the lag instead puts every
        # strip on the wall 30 m after the drone has flown past it, and with forward gaze it never
        # looks back, so the drone sees nothing and the team result is exactly the solo one.
        while state["next"] < len(strips) and strips[state["next"]]["step"] - lag <= k:
            s = strips[state["next"]]
            placed = sim.place_marker(s["truth"], s["yaw"])
            if placed != s["slot"]:
                raise AssertionError(
                    f"slot {placed} in the drone's world is slot {s['slot']} in the robot's; "
                    "the strips must be placed in the order they were mounted"
                )
            if team:
                odom.register_foreign_landmark(s["slot"], s["reported"], s["R_drop"], s["normal"])
            state["next"] += 1

    cfg = RunConfig(seed=seed, platform="drone", world=mixed_world(length),
                    odometry=OdometryConfig(num_threads=1,
                                            gicp_ratio_floor=th["drone_gicp_ratio_floor"]))
    return run_pass(cfg, ForwardGaze(), step_hook=step_hook, sim=sim)


def _one(job):
    seed, length, lag = job
    th = _thresholds()
    lead, strips, frame_error = leader(seed, length, th)
    lead_err = lead.array("along_err_m")
    out = {"seed": seed, "strips": len(strips), "leader_along_m": abs(frame_error),
           "leader_markers": lead.markers_placed}
    for team in (False, True):
        r = follower(seed, length, th, strips, team, lag)
        along = float(r.rows[-1]["along_err_m"])
        name = "team" if team else "solo"
        out[f"{name}_world_m"] = abs(along)
        # in the robot's frame: the drone's error less the robot's own, both along track
        out[f"{name}_frame_m"] = abs(along - frame_error)
        out[f"{name}_mean_abs_m"] = float(np.mean(np.abs(r.array("along_err_m"))))
        # the declared secondary: how far apart the two vehicles' frames are over the whole run,
        # the drone's step k against the robot's step k + lag, which is the same place
        drone_err = r.array("along_err_m")
        n = min(len(drone_err), len(lead_err) - lag)
        out[f"{name}_consistency_m"] = (float(np.mean(np.abs(drone_err[:n] - lead_err[lag:lag + n])))
                                        if n > 0 else float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--length", type=float, default=300.0)
    ap.add_argument("--lag", type=int, default=LAG_STEPS)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = [(s, args.length, args.lag) for s in range(args.seeds)]
    with Pool(args.workers, maxtasksperchild=2) as pool:
        rows = sorted(pool.imap_unordered(_one, jobs), key=lambda r: r["seed"])

    out = ROOT / "results" / "team_pass.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'seed':>5} {'strips':>7} {'leader':>8} {'solo world':>11} {'team world':>11} "
          f"{'solo frame':>11} {'team frame':>11}")
    for r in rows:
        print(f"{r['seed']:>5} {r['strips']:>7} {r['leader_along_m']:>8.2f} {r['solo_world_m']:>11.2f} "
              f"{r['team_world_m']:>11.2f} {r['solo_frame_m']:>11.2f} {r['team_frame_m']:>11.2f}")
    for name in ("solo", "team"):
        v = [r[f"{name}_consistency_m"] for r in rows]
        print(f"map-frame consistency, {name}: mean over the run of the gap between the two "
              f"vehicles, median {np.median(v):.2f} m, worst {np.max(v):.2f} m")
    for frame in ("frame", "world"):
        d = [r[f"solo_{frame}_m"] - r[f"team_{frame}_m"] for r in rows]
        med, lo, hi = bootstrap_ci(d)
        worse = [r["seed"] for r, x in zip(rows, d) if x < 0]
        print(f"in the robot's {frame}: median solo {np.median([r[f'solo_{frame}_m'] for r in rows]):.2f} m, "
              f"team {np.median([r[f'team_{frame}_m'] for r in rows]):.2f} m; paired {med:+.2f} m "
              f"[{lo:+.2f}, {hi:+.2f}]; team worse on seeds {worse}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
