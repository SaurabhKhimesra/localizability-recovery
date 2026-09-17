"""Sensitivity of baseline drift to the registration range cap.

This is the evidence behind ``LidarSpec.nyquist_range``. It is a sweep over a
pipeline parameter, not over a policy parameter, and it is reported in full in
the README including the configurations that fail.

    python experiments/registration_range_sweep.py --seeds 3
"""
from __future__ import annotations

import argparse
import pathlib


from locrec import LIMITED_FOV, SPINNING_360, RunConfig, run_pass
from locrec.odometry import OdometryConfig
from locrec.runner import write_csv

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    rows = []
    grid = {
        "ugv": (SPINNING_360, [2.9, 4.0, 5.7, 7.0, 8.5]),
        "drone": (LIMITED_FOV, [8.0, 10.0, 11.5, 15.0, 21.2, 28.5]),
    }
    for platform, (spec, ranges) in grid.items():
        for rr in ranges:
            for seed in range(args.seeds):
                cfg = RunConfig(
                    seed=seed,
                    platform=platform,
                    odometry=OdometryConfig(registration_range=rr),
                )
                r = run_pass(cfg)
                e = r.array("trans_err_m")
                rows.append(
                    {
                        "platform": platform,
                        "registration_range_m": rr,
                        "beam_spacing_m": spec.beam_spacing_at(rr),
                        "voxel_m": cfg.odometry.map_resolution,
                        "seed": seed,
                        "cold_start_peak_m": float(e[:20].max()),
                        "final_drift_m": r.final_translation_error,
                        "path_length_m": r.path_length,
                    }
                )
                print(
                    f"{platform:5s} rr {rr:5.1f} m (spacing {spec.beam_spacing_at(rr):.3f} m) "
                    f"seed {seed}  peak {e[:20].max():5.2f}  final {r.final_translation_error:5.2f}",
                    flush=True,
                )

    (ROOT / "results").mkdir(exist_ok=True)
    out = ROOT / "results" / "registration_range_sweep.csv"
    write_csv(rows, str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
