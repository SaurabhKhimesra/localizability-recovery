"""Milestone 3: drift against what the gaze policy costs.

    python experiments/plot_drone_cost.py results/ --out docs/

Two cost axes, because only one of them turns out to be real. Mission time is
identical across gaze policies by construction: every policy flies the same
centreline at the same speed, and yaw is decoupled from velocity, so the only
thing a gaze policy spends is slew. The mission-time panel is kept in the figure
so that claim is visible rather than asserted.

The plotting itself is the same code milestone 4 uses, imported rather than
copied, so the two figures cannot drift apart.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from summarise import drone_cost, load  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs"))
    args = ap.parse_args()

    rows = load(args.results)
    if not rows:
        raise SystemExit(f"no monte_carlo_*.csv under {args.results}")
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "drone_drift_vs_cost.png"
    drone_cost(rows, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
