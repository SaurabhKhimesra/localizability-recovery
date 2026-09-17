"""Markers spent at matched drift: the scheduler's actual claim.

The scheduler is not claiming lower drift than uniform spacing. It is claiming the
same drift for fewer markers, so the comparison is horizontal on the Pareto plot,
rather than vertical: take a policy's drift, read off how many markers uniform
spacing needs to reach it, and subtract.

    python experiments/matched_drift.py

The uniform curve is interpolated from three points and at eight seeds it is not
monotone, so a saving of a few markers means nothing. A saving of minus thirty-eight
does.
"""
import csv
import sys
import pathlib
import numpy as np
sys.path.insert(0, 'experiments')
from collections import defaultdict

def summarise(path):
    rows = list(csv.DictReader(open(path)))
    by = defaultdict(list)
    for r in rows:
        by[r["policy"]].append(r)
    out = {}
    for k, sub in by.items():
        out[k] = (
            float(np.median([float(r["markers_placed"]) for r in sub])),
            float(np.median([float(r["final_along_m"]) for r in sub])),
        )
    return out

def matched(out):
    uni = sorted(
        (v for k, v in out.items() if k.startswith("uniform")), key=lambda p: p[0]
    )
    uni = [(0.0, out["none"][1])] + uni
    xs = np.array([p[0] for p in uni])
    ys = np.array([p[1] for p in uni])
    order = np.argsort(ys)[::-1]           # drift decreasing as markers rise
    res = {}
    for k, (mk, drift) in out.items():
        if k.startswith("uniform") or k == "none":
            continue
        need = float(np.interp(drift, ys[order], xs[order]))
        res[k] = (mk, drift, need, need - mk)
    return uni, res

for world, path in (("blind", "results/milestone2b_blind.csv"), ("mixed", "results/milestone2b_mixed.csv")):
    if not pathlib.Path(path).exists():
        continue
    out = summarise(path)
    uni, res = matched(out)
    print(f"\n=== {world} world, 300 m, 8 seeds ===")
    print("uniform curve (markers -> along-track):", [(round(a), round(b, 2)) for a, b in uni])
    print(f"{'policy':<12}{'markers':>8}{'along':>8}{'uniform needs':>15}{'saving':>9}")
    for k, (mk, drift, need, save) in sorted(res.items()):
        print(f"{k:<12}{mk:>8.0f}{drift:>8.2f}{need:>15.1f}{save:>+9.1f}")
