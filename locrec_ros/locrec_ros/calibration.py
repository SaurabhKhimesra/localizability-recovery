"""Where the calibrated numbers come from, with no ROS import.

There are three: the ratio threshold, and for the UGV the two numbers the marker
scheduler spaces by. None of them has a default anywhere in this package, because a
default is a second source and the second source is the one that goes stale
(`docs/failures.md` numbers 20 and 23). They come from the file
`locrec/experiments/calibrate_thresholds.py` writes, or from a parameter given explicitly,
which wins so that a threshold can still be varied for an experiment without
editing the file.
"""
from __future__ import annotations

import json
import pathlib

__all__ = ["FILE_KEYS", "resolve_calibration"]

FILE_KEYS = {
    "ugv": {
        "ratio_threshold": "ratio_threshold",
        "marker_reliable_range_m": "marker_reliable_range_m",
        "scheduler_margin_m": "scheduler_margin_m",
    },
    # the drone recommends a yaw, not a marker, so it needs the threshold alone
    "drone": {"ratio_threshold": "drone_ratio_threshold"},
}


def resolve_calibration(
    platform: str, explicit: dict[str, float | None], thresholds_file: str | None
) -> dict[str, float]:
    """Return every calibrated value ``platform`` needs, or raise saying what is missing.

    ``explicit`` maps the names above to a value or ``None``. An explicit value wins
    over the file.
    """
    if platform not in FILE_KEYS:
        raise ValueError(f"platform must be one of {sorted(FILE_KEYS)}, got {platform!r}")
    from_file: dict = {}
    if thresholds_file:
        path = pathlib.Path(thresholds_file).expanduser()
        if not path.is_file():
            raise ValueError(f"thresholds_file {str(path)!r} does not exist")
        from_file = json.loads(path.read_text())

    out: dict[str, float] = {}
    missing = []
    for name, key in FILE_KEYS[platform].items():
        value = explicit.get(name)
        if value is None:
            value = from_file.get(key)
        if value is None:
            missing.append(f"{name} (thresholds.json key {key!r})")
            continue
        out[name] = float(value)
    if missing:
        raise ValueError(
            "no value for " + ", ".join(missing) + ". These are calibrated, they do not "
            "transfer between sensors or spaces, and nothing here ships a default. Pass "
            "thresholds_file:=<path to locrec/results/thresholds.json>, written by "
            "locrec/experiments/calibrate_thresholds.py, or give each one as a parameter."
        )
    return out
