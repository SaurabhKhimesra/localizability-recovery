"""Procedural 3D tunnel worlds for MuJoCo.

The tunnel is built as a ribbed shell: the centreline is sampled at a fixed
arclength step and each sample gets four thin box geoms (left wall, right wall,
floor, ceiling) oriented with the local heading. That makes width variation,
curvature, niches and junction openings trivial to express, at the cost of a
few hundred geoms per world.

Everything is a deterministic function of the seed. The same seed must produce
byte-identical XML on any machine, so no unseeded randomness is used anywhere.

Geom naming / grouping (used by the sensor model and by the marker scheduler):
    group 0  tunnel shell  (names: wall_*, floor_*, ceil_*, cap_*)
    group 3  markers       (names: marker_<i>), carried by mocap bodies

Marker slots are pre-allocated as mocap bodies parked below the floor. Dropping
a marker is then a write to ``data.mocap_pos``, which needs no model rebuild.
"""
from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from typing import Literal

import numpy as np

__all__ = ["WorldSpec", "TunnelWorld", "build_world"]

Side = Literal["left", "right"]


@dataclasses.dataclass(frozen=True)
class WorldSpec:
    """Knobs for the procedural generator. Defaults are the study configuration."""

    length: float = 160.0
    """Arclength of the main tunnel, metres."""
    ds: float = 1.0
    """Centreline sampling step, metres. Also the rib spacing."""
    width_mean: float = 3.2
    width_min: float = 2.2
    width_max: float = 4.6
    width_corr_length: float = 18.0
    """Correlation length of the smooth width variation, metres."""
    height: float = 2.6
    wall_thickness: float = 0.12

    n_curves: int = 2
    """Number of gentle curved sections in the main tunnel."""
    curve_radius_range: tuple[float, float] = (18.0, 45.0)
    curve_length_range: tuple[float, float] = (12.0, 26.0)

    n_junctions: int = 2
    """Side branches off the main tunnel. Each is an opening plus a stub tunnel."""
    junction_length: float = 10.0
    junction_width: float = 3.0
    junction_opening: float = 3.0
    """Arclength of the wall opening at a junction, metres."""

    n_niches: int = 4
    """Local one-sided width bumps (alcoves). Strong localizability features."""
    niche_depth_range: tuple[float, float] = (1.0, 2.2)
    niche_length_range: tuple[float, float] = (1.5, 3.5)

    straight_lead_in: float = 12.0
    """Arclength at the start with no curvature and no features."""

    heading_offset_deg: float = 0.0
    """Initial centreline heading, degrees, which rotates the whole tunnel about the
    origin.

    The estimator's voxel grids are axis aligned in its world frame and are never
    rebinned, so this is the angle between the corridor and those grids. It exists
    because the localizability ratio in a featureless tunnel depends on that angle,
    with a period of 90 degrees and a spread of 19 percent (`docs/failures.md`
    number 21), so a calibration that leaves it at zero measures one orientation of
    a periodic function. Zero reproduces the generator exactly: nothing is added to
    the heading unless the offset is non-zero."""

    structured_stretches: bool = False
    """Lay the tunnel out as alternating blind and structured runs.

    With this off, features are scattered across the whole tunnel, which is the
    milestone 1 behaviour and is what the detector trace was measured on. With it
    on, the tunnel is partitioned into runs whose lengths are drawn per seed, and
    every junction, niche and curve is confined to the structured ones. That is the
    world a scheduler can actually be tested in: a tunnel that is featureless end to
    end is equally degenerate everywhere, so there is nowhere to save a marker, and
    a scheduler cannot beat uniform spacing no matter how good it is."""
    blind_stretch_range: tuple[float, float] = (35.0, 70.0)
    structured_stretch_range: tuple[float, float] = (15.0, 30.0)
    n_marker_slots: int = 64
    marker_height: float = 1.0
    marker_width: float = 0.15
    marker_thickness: float = 0.02
    """Retroreflective target: a vertical wall-mounted strip, 1.0 m by 0.15 m.

    Mine practice uses strips rather than discs for a reason this simulator
    reproduces exactly. A 0.3 m disc subtends 1.7 deg at 10 m, and a VLP-16 puts
    its channels 2 deg apart, so the disc falls between rings and returns nothing.
    A 1.0 m strip subtends 5.7 deg and always crosses two or three channels.
    Horizontally 0.15 m subtends 0.86 deg against a 1 deg azimuth step, so a
    detection is typically one or two beams wide, which is all that is needed:
    one high-intensity return is a range plus bearing landmark."""

    def __post_init__(self) -> None:
        if self.ds <= 0:
            raise ValueError("ds must be positive")
        if not (self.width_min <= self.width_mean <= self.width_max):
            raise ValueError("width_mean must lie inside [width_min, width_max]")


@dataclasses.dataclass
class TunnelWorld:
    """A generated world: MuJoCo XML plus the ground-truth centreline."""

    spec: WorldSpec
    seed: int
    xml: str
    s: np.ndarray
    """Arclength samples of the main centreline, shape (N,)."""
    centreline: np.ndarray
    """xy positions at each sample, shape (N, 2)."""
    heading: np.ndarray
    """Tangent heading (rad) at each sample, shape (N,)."""
    width: np.ndarray
    """Full tunnel width at each sample, shape (N,)."""
    curvature: np.ndarray
    """Signed centreline curvature at each sample, 1/m, shape (N,). Non-zero stretches
    are the curved sections, which restore localizability without any feature."""
    feature_mask: np.ndarray
    """Bool per sample: True where a junction opening or niche is present.

    This is ground truth about the *world*, not about the estimator. It is used
    only to report where features are, never to choose any parameter.
    """
    junctions: list[dict]
    niches: list[dict]

    # ---- convenience ----------------------------------------------------

    def pose_at(self, s: float, height: float = 0.6) -> tuple[np.ndarray, float]:
        """Centreline position (3,) and heading at arclength s, linearly interpolated."""
        x = np.interp(s, self.s, self.centreline[:, 0])
        y = np.interp(s, self.s, self.centreline[:, 1])
        # interpolate heading through its unwrapped form
        h = np.interp(s, self.s, np.unwrap(self.heading))
        return np.array([x, y, height]), float(h)

    @property
    def total_length(self) -> float:
        return float(self.s[-1])


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def _smooth_noise(rng: np.random.Generator, n: int, ds: float, corr_length: float) -> np.ndarray:
    """Zero-mean, unit-variance-ish smooth 1-D noise by Gaussian-filtering white noise."""
    w = rng.standard_normal(n)
    sigma_samples = max(corr_length / ds, 1.0)
    half = int(np.ceil(3.0 * sigma_samples))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_samples) ** 2)
    k /= k.sum()
    sm = np.convolve(w, k, mode="same")
    sd = sm.std()
    return sm / sd if sd > 1e-9 else sm


def _build_centreline(
    spec: WorldSpec, rng: np.random.Generator, allowed: np.ndarray | None = None
):
    n = int(round(spec.length / spec.ds)) + 1
    s = np.arange(n) * spec.ds

    # curvature: zero except inside a few curved sections
    kappa = np.zeros(n)
    free_start = int(spec.straight_lead_in / spec.ds)
    for _ in range(spec.n_curves):
        seg_len = rng.uniform(*spec.curve_length_range)
        seg_n = max(int(seg_len / spec.ds), 2)
        if free_start + seg_n >= n - 2:
            break
        if allowed is not None:
            options = [
                i
                for i in range(free_start, n - seg_n - 1)
                if allowed[i : i + seg_n].all()
            ]
            if not options:
                break
            i0 = int(options[rng.integers(len(options))])
        else:
            i0 = int(rng.integers(free_start, n - seg_n - 1))
        radius = rng.uniform(*spec.curve_radius_range)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        # raised-cosine taper so heading is C1
        taper = 0.5 * (1.0 - np.cos(np.linspace(0.0, 2.0 * np.pi, seg_n)))
        kappa[i0 : i0 + seg_n] += sign * taper / radius
        free_start = i0 + seg_n + 2

    heading = np.cumsum(kappa) * spec.ds
    if spec.heading_offset_deg:
        heading = heading + np.deg2rad(spec.heading_offset_deg)
    pos = np.zeros((n, 2))
    pos[1:, 0] = np.cumsum(np.cos(heading[:-1]) * spec.ds)
    pos[1:, 1] = np.cumsum(np.sin(heading[:-1]) * spec.ds)

    width = spec.width_mean + 0.35 * _smooth_noise(rng, n, spec.ds, spec.width_corr_length)
    width = np.clip(width, spec.width_min, spec.width_max)
    return s, pos, heading, width, kappa


def _stretch_layout(spec: WorldSpec, rng: np.random.Generator, n: int) -> np.ndarray:
    """Boolean per sample: True where the tunnel is allowed to have structure."""
    allowed = np.zeros(n, dtype=bool)
    i = int(spec.straight_lead_in / spec.ds)
    blind = True  # the lead-in is blind, so the first drawn run is structured
    while i < n:
        span = rng.uniform(
            *(spec.structured_stretch_range if not blind else spec.blind_stretch_range)
        )
        j = min(i + max(int(span / spec.ds), 2), n)
        if not blind:
            allowed[i:j] = True
        i = j
        blind = not blind
    return allowed


def _place_features(
    spec: WorldSpec,
    rng: np.random.Generator,
    s: np.ndarray,
    allowed: np.ndarray | None = None,
):
    """Choose junction and niche locations. Kept clear of the lead-in and of each other."""
    n = len(s)
    lead = int(spec.straight_lead_in / spec.ds)
    occupied = np.zeros(n, dtype=bool)
    occupied[:lead] = True
    occupied[-3:] = True
    if allowed is not None:
        occupied |= ~allowed

    def _claim(width_samples: int, pad: int = 4) -> int | None:
        candidates = [
            i
            for i in range(lead, n - width_samples - 1)
            if not occupied[max(0, i - pad) : i + width_samples + pad].any()
        ]
        if not candidates:
            return None
        i = int(candidates[rng.integers(len(candidates))])
        occupied[max(0, i - pad) : i + width_samples + pad] = True
        return i

    # inside a structured stretch the features are meant to be packed, so the
    # keep-apart padding shrinks; scattered across a whole tunnel they should not be
    packed = allowed is not None
    junctions: list[dict] = []
    open_n = max(int(spec.junction_opening / spec.ds), 2)
    for _ in range(spec.n_junctions):
        i = _claim(open_n, pad=2 if packed else 6)
        if i is None:
            break
        junctions.append(
            {
                "index": i,
                "n_samples": open_n,
                "side": "left" if rng.random() < 0.5 else "right",
                "length": spec.junction_length,
                "width": spec.junction_width,
            }
        )

    niches: list[dict] = []
    for _ in range(spec.n_niches):
        ln = rng.uniform(*spec.niche_length_range)
        nn = max(int(ln / spec.ds), 2)
        i = _claim(nn, pad=1 if packed else 3)
        if i is None:
            break
        niches.append(
            {
                "index": i,
                "n_samples": nn,
                "side": "left" if rng.random() < 0.5 else "right",
                "depth": float(rng.uniform(*spec.niche_depth_range)),
            }
        )
    return junctions, niches


def _rib(parent: ET.Element, name: str, pos, size, yaw: float) -> None:
    ET.SubElement(
        parent,
        "geom",
        {
            "name": name,
            "type": "box",
            "class": "shell",
            "pos": "%.4f %.4f %.4f" % tuple(pos),
            "size": "%.4f %.4f %.4f" % tuple(size),
            "euler": "0 0 %.6f" % yaw,
        },
    )


def _emit_ribs(
    body: ET.Element,
    prefix: str,
    pos_xy: np.ndarray,
    heading: np.ndarray,
    width: np.ndarray,
    spec: WorldSpec,
    left_open: np.ndarray | None = None,
    right_open: np.ndarray | None = None,
    left_offset: np.ndarray | None = None,
    right_offset: np.ndarray | None = None,
) -> None:
    n = len(heading)
    half_len = spec.ds * 0.5 * 1.1
    ht = spec.wall_thickness * 0.5
    hh = spec.height * 0.5
    for i in range(n):
        yaw = float(heading[i])
        nx, ny = -np.sin(yaw), np.cos(yaw)  # left normal
        p = pos_xy[i]
        w = float(width[i])
        lo = float(left_offset[i]) if left_offset is not None else 0.0
        ro = float(right_offset[i]) if right_offset is not None else 0.0
        if left_open is None or not left_open[i]:
            c = (p[0] + nx * (w * 0.5 + lo), p[1] + ny * (w * 0.5 + lo), hh)
            _rib(body, f"wall_{prefix}_L{i}", c, (half_len, ht, hh), yaw)
        if right_open is None or not right_open[i]:
            c = (p[0] - nx * (w * 0.5 + ro), p[1] - ny * (w * 0.5 + ro), hh)
            _rib(body, f"wall_{prefix}_R{i}", c, (half_len, ht, hh), yaw)
        half_w = w * 0.5 + max(lo, ro) + spec.wall_thickness
        _rib(body, f"floor_{prefix}_{i}", (p[0], p[1], -ht), (half_len, half_w, ht), yaw)
        _rib(
            body,
            f"ceil_{prefix}_{i}",
            (p[0], p[1], spec.height + ht),
            (half_len, half_w, ht),
            yaw,
        )


def build_world(seed: int, spec: WorldSpec | None = None) -> TunnelWorld:
    """Generate a tunnel world. Deterministic in ``seed``."""
    spec = spec or WorldSpec()
    rng = np.random.default_rng(seed)

    n_samples = int(round(spec.length / spec.ds)) + 1
    allowed = _stretch_layout(spec, rng, n_samples) if spec.structured_stretches else None
    s, pos, heading, width, kappa = _build_centreline(spec, rng, allowed)
    junctions, niches = _place_features(spec, rng, s, allowed)
    n = len(s)

    left_open = np.zeros(n, dtype=bool)
    right_open = np.zeros(n, dtype=bool)
    left_off = np.zeros(n)
    right_off = np.zeros(n)
    feature_mask = np.zeros(n, dtype=bool)

    for j in junctions:
        sl = slice(j["index"], j["index"] + j["n_samples"])
        (left_open if j["side"] == "left" else right_open)[sl] = True
        feature_mask[sl] = True
    for nc in niches:
        sl = slice(nc["index"], nc["index"] + nc["n_samples"])
        (left_off if nc["side"] == "left" else right_off)[sl] = nc["depth"]
        feature_mask[sl] = True

    root = ET.Element("mujoco", {"model": f"tunnel_seed{seed}"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(
        root, "option", {"timestep": "0.01", "gravity": "0 0 -9.81", "integrator": "implicitfast"}
    )
    visual = ET.SubElement(root, "visual")
    visual.append(ET.Element("headlight", {"ambient": "0.4 0.4 0.4", "diffuse": "0.6 0.6 0.6"}))
    # the offscreen framebuffer defaults to 640x480, which is smaller than the
    # video frame and fails at render time rather than at compile time
    visual.append(ET.Element("global", {"offwidth": "1280", "offheight": "720"}))

    default = ET.SubElement(root, "default")
    shell = ET.SubElement(default, "default", {"class": "shell"})
    ET.SubElement(
        shell,
        "geom",
        {"group": "0", "rgba": "0.55 0.52 0.48 1", "contype": "1", "conaffinity": "1"},
    )
    mk = ET.SubElement(default, "default", {"class": "marker"})
    ET.SubElement(
        mk,
        "geom",
        {
            "group": "3",
            "type": "box",
            "size": "%.4f %.4f %.4f"
            % (spec.marker_thickness * 0.5, spec.marker_width * 0.5, spec.marker_height * 0.5),
            "rgba": "0.05 1.0 0.35 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(
        wb, "light", {"pos": "0 0 6", "dir": "0 0 -1", "directional": "true"}
    )

    _emit_ribs(wb, "main", pos, heading, width, spec, left_open, right_open, left_off, right_off)

    # end caps: a dead-end wall at each end of the main tunnel keeps the tunnel
    # bounded so rays terminate instead of escaping to max range.
    for tag, i, sgn in (("start", 0, -1.0), ("end", n - 1, 1.0)):
        yaw = float(heading[i])
        p = pos[i]
        c = (
            p[0] + sgn * np.cos(yaw) * spec.ds,
            p[1] + sgn * np.sin(yaw) * spec.ds,
            spec.height * 0.5,
        )
        _rib(
            wb,
            f"cap_{tag}",
            c,
            (spec.wall_thickness * 0.5, width[i] * 0.5 + 0.5, spec.height * 0.5),
            yaw,
        )

    # junction stub tunnels
    for k, j in enumerate(junctions):
        i = j["index"] + j["n_samples"] // 2
        yaw0 = float(heading[i])
        sgn = 1.0 if j["side"] == "left" else -1.0
        branch_yaw = yaw0 + sgn * np.pi * 0.5
        nb = max(int(j["length"] / spec.ds), 2)
        # start just outside the main tunnel wall
        r0 = width[i] * 0.5
        base = pos[i] + sgn * np.array([-np.sin(yaw0), np.cos(yaw0)]) * r0
        bpos = np.stack(
            [
                base[0] + np.cos(branch_yaw) * spec.ds * (np.arange(nb) + 0.5),
                base[1] + np.sin(branch_yaw) * spec.ds * (np.arange(nb) + 0.5),
            ],
            axis=1,
        )
        bhead = np.full(nb, branch_yaw)
        bwidth = np.full(nb, j["width"])
        _emit_ribs(wb, f"j{k}", bpos, bhead, bwidth, spec)
        # dead end of the stub
        c = (
            bpos[-1, 0] + np.cos(branch_yaw) * spec.ds,
            bpos[-1, 1] + np.sin(branch_yaw) * spec.ds,
            spec.height * 0.5,
        )
        _rib(
            wb,
            f"cap_j{k}",
            c,
            (spec.wall_thickness * 0.5, j["width"] * 0.5 + 0.3, spec.height * 0.5),
            branch_yaw,
        )

    # marker slots: mocap bodies parked below the floor
    for i in range(spec.n_marker_slots):
        b = ET.SubElement(
            wb, "body", {"name": f"marker_body_{i}", "mocap": "true", "pos": f"0 0 {-50.0 - i}"}
        )
        ET.SubElement(b, "geom", {"name": f"marker_{i}", "class": "marker"})

    xml = ET.tostring(root, encoding="unicode")
    return TunnelWorld(
        spec=spec,
        seed=seed,
        xml=xml,
        s=s,
        centreline=pos,
        heading=heading,
        width=width,
        curvature=kappa,
        feature_mask=feature_mask,
        junctions=junctions,
        niches=niches,
    )
