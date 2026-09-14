"""LiDAR sensor model on top of ``mujoco.mj_multiRay``.

Two configurations matter for this study:

* ``SPINNING_360``  a 360 deg spinning scanner with modest range (the UGV).
* ``LIMITED_FOV``   a narrow-FOV scanner whose yaw is free of the velocity
  direction (the drone).

Rays are cast from a single origin, which is exactly what ``mj_multiRay`` does,
so there is no per-beam motion distortion in this model. That is a deliberate
simplification and is listed in the README's limitations.
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from .se3 import transform_points

__all__ = ["LidarSpec", "Scan", "Lidar", "SPINNING_360", "LIMITED_FOV"]


@dataclasses.dataclass(frozen=True)
class LidarSpec:
    n_azimuth: int = 180
    n_elevation: int = 16
    fov_azimuth_deg: float = 360.0
    fov_elevation_deg: float = 30.0
    """Total vertical opening; beams are spread symmetrically about the xy plane."""
    max_range: float = 20.0
    min_range: float = 0.4
    range_sigma: float = 0.02
    """Zero-mean Gaussian range noise, metres, 1 sigma."""
    dropout: float = 0.0
    """Probability a beam returns nothing even though it hit something."""

    @property
    def n_rays(self) -> int:
        return self.n_azimuth * self.n_elevation

    @property
    def azimuth_step_rad(self) -> float:
        """Beam spacing along the dense (scanning) axis, radians."""
        return float(np.deg2rad(self.fov_azimuth_deg) / max(self.n_azimuth, 1))

    @property
    def elevation_step_rad(self) -> float:
        if self.n_elevation <= 1:
            return 0.0
        return float(np.deg2rad(self.fov_elevation_deg) / max(self.n_elevation - 1, 1))

    def beam_spacing_at(self, distance: float) -> float:
        """Spacing between neighbouring returns along the dense axis, metres."""
        return float(distance * self.azimuth_step_rad)

    def nyquist_range(self, voxel: float) -> float:
        """Distance beyond which the scan under-samples a map of this voxel pitch.

        A voxel map of pitch ``v`` can only represent structure that the scan
        samples at ``v / 2`` or finer. Past that range the nearest map point to a
        far scan point is up to a beam spacing away in the wrong direction, and
        registration inherits a systematic pull. On the drone configuration the
        effect is a cliff, not a gradient: registering the 90 deg scanner out to
        21 m (spacing = one voxel) leaves a 2.5 m cold-start excursion and 1.3 m of
        final drift, while cutting to 12 m (spacing = half a voxel) gives 0.1 m and
        0.35 m. ``experiments/registration_range_sweep.py`` reproduces the table.

        Elevation spacing is deliberately not the binding term: rings sweep along
        the tunnel axis, where the surface is smooth and a coarse sample costs
        little, whereas azimuth spacing samples across the wall, which is the
        structure registration actually uses.
        """
        step = self.azimuth_step_rad
        return float(0.5 * voxel / step) if step > 0 else float("inf")


SPINNING_360 = LidarSpec(
    n_azimuth=360, n_elevation=16, fov_azimuth_deg=360.0, fov_elevation_deg=30.0, max_range=10.0
)
"""VLP-16 class: 16 rings over 30 deg, 1 deg azimuth, short range. The UGV sensor."""

LIMITED_FOV = LidarSpec(
    n_azimuth=180, n_elevation=112, fov_azimuth_deg=90.0, fov_elevation_deg=60.0, max_range=30.0
)
"""Livox-class solid state: 90 x 60 deg at about 0.5 deg, long range. The drone sensor.

Beam counts follow the sampling rule in :meth:`LidarSpec.well_sampled_range`: at
30 m a 0.5 deg spacing puts neighbouring returns 0.26 m apart, just inside the
0.2 m map voxel. A coarser bundle at the same range is a documented failure mode,
not a cheaper approximation."""


@dataclasses.dataclass
class Scan:
    points: np.ndarray
    """Hit points in the sensor frame, shape (K, 3)."""
    ranges: np.ndarray
    """Range of each returned beam, shape (K,)."""
    geom_ids: np.ndarray
    """MuJoCo geom id hit by each returned beam, shape (K,)."""
    directions: np.ndarray
    """Unit ray direction in the sensor frame for each returned beam, (K, 3)."""
    n_cast: int

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def return_rate(self) -> float:
        return len(self) / max(self.n_cast, 1)

    def points_world(self, T_world_sensor: np.ndarray) -> np.ndarray:
        return transform_points(T_world_sensor, self.points)


def _ray_directions(spec: LidarSpec) -> np.ndarray:
    """Unit directions in the sensor frame, shape (n_rays, 3). x forward, z up."""
    fov_az = np.deg2rad(spec.fov_azimuth_deg)
    if spec.fov_azimuth_deg >= 359.999:
        az = np.arange(spec.n_azimuth) * (2.0 * np.pi / spec.n_azimuth) - np.pi
    else:
        az = np.linspace(-fov_az * 0.5, fov_az * 0.5, spec.n_azimuth)
    fov_el = np.deg2rad(spec.fov_elevation_deg)
    el = (
        np.linspace(-fov_el * 0.5, fov_el * 0.5, spec.n_elevation)
        if spec.n_elevation > 1
        else np.zeros(1)
    )
    A, E = np.meshgrid(az, el, indexing="ij")
    ce = np.cos(E)
    return np.stack([ce * np.cos(A), ce * np.sin(A), np.sin(E)], axis=-1).reshape(-1, 3)


class Lidar:
    """Stateless apart from its own RNG and cached ray table."""

    def __init__(self, spec: LidarSpec = SPINNING_360, seed: int = 0):
        self.spec = spec
        self._dirs = _ray_directions(spec)
        self._rng = np.random.default_rng(seed)
        n = spec.n_rays
        self._geomid = np.zeros(n, dtype=np.int32)
        self._dist = np.zeros(n, dtype=np.float64)

    def scan(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        T_world_sensor: np.ndarray,
        geomgroup: np.ndarray | None = None,
    ) -> Scan:
        """Cast the full ray bundle from the sensor pose and return the hits.

        ``T_world_sensor`` places the sensor; the returned points are expressed in
        the sensor frame so that odometry sees exactly what a real driver returns.
        """
        R = np.ascontiguousarray(T_world_sensor[:3, :3])
        origin = np.ascontiguousarray(T_world_sensor[:3, 3], dtype=np.float64)
        vec = np.ascontiguousarray((self._dirs @ R.T).reshape(-1), dtype=np.float64)

        self._geomid[:] = -1
        self._dist[:] = -1.0
        mujoco.mj_multiRay(
            m=model,
            d=data,
            pnt=origin,
            vec=vec,
            geomgroup=geomgroup,
            flg_static=True,
            bodyexclude=-1,
            geomid=self._geomid,
            dist=self._dist,
            normal=None,
            nray=self.spec.n_rays,
            cutoff=self.spec.max_range,
        )

        hit = (self._geomid >= 0) & (self._dist >= self.spec.min_range)
        hit &= self._dist <= self.spec.max_range
        if self.spec.dropout > 0.0:
            hit &= self._rng.random(self.spec.n_rays) >= self.spec.dropout

        rng_vals = self._dist[hit]
        if self.spec.range_sigma > 0.0:
            rng_vals = rng_vals + self._rng.normal(0.0, self.spec.range_sigma, rng_vals.shape)
        dirs = self._dirs[hit]
        return Scan(
            points=dirs * rng_vals[:, None],
            ranges=rng_vals,
            geom_ids=self._geomid[hit].copy(),
            directions=dirs,
            n_cast=self.spec.n_rays,
        )
