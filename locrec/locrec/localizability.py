"""Localizability from the registration Hessian.

``small_gicp`` returns the 6x6 Gauss-Newton information matrix H of the final
linearisation, ordered ``[rotation(0:3), translation(3:6)]`` and expressed in
the *target* frame (for us: the local map frame, i.e. the estimator's world).
That ordering is not documented in the Python bindings, so it is pinned by a
unit test (``tests/test_localizability.py::test_hessian_block_ordering``) that
builds a corridor point cloud and asserts the weak direction lands in
``H[3:6, 3:6]`` along the corridor axis.

The eigenvalues of the translational block scale with the number of
correspondences and with the noise weighting, so two quantities are reported:

* ``ratio = lambda_min / lambda_max``  scale free, the primary signal.
* ``lambda_min_per_point``             absolute, used as a sanity guard against
  the degenerate-but-tiny-scan case where the ratio looks fine only because
  every direction is badly observed.

No threshold is baked in. Thresholds are a declared study parameter and must be
calibrated on seeds disjoint from the evaluation seeds; see ``calibrate_threshold``.
"""
from __future__ import annotations

import dataclasses

import numpy as np

__all__ = [
    "Localizability",
    "LocalizabilityConfig",
    "analyse_hessian",
    "calibrate_threshold",
    "ROT_BLOCK",
    "TRANS_BLOCK",
]

ROT_BLOCK = slice(0, 3)
TRANS_BLOCK = slice(3, 6)


@dataclasses.dataclass(frozen=True)
class LocalizabilityConfig:
    """Detector thresholds. Both must be supplied explicitly by the caller.

    ``ratio_threshold``      declare degenerate when lambda_min/lambda_max falls below this.
    ``min_lambda_per_point`` declare degenerate when the absolute weakest direction
                             is below this even if the ratio looks acceptable.
    """

    ratio_threshold: float
    min_lambda_per_point: float = 0.0


@dataclasses.dataclass
class Localizability:
    eigenvalues: np.ndarray
    """Translational eigenvalues, ascending, shape (3,)."""
    eigenvectors: np.ndarray
    """Columns match ``eigenvalues``; expressed in the frame H was given in."""
    ratio: float
    """lambda_min / lambda_max of the translational block."""
    lambda_min_per_point: float
    weak_direction: np.ndarray
    """Unit vector of the least observable translation direction, shape (3,)."""
    rot_eigenvalues: np.ndarray
    rot_ratio: float
    n_points: int

    def is_degenerate(self, cfg: LocalizabilityConfig) -> bool:
        return (
            self.ratio < cfg.ratio_threshold
            or self.lambda_min_per_point < cfg.min_lambda_per_point
        )

    def in_frame(self, R_target_source: np.ndarray) -> "Localizability":
        """Re-express the directions in another frame.

        ``R_target_source`` maps a vector in the new (source) frame into the frame H
        was given in, so directions transform by its transpose.
        """
        R = np.asarray(R_target_source)
        return dataclasses.replace(
            self,
            eigenvectors=R.T @ self.eigenvectors,
            weak_direction=R.T @ self.weak_direction,
        )

    def as_row(self, prefix: str = "loc_") -> dict:
        return {
            f"{prefix}lambda_min": self.eigenvalues[0],
            f"{prefix}lambda_mid": self.eigenvalues[1],
            f"{prefix}lambda_max": self.eigenvalues[2],
            f"{prefix}ratio": self.ratio,
            f"{prefix}lambda_min_per_point": self.lambda_min_per_point,
            f"{prefix}weak_x": self.weak_direction[0],
            f"{prefix}weak_y": self.weak_direction[1],
            f"{prefix}weak_z": self.weak_direction[2],
            f"{prefix}rot_ratio": self.rot_ratio,
            f"{prefix}n_points": self.n_points,
        }


def analyse_hessian(H: np.ndarray, n_points: int) -> Localizability:
    """Eigen-analysis of the translational (and, secondarily, rotational) block."""
    H = np.asarray(H, dtype=float)
    if H.shape != (6, 6):
        raise ValueError(f"expected a 6x6 Hessian, got {H.shape}")
    Ht = 0.5 * (H[TRANS_BLOCK, TRANS_BLOCK] + H[TRANS_BLOCK, TRANS_BLOCK].T)
    Hr = 0.5 * (H[ROT_BLOCK, ROT_BLOCK] + H[ROT_BLOCK, ROT_BLOCK].T)

    ev, evec = np.linalg.eigh(Ht)
    ev = np.maximum(ev, 0.0)
    rev = np.maximum(np.linalg.eigvalsh(Hr), 0.0)

    lam_max = float(ev[-1])
    ratio = float(ev[0] / lam_max) if lam_max > 0.0 else 0.0
    rot_max = float(rev[-1])
    rot_ratio = float(rev[0] / rot_max) if rot_max > 0.0 else 0.0
    per_point = float(ev[0] / n_points) if n_points > 0 else 0.0

    weak = evec[:, 0]
    # sign is arbitrary; fix it so traces are readable
    if weak[int(np.argmax(np.abs(weak)))] < 0:
        weak = -weak
    return Localizability(
        eigenvalues=ev,
        eigenvectors=evec,
        ratio=ratio,
        lambda_min_per_point=per_point,
        weak_direction=weak,
        rot_eigenvalues=rev,
        rot_ratio=rot_ratio,
        n_points=int(n_points),
    )


def calibrate_threshold(values: np.ndarray, quantile: float) -> float:
    """Pick a threshold as a quantile of detector values on calibration runs.

    Calibration seeds must be disjoint from evaluation seeds, and the quantile
    must be fixed before any evaluation run is looked at. This function exists so
    that choice is recorded in code rather than tuned by hand.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite calibration values")
    return float(np.quantile(v, quantile))
