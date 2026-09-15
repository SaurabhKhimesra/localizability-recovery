"""The traverse property: error over legs, not over metres.

This is the claim the marker half of the project rests on, so it is asserted
rather than described. A chain of dropped markers cannot remove drift already
accumulated, but it can stop drift accumulating: each fix ties the pose to the
previous anchor with an error that is the drop-time observation plus the current
one, so error grows with the square root of the number of legs instead of
linearly with distance.

The setup is deliberately bare. No registration, no map, no MuJoCo: a straight
run with a perfect prior except for a 2 percent scale bias, which is the one error
a tunnel cannot observe away and the reason the whole project exists. Markers go
on the wall every 10 m and stay in view until the next one is dropped, so that the
per-leg error is the measurement rather than a gap in coverage. Coverage gaps are
a detection-geometry question, measured in the experiments; this is a test of the
estimator's bookkeeping.
"""
import numpy as np
import pytest

from locrec.landmarks import LandmarkSpec, measurement_information, predicted_beam_count
from locrec.lidar import SPINNING_360
from locrec.odometry import Odometry, OdometryConfig
from locrec.se3 import make_T
from locrec.sim import MarkerDetection

STEP = 0.5
SPACING_M = 10.0
SCALE_BIAS = 0.02
WALL_OFFSET = np.array([0.0, 1.5, 0.3])


def sigma_horizontal(offset: np.ndarray) -> float:
    """Standard deviation of one observation of a strip at this offset, in metres.

    Taken from the same model the estimator uses, at the worst of its two
    horizontal axes, so the bound below is the one the estimator itself claims.
    """
    n = predicted_beam_count(offset, SPINNING_360, 1.0, 0.15)
    omega = measurement_information(offset, n, SPINNING_360, LandmarkSpec(), 1.0, 0.15)
    cov = np.linalg.pinv(omega[:2, :2])
    return float(np.sqrt(np.max(np.linalg.eigvalsh(cov))))


def traverse(n_legs: int, seed: int, markers: bool = True):
    """Run a straight traverse and return the along-track error at each leg end."""
    rng = np.random.default_rng(seed)
    odom = Odometry(
        make_T(np.eye(3), [0.0, 0.0, 0.7]),
        OdometryConfig(num_threads=1),
        dt=0.5,
        lidar_spec=SPINNING_360,
    )
    sigma = sigma_horizontal(WALL_OFFSET)

    steps_per_leg = int(SPACING_M / STEP)
    x_true = 0.0
    slot = -1
    marker_world = None
    errors = []

    for k in range(n_legs * steps_per_leg + 1):
        if markers and k % steps_per_leg == 0:
            # a strip goes on the wall beside the robot: the offset is known in the
            # sensor frame, so the anchor inherits exactly the error the estimate
            # has right now and nothing else
            slot += 1
            marker_world = np.array([x_true, 0.0, 0.7]) + WALL_OFFSET
            odom.register_landmark(slot, WALL_OFFSET.copy())
            errors.append(abs(odom.T[0, 3] - x_true))

        # The robot moves, and only then does it scan, which is the order the runner
        # uses. Building the observation at the pre-step pose instead makes every
        # fix chase a stale measurement and the estimate settles into a lag.
        x_next = x_true + STEP
        observations = []
        if markers and marker_world is not None:
            offset_true = marker_world - np.array([x_next, 0.0, 0.7])
            noise = np.zeros(3)
            noise[:2] = rng.normal(0.0, sigma, size=2)
            observations.append(
                MarkerDetection(
                    slot=slot,
                    point_sensor=offset_true + noise,
                    n_beams=predicted_beam_count(offset_true, SPINNING_360, 1.0, 0.15),
                    range_m=float(np.linalg.norm(offset_true)),
                )
            )

        # the prior overstates every step by the scale bias, which is systematic
        delta = make_T(np.eye(3), [STEP * (1.0 + SCALE_BIAS), 0.0, 0.0])
        odom.step(np.zeros((0, 3)), delta, observations)
        x_true = x_next

    return np.array(errors), odom, sigma, x_true


def test_each_fix_pulls_back_to_within_three_sigma_of_its_anchor():
    """A fix removes the drift accumulated since that anchor went down, and leaves
    the drop-time error, which is what makes this a traverse and not a loop closure."""
    errors, _, sigma, _ = traverse(n_legs=20, seed=0)
    steps = np.abs(np.diff(errors))
    bound = 3.0 * np.sqrt(2.0) * sigma  # drop-time observation plus the current one
    assert steps.max() < bound, (steps.max(), bound)


def test_the_scale_state_recovers_two_thirds_of_the_bias_and_no_more():
    """What the scale state actually does, which is not what it is supposed to do.

    The prior overstates every step by 2 percent and the estimator divides it back out, so the
    estimate should converge on 1.02. It converges on 1.0137 and stays there: 1.0135 at 10 legs
    and 1.0140 at 80, spread 0.0002 across seeds. It recovers about two thirds of the bias and
    leaves the rest, systematically. In the full pipeline the leftover is 37 percent of (s - 1)
    with R squared 0.94, and because s is drawn with mean 1 that leftover averages to zero across
    seeds, which is why it reads as an unbiased estimator until it is regressed on the truth
    (docs/failures.md number 35).

    The cause is in ``Odometry._update_scale``: the predicted innovation ``L * (scale - 1)``
    describes the residual a leg would show if the prior were not being corrected, while ``step``
    has already divided the prior by ``scale``, so the filter subtracts a prediction of a
    different measurement.

    This test pins the defect rather than the intent. Fixing the innovation model should make it
    fail, and it should then be updated deliberately, because the fix moves every marker number
    in the study.
    """
    recovered = []
    for seed in range(6):
        _, odom, _, _ = traverse(n_legs=20, seed=seed)
        recovered.append((odom.scale - 1.0) / SCALE_BIAS)
    recovered = np.array(recovered)
    assert recovered.std(ddof=1) < 0.01, ("systematic, not noise", recovered)
    assert 0.60 < recovered.mean() < 0.75, ("recovers about two thirds", recovered.mean())
    # and it does not creep to the truth given four times the run
    long = np.mean([(traverse(n_legs=80, seed=s)[1].scale - 1.0) / SCALE_BIAS for s in range(3)])
    assert long < 0.75, ("still short after 80 legs", long)


def test_the_scale_state_is_recovered_from_the_marker_residuals():
    """Fix D. The prior is biased by 2 percent and nothing measures that directly,
    but the along-track residual over a leg of length L is an observation of
    (s - 1) L, so a chain of markers identifies it."""
    _, odom, _, _ = traverse(n_legs=20, seed=0)
    assert abs(odom.scale - (1.0 + SCALE_BIAS)) < 0.01, odom.scale


def test_growth_after_the_scale_state_is_between_sqrt_and_linear():
    """The traverse claim as measured, which is still not quite the claim as stated.

    Before the scale state the growth was linear: the bias was systematic, the prior
    called it white noise, and every leg inherited the same steady-state lag. With
    the bias estimated, twenty legs against five is 5.1 and forty against ten is
    2.9, against 2.0 for a square-root traverse and 4.0 for a linear one. The first
    ratio is dominated by the estimator's convergence transient, since five legs is
    barely enough to identify the scale; the second is the shape of the curve once
    it has converged, and it is much closer to a traverse than to linear growth.

    The absolute numbers halve: twenty legs is 0.11 m where it was 0.24 m.
    """
    short = np.median([traverse(5, s)[0][-1] for s in range(5)])
    long = np.median([traverse(20, s)[0][-1] for s in range(5)])
    early = np.median([traverse(10, s)[0][-1] for s in range(5)])
    late = np.median([traverse(40, s)[0][-1] for s in range(5)])
    assert long / max(short, 1e-9) > 2.6, long / short  # the transient dominates here
    assert 2.0 < late / max(early, 1e-9) < 3.6, late / early  # sqrt 2.0, linear 4.0
    assert long < 0.2, long

    per_leg = long / 20.0
    assert per_leg < 3.0 * np.sqrt(2.0) * sigma_horizontal(WALL_OFFSET), per_leg


def test_the_traverse_beats_dead_reckoning_by_the_bias_it_removes():
    errors, odom, sigma, x_true = traverse(n_legs=20, seed=0)
    with_markers = abs(odom.T[0, 3] - x_true)
    _, dead, _, x_dead = traverse(n_legs=20, seed=0, markers=False)
    without = abs(dead.T[0, 3] - x_dead)
    assert without == pytest.approx(SCALE_BIAS * x_dead, rel=0.05)
    assert with_markers < 0.1 * without, (with_markers, without)


def test_an_anchor_deep_into_the_run_is_worth_as_much_as_an_early_one():
    """The point of the relative formulation. Weighted by an absolute covariance,
    the later anchor is swamped by the distance travelled before it existed."""
    errors, _, sigma, _ = traverse(n_legs=30, seed=1)
    early = np.abs(np.diff(errors[1:6]))
    late = np.abs(np.diff(errors[-6:-1]))
    assert late.mean() < 4.0 * max(early.mean(), 1e-6), (early.mean(), late.mean())
