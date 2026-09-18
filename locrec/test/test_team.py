"""The team property: a vehicle that mounts nothing still traverses, in the frame of one that did.

``test_traverse`` asserts what a chain of markers is worth to the robot that dropped it. This
asserts what the same chain is worth to a second vehicle that carries no markers at all and has
never seen the first: told only where each strip is, how well the first robot knew that, and which
way it faces, it should hold the traverse property in the first robot's frame.

The setup is the traverse bench with the drops taken away. The follower has the same 2 percent
scale bias, the one error a tunnel cannot observe away, and the strips are already on the wall
when it arrives, each carrying the error the leader had when it bolted it on. Nothing here is a
loop closure: the follower cannot be more right about the tunnel than the leader was, and the
tests below say so, in the leader's frame and then in the world's.
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
"""Where a strip sits relative to the vehicle that mounts it, and the offset the follower's own
observations are built around."""


def sigma_horizontal(offset: np.ndarray) -> float:
    n = predicted_beam_count(offset, SPINNING_360, 1.0, 0.15)
    omega = measurement_information(offset, n, SPINNING_360, LandmarkSpec(), 1.0, 0.15)
    cov = np.linalg.pinv(omega[:2, :2])
    return float(np.sqrt(np.max(np.linalg.eigvalsh(cov))))


def leader_strips(n_legs: int, rng, drift_per_leg: float = 0.0):
    """Where the leader believes it put each strip, and how well it knew that.

    Returns (positions it reports, their true positions, the 2x2 it reports with each).
    ``drift_per_leg`` is the leader's own chain error accumulating along the tunnel, which the
    follower inherits and cannot see: the leader's frame is simply where the leader thinks it is.
    """
    sigma = sigma_horizontal(WALL_OFFSET)
    R_drop = np.diag([sigma**2, sigma**2])
    reported, truth = [], []
    for k in range(n_legs + 1):
        true = np.array([k * SPACING_M, 0.0, 0.7]) + WALL_OFFSET
        err = np.zeros(3)
        err[:2] = rng.normal(0.0, sigma, size=2)
        err[0] += k * drift_per_leg
        truth.append(true)
        reported.append(true + err)
    return reported, truth, R_drop


def follow(n_legs: int, seed: int, team: bool = True, drift_per_leg: float = 0.0):
    """Run the follower down the tunnel and return its along-track error at each leg end.

    Error is measured against the leader's frame, that is against where the leader said the
    strips are, because that is the frame the follower is localizing in. The final position is
    also returned so the world-frame question can be asked separately.
    """
    rng = np.random.default_rng(seed)
    reported, truth, R_drop = leader_strips(n_legs, rng, drift_per_leg)
    odom = Odometry(
        make_T(np.eye(3), [0.0, 0.0, 0.7]),
        OdometryConfig(num_threads=1),
        dt=0.5,
        lidar_spec=SPINNING_360,
    )
    sigma = sigma_horizontal(WALL_OFFSET)
    # the strips face back the way the leader came, which is what the follower is told
    normal = np.array([0.0, 1.0])

    steps_per_leg = int(SPACING_M / STEP)
    x_true = 0.0
    slot = -1
    errors = []

    for k in range(n_legs * steps_per_leg + 1):
        if k % steps_per_leg == 0:
            slot += 1
            if team:
                # the message: position, drop-time covariance, facing. Nothing else.
                odom.register_foreign_landmark(slot, reported[slot], R_drop, normal)
            # the leader's frame is offset from the world by its own error at this strip
            frame_offset = reported[slot][0] - truth[slot][0]
            errors.append(abs(odom.T[0, 3] - (x_true + frame_offset)))

        x_next = x_true + STEP
        observations = []
        if team and slot >= 0:
            offset_true = truth[slot] - np.array([x_next, 0.0, 0.7])
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
        delta = make_T(np.eye(3), [STEP * (1.0 + SCALE_BIAS), 0.0, 0.0])
        odom.step(np.zeros((0, 3)), delta, observations)
        x_true = x_next

    return np.array(errors), odom, sigma, x_true


def test_a_follower_that_mounts_nothing_still_traverses():
    """The claim. Solo the follower's error is the scale bias times the distance; in the
    leader's frame, on the leader's strips, it stops growing with distance."""
    errors, odom, _, x_true = follow(n_legs=20, seed=0)
    _, solo, _, x_solo = follow(n_legs=20, seed=0, team=False)
    alone = abs(solo.T[0, 3] - x_solo)
    assert alone == pytest.approx(SCALE_BIAS * x_solo, rel=0.05)
    assert errors[-1] < 0.1 * alone, (errors[-1], alone)


def test_the_follower_never_beats_the_leader_that_placed_the_strips():
    """A borrowed frame is still borrowed. With the leader's own chain running 2 cm long per leg,
    the follower tracks the leader's frame just as well and the world's just as badly, which is
    the honest result and the reason the leader's chain error is the floor on the team."""
    errors, odom, _, x_true = follow(n_legs=20, seed=0, drift_per_leg=0.02)
    in_frame = errors[-1]
    in_world = abs(odom.T[0, 3] - x_true)
    assert in_frame < 0.2, in_frame
    assert in_world > 0.3, in_world
    assert in_world == pytest.approx(20 * 0.02, abs=0.2), in_world


def test_each_fix_pulls_back_to_within_three_sigma_of_the_strip_it_used():
    """Same bound as the traverse: the leader's drop-time observation plus the follower's own."""
    errors, _, sigma, _ = follow(n_legs=20, seed=0)
    steps = np.abs(np.diff(errors))
    assert steps.max() < 3.0 * np.sqrt(2.0) * sigma, (steps.max(), 3.0 * np.sqrt(2.0) * sigma)


def test_growth_is_flat_in_the_leaders_frame_not_linear():
    short = np.median([follow(5, s)[0][-1] for s in range(5)])
    long = np.median([follow(20, s)[0][-1] for s in range(5)])
    assert long / max(short, 1e-9) < 4.0, long / max(short, 1e-9)  # linear would be 4.0
    assert long < 0.2, long


def test_a_strip_told_of_early_and_seen_late_is_not_charged_for_the_whole_run():
    """The bookkeeping that makes this work. A teammate's strips all live in one frame, so being
    fixed on any of them fixes the pose against all of them. Charging each strip for the odometry
    since the robot was told about it would make a strip first seen deep in the tunnel look
    worthless exactly where it is the only thing on offer, which is the same mistake the relative
    formulation exists to avoid."""
    rng = np.random.default_rng(0)
    reported, truth, R_drop = leader_strips(2, rng)
    odom = Odometry(make_T(np.eye(3), [0.0, 0.0, 0.7]), OdometryConfig(num_threads=1),
                    dt=0.5, lidar_spec=SPINNING_360)
    normal = np.array([0.0, 1.0])
    # told about both at the start, sees the first at once and the second much later
    odom.register_foreign_landmark(0, reported[0], R_drop, normal)
    odom.register_foreign_landmark(1, reported[1], R_drop, normal)

    def see(slot, x):
        offset = truth[slot] - np.array([x, 0.0, 0.7])
        return [MarkerDetection(slot=slot, point_sensor=offset,
                                n_beams=predicted_beam_count(offset, SPINNING_360, 1.0, 0.15),
                                range_m=float(np.linalg.norm(offset)))]

    x = 0.0
    for _ in range(int(SPACING_M / STEP)):
        x += STEP
        odom.step(np.zeros((0, 3)), make_T(np.eye(3), [STEP * (1.0 + SCALE_BIAS), 0.0, 0.0]), see(0, x))

    lm0, lm1 = odom.landmarks.get(0), odom.landmarks.get(1)
    q0 = odom._relative_covariance(lm0)
    q1 = odom._relative_covariance(lm1)
    np.testing.assert_allclose(q0, q1, atol=1e-12)
    assert float(np.trace(q1[1:, 1:])) < (SCALE_BIAS * SPACING_M) ** 2, np.trace(q1[1:, 1:])


def test_a_teammates_strip_is_never_re_surveyed():
    """Its coordinates are the shared frame's definition. Moving them to this robot's estimate
    would redefine the frame the teammate is still using, without telling it."""
    rng = np.random.default_rng(0)
    reported, truth, R_drop = leader_strips(1, rng)
    odom = Odometry(make_T(np.eye(3), [0.0, 0.0, 0.7]),
                    OdometryConfig(num_threads=1, resurvey_anchors=True),
                    dt=0.5, lidar_spec=SPINNING_360)
    odom.register_foreign_landmark(0, reported[0], R_drop, np.array([0.0, 1.0]))
    before = odom.landmarks.get(0).position.copy()
    x = 0.0
    for _ in range(10):
        x += STEP
        offset = truth[0] - np.array([x, 0.0, 0.7])
        obs = [MarkerDetection(slot=0, point_sensor=offset,
                               n_beams=predicted_beam_count(offset, SPINNING_360, 1.0, 0.15),
                               range_m=float(np.linalg.norm(offset)))]
        odom.step(np.zeros((0, 3)), make_T(np.eye(3), [STEP, 0.0, 0.0]), obs)
    np.testing.assert_allclose(odom.landmarks.get(0).position, before, atol=1e-12)


def test_the_facing_has_to_be_carried_because_the_fit_bias_is_mirrored():
    """Why the message carries a normal rather than letting the reader assume one.

    The end face the strip shows is the one toward the sensor, so the fit reads the strip short
    on the approach and long once it is passed, and the correction along the wall flips sign with
    it. A vehicle that inferred the facing from its own pose, the way a vehicle does for a strip
    it mounted itself, would apply that correction backwards on strips someone else left facing
    the other way, which is worse than not correcting at all.
    """
    from locrec.landmarks import strip_fit_bias

    normal = np.array([0.0, -1.0])       # the face points across the tunnel, at the track
    axis = np.array([1.0, 0.0])          # the strip's width axis, along the tunnel
    ahead = strip_fit_bias(np.array([4.0, 1.5]), normal, 4.27, SPINNING_360, 0.15, 0.02, 1.0, 0.2)
    passed = strip_fit_bias(np.array([-4.0, 1.5]), normal, 4.27, SPINNING_360, 0.15, 0.02, 1.0, 0.2)

    assert abs(ahead @ axis) > 1e-4, ahead
    assert (ahead @ axis) == pytest.approx(-(passed @ axis), abs=1e-12)
    assert (ahead @ normal) == pytest.approx(passed @ normal, abs=1e-12)

    # and a face the sensor cannot see gets no correction at all, rather than a mirrored one
    away = strip_fit_bias(np.array([4.0, 1.5]), -normal, 4.27, SPINNING_360, 0.15, 0.02, 1.0, 0.2)
    np.testing.assert_allclose(away, np.zeros(2), atol=1e-12)
