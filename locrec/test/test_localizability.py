"""Tests on geometry whose degeneracy structure is known before running anything."""
import numpy as np
import pytest
import small_gicp

from locrec.localizability import (
    LocalizabilityConfig,
    analyse_hessian,
    calibrate_threshold,
)


def corridor_cloud(n=4000, half_width=2.0, height=3.0, length=40.0, seed=0):
    """Two parallel walls plus a floor, running along x. Translation along x is free."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-length / 2, length / 2, n)
    z = rng.uniform(0.0, height, n)
    y = np.where(rng.random(n) < 0.5, -half_width, half_width)
    walls = np.stack([x, y, z], axis=1)
    x2 = rng.uniform(-length / 2, length / 2, n)
    y2 = rng.uniform(-half_width, half_width, n)
    floor = np.stack([x2, y2, np.zeros(n)], axis=1)
    return np.vstack([walls, floor])


def room_cloud(n=4000, seed=0):
    """A closed box. Every translation direction is observable."""
    rng = np.random.default_rng(seed)
    pts = []
    for axis, extent in enumerate((6.0, 4.0, 3.0)):
        for sign in (-1.0, 1.0):
            p = rng.uniform(-1.0, 1.0, (n // 6, 3)) * np.array([6.0, 4.0, 3.0])
            p[:, axis] = sign * extent
            pts.append(p)
    return np.vstack(pts)


def hessian_of(cloud):
    res = small_gicp.align(
        cloud,
        cloud,
        downsampling_resolution=0.1,
        max_correspondence_distance=1.0,
        num_threads=2,
    )
    return np.array(res.H), int(res.num_inliers)


def test_hessian_block_ordering():
    """Pins the undocumented small_gicp convention: H is [rotation, translation].

    In a corridor along x the free direction is a *translation* along x. If the
    ordering were the other way round this assertion would land in the rotation
    block and every localizability number downstream would be nonsense.
    """
    H, n = hessian_of(corridor_cloud())
    loc = analyse_hessian(H, n)
    assert abs(loc.weak_direction[0]) > 0.99, loc.weak_direction
    assert loc.ratio < 1e-2
    # the rotational block must not be the degenerate one here
    assert loc.rot_ratio > loc.ratio


def test_a_closed_room_is_not_degenerate():
    H, n = hessian_of(room_cloud())
    loc = analyse_hessian(H, n)
    assert loc.ratio > 0.05, loc.eigenvalues


def test_corridor_is_far_more_degenerate_than_a_room():
    corridor = analyse_hessian(*hessian_of(corridor_cloud()))
    room = analyse_hessian(*hessian_of(room_cloud()))
    assert corridor.ratio < room.ratio / 10.0


def test_weak_direction_follows_the_corridor_when_it_is_rotated():
    yaw = np.deg2rad(35.0)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    loc = analyse_hessian(*hessian_of(corridor_cloud() @ R.T))
    expected = R @ np.array([1.0, 0.0, 0.0])
    assert abs(float(loc.weak_direction @ expected)) > 0.99


def test_in_frame_rotates_the_directions():
    loc = analyse_hessian(*hessian_of(corridor_cloud()))
    yaw = np.deg2rad(90.0)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    body = loc.in_frame(R)
    np.testing.assert_allclose(body.weak_direction, R.T @ loc.weak_direction, atol=1e-12)
    np.testing.assert_allclose(body.eigenvalues, loc.eigenvalues)


def test_eigenvalues_are_ascending_and_nonnegative():
    loc = analyse_hessian(*hessian_of(corridor_cloud()))
    assert np.all(np.diff(loc.eigenvalues) >= 0)
    assert np.all(loc.eigenvalues >= 0)
    assert loc.eigenvalues[0] / loc.eigenvalues[2] == pytest.approx(loc.ratio)


def test_ratio_is_invariant_to_scaling_the_information():
    H, n = hessian_of(corridor_cloud())
    a = analyse_hessian(H, n)
    b = analyse_hessian(H * 1000.0, n)
    assert a.ratio == pytest.approx(b.ratio)
    assert b.lambda_min_per_point == pytest.approx(a.lambda_min_per_point * 1000.0)


def test_threshold_decision():
    loc = analyse_hessian(*hessian_of(corridor_cloud()))
    assert loc.is_degenerate(LocalizabilityConfig(ratio_threshold=0.05))
    assert not loc.is_degenerate(LocalizabilityConfig(ratio_threshold=1e-8))
    # the absolute guard can fire on its own
    assert loc.is_degenerate(
        LocalizabilityConfig(ratio_threshold=1e-8, min_lambda_per_point=1e12)
    )


def test_calibrate_threshold_is_a_quantile_and_validates_input():
    v = np.arange(101, dtype=float)
    assert calibrate_threshold(v, 0.1) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        calibrate_threshold(v, 1.0)
    with pytest.raises(ValueError):
        calibrate_threshold(np.array([np.nan]), 0.5)


def test_analyse_hessian_rejects_wrong_shape():
    with pytest.raises(ValueError):
        analyse_hessian(np.eye(4), 10)
