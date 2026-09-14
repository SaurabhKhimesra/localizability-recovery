import numpy as np
import pytest

from locrec.worlds import WorldSpec, build_world

SMALL = WorldSpec(length=40.0, n_curves=1, n_junctions=1, n_niches=2, n_marker_slots=8)


def test_generator_is_deterministic_in_seed():
    a = build_world(7, SMALL)
    b = build_world(7, SMALL)
    assert a.xml == b.xml
    np.testing.assert_array_equal(a.centreline, b.centreline)
    assert build_world(8, SMALL).xml != a.xml


def test_centreline_is_arclength_parameterised():
    w = build_world(3, SMALL)
    seg = np.linalg.norm(np.diff(w.centreline, axis=0), axis=1)
    np.testing.assert_allclose(seg, w.spec.ds, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(w.s, np.arange(len(w.s)) * w.spec.ds)


def test_heading_matches_the_tangent():
    w = build_world(4, SMALL)
    d = np.diff(w.centreline, axis=0)
    tangent = np.arctan2(d[:, 1], d[:, 0])
    np.testing.assert_allclose(tangent, w.heading[:-1], atol=1e-9)


def test_width_stays_inside_its_bounds():
    w = build_world(5, SMALL)
    assert w.width.min() >= SMALL.width_min - 1e-9
    assert w.width.max() <= SMALL.width_max + 1e-9


def test_features_are_placed_and_flagged():
    w = build_world(1, WorldSpec(length=120.0, n_junctions=2, n_niches=4, n_marker_slots=4))
    assert len(w.junctions) == 2
    assert len(w.niches) == 4
    assert w.feature_mask.any()
    lead = int(w.spec.straight_lead_in / w.spec.ds)
    assert not w.feature_mask[:lead].any(), "lead-in must stay featureless"
    for f in w.junctions + w.niches:
        assert w.feature_mask[f["index"] : f["index"] + f["n_samples"]].all()


def test_pose_at_interpolates_and_ends_match():
    w = build_world(2, SMALL)
    p0, h0 = w.pose_at(0.0)
    np.testing.assert_allclose(p0[:2], w.centreline[0], atol=1e-9)
    assert h0 == pytest.approx(w.heading[0], abs=1e-9)
    p_end, _ = w.pose_at(w.total_length)
    np.testing.assert_allclose(p_end[:2], w.centreline[-1], atol=1e-6)


def test_spec_rejects_inconsistent_width():
    with pytest.raises(ValueError):
        WorldSpec(width_mean=1.0, width_min=2.0, width_max=3.0)
