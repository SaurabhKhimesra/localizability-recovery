"""Where the sensor should look, on geometry whose answer is known in advance."""
import numpy as np
import pytest

from locrec.gaze import (
    ForwardGaze,
    GlanceGaze,
    GreedyGaze,
    SweepGaze,
    map_normals,
    predicted_information,
)
from locrec.lidar import LIMITED_FOV, LidarSpec

WIDE = LidarSpec(
    n_azimuth=90, n_elevation=8, fov_azimuth_deg=90.0, fov_elevation_deg=60.0, max_range=30.0
)


def wall(axis: str, offset: float, n: int = 400, span: float = 20.0, seed: int = 0):
    """A flat wall with its exact normals, so the expected answer is arithmetic."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-span / 2, span / 2, n)
    z = rng.uniform(0.0, 2.5, n)
    if axis == "x":  # a wall facing along x, i.e. a dead end across the tunnel
        pts = np.stack([np.full(n, offset), a, z], axis=1)
        nrm = np.tile([1.0, 0.0, 0.0], (n, 1))
    else:  # a side wall, normal across the tunnel
        pts = np.stack([a, np.full(n, offset), z], axis=1)
        nrm = np.tile([0.0, 1.0, 0.0], (n, 1))
    return pts, nrm


# ---- the information prediction -------------------------------------------


def test_two_parallel_side_walls_constrain_across_and_not_along():
    """The whole premise. Looking down a corridor tells you nothing about how far
    along it you are."""
    left = wall("y", 2.0)
    right = wall("y", -2.0, seed=1)
    pts = np.vstack([left[0], right[0]])
    nrm = np.vstack([left[1], right[1]])
    H = predicted_information(pts, nrm, np.zeros(3), 0.0, WIDE)
    assert H[1, 1] > 50.0 * max(H[0, 0], 1e-9), H


def test_a_wall_across_the_tunnel_constrains_along_it():
    pts, nrm = wall("x", 12.0)
    H = predicted_information(pts, nrm, np.zeros(3), 0.0, WIDE)
    assert H[0, 0] > 50.0 * max(H[1, 1], 1e-9), H


def test_points_outside_the_field_of_view_do_not_count():
    pts, nrm = wall("x", 12.0)
    ahead = predicted_information(pts, nrm, np.zeros(3), 0.0, WIDE)
    behind = predicted_information(pts, nrm, np.zeros(3), np.pi, WIDE)
    assert np.linalg.eigvalsh(ahead)[-1] > 0
    assert np.allclose(behind, 0.0)


def test_points_beyond_range_do_not_count():
    pts, nrm = wall("x", 100.0)
    H = predicted_information(pts, nrm, np.zeros(3), 0.0, WIDE)
    assert np.allclose(H, 0.0)


def test_grazing_incidence_is_charged_for():
    """A patch seen edge on returns few beams and constrains little, so it is
    weighted by the cosine to the line of sight rather than counted in full."""
    n = 300
    rng = np.random.default_rng(0)
    patch = np.stack(
        [np.full(n, 10.0), rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)], axis=1
    )
    face_on = predicted_information(
        patch, np.tile([1.0, 0.0, 0.0], (n, 1)), np.zeros(3), 0.0, WIDE
    )
    edge_on = predicted_information(
        patch, np.tile([0.0, 1.0, 0.0], (n, 1)), np.zeros(3), 0.0, WIDE
    )
    assert np.trace(face_on) > 5.0 * np.trace(edge_on)


def test_empty_map_is_no_information_not_an_error():
    H = predicted_information(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(3), 0.0, WIDE)
    assert H.shape == (3, 3)
    assert np.allclose(H, 0.0)


def test_normals_of_a_plane_point_along_its_axis():
    pts, _ = wall("y", 2.0, n=600)
    n = map_normals(pts)
    assert n.shape == pts.shape
    good = np.abs(n[:, 1]) > 0.9
    assert good.mean() > 0.9, good.mean()


def test_normals_of_a_tiny_cloud_are_zero_not_a_crash():
    assert map_normals(np.zeros((3, 3))).shape == (3, 3)


# ---- the policies ----------------------------------------------------------


class _Plat:
    def __init__(self, yaw=0.0, track=0.0):
        self.yaw = yaw
        self._track = track

    def track_yaw(self):
        return self._track

    def pose(self):
        T = np.eye(4)
        return T


class _Sim:
    def __init__(self, spec):
        self.lidar = type("L", (), {"spec": spec})()


class _Loc:
    """Stands in for the detector output. ``eigenvectors`` and ``eigenvalues``
    describe what the robot already knows, which is what the gaze objective adds
    to."""

    def __init__(self, ratio, weak_axis=0, strong=1.0, weak=1e-3):
        self.ratio = ratio
        self.eigenvectors = np.eye(3)
        lam = np.full(3, strong)
        lam[weak_axis] = weak
        order = np.argsort(lam)
        self.eigenvalues = lam[order]
        self.eigenvectors = np.eye(3)[:, order]


class _Ctx:
    def __init__(self, **kw):
        self.__dict__.setdefault("T_est", np.eye(4))
        self.__dict__.setdefault("estimated_distance", 0.0)
        self.__dict__.update(kw)


def _ctx(ratio=1e-4, points=None, normals=None, yaw=0.0, track=0.0, spec=WIDE,
         weak_axis=0):  # noqa: PLR0913
    return _Ctx(
        localizability=_Loc(ratio, weak_axis=weak_axis),
        platform=_Plat(yaw, track),
        sim=_Sim(spec),
        local_map_points=points,
        local_map_normals=normals,
    )


def test_forward_gaze_tracks_the_path():
    assert ForwardGaze()(_ctx(track=0.7))["yaw_target"] == pytest.approx(0.7)


def test_sweep_oscillates_and_returns():
    p = SweepGaze(amplitude_deg=60.0, period_m=20.0)
    offsets = [
        p(_ctx(track=0.0, ratio=1.0)).__getitem__("yaw_target")
        for _ in range(1)
    ]
    assert offsets[0] == pytest.approx(0.0, abs=1e-9)
    ctx = _ctx(track=0.0)
    ctx.estimated_distance = 5.0
    assert p(ctx)["yaw_target"] == pytest.approx(np.deg2rad(60.0), abs=1e-6)
    ctx.estimated_distance = 15.0
    assert p(ctx)["yaw_target"] == pytest.approx(-np.deg2rad(60.0), abs=1e-6)


def test_greedy_stays_forward_when_localizability_is_healthy():
    pts, nrm = wall("x", 12.0)
    out = GreedyGaze(ratio_threshold=1e-3)(_ctx(ratio=0.5, points=pts, normals=nrm, track=0.3))
    assert out["yaw_target"] == pytest.approx(0.3)


def test_greedy_turns_toward_the_only_structure_that_helps():
    """Two side walls ahead, and the angled back wall of a side branch off to the
    left. The side walls say nothing about along-track position. The branch wall
    does, and it sits outside the forward cone, so the sensor has to turn.

    The branch wall is angled rather than square across the tunnel because a flat
    surface whose normal is along x is invisible from a bearing of ninety degrees:
    seeing a face and being constrained by it are the same condition.
    """
    rng = np.random.default_rng(3)
    side_a = wall("y", 2.0, span=40.0)
    side_b = wall("y", -2.0, span=40.0, seed=1)
    n = 400
    branch = np.stack(
        [rng.uniform(5.0, 7.0, n), rng.uniform(13.0, 15.0, n), rng.uniform(0, 2.5, n)],
        axis=1,
    )
    branch_n = np.tile(np.array([-1.0, -1.0, 0.0]) / np.sqrt(2.0), (n, 1))
    pts = np.vstack([side_a[0], side_b[0], branch])
    nrm = np.vstack([side_a[1], side_b[1], branch_n])

    out = GreedyGaze(ratio_threshold=1e-3)(_ctx(points=pts, normals=nrm))
    yaw = np.rad2deg(out["yaw_target"]) % 360.0
    assert 25.0 <= yaw <= 120.0, yaw  # the branch sits at a bearing of about 65 deg


def test_greedy_falls_back_to_forward_with_an_empty_map():
    out = GreedyGaze(ratio_threshold=1e-3)(
        _ctx(points=np.zeros((0, 3)), normals=np.zeros((0, 3)), track=0.2)
    )
    assert out["yaw_target"] == pytest.approx(0.2)


def test_greedy_candidates_cover_the_full_circle():
    """A structure directly behind must be reachable, or the policy cannot use
    anything it has already driven past."""
    # a compact patch, so there is one right answer rather than a band of
    # near-equal ones that the smallest-slew tie-break would legitimately split
    behind = wall("x", -10.0, n=500, span=3.0)
    pts = behind[0]
    nrm = behind[1]
    out = GreedyGaze(ratio_threshold=1e-3)(_ctx(points=pts, normals=nrm))
    yaw = np.rad2deg(out["yaw_target"])
    # the property that matters is that the patch ends up inside the cone, not
    # that the sensor centres on it: a tie goes to the smaller slew by design
    off_boresight = abs((yaw - 180.0 + 180.0) % 360.0 - 180.0)
    assert off_boresight <= WIDE.fov_azimuth_deg / 2, yaw


def test_limited_fov_spec_is_the_one_milestone_1_measured():
    assert LIMITED_FOV.fov_azimuth_deg == 90.0
    assert LIMITED_FOV.max_range == 30.0


def test_forward_restricted_greedy_refuses_to_look_back():
    """The repair for the milestone 3 failure. The only structure worth looking at
    is behind, and the restricted policy still will not turn round, because the
    tunnel it is flying into is the tunnel it has to keep mapping."""
    behind = wall("x", -10.0, n=500, span=3.0)
    out = GreedyGaze(ratio_threshold=1e-3, cone_deg=60.0)(
        _ctx(points=behind[0], normals=behind[1], track=0.0)
    )
    yaw = np.rad2deg(out["yaw_target"])
    assert abs((yaw + 180.0) % 360.0 - 180.0) <= 60.0 + 1e-6, yaw


def test_the_cone_is_measured_from_travel_not_from_where_the_sensor_points():
    """Otherwise the sensor walks: each step's cone is centred on the last step's
    choice, and sixty degrees at a time it ends up facing backwards anyway."""
    behind = wall("x", -10.0, n=500, span=3.0)
    policy = GreedyGaze(ratio_threshold=1e-3, cone_deg=60.0)
    yaw = 0.0
    for _ in range(6):
        yaw = policy(_ctx(points=behind[0], normals=behind[1], track=0.0, yaw=yaw))[
            "yaw_target"
        ]
    assert abs(np.rad2deg(yaw)) <= 60.0 + 1e-6, np.rad2deg(yaw)


# ---- the glance policy -----------------------------------------------------


class _Spec:
    step_length = 0.5
    speed_mps = 1.0


def _glance_ctx(**kw):
    ctx = _ctx(**kw)
    ctx.platform.spec = _Spec()
    return ctx


def test_glance_turns_to_structure_behind_then_comes_back():
    """Bounded look-back: three seconds at half a second per step is six steps."""
    behind = wall("x", -10.0, n=500, span=3.0)
    p = GlanceGaze(ratio_threshold=1e-3, cooldown_s=2.0)
    yaws = [
        p(_glance_ctx(points=behind[0], normals=behind[1], track=0.0))["yaw_target"]
        for _ in range(10)
    ]
    turned = [abs((np.rad2deg(y) + 180) % 360 - 180) > 60.0 for y in yaws]
    assert turned[0], np.rad2deg(yaws[0])
    assert sum(turned) == 6, (sum(turned), np.rad2deg(yaws))
    assert not any(turned[6:]), np.rad2deg(yaws)


def test_glance_does_not_fire_where_the_geometry_is_healthy():
    behind = wall("x", -10.0, n=500, span=3.0)
    p = GlanceGaze(ratio_threshold=1e-3, cooldown_s=0.0)
    out = p(_glance_ctx(ratio=0.5, points=behind[0], normals=behind[1], track=0.3))
    assert out["yaw_target"] == pytest.approx(0.3)
    assert p.glance_fraction == 0.0


def test_glance_does_not_fire_when_there_is_nothing_behind_worth_seeing():
    """Two parallel side walls: looking back sees the same two walls, so a glance
    would spend slew for nothing."""
    left = wall("y", 2.0, span=40.0)
    right = wall("y", -2.0, span=40.0, seed=1)
    pts = np.vstack([left[0], right[0]])
    nrm = np.vstack([left[1], right[1]])
    p = GlanceGaze(ratio_threshold=1e-3, cooldown_s=0.0)
    out = p(_glance_ctx(points=pts, normals=nrm, track=0.0))
    assert out["yaw_target"] == pytest.approx(0.0)


def test_glance_respects_its_cooldown():
    behind = wall("x", -10.0, n=500, span=3.0)
    p = GlanceGaze(ratio_threshold=1e-3, hold_s=1.0, cooldown_s=5.0)
    seen = []
    for _ in range(24):
        y = p(_glance_ctx(points=behind[0], normals=behind[1], track=0.0))["yaw_target"]
        seen.append(abs((np.rad2deg(y) + 180) % 360 - 180) > 60.0)
    # two steps of glance per burst, ten steps of cooldown between bursts
    assert 0.1 < p.glance_fraction < 0.35, p.glance_fraction
    assert any(seen) and not all(seen)
