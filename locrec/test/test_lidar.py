"""Ray casting against geometry whose answer is known in closed form."""
import mujoco
import numpy as np
import pytest

from locrec.lidar import SPINNING_360, Lidar, LidarSpec
from locrec.se3 import euler_zyx, make_T
from locrec.sim import TunnelSim
from locrec.worlds import WorldSpec

BOX_ROOM = """
<mujoco>
  <worldbody>
    <geom name="xp" type="plane" pos="5 0 0" zaxis="-1 0 0" size="10 10 0.1"/>
    <geom name="xm" type="plane" pos="-5 0 0" zaxis="1 0 0" size="10 10 0.1"/>
    <geom name="yp" type="plane" pos="0 3 0" zaxis="0 -1 0" size="10 10 0.1"/>
    <geom name="ym" type="plane" pos="0 -3 0" zaxis="0 1 0" size="10 10 0.1"/>
    <geom name="zp" type="plane" pos="0 0 2" zaxis="0 0 -1" size="10 10 0.1"/>
    <geom name="zm" type="plane" pos="0 0 -2" zaxis="0 0 1" size="10 10 0.1"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def room():
    m = mujoco.MjModel.from_xml_string(BOX_ROOM)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_horizontal_ring_measures_the_known_box(room):
    m, d = room
    spec = LidarSpec(
        n_azimuth=4,
        n_elevation=1,
        fov_azimuth_deg=360.0,
        fov_elevation_deg=0.0,
        max_range=50.0,
        range_sigma=0.0,
    )
    scan = Lidar(spec, seed=0).scan(m, d, np.eye(4))
    assert len(scan) == 4
    # beams point at -x, -y, +x, +y from the ring construction
    got = np.sort(np.round(scan.ranges, 6))
    np.testing.assert_allclose(got, [3.0, 3.0, 5.0, 5.0])


def test_rotating_the_sensor_rotates_the_points_not_the_world(room):
    m, d = room
    spec = LidarSpec(
        n_azimuth=8,
        n_elevation=1,
        fov_azimuth_deg=360.0,
        fov_elevation_deg=0.0,
        max_range=50.0,
        range_sigma=0.0,
    )
    lidar = Lidar(spec, seed=0)
    a = lidar.scan(m, d, np.eye(4))
    b = lidar.scan(m, d, make_T(euler_zyx(np.pi / 2), np.zeros(3)))
    # same set of world points, so the same multiset of ranges
    np.testing.assert_allclose(np.sort(a.ranges), np.sort(b.ranges), atol=1e-9)
    # but the points themselves are expressed in the rotated sensor frame
    assert not np.allclose(np.sort(a.points, axis=0), np.sort(b.points, axis=0))


def test_translating_the_sensor_shifts_the_ranges(room):
    m, d = room
    spec = LidarSpec(
        n_azimuth=4,
        n_elevation=1,
        fov_azimuth_deg=360.0,
        fov_elevation_deg=0.0,
        max_range=50.0,
        range_sigma=0.0,
    )
    scan = Lidar(spec, seed=0).scan(m, d, make_T(np.eye(3), [1.0, 0.0, 0.0]))
    np.testing.assert_allclose(np.sort(np.round(scan.ranges, 6)), [3.0, 3.0, 4.0, 6.0])


def test_max_range_cuts_beams_off(room):
    m, d = room
    spec = LidarSpec(
        n_azimuth=4,
        n_elevation=1,
        fov_azimuth_deg=360.0,
        fov_elevation_deg=0.0,
        max_range=4.0,
        range_sigma=0.0,
    )
    scan = Lidar(spec, seed=0).scan(m, d, np.eye(4))
    assert len(scan) == 2
    np.testing.assert_allclose(scan.ranges, [3.0, 3.0])


def test_limited_fov_sees_only_the_forward_wedge(room):
    m, d = room
    spec = LidarSpec(
        n_azimuth=9,
        n_elevation=1,
        fov_azimuth_deg=90.0,
        fov_elevation_deg=0.0,
        max_range=50.0,
        range_sigma=0.0,
    )
    scan = Lidar(spec, seed=0).scan(m, d, np.eye(4))
    assert len(scan) == 9
    assert (scan.directions[:, 0] > 0).all()


def test_range_noise_is_the_only_nondeterminism(room):
    m, d = room
    quiet = LidarSpec(n_azimuth=16, n_elevation=4, max_range=50.0, range_sigma=0.0)
    a = Lidar(quiet, seed=0).scan(m, d, np.eye(4))
    b = Lidar(quiet, seed=1).scan(m, d, np.eye(4))
    np.testing.assert_allclose(a.ranges, b.ranges)

    noisy = LidarSpec(n_azimuth=16, n_elevation=4, max_range=50.0, range_sigma=0.05)
    c = Lidar(noisy, seed=0).scan(m, d, np.eye(4))
    assert not np.allclose(a.ranges, c.ranges)
    assert abs(np.mean(c.ranges - a.ranges)) < 0.05


def test_markers_are_detected_by_geom_id():
    spec = WorldSpec(length=30.0, n_curves=0, n_junctions=0, n_niches=0, n_marker_slots=4)
    sim = TunnelSim(0, spec, SPINNING_360)
    T = make_T(np.eye(3), [5.0, 0.0, 0.7])

    before = sim.scan(T)
    assert sim.is_marker_geom(before.geom_ids).sum() == 0

    pos, _ = sim.world.pose_at(8.0, 1.0)
    sim.place_marker(pos)
    after = sim.scan(T)
    assert sim.is_marker_geom(after.geom_ids).sum() > 0
    assert sim.n_markers_placed == 1
    np.testing.assert_allclose(sim.marker_positions()[0], pos, atol=1e-9)


def test_marker_slots_are_finite():
    spec = WorldSpec(length=20.0, n_curves=0, n_junctions=0, n_niches=0, n_marker_slots=2)
    sim = TunnelSim(0, spec)
    for _ in range(2):
        sim.place_marker(np.array([0.0, 0.0, 1.0]))
    with pytest.raises(RuntimeError):
        sim.place_marker(np.array([0.0, 0.0, 1.0]))
