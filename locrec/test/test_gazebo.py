"""The Gazebo export, without a Gazebo: geometry, beams, placement, and the harness hook.

The live half, a server rendering scans, is measured by
``experiments/calibrate_thresholds_gazebo.py``, which checks beam order and rotation on
every run; nothing here needs gz installed.
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from locrec import LIMITED_FOV, SPINNING_360, NoMarkers, RunConfig, TunnelSim, run_pass
from locrec import gazebo as gzl
from locrec.lidar import _ray_directions
from locrec.worlds import WorldSpec

SMALL = WorldSpec(length=40.0, n_curves=1, n_junctions=1, n_niches=2, n_marker_slots=4)


def test_every_shell_box_of_the_mujoco_model_is_exported_at_its_pose_and_full_size():
    sim = TunnelSim(3, SMALL, SPINNING_360)
    boxes = {b.name: b for b in gzl.shell_boxes(sim.world)}
    shell = [i for i in range(sim.model.ngeom)
             if not (mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("marker")]
    assert len(boxes) == len(shell)
    for i in shell:
        name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, i)
        b = boxes[name]
        np.testing.assert_allclose(b.size, 2.0 * sim.model.geom_size[i], atol=1e-4)
        np.testing.assert_allclose(b.centre, sim.data.geom_xpos[i], atol=1e-4)
        R = sim.data.geom_xmat[i].reshape(3, 3)
        assert abs(np.arctan2(np.sin(np.arctan2(R[1, 0], R[0, 0]) - b.yaw),
                              np.cos(np.arctan2(R[1, 0], R[0, 0]) - b.yaw))) < 1e-5


@pytest.mark.parametrize("spec", [SPINNING_360, LIMITED_FOV], ids=["360", "limited"])
@pytest.mark.parametrize("oversample", [1, 3, 6])
def test_every_kth_gazebo_beam_is_exactly_a_locrec_beam(spec, oversample):
    """Gazebo spreads its samples from min to max inclusive; locrec's beams must be among them."""
    sensor = ET.fromstring(gzl.lidar_sensor(spec, "lidar", "/l", 2.0, oversample))
    h = sensor.find("lidar/scan/horizontal")
    v = sensor.find("lidar/scan/vertical")
    n_h, h0, h1 = int(h.findtext("samples")), float(h.findtext("min_angle")), float(h.findtext("max_angle"))
    n_v, v0, v1 = int(v.findtext("samples")), float(v.findtext("min_angle")), float(v.findtext("max_angle"))
    az = np.linspace(h0, h1, n_h)[::oversample]
    el = np.linspace(v0, v1, n_v)
    A, E = np.meshgrid(az, el, indexing="ij")
    gz = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=-1).reshape(-1, 3)
    np.testing.assert_allclose(gz, _ray_directions(spec), atol=1e-8)


def test_a_second_copy_of_the_tunnel_side_by_side_shares_no_ground_with_the_first():
    world = TunnelSim(1, SMALL, SPINNING_360).world
    offset = gzl.side_by_side_offset(world, clearance=2.0)
    assert offset[2] == 0.0 and np.linalg.norm(offset) > 0.0

    def footprint(shift):
        cells = set()
        for b in gzl.shell_boxes(world):
            c, s = np.cos(b.yaw), np.sin(b.yaw)
            for u in np.linspace(-b.size[0] / 2, b.size[0] / 2, 5):
                for w in np.linspace(-b.size[1] / 2, b.size[1] / 2, 5):
                    x = b.centre[0] + c * u - s * w + shift[0]
                    y = b.centre[1] + s * u + c * w + shift[1]
                    cells.add((int(np.floor(x)), int(np.floor(y))))
        return cells

    assert not footprint(np.zeros(3)) & footprint(offset)


def test_a_scan_is_a_whole_number_of_steps():
    assert gzl.steps_per_scan() * gzl.STEP_SIZE == pytest.approx(0.5)
    from locrec.sim import PlatformSpec

    with pytest.raises(ValueError):
        gzl.steps_per_scan(PlatformSpec(step_length=0.5, speed_mps=1.3))


def test_run_pass_refuses_a_simulator_built_for_another_run():
    cfg = RunConfig(seed=2, platform="ugv", world=SMALL)
    with pytest.raises(ValueError, match="seed, world and LiDAR"):
        run_pass(cfg, NoMarkers(), sim=TunnelSim(3, SMALL, SPINNING_360))
    used = TunnelSim(2, SMALL, SPINNING_360)
    used.place_marker(np.array([5.0, 1.5, 1.2]))
    with pytest.raises(ValueError, match="fresh"):
        run_pass(cfg, NoMarkers(), sim=used)


def test_a_retro_return_belongs_to_the_nearest_strip_and_rock_to_none():
    """GazeboTunnelSim.marker_detections, fed a hand-made scan: no server involved."""
    sim = gzl.GazeboTunnelSim.__new__(gzl.GazeboTunnelSim)
    TunnelSim.__init__(sim, 0, WorldSpec(length=40.0, n_curves=0, n_junctions=0, n_niches=0,
                                         straight_lead_in=40.0, n_marker_slots=2), SPINNING_360)
    sim.place_marker = lambda p, yaw=0.0: TunnelSim.place_marker(sim, p, yaw)
    sim.place_marker(np.array([10.0, 1.5, 1.2]), -np.pi / 2)
    sim.place_marker(np.array([20.0, -1.5, 1.2]), np.pi / 2)
    T = np.eye(4)
    T[:3, 3] = [15.0, 0.0, 0.7]
    pts = np.array([[-5.0, 1.49, 0.5], [-5.0, 1.51, 0.4], [5.0, -1.49, 0.5], [0.0, 1.6, 0.0]])
    r = np.linalg.norm(pts, axis=1)
    scan = gzl.GazeboScan(points=pts, ranges=r, geom_ids=np.zeros(0, dtype=np.int32), directions=pts / r[:, None],
                          n_cast=SPINNING_360.n_rays, intensities=np.array([200.0, 200.0, 200.0, 0.0]),
                          T_world_sensor=T)
    found = {d.slot: d.n_beams for d in sim.marker_detections(scan, fit_strip=False)}
    assert found == {0: 2, 1: 1}


def test_the_merged_shell_is_every_box_with_faces_wound_outward():
    """Winding matters: a renderer that culls back faces would drop a face wound inward, and the
    LiDAR would see through that wall."""
    world = TunnelSim(1, SMALL, SPINNING_360).world
    boxes = gzl.shell_boxes(world)
    obj = gzl.box_mesh_obj(boxes)
    v = np.array([[float(x) for x in line.split()[1:]] for line in obj.splitlines() if line.startswith("v ")])
    vn = np.array([[float(x) for x in line.split()[1:]] for line in obj.splitlines() if line.startswith("vn ")])
    faces = [line.split()[1:] for line in obj.splitlines() if line.startswith("f ")]
    assert (len(v), len(vn), len(faces)) == (24 * len(boxes), 6 * len(boxes), 12 * len(boxes))
    for k, tri in enumerate(faces):
        a, b, c = (v[int(t.split("//")[0]) - 1] for t in tri)
        n = vn[int(tri[0].split("//")[1]) - 1]
        assert np.dot(np.cross(b - a, c - a), n) > 0.0
        centre = np.array(boxes[k // 12].centre)
        assert np.dot((a + b + c) / 3.0 - centre, n) > 0.0
    # and every vertex lies on the surface of its box, in the box's own frame
    for i, box in enumerate(boxes):
        c, s = np.cos(box.yaw), np.sin(box.yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        local = (v[24 * i: 24 * (i + 1)] - box.centre) @ R
        np.testing.assert_allclose(np.abs(local).max(axis=0), 0.5 * np.array(box.size), atol=2e-6)


def test_a_world_with_meshes_references_one_file_per_material(tmp_path):
    world = TunnelSim(1, SMALL, SPINNING_360).world
    sdf = gzl.world_sdf([(world, "tunnel", (0.0, 0.0, 0.0))], [], mesh_dir=tmp_path, ceiling_transparency=0.8)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["tunnel_ceiling.obj", "tunnel_floor.obj", "tunnel_walls.obj"]
    assert sdf.count("<mesh>") == 3 and sdf.count("<box>") == 0
    assert sdf.count("<transparency>") == 1
