"""The live views: the simulator window's display copy, and rviz's config against the code.

No window opens and no graph runs.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PKG = Path(__file__).resolve().parents[1]

mujoco = pytest.importorskip("mujoco")
from locrec_ros.sim_window import display_model  # noqa: E402

from locrec import SPINNING_360, TunnelSim, UGV  # noqa: E402
from locrec.worlds import WorldSpec  # noqa: E402


def _ceilings(model):
    return [i for i in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("ceil_")]


def test_the_window_hides_the_ceiling_in_its_own_copy_and_never_in_the_simulation():
    """MuJoCo's ray caster skips a geom whose alpha is zero. Measured on the mixed world:
    hiding the ceiling in the simulation's model loses 130 of 5522 returns in one scan, so
    only the window's display copy may hide it."""
    spec = WorldSpec(length=40.0, n_curves=0, n_junctions=0, n_niches=0, straight_lead_in=40.0,
                     width_min=3.2, width_max=3.2, width_mean=3.2)
    sim = TunnelSim(0, spec, SPINNING_360)
    shown = display_model(sim.world.xml)
    ceil = _ceilings(shown)
    assert ceil and np.all(shown.geom_rgba[ceil, 3] == 0.0)
    assert np.all(sim.model.geom_rgba[_ceilings(sim.model), 3] > 0.0)

    plat = UGV(sim.world)
    for _ in range(20):
        plat.advance()
    T = plat.pose()
    gid = np.zeros(1, dtype=np.int32)
    up = np.array([0.0, 0.0, 1.0])
    hit = mujoco.mj_ray(sim.model, sim.data, T[:3, 3].astype(float), up, None, 1, -1, gid)
    assert hit > 0.0 and (mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0])) or "").startswith("ceil_")


@pytest.mark.parametrize("config_name, mode", [("demo.rviz", "ugv"), ("gazebo_ugv.rviz", "ugv"),
                                               ("gazebo_drone.rviz", "drone"),
                                               ("gazebo_team.rviz", "team"),
                                               ("gazebo_ugv_recording.rviz", "ugv"),
                                               ("gazebo_team_recording.rviz", "team")])
def test_every_topic_rviz_listens_to_is_one_the_package_publishes(config_name, mode):
    yaml = pytest.importorskip("yaml")
    pytest.importorskip("rclpy")
    from locrec_ros.demo_viewer import rviz_topics

    config = yaml.safe_load((PKG / "rviz" / config_name).read_text())
    topics = {d["Topic"]["Value"] for d in config["Visualization Manager"]["Displays"] if "Topic" in d}
    published = rviz_topics(mode) | {"/world/outline"}
    assert topics, "the config lost its displays"
    assert topics <= published, sorted(topics - published)
    assert config["Visualization Manager"]["Global Options"]["Fixed Frame"] == "world"
    # The camera follows a frame gz_driver broadcasts in this mode. The team config was copied from
    # the drone one and kept following glance/sensor, which the team never publishes, so the view
    # sat still at the origin while both vehicles drove away from it.
    broadcast = {"ugv": {"sensor"}, "drone": {"forward/sensor", "glance/sensor"},
                 "team": {"ugv/sensor", "drone/sensor"}}[mode]
    target = config["Visualization Manager"]["Views"]["Current"].get("Target Frame")
    if target is not None and config_name != "demo.rviz":
        assert target in broadcast, (config_name, target, sorted(broadcast))


def test_an_estimator_whose_first_estimate_has_no_true_pose_is_still_aligned():
    """The simulator's first transform can be lost to discovery while the estimator's first
    estimate arrives. Aligning on the first estimate alone then never aligned, and one
    estimator's whole run was drawn and scored as missing."""
    pytest.importorskip("rclpy")
    from locrec.se3 import make_T
    from locrec_ros.demo_viewer import MODES, Track

    tr = Track(MODES["ugv"][1][1])
    tr.est = {1: make_T(np.eye(3), [0.0, 0.0, 0.0]), 2: make_T(np.eye(3), [0.5, 0.0, 0.0]),
              3: make_T(np.eye(3), [1.0, 0.0, 0.0])}
    truth = {2: make_T(np.eye(3), [10.5, 2.0, 0.7]), 3: make_T(np.eye(3), [11.0, 2.0, 0.7])}
    assert tr.world(1, truth) is not None
    np.testing.assert_allclose(tr.world(3, truth)[:3, 3], [11.0, 2.0, 0.7])
