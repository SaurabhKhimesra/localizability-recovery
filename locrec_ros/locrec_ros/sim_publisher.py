"""Drive the MuJoCo tunnel and publish its scans, so the node has something to eat.

The detector node subscribes to ``/points`` and nothing in this repository was
publishing them: milestone 5's real sequence is deferred, so without this you need
a bag before you can watch anything happen. This walks a generated tunnel at the
platform's own speed and publishes each scan as a ``PointCloud2``, plus the noisy
dead-reckoned pose as ordinary cumulative ``Odometry`` on ``/odom_prior`` and the
true pose on ``/tf`` so rviz has a frame to draw in.

    ros2 run locrec_ros sim_publisher --ros-args -p platform:=ugv -p world:=mixed

With ``drop_markers`` set it also closes the loop. It listens to the detector's
recommended action, mounts a marker strip on the wall beside the robot when told
to, and reports, once per scan and stamped like the scan, the markers it mounted
and the markers its retroreflector detector saw. Mounting and detection are
``TunnelSim.drop_marker_on_wall`` and ``TunnelSim.marker_detections``, the same two
calls ``run_pass`` makes, so nothing about markers is modelled a second time here.

It is a demonstration harness, not a benchmark. Every number in the README comes
from ``locrec/experiments/``, which runs the same simulation with no ROS in the loop and
no wall-clock pacing.
"""
from __future__ import annotations

import faulthandler
import os
import signal
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from nav_msgs.msg import Odometry as OdometryMsg
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from locrec_msgs.msg import MarkerDrop, MarkerObservation, ScanReport

from locrec import LIMITED_FOV, SPINNING_360, Drone, RunConfig, TunnelSim, UGV
from locrec.odometry import MotionPrior
from locrec.se3 import inv_T

from .demo_worlds import WORLDS, wall_edges


def to_pointcloud2(points: np.ndarray, stamp, frame_id: str) -> PointCloud2:
    """Pack an (N, 3) float array as an unorganised float32 xyz cloud."""
    pts = np.asarray(points, dtype=np.float32)
    msg = PointCloud2()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height = 1
    msg.width = int(pts.shape[0])
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.data = pts.tobytes()
    msg.is_dense = True
    return msg


def world_outline(world, stamp) -> MarkerArray:
    """The plan of the tunnel as line strips, for rviz and for the demo viewer."""
    out = MarkerArray()
    for k, line in enumerate(wall_edges(world)):
        m = Marker()
        m.header = Header(stamp=stamp, frame_id="world")
        m.ns, m.id, m.type, m.action = "walls", k, Marker.LINE_STRIP, Marker.ADD
        m.scale.x = 0.06
        m.color.r, m.color.g, m.color.b, m.color.a = 0.45, 0.55, 0.65, 1.0
        m.pose.orientation.w = 1.0
        for x, y in line:
            pt = Point()
            pt.x, pt.y, pt.z = float(x), float(y), 0.0
            m.points.append(pt)
        out.markers.append(m)
    return out


class SimPublisher(Node):
    def __init__(self) -> None:
        super().__init__("locrec_sim_publisher")
        self.declare_parameter("platform", "ugv")
        self.declare_parameter("world", "mixed")
        self.declare_parameter("length", 300.0)
        self.declare_parameter("seed", 0)
        self.declare_parameter("rate_hz", 4.0)
        self.declare_parameter("frame_id", "sensor")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("loop", False)
        self.declare_parameter("drop_markers", False)
        self.declare_parameter("action_topic", "/localizability/recommended_action")
        self.declare_parameter("window", False)

        self.platform_name = str(self.get_parameter("platform").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.seed = int(self.get_parameter("seed").value)
        self.length = float(self.get_parameter("length").value)
        self.world_name = str(self.get_parameter("world").value)

        self.drop_markers = bool(self.get_parameter("drop_markers").value)
        self._drop_requested = False
        self._last_stamp = None

        self.points_pub = self.create_publisher(PointCloud2, "/points", 10)
        self.prior_pub = self.create_publisher(OdometryMsg, "/odom_prior", 10)
        self.report_pub = self.create_publisher(ScanReport, "/scan_report", 100)
        # latched, so a viewer that starts late still gets the tunnel
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.outline_pub = self.create_publisher(MarkerArray, "/world/outline", latched)
        self.tf = TransformBroadcaster(self)
        if self.drop_markers:
            self.create_subscription(
                String, str(self.get_parameter("action_topic").value), self.on_action, 10
            )
        # the noisy increments integrated. Kept across loops, so a restart moves the
        # scans to a new tunnel but never makes the odometry jump
        self.T_odom = np.eye(4)
        self.rate_hz = max(float(self.get_parameter("rate_hz").value), 1e-6)
        self.want_window = bool(self.get_parameter("window").value)
        self.window = None
        self._build()

        self.create_timer(1.0 / self.rate_hz, self.tick)
        if self.want_window:
            self.create_timer(1.0 / 30.0, self._draw_window)
        self.get_logger().info(
            f"locrec sim publisher up: {self.platform_name} in the {self.world_name} "
            f"world, {self.length:.0f} m, seed {self.seed}"
        )

    def _build(self) -> None:
        spec = WORLDS[self.world_name](self.length)
        lidar = SPINNING_360 if self.platform_name == "ugv" else LIMITED_FOV
        self.cfg = RunConfig(
            seed=self.seed, platform=self.platform_name, world=spec, lidar=lidar
        )
        self.sim = TunnelSim(self.seed, spec, lidar)
        self.plat = (
            UGV(self.sim.world, self.cfg.platform_spec)
            if self.platform_name == "ugv"
            else Drone(self.sim.world, self.cfg.platform_spec)
        )
        dt = self.cfg.platform_spec.step_length / self.cfg.platform_spec.speed_mps
        self.prior = MotionPrior(self.cfg.prior, seed=self.seed, dt=dt)
        self.T_true = self.plat.pose()
        self._drop_requested = False
        self._last_stamp = None
        self.outline_pub.publish(world_outline(self.sim.world, self.get_clock().now().to_msg()))
        if self.want_window:
            self.close_window()
            try:
                from .sim_window import SimWindow

                self.window = SimWindow(self.sim.world.xml, self.rate_hz)
            except Exception as exc:  # noqa: BLE001  no display is a reason to run headless, not to stop
                self.window = None
                self.get_logger().warning(f"no simulator window ({type(exc).__name__}: {exc}); running without it")

    def _draw_window(self) -> None:
        if self.window is not None:
            self.window.draw()

    def close_window(self) -> bool:
        """Close the simulator window if one is open. Returns whether one was."""
        if self.window is None:
            return False
        self.window.close()
        self.window = None
        return True

    def on_action(self, msg: String) -> None:
        if msg.data == "drop_marker":
            self._drop_requested = True

    def _mount_marker(self) -> list:
        """Mount the requested marker at the pose of the last scan, before moving on.

        That is where ``run_pass`` mounts it: the policy acts on what the last scan
        told it, and the robot has not moved since. The report names the scan the
        pose belongs to, so an estimator that fell a scan behind still registers the
        marker against the right estimate.
        """
        self._drop_requested = False
        if self._last_stamp is None or self.sim.n_markers_placed >= self.sim.marker_capacity:
            return []
        try:
            slot, offset = self.sim.drop_marker_on_wall(self.T_true)
        except RuntimeError as exc:
            self.get_logger().warning(f"marker not mounted: {exc}")
            return []
        world = self.T_true[:3, :3] @ offset + self.T_true[:3, 3]
        drop = MarkerDrop(slot=int(slot), pose_stamp=self._last_stamp)
        drop.offset_sensor.x, drop.offset_sensor.y, drop.offset_sensor.z = map(float, offset)
        drop.position_world.x, drop.position_world.y, drop.position_world.z = map(float, world)
        return [drop]

    def tick(self) -> None:
        if self.plat.finished:
            if not self.loop:
                # the timer keeps firing with nothing left to publish, so say so
                # occasionally rather than at the scan rate
                self.get_logger().info("end of tunnel", throttle_duration_sec=10.0)
                return
            self.get_logger().info("end of tunnel, starting again")
            self.seed += 1
            self._build()

        drops = self._mount_marker() if self._drop_requested else []

        T_prev = self.T_true
        self.plat.advance()
        self.T_true = self.plat.pose()
        scan = self.sim.scan(self.T_true)
        if self.window is not None:
            self.window.new_scan(self.T_true, scan.points, self.sim.data.mocap_pos, self.sim.data.mocap_quat)
        stamp = self.get_clock().now().to_msg()
        self._last_stamp = stamp

        # one report per scan, empty or not, and before the scan it belongs to
        report = ScanReport(header=Header(stamp=stamp, frame_id=self.frame_id), drops=drops,
                            track_yaw=self.plat.track_yaw() if isinstance(self.plat, Drone) else float("nan"))
        for det in self.sim.marker_detections(scan):
            obs = MarkerObservation(
                slot=int(det.slot), n_beams=int(det.n_beams), range_m=float(det.range_m),
                seen_width_m=float("nan") if det.seen_width_m is None else float(det.seen_width_m),
                n_columns=int(det.n_columns or 0),
            )
            obs.point_sensor.x, obs.point_sensor.y, obs.point_sensor.z = map(float, det.point_sensor)
            report.detections.append(obs)
        self.report_pub.publish(report)

        # the scan goes out before the odometry that shares its stamp, which is the
        # order the node's pairing has to survive
        self.points_pub.publish(to_pointcloud2(scan.points, stamp, self.frame_id))

        self.T_odom = self.T_odom @ self.prior.predict(inv_T(T_prev) @ self.T_true)
        odom = OdometryMsg()
        odom.header = Header(stamp=stamp, frame_id=self.odom_frame)
        odom.child_frame_id = self.frame_id
        odom.pose.pose.position.x = float(self.T_odom[0, 3])
        odom.pose.pose.position.y = float(self.T_odom[1, 3])
        odom.pose.pose.position.z = float(self.T_odom[2, 3])
        qx, qy, qz, qw = Rotation.from_matrix(self.T_odom[:3, :3]).as_quat()
        odom.pose.pose.orientation.x = float(qx)
        odom.pose.pose.orientation.y = float(qy)
        odom.pose.pose.orientation.z = float(qz)
        odom.pose.pose.orientation.w = float(qw)
        self.prior_pub.publish(odom)

        # the true pose, so rviz has somewhere to draw the cloud. It is ground truth
        # and the detector never sees it.
        tf = TransformStamped()
        tf.header = Header(stamp=stamp, frame_id="world")
        tf.child_frame_id = self.frame_id
        tf.transform.translation.x = float(self.T_true[0, 3])
        tf.transform.translation.y = float(self.T_true[1, 3])
        tf.transform.translation.z = float(self.T_true[2, 3])
        qw = float(np.sqrt(max(0.0, 1.0 + np.trace(self.T_true[:3, :3]))) * 0.5)
        if qw > 1e-9:
            tf.transform.rotation.w = qw
            tf.transform.rotation.x = float(
                (self.T_true[2, 1] - self.T_true[1, 2]) / (4.0 * qw)
            )
            tf.transform.rotation.y = float(
                (self.T_true[0, 2] - self.T_true[2, 0]) / (4.0 * qw)
            )
            tf.transform.rotation.z = float(
                (self.T_true[1, 0] - self.T_true[0, 1]) / (4.0 * qw)
            )
        else:
            tf.transform.rotation.w = 1.0
        self.tf.sendTransform(tf)


def main(args=None) -> None:
    # rclpy's own handler stops spin on ctrl+c by shutting the context down. The Python
    # handler it chains to would also raise KeyboardInterrupt, and a terminal ctrl+c reaches
    # every process in the group while ros2 launch forwards a second one, which then lands in
    # the middle of cleanup: the process dies with -2, and the viewer's video can be cut off
    # before it is finished. Ignoring SIGINT here leaves rclpy's handler as the only one.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # kill -USR1 <pid> prints every thread's Python stack to stderr, which is how to see
    # what a node that will not stop is doing without attaching a debugger
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    rclpy.init(args=args)
    node = SimPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # ctrl+c reaches rclpy's own signal handler first, which shuts the context
        # down and makes spin raise the second of these rather than the first
        pass
    finally:
        had_window = node.close_window()
        # Destroying a node after a signal has already shut its context down can hang in
        # rclpy 7.1: destroy_node wakes the executor, whose guard condition the shutdown is
        # tearing down on another thread. Caught with faulthandler on SIGUSR1 in one of eight
        # ctrl+c runs, and it is what left a node spinning at 100 percent after a recording.
        # The process is exiting either way, so destroy the node only while the context is up.
        if rclpy.ok():
            node.destroy_node()
        rclpy.try_shutdown()
        if had_window:
            # MuJoCo's viewer closes cleanly and then segfaults in interpreter teardown (exit
            # 139, every time, measured). Everything worth keeping is done by here, so leave
            # without the teardown.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
