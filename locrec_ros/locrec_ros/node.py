"""The node. Everything it decides lives in detector.py and prior.py; this is the shell."""
from __future__ import annotations

import faulthandler
import signal

import numpy as np
import rclpy
from nav_msgs.msg import Odometry as OdometryMsg
from rclpy.exceptions import ParameterUninitializedException
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String

from locrec.sim import MarkerDetection
from locrec_msgs.msg import Localizability, ScanReport, SharedLandmark

from .calibration import resolve_calibration
from .conversions import odometry_to_pose, pointcloud2_to_xyz, stamp_to_ns
from .detector import DetectorConfig, LocalizabilityDetector
from .prior import PriorPairer

ESTIMATE_FRAME = "locrec_odom"
"""Frame of the published estimate. Its origin is the sensor pose of the first scan
the node processed, which is all an odometry can ever know about where it started; with
``start_at_odometry`` it is the frame of ``/odom_prior`` instead."""


class LocalizabilityNode(Node):
    def __init__(self) -> None:
        super().__init__("locrec_localizability")
        # calibrated, so declared by type with no default: a default here would be a
        # second source for a number that has one. See calibration.py.
        self.declare_parameter("thresholds_file", "")
        self.declare_parameter("ratio_threshold", Parameter.Type.DOUBLE)
        self.declare_parameter("marker_reliable_range", Parameter.Type.DOUBLE)
        self.declare_parameter("scheduler_margin", Parameter.Type.DOUBLE)
        self.declare_parameter("min_lambda_per_point", 0.0)
        self.declare_parameter("range", 10.0)
        self.declare_parameter("fov", 360.0)
        self.declare_parameter("azimuth_beams", 360)
        self.declare_parameter("elevation_beams", 16)
        self.declare_parameter("fov_elevation", 30.0)
        self.declare_parameter("platform", "ugv")
        self.declare_parameter("gaze", "across")
        self.declare_parameter("start_at_odometry", False)
        self.declare_parameter("registration_threads", 4)
        self.declare_parameter("max_points", 60000)
        self.declare_parameter("use_markers", False)
        self.declare_parameter("share_landmarks", False)
        self.declare_parameter("use_shared_landmarks", False)

        self.max_points = int(self.get_parameter("max_points").value)
        self.use_markers = bool(self.get_parameter("use_markers").value)
        self.share_landmarks = bool(self.get_parameter("share_landmarks").value)
        self.use_shared = bool(self.get_parameter("use_shared_landmarks").value)
        if self.share_landmarks and not self.use_markers:
            raise RuntimeError("share_landmarks needs use_markers: a vehicle can only tell the "
                               "team about strips it is mounting")
        if self.use_shared and self.use_markers:
            raise RuntimeError("use_shared_landmarks is for a vehicle that mounts nothing of its "
                               "own; with use_markers the two sets of slots would collide")
        platform = str(self.get_parameter("platform").value)
        try:
            calibrated = resolve_calibration(
                platform,
                {
                    "ratio_threshold": self._optional("ratio_threshold"),
                    "marker_reliable_range_m": self._optional("marker_reliable_range"),
                    "scheduler_margin_m": self._optional("scheduler_margin"),
                },
                str(self.get_parameter("thresholds_file").value),
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from None
        self.detector = LocalizabilityDetector(
            DetectorConfig(
                ratio_threshold=calibrated["ratio_threshold"],
                min_lambda_per_point=float(
                    self.get_parameter("min_lambda_per_point").value
                ),
                max_range=float(self.get_parameter("range").value),
                fov_azimuth_deg=float(self.get_parameter("fov").value),
                azimuth_beams=int(self.get_parameter("azimuth_beams").value),
                platform=platform,
                gaze=str(self.get_parameter("gaze").value),
                start_at_odometry=bool(self.get_parameter("start_at_odometry").value),
                registration_threads=int(self.get_parameter("registration_threads").value),
                elevation_beams=int(self.get_parameter("elevation_beams").value),
                fov_elevation_deg=float(self.get_parameter("fov_elevation").value),
                marker_reliable_range_m=calibrated.get("marker_reliable_range_m"),
                scheduler_margin_m=calibrated.get("scheduler_margin_m"),
                share_landmarks=self.share_landmarks,
                use_shared_landmarks=self.use_shared,
            )
        )
        # the report carries what markers and a gaze policy need, so either one holds each
        # scan for the report with its stamp
        self.use_report = self.use_markers or self.use_shared or self.detector.needs_track_yaw
        self.pairer = PriorPairer(require_side=self.use_report)
        self._odom_child_frame: str | None = None
        self._warned_frames = False
        self._dropped_reported = 0
        self._n_reports = 0
        self._n_shared = 0

        self.pub = self.create_publisher(Localizability, "/localizability", 10)
        self.action_pub = self.create_publisher(
            String, "/localizability/recommended_action", 10
        )
        self.estimate_pub = self.create_publisher(OdometryMsg, "/localizability/estimate", 10)
        # Reliable, keep all, and latched. A strip is told of once and used for the rest of the
        # run, so a receiver that joins late, or blinks, must still get every one of them.
        team_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL,
                              history=HistoryPolicy.KEEP_ALL)
        self.shared_pub = (self.create_publisher(SharedLandmark, "/team/landmarks", team_qos)
                           if self.share_landmarks else None)
        if self.use_shared:
            self.create_subscription(SharedLandmark, "/team/landmarks", self.on_shared, team_qos)
        self.create_subscription(PointCloud2, "/points", self.on_points, 10)
        self.create_subscription(OdometryMsg, "/odom_prior", self.on_prior, 100)
        if self.use_report:
            self.create_subscription(ScanReport, "/scan_report", self.on_report, 100)
        self.get_logger().info(
            f"locrec localizability node up: threshold {calibrated['ratio_threshold']:.4e}, "
            f"registration range {self.detector.odometry.cfg.registration_range:.2f} m, "
            "every scan waits for /odom_prior at its stamp"
            + (" and for its /scan_report" if self.use_report else "")
            + (f", gaze {self.detector.config.gaze}" if platform == "drone" else "")
        )

    def _optional(self, name: str) -> float | None:
        try:
            return float(self.get_parameter(name).value)
        except ParameterUninitializedException:
            return None

    def on_prior(self, msg: OdometryMsg) -> None:
        try:
            pose = odometry_to_pose(msg)
        except ValueError as exc:
            self.get_logger().warning(f"unusable odometry prior: {exc}")
            return
        self._odom_child_frame = msg.child_frame_id
        self._handle(self.pairer.add_odometry(stamp_to_ns(msg.header.stamp), pose))

    def on_report(self, msg: ScanReport) -> None:
        self._n_reports += 1
        drops = [
            (int(d.slot), np.array([d.offset_sensor.x, d.offset_sensor.y, d.offset_sensor.z]))
            for d in msg.drops
        ] if self.use_markers else []
        # A vehicle using a teammate's strips mounts none of its own, so it has no drops to read,
        # and it still has to read the detections: those are the strips it can see. Gating both on
        # use_markers threw away every detection the team drone made, and its team estimate came
        # out identical to its solo one, which is the same signature as failures number 32.
        detections = [
            MarkerDetection(
                slot=int(o.slot),
                point_sensor=np.array([o.point_sensor.x, o.point_sensor.y, o.point_sensor.z]),
                n_beams=int(o.n_beams),
                range_m=float(o.range_m),
                seen_width_m=None if np.isnan(o.seen_width_m) else float(o.seen_width_m),
                n_columns=int(o.n_columns) or None,
            )
            for o in msg.detections
        ] if (self.use_markers or self.use_shared) else []
        self._handle(self.pairer.add_side(stamp_to_ns(msg.header.stamp), (drops, detections, float(msg.track_yaw))))

    def on_points(self, msg: PointCloud2) -> None:
        try:
            points = pointcloud2_to_xyz(msg, max_points=self.max_points)
        except ValueError as exc:
            self.get_logger().warning(f"unusable point cloud: {exc}")
            return
        self._handle(self.pairer.add_scan(stamp_to_ns(msg.header.stamp), (msg.header, points)))
        if self.pairer.n_odometry == 0:
            self.get_logger().warning(
                "no /odom_prior received yet: scans are held for it, never processed "
                "without it",
                throttle_duration_sec=5.0,
            )
        if self.use_report and self._n_reports == 0:
            self.get_logger().warning(
                "no /scan_report has arrived: with use_markers or a gaze policy, scans are "
                "held for the report that carries their stamp",
                throttle_duration_sec=5.0,
            )

    def on_shared(self, msg: SharedLandmark) -> None:
        """A strip a teammate mounted. Registered at once, well before it comes into view.

        There is no pairing with a scan stamp here and there should not be: the anchor is a
        statement about the shared frame, not an observation of this vehicle's, and the estimator
        holds it until this vehicle sees the strip for itself.
        """
        try:
            new = self.detector.add_shared_landmark(
                msg.slot,
                np.array([msg.position.x, msg.position.y, msg.position.z]),
                np.array(msg.drop_covariance, dtype=float),
                np.array([msg.normal.x, msg.normal.y, msg.normal.z]),
            )
        except ValueError as exc:
            self.get_logger().error(f"unusable shared landmark {msg.slot}: {exc}",
                                    throttle_duration_sec=5.0)
            return
        if new:
            self._n_shared += 1
            self.get_logger().info(f"strip {msg.slot} from the team, {self._n_shared} held")

    def _handle(self, released) -> None:
        for item in released:
            header, points = item.payload
            if not self._warned_frames and self._odom_child_frame not in (None, header.frame_id):
                self._warned_frames = True
                self.get_logger().warning(
                    f"/odom_prior describes '{self._odom_child_frame}' but the cloud is in "
                    f"'{header.frame_id}': the prior is only right if the two coincide"
                )
            drops, detections, track_yaw = item.side if item.side is not None else ([], [], None)
            # a marker dropped while this node still had nothing to register against
            # would anchor to a pose the estimator never held, so such a drop cannot
            # happen: the node only recommends one after it has processed a scan
            try:
                out = self.detector.process(points, item.pose, drops, detections, track_yaw)
            except ValueError as exc:
                self.get_logger().error(f"scan not processed: {exc}", throttle_duration_sec=5.0)
                continue
            self.estimate_pub.publish(build_estimate(header, self.detector.pose))
            while self.detector.shared and self.shared_pub is not None:
                self.shared_pub.publish(build_shared(header, self.detector.shared.pop(0)))
            if out is None:
                continue
            self.pub.publish(build_message(header, out))
            self.action_pub.publish(String(data=out.action))
        if self.pairer.dropped > self._dropped_reported:
            self._dropped_reported = self.pairer.dropped
            self.get_logger().warning(
                f"{self.pairer.dropped} scans dropped rather than processed on a guess "
                f"(older than the odometry {self.pairer.dropped_stale}, waited too long "
                f"{self.pairer.dropped_overflow}, out of order {self.pairer.dropped_out_of_order}, "
                f"marker report never came {self.pairer.dropped_no_side})",
                throttle_duration_sec=5.0,
            )


def build_shared(header, rec: dict) -> SharedLandmark:
    """Assemble the team message. Separate so it can be tested without a graph."""
    msg = SharedLandmark()
    msg.header = header
    msg.slot = int(rec["slot"])
    msg.position.x, msg.position.y, msg.position.z = (float(v) for v in rec["position"])
    msg.drop_covariance = [float(v) for v in np.asarray(rec["drop_covariance"], dtype=float).ravel()]
    msg.normal.x, msg.normal.y = float(rec["normal"][0]), float(rec["normal"][1])
    msg.normal.z = 0.0
    return msg


def build_message(header, out) -> Localizability:
    """Assemble the message. Separate so it can be tested without a graph."""
    msg = Localizability()
    msg.header = header
    msg.eigenvalues = [float(v) for v in np.asarray(out.eigenvalues).reshape(3)]
    msg.weak_direction = [float(v) for v in np.asarray(out.weak_direction).reshape(3)]
    msg.ratio = float(out.ratio)
    msg.lambda_min_per_point = float(out.lambda_min_per_point)
    msg.is_degenerate = bool(out.is_degenerate)
    msg.n_points = int(out.n_points)
    msg.landmarks_used = int(out.landmarks_used)
    return msg


def build_estimate(header, pose: np.ndarray) -> OdometryMsg:
    """The estimate for one scan, stamped like the scan so it joins to the truth exactly."""
    msg = OdometryMsg()
    msg.header = Header(stamp=header.stamp, frame_id=ESTIMATE_FRAME)
    msg.child_frame_id = header.frame_id
    msg.pose.pose.position.x = float(pose[0, 3])
    msg.pose.pose.position.y = float(pose[1, 3])
    msg.pose.pose.position.z = float(pose[2, 3])
    qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
    msg.pose.pose.orientation.x = float(qx)
    msg.pose.pose.orientation.y = float(qy)
    msg.pose.pose.orientation.z = float(qz)
    msg.pose.pose.orientation.w = float(qw)
    return msg


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
    node = LocalizabilityNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # ctrl+c reaches rclpy's own signal handler first, which shuts the context
        # down and makes spin raise the second of these rather than the first
        pass
    finally:
        # Destroying a node after a signal has already shut its context down can hang in
        # rclpy 7.1: destroy_node wakes the executor, whose guard condition the shutdown is
        # tearing down on another thread. Caught with faulthandler on SIGUSR1 in one of eight
        # ctrl+c runs, and it is what left a node spinning at 100 percent after a recording.
        # The process is exiting either way, so destroy the node only while the context is up.
        if rclpy.ok():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
