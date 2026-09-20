"""Drive vehicles through the tunnel in Gazebo and publish what their LiDARs see.

The Gazebo counterpart of ``sim_publisher``. It publishes the same topics with the same
messages, so the estimator nodes and the viewer cannot tell which simulator is running.
What changes is where the scans come from: this node starts a ``gz sim`` server on the
tunnel ``locrec.gazebo`` exports, moves each vehicle through it, and takes each scan from
Gazebo's GPU LiDAR. Marker strips are spawned into the running world as they are mounted.

    ros2 run locrec_ros gz_driver --ros-args -p vehicle:=ugv -p world:=mixed -p seed:=1

Two setups:

* ``ugv``: one ground robot on the root topics (``/points``, ``/odom_prior``,
  ``/scan_report``). It mounts a marker whenever ``/markers/localizability/recommended_action``
  says ``drop_marker``.
* ``drone``: two quadrotors, ``forward`` and ``glance``, each in its own copy of the tunnel,
  side by side, on their own namespaces (``/forward/points`` and so on). Each flies the
  centreline and slews its sensor toward the yaw its own estimator recommends, at the study's
  rate limit. Both copies have the same seed, so the same tunnel and the same odometry noise,
  which is how ``locrec/experiments/drone_gaze.py`` pairs its policies.

Lockstep. After each scan the driver waits for every estimator in ``estimates`` to publish
its estimate for that scan, and for the action that follows it, before moving on. A vehicle
then acts on what the last scan told its estimator, as ``run_pass``'s policy does, however
long registration takes. Without it an action that is late for its scan is applied one scan
later, and the live run stops being the run the study measures.

Between scans each vehicle is moved through the interval in small steps, so the Gazebo
window shows it travelling rather than jumping. Only the last step of an interval renders.

Each drone also has an empty model, ``<name>_track``, moved along its position with the track
heading instead of the sensor's. Gazebo's window follows a model with its camera offset in
that model's own frame, so following the drone itself would swing the camera round every time
the sensor glances back; following the anchor keeps the camera behind the direction of travel.
"""
from __future__ import annotations

import faulthandler
import pathlib
import signal
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from nav_msgs.msg import Odometry as OdometryMsg
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from locrec import LIMITED_FOV, SPINNING_360, Drone, RunConfig, TunnelSim, UGV
from locrec import gazebo as gzl
from locrec.odometry import MotionPrior
from locrec.se3 import euler_zyx, inv_T, make_T
from locrec_msgs.msg import MarkerDrop, MarkerObservation, ScanReport

from .demo_worlds import WORLDS, wall_edges
from .sim_publisher import to_pointcloud2

GAZEBO_WORLD = "locrec_tunnel"
OVERSAMPLE = {"ugv": 6, "drone": 3}
"""Horizontal oversampling of the Gazebo sensors, as ``locrec/experiments/calibrate_thresholds_gazebo.py``
exports them. A sensor exported any other way is not the sensor its thresholds describe."""
VEHICLES = {
    "ugv": (("ugv", "ugv"),),
    "drone": (("forward", "drone"), ("glance", "drone")),
    # the team: one tunnel, a ground robot that mounts strips and a drone behind it that mounts
    # nothing. The other two modes give each vehicle its own copy of the tunnel side by side,
    # which is exactly what the team cannot have: the drone has to fly past the robot's strips.
    "team": (("ugv", "ugv"), ("drone", "drone")),
}
DEFAULT_ESTIMATES = {
    "ugv": {"ugv": ["/baseline/localizability/estimate", "/markers/localizability/estimate"]},
    "drone": {"forward": ["/forward/localizability/estimate"],
              "glance": ["/glance/localizability/estimate"]},
    "team": {"ugv": ["/markers/localizability/estimate"],
             "drone": ["/solo/localizability/estimate", "/team/localizability/estimate"]},
}
COLD_SCANS = 10
"""Scans after a vehicle sets off during which its estimators get ``startup_timeout`` rather than
``lockstep_timeout``. One was not enough: the drone's were still cold several scans in."""
TEAM_LAG_STEPS = 60
"""Scans the drone waits at the portal before it sets off, 30 m at the 0.5 m step, as
``locrec/experiments/team_pass.py`` has it. Until it does it publishes nothing, so its first scan is at
the start of the tunnel with the robot already 30 m in, which is the pass the offline experiment
measured."""


def interpolate_pose(T0: np.ndarray, T1: np.ndarray, a: float) -> np.ndarray:
    """Position linearly, yaw the shorter way round. The vehicles only ever yaw."""
    h0 = np.arctan2(T0[1, 0], T0[0, 0])
    h1 = np.arctan2(T1[1, 0], T1[0, 0])
    h = h0 + a * np.arctan2(np.sin(h1 - h0), np.cos(h1 - h0))
    T = np.eye(4)
    T[:2, :2] = [[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]]
    T[:3, 3] = (1.0 - a) * T0[:3, 3] + a * T1[:3, 3]
    return T


class Vehicle:
    """One vehicle: its copy of the tunnel, its platform, its odometry, its topics."""

    def __init__(self, node: Node, name: str, kind: str, seed: int, spec, offset, link: gzl.GazeboLink,
                 n_steps: int, action_topic: str, marker_prefix: str | None = None,
                 start_tick: int = 0, namespace: str | None = None):
        self.name, self.kind = name, kind
        self.ns = (f"/{name}" if kind != "ugv" else "") if namespace is None else namespace
        self.start_tick = int(start_tick)
        """Scan index this vehicle sets off at. Before it, the vehicle is at the portal and
        publishes nothing, so its own first scan is its first scan."""
        self.offset = np.asarray(offset, dtype=float)
        lidar = SPINNING_360 if kind == "ugv" else LIMITED_FOV
        self.cfg = RunConfig(seed=seed, platform=kind, world=spec, lidar=lidar)
        # Gazebo publishes a laser scan on the sensor's topic and the cloud under it, on /points
        self.sensor_topic = f"/{name}/lidar"
        self.gz_topic = f"{self.sensor_topic}/points"
        self.sim = gzl.GazeboTunnelSim(seed, spec, lidar, link, name, self.gz_topic, n_steps,
                                       offset=self.offset,
                                       marker_prefix=marker_prefix or f"{name}_marker")
        self.plat = (UGV(self.sim.world, self.cfg.platform_spec) if kind == "ugv"
                     else Drone(self.sim.world, self.cfg.platform_spec))
        dt = self.cfg.platform_spec.step_length / self.cfg.platform_spec.speed_mps
        self.prior = MotionPrior(self.cfg.prior, seed=seed, dt=dt)
        self.T_true = self.plat.pose()
        self.T_prev = self.T_true.copy()
        self.T_track = self.track_pose()
        self.T_track_prev = self.T_track.copy()
        # the odometry frame is the world frame at the start, as run_pass starts its estimate
        # at the platform's first pose; the prior's noise is what separates the two afterwards
        self.T_odom = self.T_true.copy()
        self.frame_id = "sensor" if not self.ns else f"{self.ns.lstrip('/')}/sensor"
        self.points_pub = node.create_publisher(PointCloud2, f"{self.ns}/points", 10)
        self.prior_pub = node.create_publisher(OdometryMsg, f"{self.ns}/odom_prior", 10)
        self.report_pub = node.create_publisher(ScanReport, f"{self.ns}/scan_report", 100)
        self.action_topic = action_topic
        self.actions: list[str] = []
        self.pending_drops: list[MarkerDrop] = []
        self.last_stamp = None
        if action_topic:
            node.create_subscription(String, action_topic, lambda m: self.actions.append(m.data), 50)

    @property
    def anchor(self) -> str | None:
        """Name of the model the Gazebo window follows for this vehicle, when it is not the vehicle."""
        return f"{self.name}_track" if self.kind == "drone" else None

    def track_pose(self) -> np.ndarray:
        """Where the vehicle is, facing the way it travels."""
        yaw = self.plat.track_yaw() if isinstance(self.plat, Drone) else float(np.arctan2(self.T_true[1, 0], self.T_true[0, 0]))
        return make_T(euler_zyx(yaw), self.T_true[:3, 3])

    def anchor_sdf(self) -> str:
        pose = self.sim.gazebo_pose(self.T_track)
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        return (f'<model name="{self.anchor}"><pose>{pose[0, 3]:.6g} {pose[1, 3]:.6g} {pose[2, 3]:.6g} 0 0 {yaw:.6g}</pose>'
                '<link name="base"><gravity>false</gravity><inertial><mass>1.0</mass>'
                '<inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz></inertia></inertial></link></model>')

    def model_sdf(self, rate: float) -> str:
        pose = self.sim.gazebo_pose(self.T_true)
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        return gzl.vehicle_model(self.name, self.kind, self.cfg.lidar_spec(), self.sensor_topic, rate,
                                 pose=(*pose[:3, 3], 0.0, 0.0, yaw),
                                 color="blue" if self.name == "forward" else "orange",
                                 oversample=OVERSAMPLE[self.kind])


class GazeboDriver(Node):
    def __init__(self) -> None:
        super().__init__("locrec_gz_driver")
        defaults = {
            "vehicle": "ugv", "world": "mixed", "length": 300.0, "seed": 1, "rate_hz": 4.0,
            "drop_markers": True, "lockstep": True, "lockstep_timeout": 20.0, "action_grace": 0.5,
            "startup_timeout": 25.0,
            "estimates": ["default"],
            "work_dir": str(pathlib.Path.home() / ".cache" / "locrec" / "gazebo"),
            "ceiling_transparency": 0.8, "server_verbosity": 1, "team_lag": TEAM_LAG_STEPS,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        def get(name):
            return self.get_parameter(name).value

        self.kind = str(get("vehicle"))
        if self.kind not in VEHICLES:
            raise ValueError(f"vehicle must be one of {sorted(VEHICLES)}, got {self.kind!r}")
        self.team = self.kind == "team"
        self.world_name = str(get("world"))
        self.length = float(get("length"))
        self.seed = int(get("seed"))
        self.rate_hz = max(float(get("rate_hz")), 1e-3)
        self.drop_markers = bool(get("drop_markers")) and self.kind in ("ugv", "team")
        self.lockstep = bool(get("lockstep"))
        self.lockstep_timeout = float(get("lockstep_timeout"))
        self.startup_timeout = float(get("startup_timeout"))
        """How long to wait for estimates of the scan on which a vehicle sets off.

        A vehicle's estimators have received nothing at all until it sets off, so discovery, the
        first pairing and the first registrations all land in the first few intervals after that,
        not in one: in the team the drone's estimators were still cold several scans in and the
        ordinary timeout fired on each of them. The window is ``COLD_SCANS`` long. Every scan after
        it uses ``lockstep_timeout``, which is the number that matters for keeping a live run equal
        to its offline pass.

        It must stay **below** the viewer's ``idle_finish``, or the viewer reads the stall as the
        end of the run and closes its video in the middle of it (``docs/failures.md`` number 37)."""
        self._cold_scan = False
        self.action_grace = float(get("action_grace"))
        estimates = [str(t) for t in get("estimates") if str(t)]
        by_vehicle = DEFAULT_ESTIMATES[self.kind]
        if estimates == ["default"]:
            self.estimate_topics = [t for topics in by_vehicle.values() for t in topics]
            self._estimate_owner = {t: n for n, topics in by_vehicle.items() for t in topics}
        else:
            # an override says nothing about who owns what, so every topic is always required
            self.estimate_topics, self._estimate_owner = estimates, {}
        self.lag = int(get("team_lag")) if self.team else 0
        self.n_steps = gzl.steps_per_scan()
        work = pathlib.Path(str(get("work_dir"))).expanduser()
        work.mkdir(parents=True, exist_ok=True)

        names = VEHICLES[self.kind]
        spec = WORLDS[self.world_name](self.length)
        if self.team:
            # one tunnel, both vehicles in it, one set of strip models between them
            offsets = [np.zeros(3)] * len(names)
        else:
            offsets = [np.zeros(3)]
            if len(names) > 1:
                offsets.append(gzl.side_by_side_offset(TunnelSim(self.seed, spec, SPINNING_360).world))
        self.link = gzl.GazeboLink(GAZEBO_WORLD,
                                   {f"/{n}/lidar/points": OVERSAMPLE[k] for n, k in names})
        self.server = None
        self.vehicles = []
        for (name, kind), offset in zip(names, offsets):
            if kind == "ugv":
                action = "/markers/localizability/recommended_action" if self.drop_markers else ""
            elif self.team:
                action = ""  # the drone of a team flies the track; nothing steers it
            else:
                action = f"/{name}/localizability/recommended_action"
            self.vehicles.append(Vehicle(
                self, name, kind, self.seed, spec, offset, self.link, self.n_steps, action,
                marker_prefix="team_marker" if self.team else None,
                start_tick=self.lag if (self.team and kind == "drone") else 0,
                namespace=f"/{name}" if self.team else None))
        self.scan_index = 0
        self._start_server(work, int(get("server_verbosity")), float(get("ceiling_transparency")))

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.outline_pub = self.create_publisher(MarkerArray, "/world/outline", latched)
        self.outline_pub.publish(self._outline())
        self.tf = TransformBroadcaster(self)
        self._est_stamps: dict[str, set[int]] = {t: set() for t in self.estimate_topics}
        for topic in self.estimate_topics:
            self.create_subscription(OdometryMsg, topic, lambda m, t=topic: self._on_estimate(t, m), 50)

        # the first interval moves nowhere: it renders the first scan at the start pose
        self.state = "move"
        self.substep = 0
        self.first = True
        self.wait_stamp: int | None = None
        self.published_at = 0.0
        self.estimates_done_at: float | None = None
        self.marks: list[int] = []
        self.create_timer(1.0 / (self.rate_hz * self.n_steps), self._tick)
        self.get_logger().info(
            f"locrec gazebo driver up: {', '.join(n for n, _ in names)} in the {self.world_name} world, "
            f"{self.length:.0f} m, seed {self.seed}; lockstep "
            + (f"on {', '.join(self.estimate_topics)}" if self.lockstep and self.estimate_topics else "off"))

    # ---- gazebo ---------------------------------------------------------------

    def _start_server(self, work: pathlib.Path, verbosity: int, transparency: float) -> None:
        rate = 1.0 / (self.n_steps * gzl.STEP_SIZE)
        tunnels, seen = [], set()
        for v in self.vehicles:
            key = tuple(np.round(v.offset, 6))
            if key in seen:      # the team shares one tunnel; building it twice would double
                continue         # every wall and halve the LiDAR's frame rate for nothing
            seen.add(key)
            tunnels.append((v.sim.world, f"tunnel_{v.name}", tuple(v.offset)))
        models = [v.model_sdf(rate) for v in self.vehicles] + [v.anchor_sdf() for v in self.vehicles if v.anchor]
        sdf = gzl.world_sdf(tunnels, models, name=GAZEBO_WORLD, ceiling_transparency=transparency,
                            mesh_dir=work / f"meshes_{self.kind}")
        path = gzl.write_world(work / f"{GAZEBO_WORLD}_{self.kind}.sdf", sdf)
        self.server = gzl.GazeboServer(path, GAZEBO_WORLD, log_path=work / f"server_{self.kind}.log",
                                       verbosity=verbosity, seed=self.seed).start()
        t0 = time.monotonic()
        self.link.wait_ready(120.0, self.server)
        # Primed as locrec/experiments/gazebo_crosscheck.py primes, with the frame thrown away, so the
        # first scan is the same render of Gazebo's seeded noise as the offline pass's first scan.
        # Keeping the priming frame as the first scan shifted every scan's noise by one render,
        # and the live run of a seed stopped being the offline run of that seed.
        self.link.set_poses({v.name: v.sim.gazebo_pose(np.eye(4)) for v in self.vehicles})
        self.link.prime(self.n_steps)
        self.get_logger().info(f"gazebo world up in {time.monotonic() - t0:.1f} s: {path}")

    def close(self) -> None:
        if self.link is not None:
            self.link.close()
            self.link = None
        if self.server is not None:
            self.server.stop()
            self.server = None

    # ---- the loop -------------------------------------------------------------

    def _on_estimate(self, topic: str, msg: OdometryMsg) -> None:
        stamps = self._est_stamps[topic]
        stamps.add(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)
        if len(stamps) > 50:
            stamps.discard(min(stamps))

    @property
    def finished(self) -> bool:
        return all(v.plat.finished for v in self.vehicles if self.started(v))

    def _tick(self) -> None:
        if self.link is None:
            return
        if self.state == "move":
            self.substep += 1
            a = self.substep / self.n_steps
            self.link.set_poses(self._poses(a))
            self.link.step(1)
            if self.substep == self.n_steps:
                self.state = "scan"
        elif self.state == "scan":
            self._publish_scans()
            self.state = "wait"
        elif self.state == "wait" and self._lockstep_ready():
            if self.finished:
                self.get_logger().info("end of tunnel", throttle_duration_sec=10.0)
                return
            self._act_and_advance()
            self.substep = 0
            self.state = "move"

    def started(self, v) -> bool:
        """Whether this vehicle has set off. A team's drone waits at the portal for the lag."""
        return self.scan_index >= v.start_tick

    def _poses(self, a: float) -> dict[str, np.ndarray]:
        """Every model's Gazebo pose a fraction ``a`` of the way through the current interval."""
        poses = {}
        for v in self.vehicles:
            poses[v.name] = v.sim.gazebo_pose(interpolate_pose(v.T_prev, v.T_true, a))
            if v.anchor:
                poses[v.anchor] = v.sim.gazebo_pose(interpolate_pose(v.T_track_prev, v.T_track, a))
        return poses

    def _publish_scans(self) -> None:
        stamp = self.get_clock().now().to_msg()
        # A vehicle that has not set off renders nothing and says nothing. Publishing its scans
        # while it sat at the portal would give its estimator a map built from one pose repeated,
        # and its first moving scan would no longer be the first scan of the offline pass.
        moving = [v for v in self.vehicles if self.started(v)]
        # this scan is inside some vehicle's first few, so its estimators are still cold: they had
        # nothing to discover, pair or register against until it set off
        self._cold_scan = any(v.start_tick <= self.scan_index < v.start_tick + COLD_SCANS
                              for v in self.vehicles)
        scans = {v.name: v.sim.frame_scan(v.T_true) for v in moving}
        for v in moving:
            scan = scans[v.name]
            report = ScanReport(header=Header(stamp=stamp, frame_id=v.frame_id), drops=v.pending_drops,
                                track_yaw=v.plat.track_yaw() if isinstance(v.plat, Drone) else float("nan"))
            v.pending_drops = []
            for det in v.sim.marker_detections(scan):
                obs = MarkerObservation(
                    slot=int(det.slot), n_beams=int(det.n_beams), range_m=float(det.range_m),
                    seen_width_m=float("nan") if det.seen_width_m is None else float(det.seen_width_m),
                    n_columns=int(det.n_columns or 0))
                obs.point_sensor.x, obs.point_sensor.y, obs.point_sensor.z = map(float, det.point_sensor)
                report.detections.append(obs)
            # report, then scan, then odometry: the order the node's pairing has to survive
            v.report_pub.publish(report)
            v.points_pub.publish(to_pointcloud2(scan.points, stamp, v.frame_id))
            if not self.first:
                v.T_odom = v.T_odom @ v.prior.predict(inv_T(v.T_prev) @ v.T_true)
            v.prior_pub.publish(self._odometry(v, stamp))
            self.tf.sendTransform(self._transform(v, stamp))
            v.last_stamp = stamp
        self.first = False
        self.wait_stamp = stamp.sec * 1_000_000_000 + stamp.nanosec
        self.published_at = time.monotonic()
        self.estimates_done_at = None
        self.marks = [len(v.actions) for v in self.vehicles]
        self.scan_index += 1

    def _lockstep_ready(self) -> bool:
        if not self.lockstep or not self.estimate_topics:
            return True
        now = time.monotonic()
        budget = self.startup_timeout if self._cold_scan else self.lockstep_timeout
        waiting_for = {v.name for v in self.vehicles if self.started(v)}
        missing = [t for t in self.estimate_topics
                   if self._estimate_owner.get(t, next(iter(waiting_for), None)) in waiting_for
                   and self.wait_stamp not in self._est_stamps[t]]
        if missing:
            if now - self.published_at > budget:
                self.get_logger().warning(
                    f"no estimate for the last scan after {budget:.0f} s from "
                    f"{', '.join(missing)}; moving on without it", throttle_duration_sec=10.0)
                return True
            return False
        if self.estimates_done_at is None:
            self.estimates_done_at = now
        # the action follows the estimate from the same node; an estimator publishes none for
        # a scan it could not register yet, so wait a moment for it and no longer
        acted = all(len(v.actions) > mark for v, mark in zip(self.vehicles, self.marks) if v.action_topic)
        return acted or now - self.estimates_done_at > self.action_grace

    def _act_and_advance(self) -> None:
        for v, mark in zip(self.vehicles, self.marks):
            if not self.started(v):
                continue
            new = v.actions[mark:] if self.lockstep else v.actions[-1:]
            action = new[-1] if new else "none"
            v.actions = []
            v.T_prev = v.T_true.copy()
            if v.kind == "ugv":
                if action == "drop_marker" and self.drop_markers:
                    self._mount(v)
            elif action.startswith("yaw_to:"):
                v.plat.command_yaw(np.deg2rad(float(action.split(":", 1)[1])))
            else:
                # no recommendation for this scan: the policies' own answer then is forward
                v.plat.track_heading()
            v.plat.advance()
            v.T_true = v.plat.pose()
            v.T_track_prev, v.T_track = v.T_track, v.track_pose()

    def _mount(self, v: Vehicle) -> None:
        """Mount the requested marker at the pose of the last scan, before moving on, where
        ``run_pass`` and ``sim_publisher`` mount it."""
        if v.last_stamp is None or v.sim.n_markers_placed >= v.sim.marker_capacity:
            return
        try:
            slot, offset = v.sim.drop_marker_on_wall(v.T_true)
        except RuntimeError as exc:
            self.get_logger().warning(f"marker not mounted: {exc}")
            return
        point_world = v.T_true[:3, :3] @ offset + v.T_true[:3, 3]
        world = point_world + v.offset
        # Everyone else in this same tunnel is looking at the same wall, so they have to be told
        # the strip is there, in the same slot, or their detectors will not associate a return
        # with it. Only the vehicle that mounted it spawns the model.
        for other in self.vehicles:
            if other is v or not np.allclose(other.offset, v.offset):
                continue
            noted = other.sim.note_marker(point_world, self._marker_yaw(v, slot))
            if noted != slot:
                self.get_logger().error(
                    f"{other.name} has the strip in slot {noted} where {v.name} has it in {slot}")
        drop = MarkerDrop(slot=int(slot), pose_stamp=v.last_stamp)
        drop.offset_sensor.x, drop.offset_sensor.y, drop.offset_sensor.z = map(float, offset)
        drop.position_world.x, drop.position_world.y, drop.position_world.z = map(float, world)
        v.pending_drops.append(drop)

    @staticmethod
    def _marker_yaw(v, slot: int) -> float:
        qw, qx, qy, qz = v.sim.data.mocap_quat[v.sim._marker_mocap_ids[slot]]
        return float(2.0 * np.arctan2(qz, qw))

    # ---- messages -------------------------------------------------------------

    def _odometry(self, v: Vehicle, stamp) -> OdometryMsg:
        odom = OdometryMsg()
        odom.header = Header(stamp=stamp, frame_id="odom" if not v.ns else f"{v.ns.lstrip('/')}/odom")
        odom.child_frame_id = v.frame_id
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, v.T_odom[:3, 3])
        qx, qy, qz, qw = Rotation.from_matrix(v.T_odom[:3, :3]).as_quat()
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y = float(qx), float(qy)
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = float(qz), float(qw)
        return odom

    def _transform(self, v: Vehicle, stamp) -> TransformStamped:
        """The true pose in the Gazebo world, for drawing. No estimator reads it."""
        T = v.sim.gazebo_pose(v.T_true)
        tf = TransformStamped()
        tf.header = Header(stamp=stamp, frame_id="world")
        tf.child_frame_id = v.frame_id
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, T[:3, 3])
        qx, qy, qz, qw = Rotation.from_matrix(T[:3, :3]).as_quat()
        tf.transform.rotation.x, tf.transform.rotation.y = float(qx), float(qy)
        tf.transform.rotation.z, tf.transform.rotation.w = float(qz), float(qw)
        return tf

    def _outline(self) -> MarkerArray:
        out = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for v in self.vehicles:
            for k, line in enumerate(wall_edges(v.sim.world)):
                m = Marker()
                m.header = Header(stamp=stamp, frame_id="world")
                m.ns, m.id, m.type, m.action = f"walls {v.name}", k, Marker.LINE_STRIP, Marker.ADD
                m.scale.x = 0.06
                m.color.r, m.color.g, m.color.b, m.color.a = 0.6, 0.6, 0.6, 1.0
                m.pose.orientation.w = 1.0
                m.points = [Point(x=float(x + v.offset[0]), y=float(y + v.offset[1]), z=0.0) for x, y in line]
                out.markers.append(m)
        return out


def main(args=None) -> None:
    # the shutdown handling every node here has; the reasons are in sim_publisher.main
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    rclpy.init(args=args)
    node = None
    try:
        node = GazeboDriver()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            if rclpy.ok():
                node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
