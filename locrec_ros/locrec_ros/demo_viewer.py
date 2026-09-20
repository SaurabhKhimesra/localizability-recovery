"""The demonstration viewer: draws what the graph publishes, and records it.

It subscribes to the scans, the true poses on /tf, the estimates, the localizability
streams, the scan reports and the tunnel plan. Ground truth is used for two things only,
drawing and scoring, and no estimator ever sees it.

Three modes, matching the setups of ``gazebo.launch.py`` (``demo.launch.py`` is ``ugv``):

* ``ugv``: one robot, two estimators on its scans, ``baseline`` (LiDAR only) and ``markers``
  (LiDAR and marker fixes). One 3D view.
* ``drone``: two drones in two copies of the tunnel, each with its own estimator,
  ``forward`` and ``glance``. One 3D view per drone.
* ``team``: a ground robot and a drone 30 m behind it in one tunnel. The robot mounts strips and
  runs ``markers``; the drone mounts nothing and runs ``solo`` and ``team`` on its own scans,
  which differ only in whether they were told the strips are there. One 3D view each.

The look is an engineering one on purpose: the 3D view is drawn flat the way rviz draws,
on rviz's grey, and the numbers are matplotlib plots with axis labels and units. The run
ends on a figure and a table rather than a slogan.

Frames are driven by scans, not by the wall clock. Output frame f shows run time
``f * playback_speed / fps``, and a scan interval is drawn once every estimate for the scan
that ends it has arrived, or ``max_wait`` seconds after that scan, whichever comes first. A
slow frame delays the video; it does not change it.

Scoring is along-track error: the estimate's offset from the true position along the
direction of travel, after aligning each estimator's frame to the world at the first scan it
processed, the only alignment an odometry is entitled to. For the robot the direction of
travel is its heading. For a drone it is the track heading from the scan report, not the
sensor's heading, which a glancing drone turns away from the tunnel. ``run_pass`` projects
its per-scan rows on the sensor's x axis, so for a drone the two agree only while it looks
forward.

Rendering is on the GPU through EGL when a context can be made (``render_gl``) and on the
CPU otherwise (``render_cpu``), with the same scene either way.
"""
from __future__ import annotations

import dataclasses
import faulthandler
import json
import pathlib
import signal
import time
import traceback

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry as OdometryMsg
from nav_msgs.msg import Path
from PIL import Image, ImageDraw, ImageFont
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA, Float64, Header
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

from locrec.odometry import voxel_keys
from locrec.se3 import inv_T
from locrec_msgs.msg import Localizability, ScanReport

from .conversions import odometry_to_pose, pointcloud2_to_xyz, stamp_to_ns
from .render_gl import look_at, perspective
from .sim_publisher import to_pointcloud2

W, H = 1920, 1080
VIEW_W = 1180
"""The 3D views fill the frame left of this; the plots fill the rest."""
PANEL_W = W - VIEW_W

# rviz's defaults, and matplotlib's tab10 for everything that is a data series
BACKGROUND = (48 / 255, 48 / 255, 48 / 255)
GRID = np.array([0.40, 0.40, 0.41])
MAP_WALL = np.array([0.78, 0.78, 0.78])
MAP_FLOOR = np.array([0.52, 0.52, 0.52])
TRUTH = np.array([1.0, 1.0, 1.0])
SCAN_OK = np.array([1.0, 1.0, 1.0])
TAB = {
    "blue": (0.122, 0.467, 0.706), "orange": (1.0, 0.498, 0.055), "green": (0.173, 0.627, 0.173),
    "red": (0.839, 0.153, 0.157), "grey": (0.498, 0.498, 0.498),
}
MAP_CEILING_CUT = 2.3
"""Map points above this height are not drawn, so the view from above sees into the tunnel."""
FONT_FAMILY = "Liberation Sans"


def rgb8(c) -> tuple[int, int, int]:
    return tuple(int(round(255 * v)) for v in c)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    from matplotlib import font_manager

    prop = font_manager.FontProperties(family=FONT_FAMILY, weight="bold" if bold else "normal")
    return ImageFont.truetype(font_manager.findfont(prop, fallback_to_default=True), size)


def heading_of(T: np.ndarray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


def wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def polyline_segments(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    return np.stack([pts[:-1], pts[1:]], axis=1) if len(pts) > 1 else np.zeros((0, 2, 3))


def arrow(p: np.ndarray, yaw: float, length: float, z: float) -> np.ndarray:
    """An rviz-style flat arrow as line segments: a shaft and two barbs."""
    f = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    left = np.array([-f[1], f[0], 0.0])
    base = np.array([p[0], p[1], z])
    tip = base + length * f
    return np.array([[base, tip], [tip, tip - 0.3 * length * f + 0.18 * length * left],
                     [tip, tip - 0.3 * length * f - 0.18 * length * left]])


class _Named:
    """A renderer whose layer names are all prefixed with one vehicle's name."""

    def __init__(self, renderer, name: str):
        self._r, self._p = renderer, f"{name}/"

    def points(self, name, *args, **kwargs):
        self._r.points(self._p + name, *args, **kwargs)

    def lines(self, name, *args, **kwargs):
        self._r.lines(self._p + name, *args, **kwargs)

    def clear(self, name):
        self._r.clear(self._p + name)


class NullRenderer:
    """Stands in when ``render`` is off: the viewer still joins, scores and publishes."""

    device = "none"

    def points(self, *args, **kwargs) -> None:
        pass

    lines = clear = hud = points

    def render(self, *args, **kwargs):
        return None

    def release(self) -> None:
        pass


@dataclasses.dataclass
class VehicleSpec:
    name: str
    frame: str
    points_topic: str
    report_topic: str
    tracks: tuple[str, ...]
    title: str
    kind: str = "ugv"
    """Which platform this is, ``ugv`` or ``drone``. It used to be read off the mode, which works
    while every vehicle in a mode is the same kind. The team has one of each, and the sensor's
    field of view, the camera's distance and the calibrated threshold all follow the kind."""


@dataclasses.dataclass
class TrackSpec:
    name: str
    label: str
    color: tuple[float, float, float]
    vehicle: str
    short: str = ""
    """Column heading for the closing table, where the full label does not fit. Three columns of
    a 1920 frame leave about 22 per cent of the width each, and "drone on the robot's strips" ran
    into its neighbour. Empty means the label is short enough to use as it is."""


MODES = {
    "ugv": (
        [VehicleSpec("ugv", "sensor", "/points", "/scan_report", ("baseline", "markers"), "")],
        [TrackSpec("baseline", "LiDAR only", TAB["blue"], "ugv"),
         TrackSpec("markers", "LiDAR + marker fixes", TAB["orange"], "ugv")],
    ),
    "drone": (
        [VehicleSpec("forward", "forward/sensor", "/forward/points", "/forward/scan_report", ("forward",),
                     "Forward gaze", "drone"),
         VehicleSpec("glance", "glance/sensor", "/glance/points", "/glance/scan_report", ("glance",),
                     "Glance gaze", "drone")],
        [TrackSpec("forward", "forward gaze", TAB["blue"], "forward"),
         TrackSpec("glance", "glance gaze", TAB["orange"], "glance")],
    ),
    # The team. One tunnel: the robot ahead mounting strips, the drone 30 m behind using them and
    # mounting none of its own. The drone's two estimators run on one stream of scans and differ
    # only in whether they were told the strips are there.
    "team": (
        [VehicleSpec("ugv", "ugv/sensor", "/ugv/points", "/ugv/scan_report", ("markers",),
                     "Ground robot, mounting strips", "ugv"),
         VehicleSpec("drone", "drone/sensor", "/drone/points", "/drone/scan_report", ("solo", "team"),
                     "Drone 30 m behind, mounting nothing", "drone")],
        [TrackSpec("markers", "robot, on its own strips", TAB["orange"], "ugv", "robot"),
         TrackSpec("solo", "drone alone", TAB["blue"], "drone", "drone alone"),
         TrackSpec("team", "drone on the robot's strips", TAB["green"], "drone", "drone + strips")],
    ),
}


def rviz_topics(mode: str) -> set[str]:
    """Every topic the viewer publishes for rviz in ``mode``. The publishers are made from
    this, and the test that holds the rviz configs to the code reads it."""
    vehicles, tracks = MODES[mode]
    out = {"/demo/image", "/demo/markers", "/demo/map"}
    out |= {f"/demo/path/truth_{v.name}" for v in vehicles}
    out |= {f"/demo/scan/{v.name}/{state}" for v in vehicles for state in ("degenerate", "constrained")}
    out |= {f"/demo/path/{t.name}" for t in tracks}
    return out


class Track:
    """One estimator: its estimates and its localizability, keyed by scan stamp."""

    def __init__(self, spec: TrackSpec) -> None:
        self.spec = spec
        self.est: dict[int, np.ndarray] = {}
        self.loc: dict[int, Localizability] = {}
        self.align: np.ndarray | None = None
        self.path: list[np.ndarray] = []

    def world(self, stamp: int, truth: dict) -> np.ndarray | None:
        """The estimate in the world frame, aligned once at the first scan for which both the
        estimate and the true pose have arrived.

        Not simply the first estimate: the simulator's first transform can go out before
        discovery has connected it to this node, while the estimator's first estimate, from a
        publisher matched long before, arrives. Aligning only on the first estimate then never
        aligned at all, and a whole run of one estimator was drawn and scored as missing.
        """
        if stamp not in self.est:
            return None
        if self.align is None:
            common = [s for s in self.est if s in truth]
            if not common:
                return None
            first = min(common)
            self.align = truth[first] @ inv_T(self.est[first])
        return self.align @ self.est[stamp]


class Vehicle:
    """One vehicle: its scans, true poses, track headings and the map drawn from them."""

    def __init__(self, spec: VehicleSpec) -> None:
        self.spec = spec
        self.truth: dict[int, np.ndarray] = {}
        self.scans: dict[int, np.ndarray] = {}
        self.track_yaw: dict[int, float] = {}
        self.truth_path: list[np.ndarray] = []
        self.map_pts = np.zeros((0, 3), dtype=np.float32)
        self.map_keys = np.zeros(0, dtype=np.int64)
        self.cam_heading: float | None = None
        self.last_view: tuple[np.ndarray, np.ndarray] | None = None

    def travel_yaw(self, stamp: int) -> float:
        """Direction of travel: the reported track heading, else the vehicle's own heading."""
        yaw = self.track_yaw.get(stamp)
        return heading_of(self.truth[stamp]) if yaw is None or not np.isfinite(yaw) else float(yaw)


class DemoViewer(Node):
    def __init__(self) -> None:
        super().__init__("locrec_demo_viewer")
        defaults = {
            "mode": "ugv", "simulator": "MuJoCo", "world_name": "mixed", "seed": 1, "length": 300.0,
            "record": "", "rate_hz": 4.0, "playback_speed": 2.75, "fps": 30,
            "renderer": "auto", "thresholds_file": "", "snapshot_dir": "",
            "publish_image": True, "max_wait": 1.0, "idle_finish": 3.0, "results_seconds": 6.0,
            "team_lag": 60,
            "render": True, "rviz_topics": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        def get(name):
            return self.get_parameter(name).value

        self.mode = str(get("mode"))
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}, got {self.mode!r}")
        self.simulator = str(get("simulator"))
        self.world_name = str(get("world_name"))
        self.seed = int(get("seed"))
        self.length = float(get("length"))
        self.rate_hz = float(get("rate_hz"))
        self.speed = float(get("playback_speed"))
        self.fps = int(get("fps"))
        self.team_lag = int(get("team_lag"))
        """Scans the drone trails the robot by in team mode, the driver's own ``team_lag``. Only
        the frame-gap panel uses it, and it has to match or that panel compares two places."""
        self.max_wait = float(get("max_wait"))
        self.idle_finish = float(get("idle_finish"))
        self.results_seconds = float(get("results_seconds"))
        self.publish_image = bool(get("publish_image"))
        snap = str(get("snapshot_dir"))
        self.snapshot_dir = pathlib.Path(snap).expanduser() if snap else None
        vehicle_specs, track_specs = MODES[self.mode]
        self.threshold = None
        self.thresholds: dict[str, float] = {}
        if str(get("thresholds_file")):
            data = json.loads(pathlib.Path(str(get("thresholds_file"))).expanduser().read_text())
            key = {"ugv": "ratio_threshold", "drone": "drone_ratio_threshold"}
            # one threshold per kind: the team has a vehicle of each and they are not the same
            # number, so a single one would call the drone degenerate by the robot's rule
            self.thresholds = {v.name: float(data[key[v.kind]]) for v in vehicle_specs}
            self.threshold = next(iter(self.thresholds.values()))
        self.has_markers = self.mode in ("ugv", "team")
        """Whether strips get mounted in this mode, which decides the third plot."""
        self.degenerate_track = {"ugv": "baseline", "team": "solo"}.get(self.mode)
        """The estimator whose degenerate scans are shaded behind the error plot: the one that is
        not getting the strips, so the shading shows what they are worth and where."""

        self.vehicles = {v.name: Vehicle(v) for v in vehicle_specs}
        self.tracks = {t.name: Track(t) for t in track_specs}
        self.lead = next(iter(self.vehicles.values()))

        self.render_frames = bool(get("render"))
        if str(get("record")) and not self.render_frames:
            raise ValueError("record needs render: a video is made of rendered frames")
        self.view_h = H // len(self.vehicles)
        # one renderer for every view; each vehicle's layers are named after it
        self.renderer = (self._make_renderer(str(get("renderer")), VIEW_W, self.view_h)
                         if self.render_frames else NullRenderer())
        self.rviz = bool(get("rviz_topics"))
        self.focal = 0.5 * VIEW_W / np.tan(np.deg2rad(33.0))
        self.proj = perspective(self.focal, self.focal, VIEW_W * 0.5, self.view_h * 0.5, VIEW_W, self.view_h)

        self.order: list[int] = []
        self.completed_at: dict[int, float] = {}
        self.marker_pos: dict[int, np.ndarray] = {}
        self.marker_stamp: dict[int, int] = {}
        self.seen: dict[int, list[int]] = {}
        self.last_scan_wall = time.monotonic()

        self.next_i = 0
        self.next_t = 0.0
        self.frames = 0
        self.failures = 0
        self.history: list[dict] = []
        self.dist = 0.0
        self.started = False
        self.finished = False
        self.panel: np.ndarray | None = None
        self._plots = None
        self._f = {"title": font(26, True), "text": font(21), "small": font(18)}

        self.writer = None
        self.video_path = None
        record = str(get("record"))
        if record:
            try:
                import imageio.v2 as imageio
                import imageio_ffmpeg  # noqa: F401  the writer needs its ffmpeg binary
            except ImportError as exc:
                raise RuntimeError(
                    "recording needs imageio and imageio-ffmpeg: pip install imageio imageio-ffmpeg"
                ) from exc
            self.video_path = pathlib.Path(record).expanduser()
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(
                str(self.video_path), fps=self.fps, codec="libx264", quality=None,
                macro_block_size=1, pixelformat="yuv420p",
                output_params=["-crf", "20", "-preset", "medium", "-movflags", "+faststart"],
            )
            self.get_logger().info(f"recording to {self.video_path}")
        if self.snapshot_dir is not None:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        topics = rviz_topics(self.mode)
        self.image_pub = self.create_publisher(ImageMsg, "/demo/image", 2)
        # What the viewer scores, as plain numbers on topics, so a plotting tool can draw them
        # live the way it would draw any other signal: |along-track error| against ground truth per
        # estimator, and in the team the drone's distance from the robot's frame at the same place.
        # Only the viewer has the truth to compute them from; no estimator sees these.
        self.error_pubs = {name: self.create_publisher(Float64, f"/demo/error/{name}", 10)
                           for name in self.tracks}
        self.gap_pubs = ({name: self.create_publisher(Float64, f"/demo/frame_gap/{name}", 10)
                          for name in ("solo", "team")} if self.mode == "team" else {})
        if self.rviz:
            self.path_pubs = {t[len("/demo/path/"):]: self.create_publisher(Path, t, 2)
                              for t in topics if t.startswith("/demo/path/")}
            self.scan_pubs = {tuple(t[len("/demo/scan/"):].split("/")): self.create_publisher(PointCloud2, t, 5)
                              for t in topics if t.startswith("/demo/scan/")}
            self.scene_pub = self.create_publisher(MarkerArray, "/demo/markers", 10)
            self.map_pub = self.create_publisher(PointCloud2, "/demo/map", 2)

        self.create_subscription(TFMessage, "/tf", self.on_tf, 200)
        for v in self.vehicles.values():
            self.create_subscription(PointCloud2, v.spec.points_topic, lambda m, v=v: self.on_points(v, m), 50)
            self.create_subscription(ScanReport, v.spec.report_topic, lambda m, v=v: self.on_report(v, m), 200)
        for name in self.tracks:
            self.create_subscription(OdometryMsg, f"/{name}/localizability/estimate",
                                     lambda m, n=name: self.on_estimate(n, m), 200)
            self.create_subscription(Localizability, f"/{name}/localizability",
                                     lambda m, n=name: self.on_loc(n, m), 200)
        self.create_timer(0.01, self._tick)
        self.get_logger().info(
            f"locrec demo viewer up ({self.mode}, {self.simulator}) on {self.renderer.device}, {W}x{H} at "
            f"{self.fps} fps, {self.speed:g}x real time, waiting for the first scan")

    def _make_renderer(self, choice: str, width: int, height: int):
        if choice not in ("auto", "gpu", "cpu"):
            raise ValueError(f"renderer must be auto, gpu or cpu, got {choice!r}")
        if choice in ("auto", "gpu"):
            try:
                from .render_gl import GLRenderer

                return GLRenderer.create(width, height)
            except Exception as exc:  # noqa: BLE001  any failure to make a context means no GPU
                if choice == "gpu":
                    raise
                self.get_logger().warning(f"no GPU context ({type(exc).__name__}: {exc}); rendering on the CPU")
        from .render_cpu import CPURenderer

        return CPURenderer.create(width, height)

    # ---- inputs -------------------------------------------------------------

    def on_points(self, v: Vehicle, msg: PointCloud2) -> None:
        stamp = stamp_to_ns(msg.header.stamp)
        v.scans[stamp] = pointcloud2_to_xyz(msg).astype(np.float32)
        self._complete(stamp)

    def on_tf(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            if t.header.frame_id != "world":
                continue
            v = next((x for x in self.vehicles.values() if x.spec.frame == t.child_frame_id), None)
            if v is None:
                continue
            stamp = stamp_to_ns(t.header.stamp)
            T = np.eye(4)
            q = t.transform.rotation
            T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            tr = t.transform.translation
            T[:3, 3] = [tr.x, tr.y, tr.z]
            v.truth[stamp] = T
            self._complete(stamp)

    def on_report(self, v: Vehicle, msg: ScanReport) -> None:
        stamp = stamp_to_ns(msg.header.stamp)
        v.track_yaw[stamp] = float(msg.track_yaw)
        for d in msg.drops:
            p = d.position_world
            self.marker_pos[int(d.slot)] = np.array([p.x, p.y, p.z])
            self.marker_stamp[int(d.slot)] = stamp
        if msg.detections:
            self.seen[stamp] = [int(o.slot) for o in msg.detections]

    def on_estimate(self, name: str, msg: OdometryMsg) -> None:
        self.tracks[name].est[stamp_to_ns(msg.header.stamp)] = odometry_to_pose(msg)

    def on_loc(self, name: str, msg: Localizability) -> None:
        self.tracks[name].loc[stamp_to_ns(msg.header.stamp)] = msg

    def _complete(self, stamp: int) -> None:
        if stamp in self.completed_at:
            return
        if not all(stamp in v.scans and stamp in v.truth for v in self.vehicles.values()):
            return
        self.completed_at[stamp] = time.monotonic()
        self.last_scan_wall = time.monotonic()
        k = len(self.order)
        while k > 0 and self.order[k - 1] > stamp:
            k -= 1
        self.order.insert(k, stamp)

    # ---- the clock ------------------------------------------------------------

    def _tick(self) -> None:
        if self.finished:
            return
        try:
            if not self.started:
                if self.order:
                    self.started = True
                    for v in self.vehicles.values():
                        self._add_to_map(v, self.order[0])
                        v.truth_path.append(v.truth[self.order[0]][:3, 3].copy())
                    self.next_i = 1
                return
            if self.next_i < len(self.order):
                cur = self.order[self.next_i]
                # Only the estimators that have started gate the interval. In the team the robot
                # drives the first 30 m alone and the drone's two estimators have said nothing
                # yet, so waiting for them made every one of those intervals sit out the full
                # max_wait and the video opened after the drone had joined, with the part the
                # story starts from missing.
                active = [tr for tr in self.tracks.values() if tr.est]
                ready = bool(active) and all(cur in tr.est for tr in active)
                if ready or time.monotonic() - self.completed_at[cur] > self.max_wait:
                    i = self.next_i
                    self.next_i += 1
                    self._render_interval(i)
                return
            if self.writer is not None and time.monotonic() - self.last_scan_wall > self.idle_finish:
                self._results()
                self.close()
                self.finished = True
        except Exception:  # noqa: BLE001  a bad frame must not take the recording down with it
            self.failures += 1
            self.get_logger().error(f"frame failed ({self.failures} so far):\n{traceback.format_exc()}")

    # ---- per scan -----------------------------------------------------------

    def _add_to_map(self, v: Vehicle, stamp: int) -> None:
        T = v.truth[stamp].astype(np.float32)
        world = v.scans[stamp] @ T[:3, :3].T + T[:3, 3]
        world = world[world[:, 2] < MAP_CEILING_CUT]
        keys = voxel_keys(world, 0.14)
        _, first = np.unique(keys, return_index=True)
        fresh = first[~np.isin(keys[first], v.map_keys)]
        v.map_pts = np.vstack([v.map_pts, world[fresh]])
        v.map_keys = np.concatenate([v.map_keys, keys[fresh]])
        colors = np.tile(MAP_WALL.astype(np.float32), (len(v.map_pts), 1))
        colors[v.map_pts[:, 2] < 0.12] = MAP_FLOOR
        self.renderer.points(f"{v.spec.name}/map", v.map_pts, colors, 2.0)

    def _along_track(self, name: str, stamp: int) -> float | None:
        tr = self.tracks[name]
        v = self.vehicles[tr.spec.vehicle]
        world = tr.world(stamp, v.truth)
        if world is None:
            return None
        yaw = v.travel_yaw(stamp)
        return float(np.dot(world[:2, 3] - v.truth[stamp][:2, 3], [np.cos(yaw), np.sin(yaw)]))

    def _degenerate(self, name: str, stamp: int) -> bool | None:
        loc = self.tracks[name].loc.get(stamp)
        return None if loc is None else bool(loc.is_degenerate)

    def _weak_axis(self, name: str, stamp: int) -> np.ndarray | None:
        """The least observable direction in the world frame, horizontal, or None if not degenerate."""
        tr = self.tracks[name]
        loc = tr.loc.get(stamp)
        if loc is None or not loc.is_degenerate:
            return None
        w = np.array(loc.weak_direction, dtype=float)
        if tr.align is not None:
            w = tr.align[:3, :3] @ w
        w[2] = 0.0
        n = np.linalg.norm(w)
        return w / n if n > 1e-6 else None

    def _render_interval(self, i: int) -> None:
        prev, cur = self.order[i - 1], self.order[i]
        lead = self.lead
        self.dist += float(np.linalg.norm(lead.truth[cur][:3, 3] - lead.truth[prev][:3, 3]))
        row = {"stamp": cur, "dist": self.dist,
               "markers": sum(1 for st in self.marker_stamp.values() if st <= cur)}
        for v in self.vehicles.values():
            self._add_to_map(v, cur)
            v.truth_path.append(v.truth[cur][:3, 3].copy())
            row[f"yaw_offset_{v.spec.name}"] = float(np.rad2deg(wrap(heading_of(v.truth[cur]) - v.travel_yaw(cur))))
        for name, tr in self.tracks.items():
            loc = tr.loc.get(cur)
            row[f"ratio_{name}"] = None if loc is None else float(loc.ratio)
            row[f"degenerate_{name}"] = None if loc is None else bool(loc.is_degenerate)
            row[f"fix_{name}"] = bool(loc is not None and loc.landmarks_used > 0)
            row[f"along_{name}"] = self._along_track(name, cur)
            world = tr.world(cur, self.vehicles[tr.spec.vehicle].truth)
            if world is not None:
                tr.path.append(world[:3, 3].copy())
        self.history.append(row)
        self._publish_metrics()
        if self.rviz:
            self._publish_rviz(cur)
        if not self.render_frames:
            self._forget(prev)
            return

        self.panel = self._draw_panel()
        for v in self.vehicles.values():
            self._static_layers(v, cur)
        t0, t1 = (i - 1) / self.rate_hz, i / self.rate_hz
        overlays = {}
        while self.next_t < t1 - 1e-9:
            a = (self.next_t - t0) * self.rate_hz
            views = []
            for name, v in self.vehicles.items():
                view = self._chase_view(v, prev, cur, a)
                if name not in overlays:
                    overlays[name] = self._overlay(v, cur)
                self.renderer.hud(overlays[name])
                self._dynamic_layers(v, prev, cur, a)
                views.append(self.renderer.render(view, self.proj, self.focal, background=BACKGROUND,
                                                  prefix=f"{name}/"))
            self._emit(self._compose(views))
            self.next_t += self.speed / self.fps
        self._forget(prev)

    def _forget(self, prev: int) -> None:
        for v in self.vehicles.values():
            for old in [s for s in v.scans if s < prev]:
                del v.scans[old]

    def _compose(self, views: list) -> np.ndarray | None:
        if any(img is None for img in views) or self.panel is None:
            return None
        return np.hstack([np.vstack(views), self.panel])

    def _static_layers(self, v: Vehicle, cur: int) -> None:
        r = _Named(self.renderer, v.spec.name)
        pts = np.array(v.truth_path)
        pts[:, 2] = 0.03
        r.lines("truth_path", polyline_segments(pts), TRUTH, 2.0)
        for k, name in enumerate(v.spec.tracks):
            path = np.array(self.tracks[name].path) if self.tracks[name].path else np.zeros((0, 3))
            if len(path):
                path[:, 2] = 0.06 + 0.03 * k
            r.lines(f"path_{name}", polyline_segments(path), self.tracks[name].spec.color, 2.0)
        # a local grid on the floor, as rviz draws one, 1 m cells around the vehicle
        c = np.floor(v.truth[cur][:2, 3])
        ticks = np.arange(-30.0, 31.0, 1.0)
        z = -0.01
        segs = [[[c[0] + t, c[1] - 30.0, z], [c[0] + t, c[1] + 30.0, z]] for t in ticks]
        segs += [[[c[0] - 30.0, c[1] + t, z], [c[0] + 30.0, c[1] + t, z]] for t in ticks]
        r.lines("grid", np.array(segs), GRID, 1.0)
        mounted = [s for s, st in self.marker_stamp.items() if st <= cur]
        # every panel that is looking at the tunnel the strips are in draws them, which in the
        # team is both: the drone's panel has to show the strips it is flying past and using
        if mounted:
            base = np.array([self.marker_pos[s] for s in mounted])
            lo, hi = base.copy(), base.copy()
            lo[:, 2] -= 0.5
            hi[:, 2] += 0.5
            r.lines("strips", np.stack([lo, hi], axis=1), TAB["green"], 6.0)
        else:
            r.clear("strips")

    def _dynamic_layers(self, v: Vehicle, prev: int, cur: int, a: float) -> None:
        r = _Named(self.renderer, v.spec.name)
        own = v.spec.tracks[-1]
        degenerate = bool(self._degenerate(own, cur))
        T = v.truth[cur].astype(np.float32)
        r.points("scan", v.scans[cur] @ T[:3, :3].T + T[:3, 3], TAB["red"] if degenerate else SCAN_OK, 3.0)
        pose = self._interp_pose(v.truth[prev], v.truth[cur], a)
        p = pose[:3, 3]
        segs, cols, widths = [arrow(p, heading_of(pose), 1.6, 0.9)], [np.tile(TRUTH, (3, 1))], [np.full(3, 3.0)]
        for name in v.spec.tracks:
            tr = self.tracks[name]
            w0, w1 = tr.world(prev, v.truth), tr.world(cur, v.truth)
            if w0 is not None and w1 is not None:
                q = self._interp_pose(w0, w1, a)
                segs.append(arrow(q[:3, 3], heading_of(q), 1.3, 0.12))
                cols.append(np.tile(tr.spec.color, (3, 1)))
                widths.append(np.full(3, 3.0))
        if v.spec.kind == "drone":
            # the sensor's horizontal field of view, 90 degrees, drawn out to 8 m
            yaw = heading_of(pose)
            for side in (-1.0, 1.0):
                e = p + 8.0 * np.array([np.cos(yaw + side * np.pi / 4), np.sin(yaw + side * np.pi / 4), 0.0])
                segs.append(np.array([[p, e]]))
                cols.append(np.array([self.tracks[own].spec.color]))
                widths.append(np.array([2.0]))
        weak = self._weak_axis(own, cur)
        if weak is not None:
            c = p.copy()
            c[2] = 1.4
            e0, e1 = c - 3.5 * weak, c + 3.5 * weak
            barbs = [[e0, e1]]
            side = np.array([-weak[1], weak[0], 0.0])
            for end, sgn in ((e0, -1.0), (e1, 1.0)):
                barbs += [[end, end - sgn * 0.6 * weak + 0.3 * side], [end, end - sgn * 0.6 * weak - 0.3 * side]]
            segs.append(np.array(barbs))
            cols.append(np.tile(TAB["red"], (len(barbs), 1)))
            widths.append(np.full(len(barbs), 2.0))
        loc = self.tracks[own].loc.get(cur)
        if v.spec.name == "ugv" and loc is not None and loc.landmarks_used > 0:
            for slot in self.seen.get(cur, []):
                if slot in self.marker_pos:
                    segs.append(np.array([[p, self.marker_pos[slot]]]))
                    cols.append(np.array([TAB["green"]]))
                    widths.append(np.array([2.0]))
        r.lines("overlay", np.concatenate(segs), np.concatenate(cols), np.concatenate(widths))

    @staticmethod
    def _interp_pose(T0: np.ndarray, T1: np.ndarray, a: float) -> np.ndarray:
        h0, h1 = heading_of(T0), heading_of(T1)
        h = h0 + a * np.arctan2(np.sin(h1 - h0), np.cos(h1 - h0))
        T = np.eye(4)
        T[:2, :2] = [[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]]
        T[:3, 3] = (1.0 - a) * T0[:3, 3] + a * T1[:3, 3]
        return T

    def _chase_view(self, v: Vehicle, prev: int, cur: int, a: float) -> np.ndarray:
        """Behind, to the left and above the direction of travel, turning with it slowly."""
        pose = self._interp_pose(v.truth[prev], v.truth[cur], a)
        y0, y1 = v.travel_yaw(prev), v.travel_yaw(cur)
        heading = y0 + a * wrap(y1 - y0)
        if v.cam_heading is None:
            v.cam_heading = heading
        v.cam_heading += (1.0 - np.exp(-(self.speed / self.fps) / 1.6)) * wrap(heading - v.cam_heading)
        fwd = np.array([np.cos(v.cam_heading), np.sin(v.cam_heading), 0.0])
        left = np.array([-fwd[1], fwd[0], 0.0])
        p = pose[:3, 3]
        near = 1.0 if v.spec.kind == "ugv" else 1.25
        eye = p - near * 8.0 * fwd - near * 7.0 * left + np.array([0.0, 0.0, near * 9.5])
        target = p + 2.5 * fwd + np.array([0.0, 0.0, 0.3])
        v.last_view = (eye, target)
        return look_at(eye, target)

    def _emit(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        if self.writer is not None:
            self.writer.append_data(frame)
        self.frames += 1
        if self.snapshot_dir is not None and self.frames % 150 == 0:
            Image.fromarray(frame).save(self.snapshot_dir / f"frame_{self.frames:05d}.png")
        if self.publish_image and self.frames % 3 == 0:
            half = np.ascontiguousarray(frame[::2, ::2])
            msg = ImageMsg(height=half.shape[0], width=half.shape[1], encoding="rgb8",
                           step=half.shape[1] * 3, data=half.tobytes())
            msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(msg)

    # ---- text on the 3D view ------------------------------------------------------

    def _overlay(self, v: Vehicle, cur: int) -> np.ndarray:
        img = Image.new("RGBA", (VIEW_W, self.view_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        f = self._f
        white, grey = (255, 255, 255, 255), (200, 200, 200, 255)
        x, y = 22, 16
        if v.spec.title:
            d.text((x, y), v.spec.title, font=f["title"], fill=white)
            y += 38
        elif v is self.lead:
            what = ("Ground robot, 360\u00b0 LiDAR" if v.spec.kind == "ugv"
                    else "Drones, 90\u00b0 LiDAR")
            d.text((x, y), f"{what}, {self.world_name} tunnel, seed {self.seed}", font=f["title"], fill=white)
            y += 38
        if v is self.lead:
            d.text((x, y), f"{self.simulator} simulation, ROS 2 Jazzy, run shown at {self.speed:g}x",
                   font=f["small"], fill=grey)
            y += 30
        own = v.spec.tracks[-1]
        loc = self.tracks[own].loc.get(cur)
        threshold = self.thresholds.get(v.spec.name, self.threshold)
        if loc is not None and threshold is not None:
            state = "degenerate" if loc.is_degenerate else "constrained"
            color = rgb8(TAB["red"]) + (255,) if loc.is_degenerate else white
            d.text((x, y), f"λmin/λmax = {loc.ratio:.2e}   threshold {threshold:.2e}   {state}",
                   font=f["text"], fill=color)
            y += 30
        d.text((x, y), f"s = {self.dist:.1f} m", font=f["text"], fill=white)

        # legend, top right, on a plain panel so it reads over the points
        entries = [("ground truth", TRUTH)] + [(self.tracks[t].spec.label, self.tracks[t].spec.color)
                                              for t in v.spec.tracks]
        if v.spec.name == "ugv":
            entries.append(("marker strip", TAB["green"]))
        entries += [("scan", SCAN_OK), ("scan, degenerate", TAB["red"])]
        width = 44 + max(d.textlength(label, font=f["small"]) for label, _ in entries) + 24
        lx, ly = VIEW_W - width - 16, 16
        d.rectangle([lx, ly, lx + width, ly + 14 + 26 * len(entries)], fill=(40, 40, 40, 235), outline=(90, 90, 90, 255))
        ly += 8
        for label, color in entries:
            d.line([(lx + 12, ly + 11), (lx + 46, ly + 11)], fill=rgb8(color) + (255,), width=4)
            d.text((lx + 56, ly), label, font=f["small"], fill=white)
            ly += 26
        return np.asarray(img)

    # ---- the plots ------------------------------------------------------------------

    def _figure(self, width_px: int, height_px: int):
        import matplotlib

        matplotlib.rcParams.update({
            "font.family": "sans-serif", "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"], "font.size": 12,
            "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": 0.7, "axes.titlesize": 12,
            "axes.labelsize": 12, "legend.fontsize": 11, "legend.frameon": True, "legend.framealpha": 1.0,
        })
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(width_px / 100.0, height_px / 100.0), dpi=100, facecolor="white")
        FigureCanvasAgg(fig)
        return fig

    def _series(self, key: str):
        xs = [r["dist"] for r in self.history if r.get(key) is not None]
        ys = [r[key] for r in self.history if r.get(key) is not None]
        return np.array(xs), np.array(ys)

    def _draw_panel(self) -> np.ndarray:
        if self._plots is None:
            fig = self._figure(PANEL_W, H)
            axes = fig.subplots(3, 1, sharex=True)
            fig.subplots_adjust(left=0.15, right=0.96, top=0.95, bottom=0.07, hspace=0.28)
            lines = {}
            ax = axes[0]
            ax.set_yscale("log")
            ax.set_ylim(3e-4, 1.0)
            ax.set_ylabel("λmin / λmax [-]")
            ax.set_title("Localizability of the registration")
            for name, tr in self.tracks.items():
                lines[f"ratio_{name}"], = ax.plot([], [], color=tr.spec.color, lw=1.4, label=tr.spec.label)
            if self.threshold is not None:
                ax.axhline(self.threshold, color="black", ls="--", lw=1.0, label="threshold")
            ax.legend(loc="upper right", ncol=1)
            ax = axes[1]
            ax.set_ylabel("|along-track error| [m]")
            ax.set_title("Estimate against ground truth")
            for name, tr in self.tracks.items():
                lines[f"along_{name}"], = ax.plot([], [], color=tr.spec.color, lw=1.6, label=tr.spec.label)
            ax.legend(loc="upper left")
            ax = axes[2]
            if self.mode == "team":
                # The middle plot is each vehicle's error against the world, and on a seed where
                # the drone is lucky alone that plot says the strips did nothing. It is the wrong
                # axis for this claim: what the team buys is agreement with the robot, not
                # accuracy against a truth neither vehicle can see. So this panel carries the
                # declared statistic, the gap between the drone's error and the robot's at the
                # same place in the tunnel, and it is labelled as the different quantity it is.
                ax.set_ylabel("distance from the robot's frame [m]")
                ax.set_title("How far the drone is from the frame it is navigating in")
                for name in ("solo", "team"):
                    lines[f"gap_{name}"], = ax.plot([], [], color=self.tracks[name].spec.color,
                                                    lw=1.6, label=self.tracks[name].spec.label)
                ax.legend(loc="upper left")
            elif self.has_markers:
                from matplotlib.ticker import MaxNLocator

                ax.set_ylabel("markers mounted [-]")
                ax.set_title("Markers, mounted where the scheduler asked")
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                lines["markers"], = ax.plot([], [], color=TAB["green"], lw=1.6, drawstyle="steps-post")
            else:
                ax.set_ylabel("sensor yaw \u2212 track heading [\u00b0]")
                ax.set_title("Where each sensor is looking")
                ax.set_ylim(-190, 190)
                ax.set_yticks([-180, -90, 0, 90, 180])
                for name, v in self.vehicles.items():
                    tr = self.tracks[v.spec.tracks[-1]]
                    lines[f"yaw_offset_{name}"], = ax.plot([], [], color=tr.spec.color, lw=1.4, label=tr.spec.label)
                ax.legend(loc="upper right")
            ax.set_xlabel("distance travelled [m]")
            for ax in axes:
                ax.set_xlim(0.0, max(self.length, 1.0))
            self._plots = (fig, axes, lines, [])
        fig, axes, lines, shades = self._plots
        for shade in shades:
            shade.remove()
        shades.clear()
        for key, line in lines.items():
            if key.startswith("gap_"):
                x, y = self._frame_gap(key[4:])
            elif key == "markers":
                x, y = self._series("markers")
            else:
                x, y = self._series(key)
                if key.startswith("along_"):
                    y = np.abs(y)
            line.set_data(x, y)
        along = [abs(r[k]) for r in self.history for k in r if k.startswith("along_") and r[k] is not None]
        axes[1].set_ylim(0.0, max(0.5, 1.15 * max(along)) if along else 0.5)
        if self.mode == "team":
            gaps = [y for name in ("solo", "team") for y in self._frame_gap(name)[1]]
            axes[2].set_ylim(0.0, max(0.5, 1.15 * max(gaps)) if gaps else 0.5)
        elif self.has_markers:
            axes[2].set_ylim(-0.5, max(3, 1.2 * self.history[-1]["markers"]) + 0.5)
        # shade where the estimator without the strips called its scan degenerate, whichever
        # mode drew the third panel
        if self.degenerate_track is not None:
            x = np.array([r["dist"] for r in self.history])
            deg = np.array([bool(r.get(f"degenerate_{self.degenerate_track}")) for r in self.history])
            if deg.any():
                shades.append(axes[1].fill_between(x, 0, 1, where=deg, step="post", color="0.88", lw=0,
                                                   transform=axes[1].get_xaxis_transform(), zorder=0))
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    # ---- rviz ---------------------------------------------------------------

    def _publish_rviz(self, cur: int) -> None:
        """What the interval knows, as rviz draws it. Everything is in the world frame."""
        from builtin_interfaces.msg import Time

        stamp = Time(sec=cur // 1_000_000_000, nanosec=cur % 1_000_000_000)
        header = Header(stamp=stamp, frame_id="world")

        def path_msg(points) -> Path:
            msg = Path(header=header)
            for p in points:
                ps = PoseStamped(header=header)
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
                ps.pose.orientation.w = 1.0
                msg.poses.append(ps)
            return msg

        markers = MarkerArray()

        def marker(ns, mid, kind, color, alpha=1.0):
            m = Marker(header=header, ns=ns, id=mid, type=kind, action=Marker.ADD)
            m.pose.orientation.w = 1.0
            m.color = ColorRGBA(r=float(color[0]), g=float(color[1]), b=float(color[2]), a=alpha)
            markers.markers.append(m)
            return m

        empty = np.zeros((0, 3), dtype=np.float32)
        maps = []
        for vi, (name, v) in enumerate(self.vehicles.items()):
            self.path_pubs[f"truth_{name}"].publish(path_msg(v.truth_path))
            T = v.truth[cur]
            world = v.scans[cur] @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)
            own = v.spec.tracks[-1]
            deg = bool(self._degenerate(own, cur))
            self.scan_pubs[(name, "degenerate")].publish(to_pointcloud2(world if deg else empty, stamp, "world"))
            self.scan_pubs[(name, "constrained")].publish(to_pointcloud2(empty if deg else world, stamp, "world"))
            maps.append(v.map_pts)
            poses = [("truth", TRUTH, T)] + [(t, self.tracks[t].spec.color, self.tracks[t].world(cur, v.truth))
                                             for t in v.spec.tracks]
            for k, (label, color, pose) in enumerate(poses):
                if pose is None:
                    continue
                m = marker(f"poses {name}", k, Marker.ARROW, color)
                m.pose.position.x, m.pose.position.y = float(pose[0, 3]), float(pose[1, 3])
                m.pose.position.z = 0.25 + 0.2 * k
                qx, qy, qz, qw = Rotation.from_euler("z", heading_of(pose)).as_quat()
                m.pose.orientation.x, m.pose.orientation.y = float(qx), float(qy)
                m.pose.orientation.z, m.pose.orientation.w = float(qz), float(qw)
                m.scale.x, m.scale.y, m.scale.z = 1.4, 0.18, 0.25
            text = marker(f"status {name}", 0, Marker.TEXT_VIEW_FACING, TAB["red"] if deg else TRUTH)
            text.pose.position.x, text.pose.position.y, text.pose.position.z = float(T[0, 3]), float(T[1, 3]), 3.4
            text.scale.z = 0.55
            loc = self.tracks[own].loc.get(cur)
            label = v.spec.title + ": " if v.spec.title else ""
            text.text = label + ("waiting" if loc is None else
                                 f"ratio {loc.ratio:.2e}, {'degenerate' if deg else 'constrained'}")
            weak = self._weak_axis(own, cur)
            axis = marker(f"weak direction {name}", vi, Marker.LINE_LIST, TAB["red"])
            axis.scale.x = 0.08
            if weak is None:
                axis.action = Marker.DELETE
            else:
                c = T[:3, 3] + np.array([0.0, 0.0, 0.8])
                axis.points = [Point(x=float(c[0] - 3.0 * weak[0]), y=float(c[1] - 3.0 * weak[1]), z=float(c[2])),
                               Point(x=float(c[0] + 3.0 * weak[0]), y=float(c[1] + 3.0 * weak[1]), z=float(c[2]))]
        for name in self.tracks:
            self.path_pubs[name].publish(path_msg(self.tracks[name].path))
        for slot, st in self.marker_stamp.items():
            if st > cur:
                continue
            p = self.marker_pos[slot]
            strip = marker("marker strips", slot, Marker.CUBE, TAB["green"])
            strip.pose.position.x, strip.pose.position.y, strip.pose.position.z = map(float, p)
            strip.scale.x, strip.scale.y, strip.scale.z = 0.1, 0.1, 1.0
        if len(self.history) % 4 == 1:
            self.map_pub.publish(to_pointcloud2(np.vstack(maps), stamp, "world"))
        self.scene_pub.publish(markers)

    # ---- the end ------------------------------------------------------------------

    def _stats(self, name: str) -> dict:
        """Mean, worst and final absolute along-track error over the whole run.

        Not the last scan alone: the dead-end wall comes into view in the last few metres,
        and on MuJoCo seed 1 the markers estimate fell from 0.24 m to 0.04 m in one scan
        there while the LiDAR-only one jumped from 1.23 m to 1.50 m, so the last scan said
        38 times better where the run said about five.
        """
        v = [abs(r[f"along_{name}"]) for r in self.history if r.get(f"along_{name}") is not None]
        if not v:
            return {"mean": None, "max": None, "final": None}
        return {"mean": float(np.mean(v)), "max": float(np.max(v)), "final": float(v[-1])}

    def _results_figure(self) -> np.ndarray:
        fig = self._figure(W, H)
        grid = fig.add_gridspec(2, 2, width_ratios=[2.1, 1.0], left=0.06, right=0.97, top=0.88, bottom=0.08,
                                wspace=0.12, hspace=0.32)
        vehicle = {"ugv": "Ground robot, 360\u00b0 LiDAR",
                   "drone": "Two drones, 90\u00b0 LiDAR",
                   "team": "Ground robot and a drone 30 m behind it, one tunnel"}[self.mode]
        fig.text(0.06, 0.945, f"{vehicle}: {self.dist:.0f} m of {self.world_name} tunnel, seed {self.seed}",
                 fontsize=20, fontweight="bold")
        fig.text(0.06, 0.912, f"{self.simulator} simulation, ROS 2 Jazzy. One run on one seed, not a benchmark.",
                 fontsize=13, color="0.3")
        x = np.array([r["dist"] for r in self.history])
        ax = fig.add_subplot(grid[0, 0])
        for name, tr in self.tracks.items():
            xs, ys = self._series(f"along_{name}")
            ax.plot(xs, np.abs(ys), color=tr.spec.color, lw=1.8, label=tr.spec.label)
        if self.degenerate_track is not None:
            deg = np.array([bool(r.get(f"degenerate_{self.degenerate_track}")) for r in self.history])
            if deg.any():
                ax.fill_between(x, 0, 1, where=deg, step="post", color="0.88", lw=0,
                                transform=ax.get_xaxis_transform(), zorder=0,
                                label=f"degenerate ({self.tracks[self.degenerate_track].spec.label})")
        ax.set_ylabel("|along-track error| [m]")
        ax.set_xlim(0, max(x[-1], 1.0))
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper left")
        ax.set_title("Along-track error")
        ax2 = fig.add_subplot(grid[1, 0], sharex=ax)
        for name, tr in self.tracks.items():
            xs, ys = self._series(f"ratio_{name}")
            ax2.plot(xs, ys, color=tr.spec.color, lw=1.2, label=tr.spec.label)
        if self.threshold is not None:
            ax2.axhline(self.threshold, color="black", ls="--", lw=1.0, label=f"threshold {self.threshold:.2e}")
        ax2.set_yscale("log")
        ax2.set_ylim(3e-4, 1.0)
        ax2.set_ylabel("λmin / λmax [-]")
        ax2.set_xlabel("distance travelled [m]")
        ax2.legend(loc="upper right")
        ax2.set_title("Localizability")

        tab = fig.add_subplot(grid[:, 1])
        tab.axis("off")
        cols = [self.tracks[t].spec.short or self.tracks[t].spec.label for t in self.tracks]
        stats = {t: self._stats(t) for t in self.tracks}

        def cell(v, unit=" m"):
            return "n/a" if v is None else f"{v:.2f}{unit}"

        rows = [["mean |error|"] + [cell(stats[t]["mean"]) for t in self.tracks],
                ["worst |error|"] + [cell(stats[t]["max"]) for t in self.tracks],
                ["final |error|"] + [cell(stats[t]["final"]) for t in self.tracks]]
        n_deg = {t: sum(1 for r in self.history if r.get(f"degenerate_{t}")) for t in self.tracks}
        rows.append(["degenerate scans"] + [f"{100.0 * n_deg[t] / max(len(self.history), 1):.0f} %" for t in self.tracks])
        if self.mode == "ugv":
            rows.append(["markers mounted", "0", f"{len(self.marker_stamp)}"])
            rows.append(["scans with a marker fix", "0", f"{sum(1 for r in self.history if r.get('fix_markers'))}"])
        elif self.mode == "team":
            # who mounted what, which is the whole point: the drone's column is a zero
            rows.append(["strips mounted"]
                        + [f"{len(self.marker_stamp)}" if self.tracks[t].spec.vehicle == "ugv" else "0"
                           for t in self.tracks])
            rows.append(["scans with a strip fix"]
                        + [f"{sum(1 for r in self.history if r.get(f'fix_{t}'))}" for t in self.tracks])
            # the declared statistic, and the reason the middle plot is not the whole story
            gap = {t: self._frame_gap(t)[1] for t in self.tracks if self.tracks[t].spec.vehicle != "ugv"}
            rows.append(["mean frame gap"]
                        + [("-" if t not in gap or not len(gap[t]) else f"{float(np.mean(gap[t])):.2f} m")
                           for t in self.tracks])
            rows.append(["worst frame gap"]
                        + [("-" if t not in gap or not len(gap[t]) else f"{float(np.max(gap[t])):.2f} m")
                           for t in self.tracks])
        else:
            yaw = {v: np.array([abs(r[f"yaw_offset_{v}"]) for r in self.history]) for v in self.vehicles}
            rows.append(["sensor > 45\u00b0 off track"] + [f"{100.0 * np.mean(yaw[v] > 45.0):.0f} %" for v in self.vehicles])
        # one width per column, however many estimators there are: the team has three
        label_w = 0.40 if len(cols) < 3 else 0.34
        table = tab.table(cellText=rows, colLabels=[""] + cols, loc="upper left", cellLoc="center",
                          colWidths=[label_w] + [(1.0 - label_w) / len(cols)] * len(cols),
                          bbox=[0.0, 0.50, 1.0, 0.44])
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        for (row, col), c in table.get_celld().items():
            c.set_edgecolor("0.6")
            if row == 0:
                c.set_text_props(fontweight="bold")
            if col == 0:
                c.set_text_props(ha="left")
                c.PAD = 0.04
        tab.text(0.0, 0.44, "\n".join(self._notes()), fontsize=12.5, va="top", color="0.2",
                 transform=tab.transAxes, linespacing=1.5)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    def _publish_metrics(self) -> None:
        """The latest row's scores, one Float64 per signal, for a live plot."""
        row = self.history[-1]
        for name, pub in self.error_pubs.items():
            e = row.get(f"along_{name}")
            if e is not None:
                pub.publish(Float64(data=abs(float(e))))
        i = len(self.history) - 1
        if self.gap_pubs and i >= self.team_lag:
            # the drone at this row is where the robot was team_lag rows ago (see _frame_gap)
            theirs = self.history[i - self.team_lag].get("along_markers")
            for name, pub in self.gap_pubs.items():
                mine = row.get(f"along_{name}")
                if mine is not None and theirs is not None:
                    pub.publish(Float64(data=abs(float(mine) - float(theirs))))

    def _frame_gap(self, name: str):
        """The drone's distance from the robot's frame, **at the same place in the tunnel**.

        The drone trails the robot by ``team_lag`` scans, so at any row the two vehicles are
        30 m apart and comparing them there compares different places. The drone at row i is where
        the robot was at row i minus the lag, and that is the row its error is measured against;
        the x value is the drone's own distance travelled.

        The lag is configured, not inferred. Inferring it from the first row carrying one of the
        drone's tracks gives zero, because the viewer only draws an interval once every vehicle
        has a scan at that stamp and the drone's first scan is the first such stamp. That read the
        two vehicles 30 m apart as though they were together and put 0.17 m on screen where the
        truth is 0.10 m."""
        lag = self.team_lag
        xs, ys = [], []
        for i in range(lag, len(self.history)):
            mine = self.history[i].get(f"along_{name}")
            theirs = self.history[i - lag].get("along_markers")
            if mine is None or theirs is None:
                continue
            xs.append(self.history[i - lag]["dist"])
            ys.append(abs(float(mine) - float(theirs)))
        return np.array(xs), np.array(ys)

    def _notes(self) -> list[str]:
        """What one run cannot show, in words. The numbers are the study's grids, cited by file."""
        if self.mode == "team":
            return [
                "One tunnel. The robot goes first and mounts a strip",
                "wherever its own registration goes degenerate. The",
                "drone follows 30 m behind, mounts nothing, and runs",
                "two estimators on one stream of scans: one that",
                "ignores the strips and one told only where each is,",
                "how well the robot knew that, and which way it faces.",
                "",
                "Over 8 seeds at 300 m on this sensor, the drone that",
                "uses the strips ends within 1 to 14 cm of the robot's",
                "frame on every seed, against a median 1.85 m alone;",
                "paired benefit +1.78 m [+0.73, +3.41]. Over the whole",
                "run the two vehicles' frames stay a median 0.06 m",
                "apart, worst 0.17 m, against 0.86 m and 2.61 m for the",
                "drone alone (locrec/results/team_pass_gazebo.csv).",
                "",
                "It agrees with the robot; it does not beat it. In the",
                "world frame the drone inherits the robot's own chain",
                "error, and on the three seeds where the drone was",
                "lucky alone, taking the robot's frame made it worse.",
                "That is the honest shape of the result: a shared",
                "frame, not free accuracy.",
            ]
        if self.mode == "ugv":
            notes = [
                "Both estimators register the same scans with the same",
                "odometry noise. The marker estimator also corrects",
                "against strips mounted where its scheduler asked.",
                "Mean and worst are over every scan, not the last one.",
                "",
            ]
            if self.simulator == "Gazebo":
                notes += [
                    "Over 8 seeds on this sensor, median final along-track",
                    "error fell from 2.21 m to 0.85 m with markers, better",
                    "on 5 of 8 seeds (locrec/results/gazebo_crosscheck_ugv.csv).",
                ]
            else:
                notes += ["Markers do not win on every seed: see locrec_ros/README.md."]
            return notes
        notes = [
            "Both drones fly the same tunnel with the same odometry",
            "noise. The glance drone may turn its sensor back to",
            "structure for up to 3 s when its scans go degenerate,",
            "then must look forward for 5 s.",
            "",
            "8 seeds of this world, final total error (the README's",
            "Total: the hypotenuse of median along and lateral):",
            "MuJoCo's grid had glance ahead, 1.17 m against 1.59 m.",
        ]
        if self.simulator == "Gazebo":
            notes += [
                "On Gazebo's LiDAR they are level, 1.89 m against 1.91 m,",
                "and glance ends further off on 5 of 8 seeds",
                "(locrec/results/gazebo_crosscheck_drone.csv). No difference",
                "either way survives 8 seeds.",
            ]
        return notes

    def _results(self) -> None:
        if not self.history:
            return
        figure = self._results_figure()
        for _ in range(int(round(self.results_seconds * self.fps))):
            self._emit(figure)

    # ---- shutdown -------------------------------------------------------------

    def close(self) -> None:
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None
        csv_path = self.video_path.with_suffix(".csv")
        keys = [k for k in (self.history[0] if self.history else {"dist": 0.0}) if k != "stamp"]

        def cell(value) -> str:
            if value is None:
                return ""
            if isinstance(value, bool):
                return str(int(value))
            return f"{value:.6g}" if isinstance(value, float) else str(value)

        with open(csv_path, "w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in self.history:
                fh.write(",".join(cell(r.get(k)) for k in keys) + "\n")
        summary = ", ".join(
            f"{self.tracks[t].spec.label} mean {s['mean']:.2f} m worst {s['max']:.2f} m" if s["mean"] is not None
            else f"{self.tracks[t].spec.label} n/a" for t, s in ((t, self._stats(t)) for t in self.tracks))
        self.get_logger().info(
            f"video closed: {self.video_path}, {self.frames} frames, {self.frames / self.fps:.1f} s; "
            f"{len(self.history)} scan intervals at {self.speed:g}x; {self.failures} failed frames; "
            f"along-track error: {summary}; {len(self.marker_stamp)} markers; per-scan numbers in {csv_path}")


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
    node = DemoViewer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
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
