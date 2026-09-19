"""Gazebo as a second simulator for the same tunnels, with no ROS import.

``TunnelSim`` ray casts MuJoCo geometry. This module puts the identical tunnel into
Gazebo (gz-sim 8) and takes the LiDAR from Gazebo's GPU ray sensor instead, so the same
harness, the same estimator and the same policies can run on a second, independent
sensor model. Three pieces:

* **Export.** Every box of ``build_world``'s XML becomes a Gazebo visual with the same
  pose and size (MuJoCo box sizes are half extents, SDF's are full ones). They are read
  back from the XML rather than regenerated, so the two simulators cannot drift apart.
  Gazebo's GPU LiDAR renders visuals, not collision shapes, so the shell has no
  collision geometry, and no physics acts on anything: vehicles are placed pose by
  pose, as ``run_pass`` places them.
* **Link.** A paused ``gz sim`` server, stepped over gz-transport. A scan is taken by
  setting the vehicle's pose, stepping until the sensor is due, and waiting for the
  cloud stamped with that simulated time, so every scan belongs to exactly one pose.
* **GazeboTunnelSim.** A ``TunnelSim`` whose ``scan`` is Gazebo's. ``run_pass`` takes it
  unchanged.

What is still MuJoCo, on purpose: the sideways probe that measures the wall before a
marker is mounted (``TunnelSim.drop_marker_on_wall``), which is one ray against the same
geometry, and the ground truth of which strip a return belongs to.

What differs from MuJoCo, on purpose:

* The vehicle's own visuals are hidden from its LiDAR with Gazebo's visibility flags.
  MuJoCo draws no vehicle, so nothing of one can be in its scans.
* A marker strip carries a ``laser_retro`` value, and Gazebo returns it as the intensity
  of every beam that hits the strip. Returns are picked out by that intensity, the way a
  retroreflector is picked out of rock. MuJoCo picks them by geom id. Which strip a
  return belongs to is still ground truth in both: the nearest mounted strip.

Measured differences between the two sensors are in ``docs/DECISIONS.md``.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

from .landmarks import beam_columns, fit_strip_centre
from .lidar import LidarSpec, Scan
from .sim import MarkerDetection, PlatformSpec, TunnelSim
from .worlds import TunnelWorld, WorldSpec

__all__ = [
    "LIDAR_SEES", "HIDDEN_FROM_LIDAR", "MARKER_RETRO", "STEP_SIZE",
    "ShellBox", "shell_boxes", "box_mesh_obj", "MESH_GROUPS", "horizontal_samples", "lidar_sensor", "vehicle_model", "marker_model", "world_sdf",
    "side_by_side_offset", "steps_per_scan",
    "GazeboServer", "GazeboLink", "LidarFrame", "GazeboScan", "GazeboTunnelSim",
]

LIDAR_SEES = 0x1
"""Visibility mask of every LiDAR, and a bit of every visual it should see."""
HIDDEN_FROM_LIDAR = 0x2
"""Visibility flags of vehicle visuals: drawn in the GUI, absent from scans."""
MARKER_RETRO = 200.0
"""``laser_retro`` of a marker strip. Rock has none, and its returns read 0."""
STEP_SIZE = 0.05
"""Simulated seconds per server iteration. A scan interval is several iterations, so a
viewer sees the vehicle move between scans instead of jumping."""
MARKER_ASSOCIATION_RADIUS = 0.6
"""A retro return further than this from every mounted strip's centre belongs to none.
Half the strip's height plus a margin for range noise."""

WALL_RGB = (0.62, 0.62, 0.62)
FLOOR_RGB = (0.42, 0.42, 0.42)
MARKER_RGB = (0.173, 0.627, 0.173)  # matplotlib tab:green
VEHICLE_RGB = {"orange": (1.0, 0.498, 0.055), "blue": (0.122, 0.467, 0.706)}  # tab:orange, tab:blue
_SYSTEM_EGL_VENDORS = "/usr/share/glvnd/egl_vendor.d"


def steps_per_scan(platform: PlatformSpec = PlatformSpec(), step_size: float = STEP_SIZE) -> int:
    """Server iterations in one scan interval: the platform's time per step over ``step_size``."""
    period = platform.step_length / max(platform.speed_mps, 1e-9)
    n = int(round(period / step_size))
    if n < 1 or abs(n * step_size - period) > 1e-9:
        raise ValueError(f"scan period {period} s is not a whole number of {step_size} s steps")
    return n


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ShellBox:
    name: str
    centre: tuple[float, float, float]
    size: tuple[float, float, float]
    """Full extents, metres, along the box's own axes."""
    yaw: float


def shell_boxes(world: TunnelWorld) -> list[ShellBox]:
    """Every tunnel box in the world's MuJoCo XML, markers excluded."""
    root = ET.fromstring(world.xml)
    compiler = root.find("compiler")
    if compiler is None or compiler.get("angle") != "radian":
        raise ValueError('expected <compiler angle="radian">; euler angles would be misread')
    boxes = []
    for geom in root.find("worldbody").findall("geom"):
        if geom.get("class") != "shell" or geom.get("type") != "box":
            continue
        euler = [float(v) for v in geom.get("euler", "0 0 0").split()]
        if abs(euler[0]) > 1e-9 or abs(euler[1]) > 1e-9:
            raise ValueError(f"{geom.get('name')}: only yaw is exported, got euler {euler}")
        half = [float(v) for v in geom.get("size").split()]
        boxes.append(ShellBox(
            name=geom.get("name"),
            centre=tuple(float(v) for v in geom.get("pos").split()),
            size=tuple(2.0 * h for h in half),
            yaw=euler[2],
        ))
    return boxes


def _f(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _material(rgb, emissive: float = 0.0) -> str:
    amb = [0.6 * c for c in rgb]
    em = [emissive * c for c in rgb]
    return (f"<material><ambient>{_f(amb)} 1</ambient><diffuse>{_f(rgb)} 1</diffuse>"
            f"<specular>0.1 0.1 0.1 1</specular><emissive>{_f(em)} 1</emissive></material>")


def horizontal_samples(spec: LidarSpec, oversample: int) -> int:
    """Beams Gazebo renders across the field of view so that every ``oversample``-th one is
    exactly a beam of ``spec``.

    Gazebo spreads its samples from ``min_angle`` to ``max_angle`` inclusive. A full circle
    in locrec starts at -180 degrees and stops one step short of +180, so ``n * k`` samples
    stopping one fine step short put every k-th on a locrec beam. A limited field of view
    in locrec is ``np.linspace`` over it, and ``k * (n - 1) + 1`` samples over the same span
    put every k-th on one.
    """
    k = int(oversample)
    if k < 1:
        raise ValueError("oversample must be at least 1")
    if spec.fov_azimuth_deg >= 359.999:
        return spec.n_azimuth * k
    return k * (spec.n_azimuth - 1) + 1


def lidar_sensor(spec: LidarSpec, name: str, topic: str, update_rate: float, oversample: int = 1) -> str:
    """A ``gpu_lidar`` casting the beams ``locrec.lidar.Lidar`` casts with this spec, and
    ``oversample - 1`` more between each horizontal pair, which the reader drops.

    The extra beams are there for the depth texture, not for their ranges. Gazebo renders
    each 90 degree cube face at a size set by the beam count, rounded up to a power of two
    and clamped to 128..1024 texels, and then reads every range out of that texture. At
    360 beams a face is 128 texels, 0.70 degrees each, and a beam that meets a wall or the
    floor at a grazing angle picks up a range error of tens of centimetres.
    """
    if spec.dropout > 0.0:
        raise ValueError("beam dropout has no Gazebo counterpart here")
    n = horizontal_samples(spec, oversample)
    if spec.fov_azimuth_deg >= 359.999:
        h_min, h_max = -np.pi, np.pi - 2.0 * np.pi / n
    else:
        half = 0.5 * np.deg2rad(spec.fov_azimuth_deg)
        h_min, h_max = -half, half
    half_el = 0.5 * np.deg2rad(spec.fov_elevation_deg) if spec.n_elevation > 1 else 0.0
    noise = ""
    if spec.range_sigma > 0.0:
        noise = (f"<noise><type>gaussian</type><mean>0</mean>"
                 f"<stddev>{spec.range_sigma:.6g}</stddev></noise>")
    return f"""
        <sensor name="{name}" type="gpu_lidar">
          <topic>{topic}</topic>
          <update_rate>{update_rate:.9g}</update_rate>
          <always_on>true</always_on>
          <visualize>false</visualize>
          <lidar>
            <scan>
              <horizontal><samples>{n}</samples><resolution>1</resolution>
                <min_angle>{h_min:.9f}</min_angle><max_angle>{h_max:.9f}</max_angle></horizontal>
              <vertical><samples>{spec.n_elevation}</samples><resolution>1</resolution>
                <min_angle>{-half_el:.9f}</min_angle><max_angle>{half_el:.9f}</max_angle></vertical>
            </scan>
            <range><min>{spec.min_range:.6g}</min><max>{spec.max_range:.6g}</max><resolution>0.001</resolution></range>
            {noise}
            <visibility_mask>{LIDAR_SEES}</visibility_mask>
          </lidar>
        </sensor>"""


def _hidden_visual(name: str, pose, geometry: str, rgb) -> str:
    return (f'<visual name="{name}"><pose>{_f(pose)}</pose><geometry>{geometry}</geometry>'
            f"{_material(rgb)}<cast_shadows>false</cast_shadows>"
            f"<visibility_flags>{HIDDEN_FROM_LIDAR}</visibility_flags></visual>")


def _ugv_body(rgb) -> str:
    """A small four wheeled rover under its sensor. The model frame is the sensor frame."""
    dark = (0.15, 0.15, 0.15)
    parts = [
        _hidden_visual("body", (0.0, 0.0, -0.42, 0, 0, 0), "<box><size>0.9 0.56 0.22</size></box>", rgb),
        _hidden_visual("mast", (0.0, 0.0, -0.18, 0, 0, 0),
                       "<cylinder><radius>0.04</radius><length>0.26</length></cylinder>", dark),
        # named apart from the sensor: a visual and a sensor with one name in one link
        # collide in the render scene, and the GPU LiDAR then crashes on its first frame
        _hidden_visual("lidar_housing", (0.0, 0.0, -0.02, 0, 0, 0),
                       "<cylinder><radius>0.06</radius><length>0.07</length></cylinder>", dark),
    ]
    for k, (dx, dy) in enumerate(((0.3, 0.33), (0.3, -0.33), (-0.3, 0.33), (-0.3, -0.33))):
        parts.append(_hidden_visual(
            f"wheel_{k}", (dx, dy, -0.55, 1.5708, 0, 0),
            "<cylinder><radius>0.15</radius><length>0.1</length></cylinder>", dark))
    return "\n".join(parts)


def _drone_body(rgb) -> str:
    """An X quadrotor around its sensor. The model frame is the sensor frame."""
    dark = (0.15, 0.15, 0.15)
    parts = [
        _hidden_visual("hub", (0.0, 0.0, -0.12, 0, 0, 0), "<box><size>0.24 0.18 0.08</size></box>", rgb),
        _hidden_visual("lidar_housing", (0.0, 0.0, -0.04, 0, 0, 0), "<box><size>0.08 0.1 0.07</size></box>", dark),
    ]
    for k, yaw in enumerate((0.7854, 2.3562, -2.3562, -0.7854)):
        c, s = np.cos(yaw), np.sin(yaw)
        parts.append(_hidden_visual(
            f"arm_{k}", (0.19 * c, 0.19 * s, -0.12, 0, 0, yaw), "<box><size>0.38 0.03 0.02</size></box>", dark))
        parts.append(_hidden_visual(
            f"rotor_{k}", (0.36 * c, 0.36 * s, -0.09, 0, 0, 0),
            "<cylinder><radius>0.13</radius><length>0.01</length></cylinder>", rgb))
    return "\n".join(parts)


def vehicle_model(name: str, kind: str, spec: LidarSpec, topic: str, update_rate: float,
                  pose=(0, 0, 0, 0, 0, 0), color: str = "orange", oversample: int = 1) -> str:
    """A kinematic vehicle carrying one LiDAR at its model origin.

    ``topic`` is the sensor's own topic. Gazebo publishes a laser scan there and the point
    cloud this module reads on ``topic + "/points"``.

    The model has mass so the physics system gives it a pose that ``set_pose`` can move,
    no collision shape so nothing pushes it, and no gravity so it stays where it is put.
    """
    body = {"ugv": _ugv_body, "drone": _drone_body}[kind](VEHICLE_RGB[color])
    return f"""
    <model name="{name}">
      <pose>{_f(pose)}</pose>
      <link name="base">
        <gravity>false</gravity>
        <inertial><mass>1.0</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz></inertia></inertial>
        {body}
        {lidar_sensor(spec, "lidar", topic, update_rate, oversample)}
      </link>
    </model>"""


def marker_model(name: str, position, yaw: float, spec: WorldSpec) -> str:
    """A retroreflective strip, placed as ``TunnelSim.place_marker`` places one: its x is its thickness."""
    size = _f((spec.marker_thickness, spec.marker_width, spec.marker_height))
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <pose>{_f((*position, 0.0, 0.0, yaw))}</pose>
    <link name="strip">
      <visual name="strip">
        <geometry><box><size>{size}</size></box></geometry>
        {_material(MARKER_RGB, emissive=0.35)}
        <laser_retro>{MARKER_RETRO:.6g}</laser_retro>
        <cast_shadows>false</cast_shadows>
      </visual>
    </link>
  </model>
</sdf>"""


MESH_GROUPS = (("walls", ("wall_", "cap_")), ("floor", ("floor_",)), ("ceiling", ("ceil_",)))
"""How the shell is merged: one mesh per material."""

# the six faces of a unit cube, each as its outward normal and four corners counter-clockwise
# seen from outside, so a renderer that culls back faces keeps every face the LiDAR can see
_CUBE_FACES = (
    ((1, 0, 0), ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
    ((-1, 0, 0), ((-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1))),
    ((0, 1, 0), ((1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1))),
    ((0, -1, 0), ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
    ((0, 0, 1), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
    ((0, 0, -1), ((-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1))),
)


def box_mesh_obj(boxes: list[ShellBox]) -> str:
    """Boxes as one Wavefront OBJ mesh: 24 vertices, 6 normals and 12 triangles per box."""
    import io

    if not boxes:
        raise ValueError("no boxes to mesh")
    normals = np.array([f[0] for f in _CUBE_FACES], dtype=float)  # (6, 3)
    corners = np.array([f[1] for f in _CUBE_FACES], dtype=float)  # (6, 4, 3)
    half = 0.5 * np.array([b.size for b in boxes], dtype=float)  # (N, 3)
    centre = np.array([b.centre for b in boxes], dtype=float)
    c, s = np.cos([b.yaw for b in boxes]), np.sin([b.yaw for b in boxes])
    R = np.zeros((len(boxes), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1.0
    local = corners[None, :, :, :] * half[:, None, None, :]  # (N, 6, 4, 3)
    verts = np.einsum("nij,nfcj->nfci", R, local) + centre[:, None, None, :]
    norms = np.einsum("nij,fj->nfi", R, normals)  # (N, 6, 3)
    n = len(boxes)
    v_index = np.arange(n * 24).reshape(n, 6, 4) + 1
    n_index = np.arange(n * 6).reshape(n, 6) + 1
    quads = np.stack([v_index[..., 0], v_index[..., 1], v_index[..., 2], v_index[..., 3]], axis=-1)
    tri = np.concatenate([quads[..., [0, 1, 2]], quads[..., [0, 2, 3]]], axis=1)  # (N, 12, 3)
    tri_n = np.concatenate([n_index, n_index], axis=1)  # (N, 12)
    out = io.StringIO()
    out.write(f"# locrec tunnel shell, {n} boxes\n")
    np.savetxt(out, verts.reshape(-1, 3), fmt="v %.6f %.6f %.6f")
    np.savetxt(out, norms.reshape(-1, 3), fmt="vn %.6f %.6f %.6f")
    faces = np.column_stack([tri.reshape(-1, 3), np.repeat(tri_n.reshape(-1, 1), 3, axis=1)])
    np.savetxt(out, faces[:, [0, 3, 1, 4, 2, 5]], fmt="f %d//%d %d//%d %d//%d")
    return out.getvalue()


def _tunnel_model(world: TunnelWorld, name: str, offset, ceiling_transparency: float,
                  mesh_dir: str | os.PathLike | None = None) -> str:
    """The shell as one static model: one visual per box, or with ``mesh_dir`` one mesh per material.

    Merged is what to run. Gazebo draws each visual separately for every face of the LiDAR's
    cube map, so at 1214 boxes the server spent about 50 ms of CPU per scan submitting draw
    calls while the GPU sat at 13 percent; the meshes are the same triangles in three draws.
    """
    if mesh_dir is not None:
        mesh_dir = pathlib.Path(mesh_dir)
        mesh_dir.mkdir(parents=True, exist_ok=True)
        boxes = shell_boxes(world)
        visuals = []
        for group, prefixes in MESH_GROUPS:
            members = [b for b in boxes if b.name.startswith(prefixes)]
            if not members:
                continue
            path = (mesh_dir / f"{name}_{group}.obj").resolve()
            path.write_text(box_mesh_obj(members))
            rgb = FLOOR_RGB if group == "floor" else WALL_RGB
            extra = ""
            if group == "ceiling" and ceiling_transparency > 0.0:
                extra = f"<transparency>{ceiling_transparency:.3g}</transparency>"
            visuals.append(
                f'<visual name="{group}"><geometry><mesh><uri>file://{path}</uri></mesh></geometry>'
                f"{_material(rgb)}{extra}<cast_shadows>false</cast_shadows></visual>")
        body = "\n        ".join(visuals)
        return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{_f((*offset, 0.0, 0.0, 0.0))}</pose>
      <link name="shell">
        {body}
      </link>
    </model>"""
    visuals = []
    for box in shell_boxes(world):
        rgb = FLOOR_RGB if box.name.startswith("floor") else WALL_RGB
        extra = ""
        if box.name.startswith("ceil_") and ceiling_transparency > 0.0:
            # measured: a transparent ceiling still stops every beam it stopped when opaque
            extra = f"<transparency>{ceiling_transparency:.3g}</transparency>"
        visuals.append(
            f'<visual name="{box.name}"><pose>{_f((*box.centre, 0.0, 0.0, box.yaw))}</pose>'
            f"<geometry><box><size>{_f(box.size)}</size></box></geometry>"
            f"{_material(rgb)}{extra}<cast_shadows>false</cast_shadows></visual>")
    body = "\n        ".join(visuals)
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{_f((*offset, 0.0, 0.0, 0.0))}</pose>
      <link name="shell">
        {body}
      </link>
    </model>"""


def world_sdf(tunnels: list[tuple[TunnelWorld, str, tuple]], vehicles: list[str], name: str = "tunnel",
              step_size: float = STEP_SIZE, real_time_factor: float = 50.0,
              ceiling_transparency: float = 0.0, mesh_dir: str | os.PathLike | None = None) -> str:
    """The SDF world: tunnel copies ``(world, model name, xyz offset)``, vehicle models, a light.

    ``real_time_factor`` only paces the server loop. The server is stepped while paused,
    so it has to be fast enough not to be the limit, and slow enough not to spin a core.
    ``mesh_dir`` merges each shell into meshes written there (see ``_tunnel_model``).
    """
    shells = "\n".join(_tunnel_model(w, n, off, ceiling_transparency, mesh_dir) for w, n, off in tunnels)
    models = "\n".join(vehicles)
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="{name}">
    <physics name="kinematic" type="ignored">
      <max_step_size>{step_size:.9g}</max_step_size>
      <real_time_factor>{real_time_factor:.6g}</real_time_factor>
    </physics>
    <gravity>0 0 0</gravity>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <scene>
      <ambient>0.55 0.55 0.55 1</ambient>
      <background>0.8 0.8 0.8 1</background>
      <grid>false</grid>
      <shadows>false</shadows>
    </scene>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 30 0 0 0</pose>
      <diffuse>0.85 0.85 0.85 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>-0.4 0.2 -0.9</direction>
    </light>
{shells}
{models}
  </world>
</sdf>
"""


def side_by_side_offset(world: TunnelWorld, clearance: float = 4.0, cell: float = 0.5) -> np.ndarray:
    """The shortest sideways shift that puts a second copy of the tunnel clear of the first.

    Sideways means across the line from the tunnel's start to its end. The footprint of
    every box is rasterised and grown by ``clearance``; a copy that overlaps nothing of it
    cannot share a wall with the original, and a closed tunnel then cannot see the other.
    """
    boxes = shell_boxes(world)
    occupied = set()
    for b in boxes:
        c, s = np.cos(b.yaw), np.sin(b.yaw)
        hx, hy = 0.5 * b.size[0] + clearance, 0.5 * b.size[1] + clearance
        u = np.arange(-hx, hx + cell, cell)
        v = np.arange(-hy, hy + cell, cell)
        U, V = np.meshgrid(u, v, indexing="ij")
        x = b.centre[0] + c * U - s * V
        y = b.centre[1] + s * U + c * V
        occupied.update(zip(np.floor(x / cell).astype(int).ravel(), np.floor(y / cell).astype(int).ravel()))
    chord = world.centreline[-1] - world.centreline[0]
    normal = np.array([-chord[1], chord[0]]) / max(np.linalg.norm(chord), 1e-9)
    own = {(i, j) for i, j in occupied}
    for k in range(1, 4000):
        shift = normal * k * cell
        di, dj = int(round(shift[0] / cell)), int(round(shift[1] / cell))
        if not any((i + di, j + dj) in own for i, j in own):
            return np.array([di * cell, dj * cell, 0.0])
    raise RuntimeError("no clear sideways offset found")


# ---------------------------------------------------------------------------
# server and link
# ---------------------------------------------------------------------------


def _death_signal():
    """A function for the child to run before exec, asking Linux to send it SIGTERM when its
    parent exits. libc is loaded here, in the parent: loading a library in a child forked from
    a process with threads can deadlock on the loader's lock."""
    import ctypes

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        return None
    return lambda: libc.prctl(1, 15)  # PR_SET_PDEATHSIG, SIGTERM


class GazeboServer:
    """A ``gz sim`` server on a world file: headless, paused, rendering through EGL."""

    def __init__(self, sdf_path: str | os.PathLike, world_name: str, log_path: str | os.PathLike | None = None,
                 verbosity: int = 1, seed: int | None = None):
        self.sdf_path = str(sdf_path)
        self.world_name = world_name
        self.log_path = str(log_path) if log_path else None
        self.verbosity = int(verbosity)
        self.seed = seed
        """Seeds Gazebo's own random numbers, which draw the LiDAR's range noise."""
        self.proc: subprocess.Popen | None = None
        self._log = None

    def start(self) -> "GazeboServer":
        env = dict(os.environ)
        # inside a conda environment the EGL loader finds no vendor unless told where the
        # system's are, and the GPU LiDAR then fails without saying why
        if "__EGL_VENDOR_LIBRARY_DIRS" not in env and os.path.isdir(_SYSTEM_EGL_VENDORS):
            env["__EGL_VENDOR_LIBRARY_DIRS"] = _SYSTEM_EGL_VENDORS
        # NVIDIA's driver spins a core while it waits for the GPU; yielding instead took the
        # server from 4.6 s to 3.4 s of CPU over 200 scans at the same scan time
        env.setdefault("__GL_YIELD", "USLEEP")
        self._log = open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
        cmd = ["gz", "sim", "-s", "--headless-rendering", "-v", str(self.verbosity)]
        # A server outlives a parent that dies without cleaning up, and keeps its topics and the
        # GPU, so it is told to stop when its parent goes. It also runs in a session of its own:
        # a terminal's ctrl+c reaches the whole group, and a server that had that SIGINT and then
        # the owner's SIGTERM took more than 10 s to go, long enough for ros2 launch to kill the
        # owner and leave the server running.
        if self.seed is not None:
            cmd += ["--seed", str(int(self.seed))]
        self.proc = subprocess.Popen(
            cmd + [self.sdf_path],
            stdout=self._log, stderr=subprocess.STDOUT, env=env, preexec_fn=_death_signal(),
            start_new_session=True,
        )
        return self

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                # inside ros2 launch's 5 s between SIGINT and SIGTERM, with the link's close
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()
        self._log = None


@dataclasses.dataclass
class LidarFrame:
    stamp: float
    """Simulated time the frame was rendered at, seconds."""
    xyz: np.ndarray
    """(N, 3) points in the sensor frame, one per beam; beams with no return are not finite."""
    intensity: np.ndarray
    """(N,) the ``laser_retro`` of the surface each beam hit."""


def _decode_cloud(msg, stride: int = 1) -> LidarFrame:
    """Unpack Gazebo's organised cloud, keeping every ``stride``-th column: rows are rings
    from the lowest up, columns are horizontal angles from the minimum up."""
    fields = {f.name: f for f in msg.field}
    for key in ("x", "y", "z", "intensity"):
        if key not in fields or fields[key].datatype != 6:  # FLOAT32
            raise ValueError(f"expected a float32 '{key}' field in the LiDAR cloud")
    n = msg.width * msg.height
    dt = np.dtype({
        "names": ["x", "y", "z", "intensity"], "formats": ["<f4"] * 4,
        "offsets": [fields[k].offset for k in ("x", "y", "z", "intensity")],
        "itemsize": msg.point_step,
    })
    arr = np.frombuffer(msg.data, dtype=dt, count=n).reshape(msg.height, msg.width)[:, ::stride].reshape(-1)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(float)
    return LidarFrame(stamp=msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9,
                      xyz=xyz, intensity=arr["intensity"].astype(float))


def _receive_frames(conn, topics: dict[str, int]) -> None:
    """Child process: subscribe to the LiDAR topics and pass every frame up the pipe.

    It ignores SIGINT. A terminal's ctrl+c reaches every process in the group, and the parent
    ends this one by closing the pipe, after it has finished with the frames."""
    import signal as _signal

    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
    from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
    from gz.transport13 import Node

    node = Node()
    lock = threading.Lock()

    def forward(topic: str, stride: int):
        def on_cloud(msg):
            frame = _decode_cloud(msg, stride)
            with lock:
                try:
                    conn.send((topic, frame.stamp, frame.xyz.astype(np.float32), frame.intensity.astype(np.float32)))
                except OSError:
                    pass
        return on_cloud

    for topic, stride in topics.items():
        if not node.subscribe(PointCloudPacked, topic, forward(topic, int(stride))):
            conn.send(("error", f"could not subscribe to {topic}", None, None))
            return
    with lock:
        conn.send(("ready", 0.0, None, None))
    try:
        conn.recv()  # returns, or raises, when the parent closes its end
    except (EOFError, OSError):
        pass


class GazeboLink:
    """The client side: place models, step the paused world, collect LiDAR frames, spawn.

    Frames are received in a child process, never in this one. gz-transport's Python
    ``request`` holds the interpreter lock for as long as it waits for its reply (measured:
    a background thread ran once in a 1.5 s request, against about 150 times with the lock
    free), and a subscription callback needs that lock to run. A frame that arrives during a
    request therefore stalls the transport thread that would have delivered the reply, and
    the request times out: in the first version of this class the calibration capture
    stopped that way at a step. A process that only subscribes and a process that only
    requests cannot block each other.

    ``topics`` maps each LiDAR's cloud topic to the ``oversample`` it was exported with.

    The child is started with multiprocessing's spawn method, which imports the calling
    script again, so a script that makes a link needs the usual ``if __name__ == "__main__"``
    guard.
    """

    def __init__(self, world_name: str, topics: dict[str, int], step_size: float = STEP_SIZE):
        import multiprocessing as mp

        from gz.transport13 import Node

        self.node = Node()
        self.world = world_name
        self.topics = dict(topics)
        self.step_size = float(step_size)
        self.iterations = 0
        self._frames: dict[str, LidarFrame] = {}
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe(duplex=True)
        self._receiver = ctx.Process(target=_receive_frames, args=(child, self.topics), daemon=True)
        self._receiver.start()
        child.close()
        if not self._conn.poll(60.0):
            raise TimeoutError("the LiDAR receiver did not start")
        kind, message, _, _ = self._conn.recv()
        if kind != "ready":
            raise RuntimeError(message)

    @property
    def sim_time(self) -> float:
        """Simulated time after every step requested so far."""
        return self.iterations * self.step_size

    def close(self) -> None:
        if self._receiver is None:
            return
        self._conn.close()
        self._receiver.join(timeout=3.0)
        if self._receiver.is_alive():
            self._receiver.terminate()
            self._receiver.join(timeout=3.0)
        self._receiver = None

    def _call(self, service: str, request, request_type, timeout_ms: int = 5000, quiet: bool = False) -> bool:
        """True when the server accepted the request. Raises when it did not answer, unless
        ``quiet``, which is for polling a server that may not be up yet."""
        from gz.msgs10.boolean_pb2 import Boolean

        t0 = time.monotonic()
        ok, reply = self.node.request(service, request, request_type, Boolean, timeout_ms)
        if not ok and not quiet:
            raise TimeoutError(f"{service}: no answer after {time.monotonic() - t0:.1f} s")
        return bool(ok and reply.data)

    def wait_ready(self, timeout: float = 60.0, server: GazeboServer | None = None) -> None:
        from gz.msgs10.world_control_pb2 import WorldControl

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if server is not None and not server.alive:
                raise RuntimeError(f"gz sim exited before the world came up (log: {server.log_path})")
            wc = WorldControl()
            wc.pause = True
            if self._call(f"/world/{self.world}/control", wc, WorldControl, 500, quiet=True):
                return
            time.sleep(0.2)
        raise TimeoutError(f"world '{self.world}' did not come up in {timeout:.0f} s")

    def set_poses(self, poses: dict[str, np.ndarray]) -> None:
        """Place models by name. The server queues the command and applies it before any
        iteration stepped after this returns."""
        from gz.msgs10.pose_pb2 import Pose
        from gz.msgs10.pose_v_pb2 import Pose_V
        from scipy.spatial.transform import Rotation

        req = Pose_V()
        for name, T in poses.items():
            p: Pose = req.pose.add()
            p.name = name
            p.position.x, p.position.y, p.position.z = (float(v) for v in T[:3, 3])
            qx, qy, qz, qw = Rotation.from_matrix(T[:3, :3]).as_quat()
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (
                float(qx), float(qy), float(qz), float(qw))
        if not self._call(f"/world/{self.world}/set_pose_vector", req, Pose_V):
            raise RuntimeError(f"set_pose_vector refused for {sorted(poses)}")

    def step(self, n: int = 1) -> None:
        from gz.msgs10.world_control_pb2 import WorldControl

        wc = WorldControl()
        wc.pause = True
        wc.multi_step = int(n)
        if not self._call(f"/world/{self.world}/control", wc, WorldControl):
            raise RuntimeError("world control refused a step")
        self.iterations += int(n)

    def wait_frame(self, topic: str, stamp: float, timeout: float = 30.0) -> LidarFrame:
        """The frame ``topic`` rendered at simulated time ``stamp``; raises if it never comes."""
        if topic not in self.topics:
            raise KeyError(f"{topic} is not one of this link's topics {sorted(self.topics)}")
        deadline = time.monotonic() + timeout
        while True:
            have = self._frames.get(topic)
            if have is not None and have.stamp >= stamp - 1e-6:
                if abs(have.stamp - stamp) > 1e-6:
                    raise RuntimeError(f"{topic}: expected the frame at t={stamp:.3f} s, got t={have.stamp:.3f} s")
                return have
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not self._conn.poll(remaining):
                raise TimeoutError(f"no frame on {topic} at t={stamp:.3f} s within {timeout:.0f} s "
                                   f"(latest {have.stamp if have else None})")
            got, t, xyz, inten = self._conn.recv()
            if got in self.topics:
                self._frames[got] = LidarFrame(stamp=t, xyz=xyz.astype(float), intensity=inten.astype(float))

    def prime(self, n_steps: int, timeout: float = 20.0, attempts: int = 5) -> None:
        """Step until every topic has delivered a frame.

        A subscriber is only served once discovery has matched it to the publisher, so the
        first frames can be lost. Stepping with nothing moving renders the same scene again,
        so repeating is harmless.
        """
        for _ in range(attempts):
            self.step(n_steps)
            try:
                for topic in self.topics:
                    self.wait_frame(topic, self.sim_time, timeout)
                return
            except TimeoutError:
                continue
        raise TimeoutError(f"no LiDAR frames after {attempts} attempts")

    def spawn(self, sdf: str) -> None:
        from gz.msgs10.entity_factory_pb2 import EntityFactory

        req = EntityFactory()
        req.sdf = sdf
        req.allow_renaming = False
        if not self._call(f"/world/{self.world}/create", req, EntityFactory):
            raise RuntimeError("the server refused to spawn a model")


# ---------------------------------------------------------------------------
# the harness side
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GazeboScan(Scan):
    intensities: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    """``laser_retro`` of each returned beam."""
    T_world_sensor: np.ndarray | None = None
    """The pose the scan was taken at. Only ground truth association reads it."""


def scan_from_frame(frame: LidarFrame, spec: LidarSpec, T_world_sensor: np.ndarray | None = None) -> GazeboScan:
    """Keep the beams that returned inside the sensor's range, as ``Lidar.scan`` does."""
    rng = np.linalg.norm(frame.xyz, axis=1)
    hit = np.isfinite(rng) & (rng >= spec.min_range) & (rng <= spec.max_range)
    pts = frame.xyz[hit]
    r = rng[hit]
    return GazeboScan(
        points=pts, ranges=r, geom_ids=np.zeros(0, dtype=np.int32),
        directions=pts / np.maximum(r[:, None], 1e-12), n_cast=spec.n_rays,
        intensities=frame.intensity[hit], T_world_sensor=None if T_world_sensor is None else T_world_sensor.copy(),
    )


class GazeboTunnelSim(TunnelSim):
    """A ``TunnelSim`` whose LiDAR is Gazebo's, for ``run_pass`` and for the ROS driver.

    The MuJoCo model is still built: it holds ground truth, probes the wall before a marker
    is mounted, and keeps the marker slots, so ``drop_marker_on_wall`` is the inherited one.
    Every strip it places is also spawned in Gazebo, where the LiDAR can see it.

    ``offset`` is where this copy of the tunnel sits in the Gazebo world. Scans are in the
    sensor frame and do not see it; poses and spawned strips are shifted by it.
    """

    def __init__(self, seed: int, world_spec: WorldSpec | None, lidar_spec: LidarSpec, link: GazeboLink,
                 vehicle: str, lidar_topic: str, n_steps: int, offset=(0.0, 0.0, 0.0),
                 marker_prefix: str = "marker", sensor_seed: int | None = None):
        super().__init__(seed, world_spec, lidar_spec, sensor_seed)
        self.link = link
        self.vehicle = vehicle
        self.lidar_topic = lidar_topic
        self.n_steps = int(n_steps)
        self.offset = np.asarray(offset, dtype=float)
        self.marker_prefix = marker_prefix
        self._yaws: list[float] = []

    def gazebo_pose(self, T_world_sensor: np.ndarray) -> np.ndarray:
        T = np.array(T_world_sensor, dtype=float)
        T[:3, 3] += self.offset
        return T

    def scan(self, T_world_sensor: np.ndarray) -> GazeboScan:
        self.link.set_poses({self.vehicle: self.gazebo_pose(T_world_sensor)})
        self.link.step(self.n_steps)
        return self.frame_scan(T_world_sensor)

    def frame_scan(self, T_world_sensor: np.ndarray, timeout: float = 30.0) -> GazeboScan:
        """The frame due at the current simulated time, for a caller that stepped itself."""
        frame = self.link.wait_frame(self.lidar_topic, self.link.sim_time, timeout)
        return scan_from_frame(frame, self.lidar.spec, T_world_sensor)

    def place_marker(self, position: np.ndarray, yaw: float = 0.0) -> int:
        slot = super().place_marker(position, yaw)
        self._yaws.append(float(yaw))
        self.link.spawn(marker_model(f"{self.marker_prefix}_{slot}",
                                     np.asarray(position, dtype=float) + self.offset, yaw, self.world.spec))
        return slot

    def note_marker(self, position: np.ndarray, yaw: float = 0.0) -> int:
        """Record a strip another vehicle in this same tunnel has already mounted.

        The model is in the Gazebo world once, put there by whoever mounted it, and this copy of
        the tunnel only needs to know it is there: where it is, so the detector can associate a
        return with a slot, and in which slot, so the two vehicles agree on the numbering. Calling
        ``place_marker`` instead would spawn a second model in the same place, and the LiDAR would
        see a strip 4 cm thick.
        """
        return TunnelSim.place_marker(self, position, yaw)

    def marker_detections(self, scan: Scan, fit_strip: bool = True) -> list[MarkerDetection]:
        """Retro returns grouped by the mounted strip they are nearest to."""
        if not isinstance(scan, GazeboScan) or scan.T_world_sensor is None or self.n_markers_placed == 0:
            return []
        retro = scan.intensities >= 0.5 * MARKER_RETRO
        if not retro.any():
            return []
        T = scan.T_world_sensor
        world = scan.points[retro] @ T[:3, :3].T + T[:3, 3]
        centres = self.marker_positions()
        d = np.linalg.norm(world[:, None, :] - centres[None, :, :], axis=2)
        nearest = np.argmin(d, axis=1)
        close = d[np.arange(len(nearest)), nearest] <= MARKER_ASSOCIATION_RADIUS
        idx = np.flatnonzero(retro)
        spec = self.world.spec
        out = []
        for slot in np.unique(nearest[close]):
            pts = scan.points[idx[close & (nearest == slot)]]
            if fit_strip:
                point, seen = fit_strip_centre(pts, self.lidar.spec, spec.marker_width, spec.marker_thickness)
            else:
                point, seen = pts.mean(axis=0), None
            out.append(MarkerDetection(slot=int(slot), point_sensor=point, n_beams=int(len(pts)),
                                       range_m=float(np.linalg.norm(point)), seen_width_m=seen,
                                       n_columns=beam_columns(pts, self.lidar.spec)))
        return out


def write_world(path: str | os.PathLike, sdf: str) -> pathlib.Path:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(sdf)
    return p
