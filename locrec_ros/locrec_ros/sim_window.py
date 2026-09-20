"""A live window on the simulation itself: MuJoCo's own viewer, following the robot.

It shows the tunnel the LiDAR is ray cast against, the robot, its rays and their hits,
and each marker strip appearing on the wall as the simulator mounts it, with a beacon
over it so it can be seen from a distance. The camera follows the robot; drag to orbit
and scroll to zoom, as in any MuJoCo viewer.

The window draws its own copy of the world, compiled from the same XML, with the
ceiling hidden and the walls part transparent so the inside can be seen. It never
touches the simulation's model: MuJoCo's ray caster skips a geom whose alpha is zero,
so hiding the ceiling in the model the LiDAR uses would quietly delete the ceiling from
every scan. Marker positions are copied across from the simulation on every scan.

The window is opened through X11 (XWayland on a Wayland desktop), which gives it a title
bar to move it by; GLFW's Wayland backend opens it undecorated.
"""
from __future__ import annotations

import os
import time

import numpy as np

__all__ = ["SimWindow", "display_model"]

MARKER_PARKED_BELOW = -40.0
"""Unmounted marker slots are parked under the floor; anything below this is not mounted."""


def _heading(T: np.ndarray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


def display_model(world_xml: str):
    """The world compiled again for drawing only: ceilings hidden, walls part transparent."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(world_xml)
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name.startswith("ceil_"):
            model.geom_rgba[i, 3] = 0.0
        elif name.startswith("wall_"):
            model.geom_rgba[i] = (0.62, 0.60, 0.56, 0.55)
        elif name.startswith("floor_"):
            model.geom_rgba[i] = (0.30, 0.31, 0.34, 1.0)
    model.vis.headlight.ambient[:] = (0.42, 0.42, 0.45)
    model.vis.headlight.diffuse[:] = (0.62, 0.62, 0.62)
    model.vis.headlight.specular[:] = (0.15, 0.15, 0.15)
    return model


class SimWindow:
    """Open, feed with each scan, draw on a timer, close. Every method is safe once closed."""

    def __init__(self, world_xml: str, rate_hz: float, rays: int = 180):
        os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
        import mujoco
        import mujoco.viewer

        self.mj = mujoco
        self.period = 1.0 / max(rate_hz, 1e-6)
        self.rays = int(rays)
        self.model = display_model(world_xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        self._T_prev: np.ndarray | None = None
        self._T_cur: np.ndarray | None = None
        self._t_cur = time.monotonic()
        self._hits = np.zeros((0, 3))
        self._look: np.ndarray | None = None
        with self.viewer.lock():
            self.viewer.cam.distance = 11.0
            self.viewer.cam.elevation = -36.0

    @property
    def running(self) -> bool:
        return self.viewer is not None and self.viewer.is_running()

    def new_scan(self, T_world_sensor: np.ndarray, points_sensor: np.ndarray,
                 mocap_pos: np.ndarray, mocap_quat: np.ndarray) -> None:
        if not self.running:
            return
        with self.viewer.lock():
            if self._T_cur is None:
                self.viewer.cam.azimuth = np.rad2deg(_heading(T_world_sensor)) + 145.0
            self.data.mocap_pos[:] = mocap_pos
            self.data.mocap_quat[:] = mocap_quat
            self.mj.mj_forward(self.model, self.data)
        self._T_prev = T_world_sensor.copy() if self._T_cur is None else self._T_cur
        self._T_cur = T_world_sensor.copy()
        self._t_cur = time.monotonic()
        pts = np.asarray(points_sensor, dtype=float)
        pts = pts[:: max(1, len(pts) // self.rays)]
        self._hits = pts @ T_world_sensor[:3, :3].T + T_world_sensor[:3, 3]

    def pose_now(self) -> tuple[np.ndarray, np.ndarray]:
        """The robot's position and rotation, eased between the last two scans."""
        a = min((time.monotonic() - self._t_cur) / self.period, 1.0)
        pos = (1.0 - a) * self._T_prev[:3, 3] + a * self._T_cur[:3, 3]
        h0, h1 = _heading(self._T_prev), _heading(self._T_cur)
        h = h0 + a * np.arctan2(np.sin(h1 - h0), np.cos(h1 - h0))
        R = np.array([[np.cos(h), -np.sin(h), 0.0], [np.sin(h), np.cos(h), 0.0], [0.0, 0.0, 1.0]])
        return pos, R

    def decorate(self, scn, pos: np.ndarray, R: np.ndarray) -> None:
        """Add the robot, its rays, their hits and the marker beacons to a scene."""
        mj = self.mj
        eye3 = np.eye(3).reshape(9)

        def add(kind, size, p, mat, rgba):
            if scn.ngeom >= scn.maxgeom:
                return None
            g = scn.geoms[scn.ngeom]
            mj.mjv_initGeom(g, kind, np.asarray(size, dtype=float), np.asarray(p, dtype=float),
                            np.asarray(mat, dtype=float).reshape(9), np.asarray(rgba, dtype=np.float32))
            scn.ngeom += 1
            return g

        G = mj.mjtGeom
        add(G.mjGEOM_BOX, (0.62, 0.40, 0.18), pos - np.array([0.0, 0.0, 0.40]), R, (1.0, 0.62, 0.05, 1.0))
        add(G.mjGEOM_BOX, (0.20, 0.30, 0.06), pos + R @ np.array([0.38, 0.0, -0.18]), R, (0.15, 0.15, 0.18, 1.0))
        for dx in (-0.42, 0.42):
            for dy in (-0.46, 0.46):
                add(G.mjGEOM_SPHERE, (0.16, 0.0, 0.0), pos + R @ np.array([dx, dy, -0.54]), eye3,
                    (0.06, 0.06, 0.07, 1.0))
        add(G.mjGEOM_CYLINDER, (0.13, 0.13, 0.09), pos, eye3, (0.10, 0.10, 0.12, 1.0))
        add(G.mjGEOM_CYLINDER, (0.135, 0.135, 0.02), pos + np.array([0.0, 0.0, 0.03]), eye3, (0.2, 0.85, 1.0, 1.0))
        for p in self._hits:
            g = add(G.mjGEOM_LINE, (0.0, 0.0, 0.0), pos, eye3, (0.25, 0.85, 1.0, 0.22))
            if g is not None:
                mj.mjv_connector(g, G.mjGEOM_LINE, 1.2, pos, p)
            add(G.mjGEOM_SPHERE, (0.045, 0.0, 0.0), p, eye3, (0.35, 0.92, 1.0, 1.0))
        for p in self.data.mocap_pos:
            if p[2] > MARKER_PARKED_BELOW:
                add(G.mjGEOM_CYLINDER, (0.16, 0.16, 1.9), (p[0], p[1], 1.9), eye3, (0.1, 1.0, 0.45, 0.35))

    def draw(self) -> None:
        if not self.running or self._T_cur is None:
            return
        pos, R = self.pose_now()
        with self.viewer.lock():
            scn = self.viewer.user_scn
            scn.ngeom = 0
            self.decorate(scn, pos, R)
            look = pos + np.array([0.0, 0.0, -0.3])
            self._look = look if self._look is None else self._look + 0.25 * (look - self._look)
            self.viewer.cam.lookat[:] = self._look
        self.viewer.sync()

    def close(self) -> None:
        if self.viewer is None:
            return
        self.viewer.close()
        for _ in range(100):
            if not self.viewer.is_running():
                break
            time.sleep(0.02)
        self.viewer = None
