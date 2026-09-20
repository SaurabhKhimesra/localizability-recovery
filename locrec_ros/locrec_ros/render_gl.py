"""GPU renderer for the demonstration viewer, with no ROS import.

Flat, the way rviz draws: opaque square points and solid lines, depth tested, on a plain
background, with an overlay image composited on top. It runs headless through EGL, so it
needs a GPU driver and no display.

Inside a conda environment on a glvnd system, the environment's libEGL finds no vendor
unless ``__EGL_VENDOR_LIBRARY_DIRS`` points at the system's vendor directory, and EGL
then fails with an error that does not say why. ``create`` sets it when the caller has
not, before the first context exists.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ["GLRenderer", "look_at", "perspective"]

_SYSTEM_EGL_VENDORS = "/usr/share/glvnd/egl_vendor.d"

_POINT_VS = """
#version 330
uniform mat4 view_proj;
in vec3 in_pos;
in vec3 in_color;
in float in_size;
out vec3 v_color;
void main() {
    gl_Position = view_proj * vec4(in_pos, 1.0);
    gl_PointSize = in_size;
    v_color = in_color;
}
"""

_POINT_FS = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() { f_color = vec4(v_color, 1.0); }
"""

_LINE_VS = """
#version 330
uniform mat4 view_proj;
in vec3 in_pos;
in vec3 in_color;
in float in_width;
out vec3 g_color;
out float g_width;
void main() {
    gl_Position = view_proj * vec4(in_pos, 1.0);
    g_color = in_color;
    g_width = in_width;
}
"""

_LINE_GS = """
#version 330
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
uniform vec2 viewport;
in vec3 g_color[];
in float g_width[];
out vec3 v_color;
void main() {
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    const float near_w = 0.1;
    if (p0.w < near_w && p1.w < near_w) return;
    if (p0.w < near_w) p0 = mix(p0, p1, (near_w - p0.w) / (p1.w - p0.w));
    if (p1.w < near_w) p1 = mix(p1, p0, (near_w - p1.w) / (p0.w - p1.w));
    vec2 s0 = p0.xy / p0.w * viewport * 0.5;
    vec2 s1 = p1.xy / p1.w * viewport * 0.5;
    vec2 dir = s1 - s0;
    float len = length(dir);
    if (len < 1e-3) return;
    vec2 n = vec2(-dir.y, dir.x) / len * g_width[0] * 0.5 / (viewport * 0.5);
    gl_Position = vec4(p0.xy + n * p0.w, p0.zw); v_color = g_color[0]; EmitVertex();
    gl_Position = vec4(p0.xy - n * p0.w, p0.zw); v_color = g_color[0]; EmitVertex();
    gl_Position = vec4(p1.xy + n * p1.w, p1.zw); v_color = g_color[1]; EmitVertex();
    gl_Position = vec4(p1.xy - n * p1.w, p1.zw); v_color = g_color[1]; EmitVertex();
    EndPrimitive();
}
"""

_LINE_FS = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() { f_color = vec4(v_color, 1.0); }
"""

_QUAD_VS = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() {
    uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_OVERLAY_FS = """
#version 330
uniform sampler2D hud;
in vec2 uv;
out vec4 f_color;
void main() {
    vec4 h = texture(hud, vec2(uv.x, 1.0 - uv.y));
    f_color = vec4(h.rgb, h.a);
}
"""


def perspective(fx: float, fy: float, cx: float, cy: float, width: int, height: int,
                near: float = 0.1, far: float = 600.0) -> np.ndarray:
    """OpenGL projection from pinhole intrinsics, with ``cy`` measured from the top."""
    P = np.zeros((4, 4))
    P[0, 0] = 2.0 * fx / width
    P[0, 2] = 1.0 - 2.0 * cx / width
    P[1, 1] = 2.0 * fy / height
    P[1, 2] = 2.0 * cy / height - 1.0
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World to camera, OpenGL convention: the camera looks down its own -z."""
    f = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    V = np.eye(4)
    V[0, :3], V[1, :3], V[2, :3] = s, u, -f
    V[:3, 3] = -V[:3, :3] @ np.asarray(eye, dtype=float)
    return V


class GLRenderer:
    """Layers of points and lines, drawn flat. Layers persist until replaced.

    Colors are plain RGB in 0..1, point sizes are pixels, line widths are pixels.
    """

    def __init__(self, ctx, width: int, height: int):
        import moderngl

        self.mgl = moderngl
        self.ctx = ctx
        self.width, self.height = int(width), int(height)
        self.point_prog = ctx.program(vertex_shader=_POINT_VS, fragment_shader=_POINT_FS)
        self.line_prog = ctx.program(
            vertex_shader=_LINE_VS, geometry_shader=_LINE_GS, fragment_shader=_LINE_FS)
        self.overlay_prog = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_OVERLAY_FS)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._quad_vbo = ctx.buffer(quad.tobytes())
        self._quad = ctx.vertex_array(self.overlay_prog, [(self._quad_vbo, "2f", "in_pos")])

        # multisampled so thin lines and points do not crawl as the camera moves
        self._msaa_color = ctx.renderbuffer((self.width, self.height), 3, samples=4)
        self._msaa_depth = ctx.depth_renderbuffer((self.width, self.height), samples=4)
        self._msaa_fbo = ctx.framebuffer(color_attachments=[self._msaa_color],
                                         depth_attachment=self._msaa_depth)
        self._out_tex = ctx.texture((self.width, self.height), 3, dtype="f1")
        self._out_fbo = ctx.framebuffer(color_attachments=[self._out_tex])
        self.hud_tex = ctx.texture((self.width, self.height), 4, dtype="f1")
        self.hud_tex.write(np.zeros((self.height, self.width, 4), dtype=np.uint8).tobytes())
        self._layers: dict[str, tuple] = {}

    @classmethod
    def create(cls, width: int, height: int) -> "GLRenderer":
        if "__EGL_VENDOR_LIBRARY_DIRS" not in os.environ and os.path.isdir(_SYSTEM_EGL_VENDORS):
            os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = _SYSTEM_EGL_VENDORS
        import moderngl

        ctx = moderngl.create_standalone_context(backend="egl", require=330)
        return cls(ctx, width, height)

    @property
    def device(self) -> str:
        return str(self.ctx.info.get("GL_RENDERER", "unknown"))

    # ---- layers ------------------------------------------------------------

    def _upload(self, name: str, prog, data: np.ndarray, fmt: str, attrs: tuple, mode, n: int):
        old = self._layers.pop(name, None)
        if n == 0:
            if old is not None:
                old[0].release()
                old[1].release()
            return
        raw = np.ascontiguousarray(data, dtype="f4").tobytes()
        if old is not None and old[0].size >= len(raw):
            vbo, vao = old[0], old[1]
            vbo.orphan(vbo.size)
            vbo.write(raw)
        else:
            if old is not None:
                old[0].release()
                old[1].release()
            # room to grow, so a layer that gains points each scan is not reallocated each scan
            vbo = self.ctx.buffer(reserve=max(len(raw), 4096) * 2, dynamic=True)
            vbo.write(raw)
            vao = self.ctx.vertex_array(prog, [(vbo, fmt, *attrs)])
        self._layers[name] = (vbo, vao, mode, n)

    def points(self, name: str, positions, colors, sizes) -> None:
        pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        n = pos.shape[0]
        col = np.broadcast_to(np.asarray(colors, dtype=np.float32), (n, 3))
        size = np.broadcast_to(np.asarray(sizes, dtype=np.float32), (n,))[:, None]
        self._upload(name, self.point_prog, np.hstack([pos, col, size]), "3f 3f 1f",
                     ("in_pos", "in_color", "in_size"), self.mgl.POINTS, n)

    def lines(self, name: str, segments, colors, widths_px) -> None:
        seg = np.asarray(segments, dtype=np.float32).reshape(-1, 2, 3)
        n = seg.shape[0]
        col = np.asarray(colors, dtype=np.float32)
        col = np.broadcast_to(col if col.ndim == 3 else col.reshape(-1, 1, 3) if col.ndim == 2 else col, (n, 2, 3))
        width = np.broadcast_to(np.asarray(widths_px, dtype=np.float32).reshape(-1, 1, 1), (n, 2, 1))
        data = np.concatenate([seg, col, width], axis=2).reshape(-1, 7)
        self._upload(name, self.line_prog, data, "3f 3f 1f",
                     ("in_pos", "in_color", "in_width"), self.mgl.LINES, 2 * n)

    def clear(self, name: str) -> None:
        self._upload(name, None, np.zeros((0, 7)), "", (), None, 0)

    def hud(self, rgba: np.ndarray) -> None:
        img = np.ascontiguousarray(rgba, dtype=np.uint8)
        assert img.shape == (self.height, self.width, 4), img.shape
        self.hud_tex.write(img.tobytes())

    # ---- frame -------------------------------------------------------------

    def _bind(self, fbo) -> None:
        # moderngl keeps a viewport per framebuffer and assigning ctx.viewport rewrites the
        # one currently bound, so every viewport is set after its framebuffer is bound
        fbo.use()
        fbo.viewport = (0, 0, self.width, self.height)
        self.ctx.viewport = (0, 0, self.width, self.height)

    def render(self, view: np.ndarray, proj: np.ndarray, focal_px: float = 0.0,
               background=(0.188, 0.188, 0.188), prefix: str = "") -> np.ndarray:
        """One frame of the layers whose names start with ``prefix``, so one renderer can
        draw several views of several scenes. One renderer, not one per view: moderngl
        contexts share nothing but are not switched for you, and a second standalone
        context once received the first one's overlay texture."""
        ctx = self.ctx
        mgl = self.mgl
        vp = (proj @ view).T.astype("f4").tobytes()  # column-major for GLSL
        self.point_prog["view_proj"].write(vp)
        self.line_prog["view_proj"].write(vp)
        self.line_prog["viewport"].value = (float(self.width), float(self.height))

        self._bind(self._msaa_fbo)
        self._msaa_fbo.clear(*background, 1.0, depth=1.0)
        ctx.enable(mgl.DEPTH_TEST | mgl.PROGRAM_POINT_SIZE)
        ctx.disable(mgl.BLEND)
        for name, (_, vao, mode, n) in self._layers.items():
            if name.startswith(prefix):
                vao.render(mode, vertices=n)

        ctx.copy_framebuffer(self._out_fbo, self._msaa_fbo)
        self._bind(self._out_fbo)
        ctx.disable(mgl.DEPTH_TEST)
        ctx.enable(mgl.BLEND)
        ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        self.hud_tex.use(0)
        self.overlay_prog["hud"].value = 0
        self._quad.render(mgl.TRIANGLE_STRIP)
        frame = np.frombuffer(self._out_fbo.read(components=3, dtype="f1"), dtype=np.uint8)
        return np.flipud(frame.reshape(self.height, self.width, 3))

    def release(self) -> None:
        for vbo, vao, _, _ in self._layers.values():
            vao.release()
            vbo.release()
        self._layers.clear()
        self.ctx.release()
