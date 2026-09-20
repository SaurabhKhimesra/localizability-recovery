"""CPU fallback with the same interface as ``GLRenderer``, for machines without EGL.

It draws the same layers the same flat way, points as small squares and lines on top,
nearest point wins, only slower. The viewer picks it when no GPU context can be made, so a
machine without a driver still gets a video rather than a crash.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

__all__ = ["CPURenderer"]


class CPURenderer:
    device = "cpu"

    def __init__(self, width: int, height: int):
        self.width, self.height = int(width), int(height)
        self._points: dict[str, tuple] = {}
        self._lines: dict[str, tuple] = {}
        self._hud = np.zeros((self.height, self.width, 4), dtype=np.uint8)

    @classmethod
    def create(cls, width: int, height: int) -> "CPURenderer":
        return cls(width, height)

    def points(self, name: str, positions, colors, sizes) -> None:
        pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        n = pos.shape[0]
        if n == 0:
            self._points.pop(name, None)
            return
        col = np.broadcast_to(np.asarray(colors, dtype=np.float32), (n, 3))
        size = np.broadcast_to(np.asarray(sizes, dtype=np.float32), (n,))
        self._points[name] = (pos, col, size)

    def lines(self, name: str, segments, colors, widths_px) -> None:
        seg = np.asarray(segments, dtype=np.float32).reshape(-1, 2, 3)
        n = seg.shape[0]
        if n == 0:
            self._lines.pop(name, None)
            return
        col = np.asarray(colors, dtype=np.float32)
        if col.ndim == 3:
            col = col[:, 0, :]
        col = np.broadcast_to(col.reshape(-1, 3), (n, 3))
        width = np.broadcast_to(np.asarray(widths_px, dtype=np.float32).reshape(-1), (n,))
        self._lines[name] = (seg, col, width)

    def clear(self, name: str) -> None:
        self._points.pop(name, None)
        self._lines.pop(name, None)

    def hud(self, rgba: np.ndarray) -> None:
        self._hud = np.asarray(rgba, dtype=np.uint8)

    def _project(self, pts: np.ndarray, vp: np.ndarray):
        h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)]) @ vp.T
        w = h[:, 3]
        safe = np.maximum(w, 1e-6)
        return (h[:, 0] / safe * 0.5 + 0.5) * self.width, (0.5 - h[:, 1] / safe * 0.5) * self.height, w

    def render(self, view, proj, focal_px: float = 0.0, background=(0.188, 0.188, 0.188),
               prefix: str = "") -> np.ndarray:
        vp = (np.asarray(proj) @ np.asarray(view)).astype(np.float32)
        img = np.empty((self.height, self.width, 3), dtype=np.float32)
        img[:] = background
        depth = np.full((self.height, self.width), np.inf, dtype=np.float32)
        for name, (pos, col, size) in self._points.items():
            if not name.startswith(prefix):
                continue
            x, y, w = self._project(pos, vp)
            ok = (w > 0.1) & (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
            if not ok.any():
                continue
            xi, yi, wi, ci = x[ok].astype(np.int64), y[ok].astype(np.int64), w[ok], col[ok]
            half = int(max(np.max(size) // 2, 0))
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    xx, yy = np.clip(xi + dx, 0, self.width - 1), np.clip(yi + dy, 0, self.height - 1)
                    order = np.argsort(-wi)  # far first, so near points overwrite
                    xx, yy, ww, cc = xx[order], yy[order], wi[order], ci[order]
                    nearer = ww < depth[yy, xx]
                    img[yy[nearer], xx[nearer]] = cc[nearer]
                    depth[yy[nearer], xx[nearer]] = ww[nearer]
        canvas = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        draw = ImageDraw.Draw(canvas)
        for name, (seg, col, width) in self._lines.items():
            if not name.startswith(prefix):
                continue
            x, y, w = self._project(seg.reshape(-1, 3), vp)
            x, y, w = x.reshape(-1, 2), y.reshape(-1, 2), w.reshape(-1, 2)
            for k in np.flatnonzero((w > 0.1).all(axis=1)):
                if max(abs(x[k]).max(), abs(y[k]).max()) > 16000:
                    continue
                rgb = tuple(int(v) for v in np.clip(col[k] * 255, 0, 255))
                draw.line([(x[k, 0], y[k, 0]), (x[k, 1], y[k, 1])], fill=rgb, width=max(int(round(width[k])), 1))
        out = np.asarray(canvas, dtype=np.float32) / 255.0
        a = self._hud[..., 3:4].astype(np.float32) / 255.0
        out = out * (1 - a) + self._hud[..., :3].astype(np.float32) / 255.0 * a
        return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)

    def release(self) -> None:
        self._points.clear()
        self._lines.clear()
