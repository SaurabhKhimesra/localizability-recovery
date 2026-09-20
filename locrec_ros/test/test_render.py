"""The demo renderers: projection, both backends, and the viewport regression.

Both backends draw flat, the way rviz does: opaque square points of a size in pixels, lines
of a width in pixels, depth tested, on a plain background. The CPU backend always runs. The
GPU backend runs where an EGL context can be made and skips itself elsewhere, so the suite
still passes on a machine without a driver.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from locrec_ros.render_cpu import CPURenderer  # noqa: E402
from locrec_ros.render_gl import look_at, perspective  # noqa: E402

W, H = 320, 180
FOCAL = 200.0
CX, CY = 160.0, 90.0
GREY = (0.188, 0.188, 0.188)


def _camera():
    eye = np.array([0.0, 0.0, 5.0])
    view = look_at(eye, np.array([10.0, 0.0, 5.0]))
    return view, perspective(FOCAL, FOCAL, CX, CY, W, H)


def _gpu():
    try:
        from locrec_ros.render_gl import GLRenderer

        return GLRenderer.create(W, H)
    except Exception as exc:  # noqa: BLE001  no context is a skip, not a failure
        pytest.skip(f"no EGL context here: {type(exc).__name__}")


def _renderer(backend):
    return CPURenderer.create(W, H) if backend == "cpu" else _gpu()


def test_a_point_straight_ahead_lands_on_the_principal_point():
    view, proj = _camera()
    h = proj @ view @ np.array([10.0, 0.0, 5.0, 1.0])
    u = (h[0] / h[3] * 0.5 + 0.5) * W
    v = (0.5 - h[1] / h[3] * 0.5) * H
    assert abs(u - CX) < 1e-6 and abs(v - CY) < 1e-6
    # and something up and to the left of it lands up and to the left in the image
    h = proj @ view @ np.array([10.0, 1.0, 6.0, 1.0])
    assert (h[0] / h[3] * 0.5 + 0.5) * W < CX and (0.5 - h[1] / h[3] * 0.5) * H < CY


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_point_is_drawn_in_its_own_colour_where_it_projects_on_the_background(backend):
    r = _renderer(backend)
    view, proj = _camera()
    r.points("dot", np.array([[10.0, 0.0, 5.0]]), np.array([0.0, 1.0, 0.0]), 7.0)
    frame = r.render(view, proj, FOCAL, background=GREY).astype(np.int32)
    assert np.all(np.abs(frame[int(CY), int(CX)] - [0, 255, 0]) <= 2), frame[int(CY), int(CX)]
    assert np.all(np.abs(np.median(frame.reshape(-1, 3), axis=0) - 48) <= 1)
    # flat: no glow around it, the frame is background a few pixels away
    assert np.all(np.abs(frame[int(CY), int(CX) + 8] - 48) <= 2)
    r.release()


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_the_nearer_of_two_points_on_one_pixel_is_the_one_drawn(backend):
    r = _renderer(backend)
    view, proj = _camera()
    # the far red point is added last, so only depth can keep it hidden
    r.points("near", np.array([[10.0, 0.0, 5.0]]), np.array([0.0, 0.0, 1.0]), 6.0)
    r.points("far", np.array([[40.0, 0.0, 5.0]]), np.array([1.0, 0.0, 0.0]), 6.0)
    frame = r.render(view, proj, FOCAL, background=GREY).astype(np.int32)
    assert np.all(np.abs(frame[int(CY), int(CX)] - [0, 0, 255]) <= 2), frame[int(CY), int(CX)]
    r.release()


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_line_is_as_wide_as_asked_in_pixels(backend):
    r = _renderer(backend)
    view, proj = _camera()
    r.lines("bar", np.array([[[10.0, -1.0, 5.0], [10.0, 1.0, 5.0]]]), np.array([1.0, 1.0, 1.0]), 5.0)
    frame = r.render(view, proj, FOCAL, background=GREY).astype(np.int32)
    column = frame[:, int(CX)].sum(axis=1)
    lit = np.flatnonzero(column > 3 * 200)
    assert 4 <= lit.size <= 7, lit.size
    r.release()


def test_the_renderer_never_leaves_a_shrunken_copy_of_the_scene_in_a_corner():
    """moderngl keeps a viewport per framebuffer. Setting the viewport before binding a
    target once shrank the previous one, so from the second frame on a quarter size copy of
    the whole scene appeared in the bottom left corner."""
    r = _gpu()
    view, proj = _camera()
    r.points("dot", np.array([[10.0, 0.0, 5.0]]), np.array([1.0, 1.0, 1.0]), 9.0)
    first = r.render(view, proj, FOCAL, background=GREY).astype(np.int32)
    for _ in range(3):
        later = r.render(view, proj, FOCAL, background=GREY).astype(np.int32)
    np.testing.assert_allclose(later, first, atol=2)
    corner = later[H - H // 3:, : W // 3]
    assert np.all(np.abs(corner - 48) <= 2)
    r.release()


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_prefix_draws_one_views_layers_and_nothing_of_the_others(backend):
    """The viewer draws every view with one renderer: each vehicle's layers carry its name."""
    r = _renderer(backend)
    view, proj = _camera()
    r.points("forward/dot", np.array([[10.0, 0.0, 5.0]]), np.array([0.0, 1.0, 0.0]), 7.0)
    r.points("glance/dot", np.array([[10.0, 1.0, 5.0]]), np.array([1.0, 0.0, 0.0]), 7.0)
    frame = r.render(view, proj, FOCAL, background=GREY, prefix="forward/").astype(np.int32)
    red = (frame[..., 0] > 200) & (frame[..., 1] < 60)
    assert not red.any()
    assert np.all(np.abs(frame[int(CY), int(CX)] - [0, 255, 0]) <= 2)
    r.release()
