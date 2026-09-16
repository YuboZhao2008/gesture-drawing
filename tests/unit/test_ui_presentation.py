"""Presentation regressions: no camera, native window, or pixel-golden dependency."""

from pathlib import Path

import numpy as np

from gesture_control.core.events import Command, TemporalOutput
from gesture_control.core.types import (
    FrameContext,
    Gesture,
    HandFrame,
    HandLandmarks,
    StableGesture,
)
from gesture_control.ui.demo import DemoUI
from gesture_control.ui.drawing import DrawingCanvas
from gesture_control.ui.overlay import essential_state


def test_essential_status_uses_visible_stable_hand(
    context: FrameContext,
    hand: HandLandmarks,
) -> None:
    stable = TemporalOutput(
        context, stable_gestures=(StableGesture(hand.hand_id, Gesture.OPEN_PALM, 0.96, 0),)
    )
    assert essential_state(HandFrame(context, (hand,)), stable) == ("OPEN PALM", "Confidence 96%")
    assert essential_state(HandFrame(context), stable)[0] == "NO HAND"
    assert essential_state(HandFrame(context, (hand,)), TemporalOutput(context))[0] == "ACQUIRING"


def test_preview_resize_preserves_strokes_and_export(context: FrameContext) -> None:
    canvas = DrawingCanvas(1920, 1080)
    canvas.execute(Command(context, "stroke.begin", {"x_px": 100, "y_px": 100}, "hand"))
    canvas.execute(Command(context, "stroke.point", {"x_px": 800, "y_px": 600}, "hand"))
    original = canvas.pixels.copy()
    for size in ((640, 360), (320, 180), (960, 540), (640, 360)):
        preview = canvas.render(size)
        assert preview.shape == (size[1], size[0], 3)
        preview[:] = 0  # Display buffers must not alias the persistent canvas.
        assert np.array_equal(canvas.pixels, original)
    assert canvas.pen_down and canvas.cursor == (800, 600)


def test_debug_and_calibration_feedback_do_not_change_canvas(tmp_path: Path) -> None:
    ui = DemoUI(calibration_path=tmp_path / "calibration.json")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    pixels = ui.canvas.pixels.copy()
    assert not ui.diagnostics
    normal = ui.render(image)
    ui.handle_key(ord("d"))
    assert ui.diagnostics
    ui.handle_key(ord("c"))
    assert ui.session is not None and len(ui.session.points) == 0
    ui.handle_key(32)
    assert ui.feedback_error and "No pointer" in ui.status
    error_image = ui.render(image)
    assert not np.array_equal(normal, error_image)
    ui.pointer = (0.1, 0.1)
    ui.handle_key(32)
    assert not ui.feedback_error and len(ui.session.points) == 1
    assert "TOP RIGHT" in ui.status
    for point in ((0.9, 0.1), (0.9, 0.9), (0.1, 0.9)):
        ui.pointer = point
        ui.handle_key(32)
    assert ui.session is None and ui.mapper.enabled
    assert "Calibrated" in ui.status
    assert np.array_equal(ui.canvas.pixels, pixels)
    assert not image.any()


def test_render_buffers_and_cached_chrome_are_independent(tmp_path: Path) -> None:
    ui = DemoUI(calibration_path=tmp_path / "none.json")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    first = ui.render(image)
    original = first.copy()
    first[:] = 255
    assert np.array_equal(original, ui.render(image))
