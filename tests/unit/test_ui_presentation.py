"""Presentation regressions: no camera, native window, or pixel-golden dependency."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from gesture_control.core.events import Command, TemporalOutput
from gesture_control.core.types import (
    FrameContext,
    Gesture,
    HandFrame,
    HandLandmarks,
    StableGesture,
)
from gesture_control.ui import overlay
from gesture_control.ui.calibration import UNIT_CORNERS, Calibration, CalibrationSession
from gesture_control.ui.demo import DemoUI
from gesture_control.ui.drawing import DrawingCanvas, Image
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


@pytest.mark.parametrize(
    "size", [(1, 1), (32, 24), (320, 240), (16, 1024), (1024, 16), (640, 480), (1920, 1080)]
)
@pytest.mark.parametrize("mode", ["normal", "debug", "calibration", "eraser"])
def test_small_layout_labels_fit_without_overlap(
    size: tuple[int, int],
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boxes: list[tuple[int, int, int, int, str]] = []
    original_label = overlay.label

    def measured_label(
        image: Image,
        text: str,
        position: tuple[int, int],
        *,
        color: tuple[int, int, int] = overlay.INK,
        scale: float = 0.5,
    ) -> None:
        if image.shape[1] >= 1076 and text:
            (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            x, y = position
            assert 0 <= x and x + w < image.shape[1], text
            assert 0 <= y - h and y + baseline < image.shape[0], text
            boxes.append((x, y - h, x + w, y + baseline, text))
        original_label(image, text, position, color=color, scale=scale)

    overlay._chrome.cache_clear()
    monkeypatch.setattr(overlay, "label", measured_label)
    ui = DemoUI(width=size[0], height=size[1], calibration_path=tmp_path / "none.json")
    if mode == "eraser":
        ui.mapper.tool = "eraser"
        ui.canvas.execute(Command(FrameContext(0, 0, 640, 480), "tool.select", {"tool": "eraser"}))
    elif mode == "debug":
        ui.handle_key(ord("d"))
    elif mode == "calibration":
        ui.handle_key(ord("c"))
    pixels = ui.canvas.pixels.copy()
    rendered = ui.render(np.zeros((480, 640, 3), dtype=np.uint8))
    assert rendered.shape[1] >= 1076 and rendered.shape[0] >= 486
    for index, a in enumerate(boxes):
        for b in boxes[index + 1 :]:
            assert not (max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])), (
                a[4],
                b[4],
            )
    assert np.array_equal(pixels, ui.canvas.pixels)


def test_fourth_corner_save_failure_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = DemoUI(calibration_path=tmp_path / "calibration.json")
    previous = Calibration(UNIT_CORNERS)
    ui.mapper.calibration = previous
    ui.handle_key(ord("c"))
    for point in UNIT_CORNERS[:3]:
        ui.pointer = point
        ui.handle_key(32)
    save = Calibration.save

    def fail_save(self: Calibration, path: str | Path) -> None:
        raise OSError("Disk unavailable")

    monkeypatch.setattr(Calibration, "save", fail_save)
    ui.pointer = UNIT_CORNERS[3]
    ui.handle_key(32)
    assert ui.session is not None and len(ui.session.points) == 3
    assert not ui.mapper.enabled and ui.mapper.calibration is previous
    assert ui.feedback_error and "SPACE to retry" in ui.status
    ui.render(np.zeros((480, 640, 3), dtype=np.uint8))
    monkeypatch.setattr(Calibration, "save", save)
    ui.handle_key(32)
    assert ui.session is None and ui.mapper.enabled and not ui.feedback_error
    assert Calibration.load(ui.calibration_path) == previous


def test_invalid_fourth_corner_and_out_of_range_session_render(tmp_path: Path) -> None:
    ui = DemoUI(calibration_path=tmp_path / "calibration.json")
    ui.handle_key(ord("c"))
    for point in (*UNIT_CORNERS[:3], UNIT_CORNERS[0]):
        ui.pointer = point
        ui.handle_key(32)
    assert ui.session is not None and len(ui.session.points) == 3
    assert ui.feedback_error and not ui.mapper.enabled
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    ui.render(image)
    # Defensive support for callers restoring/mutating a public session incorrectly.
    ui.session = CalibrationSession(list(UNIT_CORNERS))
    assert "restart" in ui.session.prompt
    ui.render(image)
    ui.handle_key(32)
    assert len(ui.session.points) == 4 and ui.feedback_error
    ui.handle_key(ord("c"))
    assert len(ui.session.points) == 0 and not ui.feedback_error


@pytest.mark.parametrize("poll_available", [True, False])
@pytest.mark.parametrize("exit_mode", ["key", "window", "removed", "continue"])
def test_present_pumps_once_and_cleans_up(
    poll_available: bool,
    exit_mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    key = 27 if exit_mode == "key" else ord("d")

    def poll() -> int:
        calls.append("poll")
        return key

    def wait(delay: int) -> int:
        assert delay == 1
        calls.append("wait")
        return key

    def visible(name: str, prop: int) -> float:
        if exit_mode == "removed":
            raise cv2.error("Native window already removed")
        return 0 if exit_mode == "window" else 1

    monkeypatch.setattr(cv2, "namedWindow", lambda *args: calls.append("create"))
    monkeypatch.setattr(cv2, "resizeWindow", lambda *args: calls.append("resize"))
    monkeypatch.setattr(cv2, "imshow", lambda *args: calls.append("show"))
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args: calls.append("destroy"))
    monkeypatch.setattr(cv2, "getWindowProperty", visible)
    monkeypatch.setattr(cv2, "waitKey", wait)
    if poll_available:
        monkeypatch.setattr(cv2, "pollKey", poll, raising=False)
    else:
        monkeypatch.delattr(cv2, "pollKey", raising=False)
    ui = DemoUI(calibration_path=tmp_path / "none.json")
    ui.canvas.pen_down = True
    dispatches: list[int] = []
    handle = ui.handle_key

    def handle_once(code: int) -> None:
        dispatches.append(code)
        handle(code)

    monkeypatch.setattr(ui, "handle_key", handle_once)
    running = ui.present(np.zeros((480, 640, 3), dtype=np.uint8))
    assert dispatches == [key]
    assert calls[:4] == ["create", "resize", "show", "poll" if poll_available else "wait"]
    assert running == (exit_mode == "continue")
    if exit_mode != "key":
        assert ui.diagnostics  # The D key reached the centralized handler once.
    if not running:
        assert not ui.canvas.pen_down and ui.mapper.hand_id is None
        assert calls.count("destroy") == 1
        assert not ui.present(np.zeros((480, 640, 3), dtype=np.uint8))
        assert dispatches == [key]
    ui.close()
    ui.close()
    assert calls.count("destroy") == 1


def test_present_reports_elapsed_fps_and_sizes_window_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gesture_control.ui import demo

    timestamps = [0.0]
    for index in range(30):
        timestamps.append(timestamps[-1] + (0.01 if index % 2 == 0 else 2 / 30 - 0.01))
    clock = iter(timestamps)
    monkeypatch.setattr(demo, "perf_counter", lambda: next(clock))
    sizes: list[tuple[int, int]] = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda name, w, h: sizes.append((w, h)))
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(cv2, "pollKey", lambda: -1, raising=False)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args: None)
    ui = DemoUI(calibration_path=tmp_path / "none.json")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = ui.render(image)
    monkeypatch.setattr(ui, "render", lambda image: rendered)
    for _ in timestamps:
        assert ui.present(image)
    # Thirty frames across one second, despite alternating fast/slow intervals.
    assert ui.fps == pytest.approx(30)
    assert sizes == [(rendered.shape[1], rendered.shape[0])]
    ui.close()


@pytest.mark.parametrize("count", [1, 2, 3])
def test_repeated_calibration_capture_does_not_advance(count: int, tmp_path: Path) -> None:
    ui = DemoUI(calibration_path=tmp_path / "calibration.json")
    ui.handle_key(ord("c"))
    for point in UNIT_CORNERS[:count]:
        ui.pointer = point
        ui.handle_key(32)
    ui.handle_key(32)
    assert ui.session is not None and ui.session.points == list(UNIT_CORNERS[:count])
    assert ui.feedback_error and "SPACE to retry" in ui.status
    assert not ui.mapper.enabled
    for point in UNIT_CORNERS[count:]:
        ui.pointer = point
        ui.handle_key(32)
    assert ui.session is None and ui.mapper.enabled and not ui.feedback_error


def test_invalid_fourth_capture_preserves_saved_transform_and_retries(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    previous = Calibration(UNIT_CORNERS)
    previous.save(path)
    saved = path.read_bytes()
    ui = DemoUI(calibration_path=path)
    ui.handle_key(ord("c"))
    for point in ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.8, 0.2)):
        ui.pointer = point
        ui.handle_key(32)
    assert ui.session is not None and len(ui.session.points) == 3
    assert ui.mapper.calibration == previous and not ui.mapper.enabled
    assert path.read_bytes() == saved
    assert "SPACE to retry" in ui.status
    ui.pointer = (0.1, 0.9)
    ui.handle_key(32)
    assert ui.session is None and ui.mapper.enabled
    assert Calibration.load(path) == ui.mapper.calibration != previous


def test_camera_is_secondary_and_canvas_pixels_are_preserved(tmp_path: Path) -> None:
    ui = DemoUI(calibration_path=tmp_path / "none.json")
    camera = np.full((480, 640, 3), (40, 50, 60), dtype=np.uint8)
    camera[100:300, 200:500] = (90, 100, 110)
    rendered = ui.render(camera)
    # Camera is aspect-preserving at 400x300; the paper remains 640x480.
    expected = cv2.resize(camera, (400, 300), interpolation=cv2.INTER_AREA)
    assert np.array_equal(rendered[190:490, 12:412], expected)
    assert np.array_equal(rendered[100:580, 424:1064], ui.canvas.pixels)


def test_save_exports_paper_without_cursor(tmp_path: Path, context: FrameContext) -> None:
    path = tmp_path / "drawing.png"
    ui = DemoUI(calibration_path=tmp_path / "none.json", drawing_path=path)
    ui.canvas.execute(Command(context, "stroke.begin", {"x_px": 30, "y_px": 40}, "hand"))
    ui.canvas.execute(Command(context, "stroke.point", {"x_px": 80, "y_px": 90}, "hand"))
    ui.handle_key(ord("s"))
    assert not ui.feedback_error
    saved = cv2.imread(str(path))
    assert saved is not None and np.array_equal(saved, ui.canvas.pixels)
    assert ui.canvas.pen_down
