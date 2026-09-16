"""Real geometry, normalization, temporal policy, calibration and raster adapter."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gesture_control.app import GestureControlApp
from gesture_control.core.config import SmoothingConfig, TemporalConfig
from gesture_control.core.events import Command
from gesture_control.core.events import GestureEventType as Kind
from gesture_control.core.types import FrameContext, Gesture, HandFrame, HandLandmarks, Landmark3D
from gesture_control.recognition import GeometricGestureRecognizer
from gesture_control.temporal import GestureTemporalProcessor
from gesture_control.ui.calibration import Calibration
from gesture_control.ui.demo import DemoUI
from gesture_control.vision import PalmNormalizer


def source_hand(
    gesture: Gesture, *, x: float = 0.5, y: float = 0.8, hand_id: str = "test-hand"
) -> HandLandmarks:
    """Anatomical chains projected into a 640x480 source, with a 0.2-width palm."""
    points = [(0.0, 0.0, 0.0), (-0.5, -0.2, 0), (-0.8, -0.4, 0), (-1.1, -0.6, 0), (-1.4, -0.8, 0)]
    for index, offset in enumerate((-0.4, 0, 0.35, 0.65)):
        extended = index == 0 or gesture is Gesture.OPEN_PALM
        points.extend(
            [
                (offset, -1, 0),
                (offset, -1.4, 0),
                (offset, -1.7 if extended else -1.1, 0 if extended else -0.1),
                (offset, -2 if extended else -0.85, 0 if extended else -0.15),
            ]
        )
    if gesture is Gesture.PINCH:
        tip = points[8]
        points[4] = (tip[0] + 0.08, tip[1], tip[2])
    if gesture is Gesture.UNKNOWN:
        points = [(0, 0, 0)] * 21
    return HandLandmarks(
        hand_id, tuple(Landmark3D(x + a * 0.2, y + b * 0.2 / 0.75, c * 0.2) for a, b, c in points)
    )


class UnusedDetector:
    def detect(self, image: object, context: FrameContext) -> HandFrame:
        raise AssertionError("Landmark replay must bypass detection")


class Pipeline:
    def __init__(self, path: Path, *, dwell: float = 0) -> None:
        self.ui = DemoUI(width=301, height=201, calibration_path=path / "calibration.json")
        self.recognizer = GeometricGestureRecognizer(PalmNormalizer())
        self.temporal = GestureTemporalProcessor(
            TemporalConfig(
                gesture_dwell_s=dwell, reacquisition_delay_s=dwell, hand_loss_grace_s=0.2
            ),
            smoothing=SmoothingConfig(enabled=False),
        )
        self.app = GestureControlApp(
            detector=UnusedDetector(),
            recognizer=self.recognizer,
            temporal=self.ui.wrap_temporal(self.temporal),
            mapper=self.ui.mapper,
            adapter=self.ui.canvas,
        )
        self.frame_id = 0

    def step(self, hand: HandLandmarks | None) -> tuple[Command, ...]:
        context = FrameContext(self.frame_id, self.frame_id * 0.05, 640, 480)
        self.frame_id += 1
        commands = self.app.process_landmarks(HandFrame(context, () if hand is None else (hand,)))
        assert all(command.context == context for command in commands)
        assert all(command.hand_id == "test-hand" for command in commands)
        return commands


def names(commands: tuple[Command, ...]) -> list[str]:
    return [command.name for command in commands]


def test_point_uses_source_coordinates_and_does_not_draw(tmp_path: Path) -> None:
    p = Pipeline(tmp_path)
    hand = source_hand(Gesture.POINT)
    before = p.ui.canvas.pixels.copy()
    assert names(p.step(hand)) == ["pointer.move"]
    assert p.ui.predictions[0].gesture is Gesture.POINT
    assert p.ui.output is not None
    pointer = p.ui.output.pointer_events[0]
    assert (pointer.x, pointer.y) == pytest.approx((hand.landmarks[8].x, hand.landmarks[8].y))
    assert p.ui.canvas.cursor == (126, 53)
    assert np.array_equal(before, p.ui.canvas.pixels)


def test_pinch_order_begin_continue_end_once(tmp_path: Path) -> None:
    p = Pipeline(tmp_path)
    p.step(source_hand(Gesture.POINT))
    assert names(p.step(source_hand(Gesture.PINCH))) == ["pointer.move", "stroke.begin"]
    assert p.ui.output is not None
    assert [e.kind for e in p.ui.output.gesture_events] == [
        Kind.GESTURE_ENDED,
        Kind.GESTURE_STARTED,
        Kind.PINCH_STARTED,
    ]
    assert names(p.step(source_hand(Gesture.PINCH, x=0.6))) == ["pointer.move", "stroke.point"]
    assert p.ui.canvas.pen_down
    assert names(p.step(source_hand(Gesture.POINT))) == ["stroke.end", "pointer.move"]
    assert [e.kind for e in p.ui.output.gesture_events] == [
        Kind.GESTURE_ENDED,
        Kind.PINCH_ENDED,
        Kind.GESTURE_STARTED,
    ]
    assert not p.ui.canvas.pen_down
    assert names(p.step(source_hand(Gesture.POINT))) == ["pointer.move"]


def test_loss_releases_immediately_and_never_connects_across_gap(tmp_path: Path) -> None:
    p = Pipeline(tmp_path)
    p.step(source_hand(Gesture.PINCH, x=0.3))
    assert names(p.step(None)) == ["stroke.end"]
    assert p.ui.predictions == () and p.ui.frame is not None and p.ui.frame.hands == ()
    assert not p.ui.canvas.pen_down
    assert names(p.step(None)) == []
    assert names(p.step(source_hand(Gesture.PINCH, x=0.8))) == ["pointer.move", "stroke.begin"]
    # Both isolated start dots lie on this row, far either side of the midpoint.
    assert tuple(p.ui.canvas.pixels[53, 140]) == (245, 247, 250)


def test_visible_unknown_is_distinct_from_absence_and_releases(tmp_path: Path) -> None:
    p = Pipeline(tmp_path)
    p.step(source_hand(Gesture.PINCH))
    assert names(p.step(source_hand(Gesture.UNKNOWN))) == ["stroke.end"]
    assert p.ui.predictions[0].gesture is Gesture.UNKNOWN
    assert p.ui.output is not None
    assert p.ui.output.stable_gestures[0].gesture is Gesture.UNKNOWN
    assert p.ui.output.pointer_events == ()
    assert names(p.step(None)) == []
    assert p.ui.predictions == ()


def test_open_palm_is_neutral(tmp_path: Path) -> None:
    p = Pipeline(tmp_path)
    assert names(p.step(source_hand(Gesture.OPEN_PALM))) == []
    assert p.ui.predictions[0].gesture is Gesture.OPEN_PALM
    p.step(source_hand(Gesture.PINCH))
    assert names(p.step(source_hand(Gesture.OPEN_PALM))) == ["stroke.end"]


def test_calibration_source_to_workspace_to_pixels(tmp_path: Path) -> None:
    p = Pipeline(tmp_path)
    p.ui.mapper.calibration = Calibration(((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)))
    p.step(source_hand(Gesture.POINT, x=0.58, y=0.5 + 0.4 / 0.75))
    assert p.ui.pointer == pytest.approx((0.5, 0.5))
    assert p.ui.canvas.cursor == (150, 100)
    # Rendering/resizing the preview cannot change interaction coordinates.
    p.ui.render(np.zeros((480, 640, 3), dtype=np.uint8))
    assert p.ui.canvas.cursor == (150, 100)


def test_timestamps_drive_dwell_and_loss_reacquisition(tmp_path: Path) -> None:
    p = Pipeline(tmp_path, dwell=0.1)
    hand = source_hand(Gesture.PINCH)
    assert p.step(hand) == ()
    assert p.step(hand) == ()
    assert names(p.step(hand)) == ["pointer.move", "stroke.begin"]
    assert names(p.step(None)) == ["stroke.end"]
    assert p.step(hand) == ()
    assert p.step(hand) == ()
    assert names(p.step(hand)) == ["pointer.move", "stroke.begin"]


@pytest.mark.parametrize("key", [ord("r"), ord("c"), 27])
def test_controls_release_and_require_fresh_pinch(tmp_path: Path, key: int) -> None:
    p = Pipeline(tmp_path)
    p.step(source_hand(Gesture.PINCH))
    p.ui.handle_key(key)
    assert not p.ui.canvas.pen_down and p.ui.mapper.hand_id is None
    assert "stroke.begin" not in names(p.step(source_hand(Gesture.PINCH)))


def test_low_confidence_cannot_draw(tmp_path: Path) -> None:
    p = Pipeline(tmp_path)
    assert p.step(replace(source_hand(Gesture.PINCH), confidence=0.3)) == ()
    assert p.ui.predictions[0].gesture is Gesture.UNKNOWN


def test_large_workspace_display_preserves_pointer_and_export_resolution(tmp_path: Path) -> None:
    import cv2

    ui = DemoUI(
        width=1920,
        height=1080,
        calibration_path=tmp_path / "calibration.json",
        drawing_path=tmp_path / "drawing.png",
    )
    context = FrameContext(0, 0, 640, 480)
    ui.canvas.execute(Command(context, "pointer.move", {"x_px": 959, "y_px": 539}, "test-hand"))
    rendered = ui.render(np.zeros((480, 640, 3), dtype=np.uint8))
    assert rendered.shape == (606, 1316, 3)
    assert ui.canvas.cursor == (959, 539)
    ui.handle_key(ord("s"))
    saved = cv2.imread(str(tmp_path / "drawing.png"))
    assert saved is not None and saved.shape == (1080, 1920, 3)
    assert np.array_equal(saved, ui.canvas.pixels)
