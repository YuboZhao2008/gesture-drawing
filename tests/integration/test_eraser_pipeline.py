"""Real landmark geometry through temporal processing, mapping and ink erasure."""

from dataclasses import replace

import numpy as np
import pytest

from gesture_control.app import GestureControlApp
from gesture_control.core.config import SmoothingConfig, TemporalConfig
from gesture_control.core.events import Command, TemporalOutput
from gesture_control.core.types import (
    FrameContext,
    Gesture,
    GesturePrediction,
    HandFrame,
    HandLandmarks,
)
from gesture_control.recognition import GeometricGestureRecognizer
from gesture_control.temporal import GestureTemporalProcessor
from gesture_control.ui.calibration import Calibration
from gesture_control.ui.drawing import DrawingCanvas, DrawingMapper
from gesture_control.vision import PalmNormalizer

from .test_pipeline import UnusedDetector, source_hand


def pose(gesture: Gesture, *, x: float = 0.5, hand_id: str = "a") -> HandLandmarks:
    hand = source_hand(gesture, x=x, hand_id=hand_id)
    if gesture is Gesture.TWO_FINGER:
        points = list(hand.landmarks)
        palm = source_hand(Gesture.OPEN_PALM, x=x, hand_id=hand_id)
        points[9:13] = palm.landmarks[9:13]
        hand = replace(hand, landmarks=tuple(points))
    return hand


class EvidenceTemporal:
    """Exercise the documented observer tap independently of UI presentation."""

    def __init__(self, mapper: DrawingMapper) -> None:
        self.mapper = mapper
        self.processor = GestureTemporalProcessor(
            TemporalConfig(gesture_dwell_s=0.12, reacquisition_delay_s=0.1),
            smoothing=SmoothingConfig(enabled=False),
        )

    def process(
        self, frame: HandFrame, predictions: tuple[GesturePrediction, ...]
    ) -> TemporalOutput:
        output = self.processor.process(frame, predictions)
        self.mapper.observe_evidence(frame, predictions)
        return output

    def reset(self) -> None:
        self.processor.reset()
        self.mapper.reset()


class EraserPipeline:
    def __init__(self, *, calibrated: bool = False) -> None:
        self.mapper = DrawingMapper(301, 201)
        if calibrated:
            self.mapper.calibration = Calibration(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))
        self.canvas = DrawingCanvas(301, 201)
        self.recognizer = GeometricGestureRecognizer(PalmNormalizer())
        self.temporal = EvidenceTemporal(self.mapper)
        self.app = GestureControlApp(
            detector=UnusedDetector(),
            recognizer=self.recognizer,
            temporal=self.temporal,
            mapper=self.mapper,
            adapter=self.canvas,
        )
        self.frame_id = 0

    def step(self, hand: HandLandmarks | None) -> tuple[Command, ...]:
        context = FrameContext(self.frame_id, self.frame_id / 20, 640, 480)
        self.frame_id += 1
        commands = self.app.process_landmarks(HandFrame(context, () if hand is None else (hand,)))
        assert all(command.context == context for command in commands)
        assert self.canvas.tool == self.mapper.tool
        return commands

    def hold(
        self, gesture: Gesture, *, count: int = 16, x: float = 0.5, hand_id: str = "a"
    ) -> list[Command]:
        return [
            command
            for _ in range(count)
            for command in self.step(pose(gesture, x=x, hand_id=hand_id))
        ]


def names(commands: list[Command] | tuple[Command, ...]) -> list[str]:
    return [command.name for command in commands]


@pytest.mark.parametrize("calibrated", [False, True])
def test_real_geometry_pencil_to_eraser_motion_release_and_return(calibrated: bool) -> None:
    p = EraserPipeline(calibrated=calibrated)
    p.hold(Gesture.PINCH, x=0.3, count=4)
    p.hold(Gesture.PINCH, x=0.7, count=4)
    drawn = p.canvas.pixels.copy()
    assert np.any(drawn != (245, 247, 250))
    assert names(p.hold(Gesture.TWO_FINGER)) == ["stroke.end", "tool.select"]
    assert p.canvas.tool == "eraser"
    assert np.array_equal(drawn, p.canvas.pixels)
    erase = p.hold(Gesture.PINCH, x=0.3, count=4)
    assert names(erase).count("stroke.begin") == 1
    p.hold(Gesture.PINCH, x=0.7, count=4)
    assert np.all(p.canvas.pixels == (245, 247, 250))
    released = p.hold(Gesture.POINT, count=5)
    assert names(released).count("stroke.end") == 1
    assert not p.canvas.pen_down
    assert names(p.hold(Gesture.TWO_FINGER)) == ["tool.select"]
    assert p.canvas.tool == "pencil"
    assert names(p.hold(Gesture.PINCH, count=4)).count("stroke.begin") == 1
    assert np.any(p.canvas.pixels != (245, 247, 250))


def test_eraser_direct_changed_id_reacquires_without_erasing_bridge() -> None:
    p = EraserPipeline()
    p.hold(Gesture.PINCH, x=0.3, count=4)
    p.hold(Gesture.PINCH, x=0.7, count=4)
    p.hold(Gesture.TWO_FINGER)
    p.hold(Gesture.PINCH, x=0.3, count=4)
    before = p.canvas.pixels.copy()
    assert names(p.step(pose(Gesture.PINCH, x=0.7, hand_id="b"))) == ["stroke.end"]
    assert not p.canvas.pen_down
    resumed = p.hold(Gesture.PINCH, x=0.7, hand_id="b", count=4)
    assert names(resumed).count("stroke.begin") == 1
    assert p.canvas.hand_id == "b"
    assert np.array_equal(before[53, 130], p.canvas.pixels[53, 130])
    assert tuple(p.canvas.pixels[53, 130]) != (245, 247, 250)
    assert names(p.step(None)) == ["stroke.end"]
    assert names(p.step(None)) == []


def test_real_tool_pose_noise_cannot_accumulate_stable_dwell() -> None:
    p = EraserPipeline()
    p.hold(Gesture.TWO_FINGER, count=12)
    p.step(pose(Gesture.UNKNOWN))
    assert p.mapper.tool_switch_progress == 0
    assert names(p.hold(Gesture.TWO_FINGER, count=12)) == []
    p.step(None)
    assert p.canvas.tool == "pencil"
    assert names(p.hold(Gesture.TWO_FINGER)) == ["tool.select"]


def test_stream_reset_requires_new_tool_dwell_and_clears_reacquisition_state() -> None:
    p = EraserPipeline()
    p.hold(Gesture.TWO_FINGER)
    p.hold(Gesture.PINCH, count=4)
    p.temporal.reset()
    p.canvas.release()
    p.frame_id = 0
    assert p.mapper.tool == p.canvas.tool == "eraser"
    assert names(p.step(pose(Gesture.PINCH))) == []
    assert names(p.hold(Gesture.PINCH, count=4)).count("stroke.begin") == 1
    assert p.canvas.pen_down
