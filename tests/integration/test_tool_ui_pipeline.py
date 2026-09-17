"""Tool gestures through the production UI observer at different frame cadences."""

from math import ceil
from pathlib import Path

import numpy as np
import pytest

from gesture_control.app import GestureControlApp
from gesture_control.core.config import SmoothingConfig
from gesture_control.core.events import Command
from gesture_control.core.types import FrameContext, Gesture, HandFrame
from gesture_control.recognition import GeometricGestureRecognizer
from gesture_control.temporal import GestureTemporalProcessor
from gesture_control.ui.demo import DemoUI
from gesture_control.vision import PalmNormalizer

from .test_eraser_pipeline import pose
from .test_pipeline import UnusedDetector


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_actual_ui_observer_switches_and_erases_at_multiple_rates(tmp_path: Path, fps: int) -> None:
    ui = DemoUI(width=301, height=201, calibration_path=tmp_path / "none.json")
    app = GestureControlApp(
        detector=UnusedDetector(),
        recognizer=GeometricGestureRecognizer(PalmNormalizer()),
        temporal=ui.wrap_temporal(
            GestureTemporalProcessor(smoothing=SmoothingConfig(enabled=False))
        ),
        mapper=ui.mapper,
        adapter=ui.canvas,
    )
    frame_id = 0

    def hold(gesture: Gesture | None, seconds: float, x: float = 0.5) -> list[Command]:
        nonlocal frame_id
        commands: list[Command] = []
        for _ in range(ceil(seconds * fps)):
            context = FrameContext(frame_id, frame_id / fps, 640, 480)
            frame_id += 1
            hands = () if gesture is None else (pose(gesture, x=x),)
            commands.extend(app.process_landmarks(HandFrame(context, hands)))
            assert ui.mapper.tool == ui.canvas.tool
        return commands

    hold(Gesture.POINT, 0.4)
    hold(Gesture.PINCH, 0.2, 0.3)
    hold(Gesture.PINCH, 0.2, 0.7)
    drawn = ui.canvas.pixels.copy()
    assert np.any(drawn != (245, 247, 250))
    commands = hold(Gesture.TWO_FINGER, 0.85)
    assert [command.name for command in commands] == ["stroke.end", "tool.select"]
    assert ui.canvas.tool == "eraser"
    assert np.array_equal(drawn, ui.canvas.pixels)
    hold(Gesture.PINCH, 0.2, 0.3)
    hold(Gesture.PINCH, 0.2, 0.7)
    assert np.all(ui.canvas.pixels == (245, 247, 250))
    assert [command.name for command in hold(None, 1 / fps)] == ["stroke.end"]
    assert not ui.canvas.pen_down
    hold(Gesture.POINT, 0.4)
    assert [command.name for command in hold(Gesture.TWO_FINGER, 0.85)] == ["tool.select"]
    assert ui.canvas.tool == "pencil"
    ui.close()
