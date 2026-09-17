"""Deterministic tool intent, stroke safety, and genuine ink removal."""

from dataclasses import replace

import numpy as np
import pytest

from gesture_control.core.config import SmoothingConfig, TemporalConfig
from gesture_control.core.events import (
    Command,
    GestureEvent,
    GestureEventType,
    PointerEvent,
    TemporalOutput,
)
from gesture_control.core.types import (
    FrameContext,
    Gesture,
    GesturePrediction,
    HandFrame,
    HandLandmarks,
    Landmark3D,
)
from gesture_control.temporal import GestureTemporalProcessor
from gesture_control.ui.calibration import Calibration
from gesture_control.ui.drawing import DrawingCanvas, DrawingMapper


class Interaction:
    def __init__(self) -> None:
        self.mapper = DrawingMapper(201, 201)
        self.canvas = DrawingCanvas(201, 201)
        self.temporal = GestureTemporalProcessor(
            TemporalConfig(gesture_dwell_s=0, reacquisition_delay_s=0),
            smoothing=SmoothingConfig(enabled=False),
        )
        self.frame_id = 0

    def step(
        self,
        time: float,
        gesture: Gesture | None,
        *,
        confidence: float = 0.95,
        hand_confidence: float = 1.0,
        hand_id: str = "a",
        point: tuple[float, float] = (0.5, 0.5),
        pinch_distance: float | None = None,
    ) -> tuple[Command, ...]:
        context = FrameContext(self.frame_id, time, 640, 480)
        self.frame_id += 1
        hand = HandLandmarks(hand_id, (Landmark3D(*point),) * 21, confidence=hand_confidence)
        frame = HandFrame(context, () if gesture is None else (hand,))
        predictions = (
            ()
            if gesture is None
            else (GesturePrediction(hand_id, gesture, confidence, pinch_distance),)
        )
        output = self.temporal.process(frame, predictions)
        self.mapper.observe_evidence(frame, predictions)
        commands = self.mapper.map(output)
        for command in commands:
            self.canvas.execute(command)
        assert self.mapper.tool == self.canvas.tool
        return commands

    def hold(
        self, start: float, gesture: Gesture, *, count: int = 15, hand_id: str = "a"
    ) -> list[Command]:
        return [
            command
            for i in range(count)
            for command in self.step(start + i * 0.05, gesture, hand_id=hand_id)
        ]


def names(commands: tuple[Command, ...] | list[Command]) -> list[str]:
    return [command.name for command in commands]


def test_deliberate_enter_exit_and_held_gesture_never_repeats() -> None:
    session = Interaction()
    commands = session.hold(0, Gesture.TWO_FINGER)
    assert names(commands) == ["tool.select"]
    assert commands[0].context.timestamp_s == pytest.approx(0.65)
    assert session.canvas.tool == "eraser"
    assert session.mapper.tool_switch_cooldown_remaining_s == pytest.approx(0.75)
    assert names(session.hold(0.75, Gesture.TWO_FINGER, count=45)) == []
    session.hold(3, Gesture.POINT, count=5)
    assert names(session.hold(3.25, Gesture.TWO_FINGER)) == ["tool.select"]
    assert session.canvas.tool == "pencil"


@pytest.mark.parametrize(
    "gesture", [Gesture.POINT, Gesture.FIST, Gesture.OPEN_PALM, Gesture.UNKNOWN]
)
def test_other_gestures_do_not_select_tools(gesture: Gesture) -> None:
    session = Interaction()
    assert "tool.select" not in names(session.hold(0, gesture, count=30))
    assert session.canvas.tool == "pencil"


@pytest.mark.parametrize("interruption", [None, Gesture.UNKNOWN, Gesture.POINT])
def test_noise_and_brief_occlusion_restart_full_dwell(interruption: Gesture | None) -> None:
    session = Interaction()
    session.hold(0, Gesture.TWO_FINGER, count=12)
    session.step(0.6, interruption)
    assert session.mapper.tool_switch_progress == 0
    assert names(session.hold(0.65, Gesture.TWO_FINGER, count=13)) == []
    assert names(session.step(1.3, Gesture.TWO_FINGER)) == ["tool.select"]


@pytest.mark.parametrize("field", ["confidence", "hand_confidence"])
def test_low_confidence_never_accumulates_tool_dwell(field: str) -> None:
    session = Interaction()
    for i in range(40):
        kwargs = {field: 0.79}
        assert names(session.step(i * 0.05, Gesture.TWO_FINGER, **kwargs)) == []  # type: ignore[arg-type]
    assert session.mapper.tool_switch_progress == 0
    assert session.canvas.tool == "pencil"


def test_one_frame_tool_noise_and_gap_cannot_switch() -> None:
    session = Interaction()
    for time, gesture in (
        (0, Gesture.TWO_FINGER),
        (0.05, Gesture.POINT),
        (0.1, Gesture.TWO_FINGER),
        (2, Gesture.TWO_FINGER),
    ):
        assert "tool.select" not in names(session.step(time, gesture))
    assert session.mapper.tool_switch_progress == 0
    assert session.canvas.tool == "pencil"


def test_changed_hand_cannot_inherit_tool_dwell() -> None:
    session = Interaction()
    session.hold(0, Gesture.TWO_FINGER, count=13)
    assert names(session.step(0.65, Gesture.TWO_FINGER, hand_id="b")) == []
    assert session.mapper.tool_switch_progress == 0
    assert names(session.hold(0.7, Gesture.TWO_FINGER, count=13, hand_id="b")) == ["tool.select"]


def test_cooldown_and_deliberate_release_both_required() -> None:
    session = Interaction()
    session.hold(0, Gesture.TWO_FINGER)
    session.step(0.75, Gesture.POINT)
    session.hold(0.8, Gesture.TWO_FINGER, count=25)
    assert session.canvas.tool == "eraser"  # One-frame release did not rearm.
    session.hold(2.05, Gesture.UNKNOWN, count=5)
    session.hold(2.3, Gesture.TWO_FINGER)
    assert session.canvas.tool == "eraser"  # UNKNOWN is not deliberate release.
    session.hold(3.05, Gesture.POINT, count=5)
    session.hold(3.3, Gesture.TWO_FINGER)
    assert session.canvas.tool == "pencil"


def test_raw_tool_evidence_ends_stale_pinch_without_stray_stroke() -> None:
    session = Interaction()
    session.step(0, Gesture.PINCH, pinch_distance=0.05)
    before = session.canvas.pixels.copy()
    commands = session.step(0.05, Gesture.TWO_FINGER, pinch_distance=0.05, point=(0.8, 0.8))
    assert names(commands) == ["stroke.end"]
    assert not session.canvas.pen_down
    assert np.array_equal(before, session.canvas.pixels)
    for i in range(1, 14):
        commands = session.step(0.05 + i * 0.05, Gesture.TWO_FINGER, pinch_distance=0.05)
        assert not any(c.name.startswith("stroke") for c in commands)
    assert session.canvas.tool == "eraser"


def test_tool_transition_does_not_consume_second_hand_pinch() -> None:
    session = Interaction()
    session.hold(0, Gesture.TWO_FINGER, count=13)
    context = FrameContext(13, 0.65, 640, 480)
    hands = tuple(HandLandmarks(id_, (Landmark3D(0.5, 0.5),) * 21) for id_ in ("a", "b"))
    session.mapper.observe_evidence(
        HandFrame(context, hands),
        (GesturePrediction("a", Gesture.TWO_FINGER, 1), GesturePrediction("b", Gesture.PINCH, 1)),
    )
    output = TemporalOutput(
        context,
        gesture_events=(GestureEvent(context, "b", GestureEventType.PINCH_STARTED, Gesture.PINCH),),
        pointer_events=(PointerEvent(context, "b", 0.5, 0.5),),
    )
    assert names(session.mapper.map(output)) == ["tool.select"]
    assert not session.mapper.stroke_active


def test_other_hand_cannot_switch_during_owned_stroke() -> None:
    session = Interaction()
    session.step(0, Gesture.PINCH)
    for i in range(1, 20):
        context = FrameContext(i, i * 0.05, 640, 480)
        hands = tuple(HandLandmarks(id_, (Landmark3D(0.5, 0.5),) * 21) for id_ in ("b", "a"))
        session.mapper.observe_evidence(
            HandFrame(context, hands),
            (
                GesturePrediction("b", Gesture.TWO_FINGER, 1),
                GesturePrediction("a", Gesture.PINCH, 1),
            ),
        )
        assert names(session.mapper.map(TemporalOutput(context))) == []
    assert session.mapper.tool == "pencil"


def test_erase_motion_release_loss_new_id_and_calibration() -> None:
    session = Interaction()
    session.mapper.calibration = Calibration(((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)))
    session.step(0, Gesture.PINCH, point=(0.3, 0.5))
    session.step(0.05, Gesture.PINCH, point=(0.7, 0.5))
    session.step(0.1, Gesture.POINT)
    ink = session.canvas.pixels.copy()
    session.hold(0.15, Gesture.TWO_FINGER)
    assert names(session.step(0.9, Gesture.PINCH, point=(0.3, 0.5))) == [
        "pointer.move",
        "stroke.begin",
    ]
    assert session.canvas.cursor == (33, 100)
    assert names(session.step(0.95, Gesture.PINCH, point=(0.4, 0.5))) == [
        "pointer.move",
        "stroke.point",
    ]
    assert tuple(session.canvas.pixels[100, 50]) == (245, 247, 250)
    assert np.array_equal(session.canvas.pixels[100, 100], ink[100, 100])
    assert names(session.step(1, None)) == ["stroke.end"]
    assert names(session.step(1.05, None)) == []
    assert names(session.step(1.1, Gesture.PINCH, hand_id="b", point=(0.7, 0.5))) == [
        "pointer.move",
        "stroke.begin",
    ]
    assert np.array_equal(
        session.canvas.pixels[100, 100], ink[100, 100]
    )  # No bridge across hand IDs.
    assert names(session.step(1.15, Gesture.POINT, hand_id="b")) == ["stroke.end", "pointer.move"]
    assert not session.canvas.pen_down


def test_clear_reset_preserves_tool_and_requires_fresh_pinch() -> None:
    session = Interaction()
    session.hold(0, Gesture.TWO_FINGER)
    session.step(0.75, Gesture.PINCH)
    session.canvas.clear()
    session.mapper.reset()
    assert session.mapper.tool == session.canvas.tool == "eraser"
    assert session.mapper.tool_switch_progress == 0
    assert names(session.step(0.8, Gesture.PINCH)) == ["pointer.move"]
    session.step(0.85, Gesture.POINT)
    assert names(session.step(0.9, Gesture.PINCH)) == ["pointer.move", "stroke.begin"]


def test_reset_or_disabled_calibration_cancels_pending_dwell() -> None:
    session = Interaction()
    session.hold(0, Gesture.TWO_FINGER, count=13)
    session.mapper.enabled = False
    assert names(session.step(0.65, Gesture.TWO_FINGER)) == []
    session.mapper.enabled = True
    assert names(session.step(0.7, Gesture.TWO_FINGER)) == []
    assert session.mapper.tool_switch_progress == 0
    session.mapper.reset()
    assert names(session.step(0.75, Gesture.TWO_FINGER)) == []
    assert session.mapper.tool_switch_progress == 0


def test_eraser_removes_ink_and_restores_nonuniform_background() -> None:
    background = np.zeros((101, 101, 3), dtype=np.uint8)
    background[:, :, 0] = np.arange(101, dtype=np.uint8)
    background[:, :, 1] = 70
    canvas = DrawingCanvas(101, 101, background=background)
    context = FrameContext(0, 0, 640, 480)
    canvas.execute(Command(context, "stroke.begin", {"x_px": 10, "y_px": 50}, "a"))
    canvas.execute(Command(context, "stroke.point", {"x_px": 90, "y_px": 50}, "a"))
    canvas.execute(Command(context, "tool.select", {"tool": "eraser"}, "a"))
    canvas.execute(Command(context, "stroke.begin", {"x_px": 10, "y_px": 50}, "a"))
    canvas.execute(Command(context, "stroke.point", {"x_px": 90, "y_px": 50}, "a"))
    assert np.array_equal(canvas.pixels, background)
    assert not canvas._ink.any()
    canvas.clear()
    assert canvas.tool == "eraser" and not canvas.pen_down
    assert np.array_equal(canvas.pixels, background)


def test_eraser_cursor_scale_and_render_never_change_export_pixels() -> None:
    canvas = DrawingCanvas(1920, 1080)
    canvas.tool = "eraser"
    canvas.cursor = (960, 540)
    assert canvas.eraser_radius_px == 32
    before = canvas.pixels.copy()
    preview = canvas.render((640, 360))
    assert preview.shape == (360, 640, 3)
    assert np.array_equal(before, canvas.pixels)


def test_invalid_tool_background_and_evidence_rejected() -> None:
    context = FrameContext(0, 0, 640, 480)
    with pytest.raises(ValueError):
        DrawingCanvas().execute(Command(context, "tool.select", {"tool": "paint"}))
    with pytest.raises(ValueError):
        DrawingCanvas(10, 10, background=np.zeros((20, 10, 3), dtype=np.uint8))
    mapper = DrawingMapper()
    with pytest.raises(ValueError):
        mapper.observe_evidence(HandFrame(context), (GesturePrediction("a", Gesture.POINT, 1),))
    mapper.observe_evidence(HandFrame(context), ())
    with pytest.raises(ValueError):
        mapper.observe_evidence(HandFrame(context), ())
    with pytest.raises(ValueError):
        mapper.observe_evidence(
            HandFrame(replace(context, frame_id=1, timestamp_s=0)),
            (GesturePrediction("a", Gesture.POINT, 1),),
        )


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_switch_timing(value: float) -> None:
    with pytest.raises(ValueError):
        DrawingMapper(tool_switch_dwell_s=value)


@pytest.mark.parametrize("value", [0, -1, 1.1, True, float("nan")])
def test_invalid_switch_confidence(value: float) -> None:
    with pytest.raises(ValueError):
        DrawingMapper(tool_switch_min_confidence=value)
