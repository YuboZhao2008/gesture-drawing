"""Synthetic stream times exercise semantics independently of execution speed."""

from dataclasses import replace

import pytest

from gesture_control.core.config import SmoothingConfig, TemporalConfig
from gesture_control.core.events import GestureEvent, TemporalOutput
from gesture_control.core.events import GestureEventType as E
from gesture_control.core.protocols import TemporalProcessor
from gesture_control.core.types import (
    FrameContext,
    Gesture,
    GesturePrediction,
    HandFrame,
    HandLandmarks,
    Landmark3D,
)
from gesture_control.temporal import GestureTemporalProcessor, HistoryConfig


class Stream:
    def __init__(
        self, *, dwell: float = 0.12, delay: float = 0, grace: float = 0.15, smoothing: bool = False
    ) -> None:
        self.processor: TemporalProcessor = GestureTemporalProcessor(
            TemporalConfig(dwell, delay, grace, 0.1), smoothing=SmoothingConfig(enabled=smoothing)
        )
        self.frame_id = 0

    def step(
        self,
        time: float,
        gesture: Gesture | None = Gesture.POINT,
        confidence: float = 0.9,
        distance: float | None = None,
        hand_id: str = "a",
        x: float = 0.5,
    ) -> TemporalOutput:
        context = FrameContext(self.frame_id, time, 640, 480)
        self.frame_id += 1
        hand = HandLandmarks(hand_id, tuple(Landmark3D(x, 0.5) for _ in range(21)))
        frame = HandFrame(context, () if gesture is None else (hand,))
        predictions = (
            () if gesture is None else (GesturePrediction(hand_id, gesture, confidence, distance),)
        )
        return self.processor.process(frame, predictions)


def kinds(output: TemporalOutput) -> list[E]:
    return [event.kind for event in output.gesture_events]


@pytest.mark.parametrize(
    "times",
    [
        [i / 30 for i in range(10)],
        [i / 60 for i in range(20)],
        [0, 0.011, 0.055, 0.119, 0.121, 0.2, 0.29],
    ],
)
def test_dwell_at_variable_frame_rates(times: list[float]) -> None:
    stream = Stream()
    starts: list[GestureEvent] = []
    for time in times:
        output = stream.step(time)
        starts.extend(e for e in output.gesture_events if e.kind is E.GESTURE_STARTED)
        if time < 0.12:
            assert output.stable_gestures[0].gesture is Gesture.NONE
    assert len(starts) == 1
    assert starts[0].context.timestamp_s == next(t for t in times if t >= 0.12)


def test_one_frame_noise_and_transition_order() -> None:
    stream = Stream()
    for time in (0, 0.06, 0.12):
        stream.step(time)
    stream.step(0.15, Gesture.FIST)
    assert stream.step(0.18).stable_gestures[0].gesture is Gesture.POINT
    stream.step(0.21, Gesture.FIST)
    stream.step(0.27, Gesture.FIST)
    output = stream.step(0.33, Gesture.FIST)
    assert kinds(output) == [E.GESTURE_ENDED, E.GESTURE_STARTED]
    assert [e.gesture for e in output.gesture_events] == [Gesture.POINT, Gesture.FIST]
    assert output.gesture_events[0].duration_s == pytest.approx(0.21)


def test_pinch_hysteresis_and_companion_order() -> None:
    stream = Stream(dwell=0.1)
    stream.step(0, Gesture.PINCH, distance=0.24)
    out = stream.step(0.1, Gesture.PINCH, distance=0.24)
    assert kinds(out) == [E.GESTURE_STARTED, E.PINCH_STARTED]
    for time, distance in ((0.15, 0.26), (0.2, 0.34), (0.25, 0.27)):
        out = stream.step(time, Gesture.UNKNOWN, confidence=0, distance=distance)
        assert out.stable_gestures[0].gesture is Gesture.PINCH
        assert E.PINCH_ENDED not in kinds(out)
    assert E.GESTURE_HELD in kinds(stream.step(0.3, Gesture.PINCH, distance=0.2))
    stream.step(0.35, Gesture.POINT, distance=0.4)
    out = stream.step(0.45, Gesture.POINT, distance=0.4)
    assert kinds(out) == [E.GESTURE_ENDED, E.PINCH_ENDED, E.GESTURE_STARTED]


def test_exit_band_cannot_start_pinch() -> None:
    stream = Stream(dwell=0)
    out = stream.step(0, Gesture.PINCH, distance=0.3)
    assert E.PINCH_STARTED not in kinds(out)


def test_dropout_releases_pinch_immediately_and_reacquisition_is_gated() -> None:
    stream = Stream(dwell=0.05, delay=0.1)
    for time in (0, 0.05, 0.1):
        stream.step(time, Gesture.PINCH, distance=0.1)
    out = stream.step(0.12, None)
    assert kinds(out) == [E.GESTURE_ENDED, E.PINCH_ENDED]
    assert not out.pointer_events
    for time in (0.14, 0.19, 0.23):
        assert E.PINCH_STARTED not in kinds(stream.step(time, Gesture.PINCH, distance=0.1))
    assert E.PINCH_STARTED in kinds(stream.step(0.25, Gesture.PINCH, distance=0.1))
    out = stream.step(0.41, None)
    assert kinds(out) == [E.GESTURE_ENDED, E.PINCH_ENDED, E.HAND_LOST]
    assert out.stable_gestures == ()
    assert not stream.step(0.5, None).gesture_events


def test_point_loss_grace_and_reacquisition() -> None:
    stream = Stream(dwell=0, delay=0.1)
    stream.step(0)
    stream.step(0.1)
    out = stream.step(0.15, None)
    assert out.stable_gestures[0].gesture is Gesture.POINT
    assert not out.gesture_events and not out.pointer_events
    out = stream.step(0.26, None)
    assert kinds(out) == [E.GESTURE_ENDED, E.HAND_LOST]
    out = stream.step(0.3)
    assert kinds(out) == [E.HAND_ACQUIRED]
    assert out.stable_gestures[0].gesture is Gesture.NONE


def test_long_gap_does_not_count_as_dwell() -> None:
    stream = Stream()
    stream.step(0)
    out = stream.step(2)
    assert kinds(out) == [E.HAND_LOST, E.HAND_ACQUIRED]
    assert out.stable_gestures[0].gesture is Gesture.NONE


def test_confidence_gating_unknown_and_recovery() -> None:
    stream = Stream(dwell=0.1)
    stream.step(0)
    stream.step(0.1)
    assert not stream.step(0.15, confidence=0.2).pointer_events
    out = stream.step(0.25, confidence=0.2)
    assert kinds(out) == [E.GESTURE_ENDED]
    assert out.stable_gestures[0].gesture is Gesture.UNKNOWN
    stream.step(0.3)
    assert E.GESTURE_STARTED in kinds(stream.step(0.4))


def test_reset_and_invalid_stream_inputs() -> None:
    stream = Stream(dwell=0)
    stream.step(1, Gesture.PINCH)
    with pytest.raises(ValueError, match="timestamps"):
        stream.step(0)
    stream.processor.reset()
    out = stream.step(0)
    assert kinds(out) == [E.HAND_ACQUIRED, E.GESTURE_STARTED]
    assert E.PINCH_ENDED not in kinds(out)
    context = FrameContext(99, 1, 640, 480)
    with pytest.raises(ValueError, match="exactly one"):
        stream.processor.process(HandFrame(context), (GesturePrediction("x", Gesture.FIST, 1),))


def test_multiple_tracks_are_isolated_and_missing_track_ends() -> None:
    processor = GestureTemporalProcessor(TemporalConfig(0, 0, 0.15, 0.1))
    a = HandLandmarks("a", tuple(Landmark3D(0.5, 0.5) for _ in range(21)))
    b = replace(a, hand_id="b")
    out = processor.process(
        HandFrame(FrameContext(0, 0, 640, 480), (a, b)),
        (
            GesturePrediction("a", Gesture.PINCH, 0.9, 0.1),
            GesturePrediction("b", Gesture.FIST, 0.9),
        ),
    )
    assert [s.gesture for s in out.stable_gestures] == [Gesture.PINCH, Gesture.FIST]
    out = processor.process(
        HandFrame(FrameContext(1, 0.05, 640, 480), (b,)),
        (GesturePrediction("b", Gesture.FIST, 0.9),),
    )
    assert all(e.hand_id == "a" for e in out.gesture_events)
    assert out.stable_gestures[1].gesture is Gesture.FIST


def test_pointer_clipping_and_time_based_smoothing() -> None:
    def endpoint(times: list[float]) -> float:
        stream = Stream(dwell=0, smoothing=True)
        stream.step(0, x=0)
        out = stream.step(times[0], x=1)
        for time in times[1:]:
            out = stream.step(time, x=1)
        return out.pointer_events[0].x

    assert endpoint([0.1]) == pytest.approx(endpoint([0.05, 0.1]))
    assert Stream(dwell=0).step(0, x=2).pointer_events[0].x == 1


def test_history_configuration_validation() -> None:
    for value in (0, -1, float("nan"), True):
        with pytest.raises(ValueError):
            HistoryConfig(window_s=value)
    with pytest.raises(ValueError):
        HistoryConfig(max_samples=0)


def test_pointer_events_continue_between_held_events() -> None:
    stream = Stream(dwell=0)
    assert kinds(stream.step(0, Gesture.PINCH)) == [
        E.HAND_ACQUIRED,
        E.GESTURE_STARTED,
        E.PINCH_STARTED,
    ]
    for time in (0.01, 0.02, 0.04, 0.08):
        out = stream.step(time, Gesture.PINCH)
        assert not out.gesture_events
        assert len(out.pointer_events) == 1
        assert out.pointer_events[0].context == out.context
    assert kinds(stream.step(0.1, Gesture.PINCH)) == [E.GESTURE_HELD]


def test_zero_grace_only_loses_an_absent_hand() -> None:
    stream = Stream(dwell=0.1, grace=0)
    stream.step(0)
    assert kinds(stream.step(0.1)) == [E.GESTURE_STARTED]
    assert kinds(stream.step(0.11, None)) == [E.GESTURE_ENDED, E.HAND_LOST]


def test_repeated_timestamps_do_not_advance_dwell() -> None:
    stream = Stream()
    for _ in range(300):
        assert stream.step(0).stable_gestures[0].gesture is Gesture.NONE
    assert E.GESTURE_STARTED in kinds(stream.step(0.12))


def test_pinch_exit_threshold_is_inclusive() -> None:
    stream = Stream(dwell=0)
    stream.step(0, Gesture.PINCH, distance=0.25)
    out = stream.step(0.01, Gesture.POINT, distance=0.35)
    assert kinds(out) == [E.GESTURE_ENDED, E.PINCH_ENDED, E.GESTURE_STARTED]


def test_hand_confidence_gates_pinch_distance() -> None:
    processor = GestureTemporalProcessor(TemporalConfig(0, 0, 0.15, 0.1))
    hand = HandLandmarks("a", tuple(Landmark3D(0.5, 0.5) for _ in range(21)))
    processor.process(
        HandFrame(FrameContext(0, 0, 640, 480), (hand,)),
        (GesturePrediction("a", Gesture.PINCH, 0.9, 0.1),),
    )
    out = processor.process(
        HandFrame(FrameContext(1, 0.01, 640, 480), (replace(hand, confidence=0.1),)),
        (GesturePrediction("a", Gesture.PINCH, 0.9, 0.1),),
    )
    assert kinds(out) == [E.GESTURE_ENDED, E.PINCH_ENDED]
    assert not out.pointer_events
