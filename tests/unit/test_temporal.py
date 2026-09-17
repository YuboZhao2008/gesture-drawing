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
from gesture_control.temporal import GestureTemporalProcessor, HistoryConfig, PointerFilterConfig


class Stream:
    def __init__(
        self,
        *,
        dwell: float = 0.12,
        delay: float = 0,
        grace: float = 0.15,
        smoothing: bool = False,
        adaptive: bool = True,
    ) -> None:
        self.processor: TemporalProcessor = GestureTemporalProcessor(
            TemporalConfig(dwell, delay, grace, 0.1),
            smoothing=SmoothingConfig(enabled=smoothing),
            pointer_filter=PointerFilterConfig(adaptive=adaptive),
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
        stream = Stream(dwell=0, smoothing=True, adaptive=False)
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


@pytest.mark.parametrize("gesture", [Gesture.POINT, Gesture.PINCH])
def test_acquisition_boundary_emits_pointer_with_gesture_start(gesture: Gesture) -> None:
    stream = Stream(delay=0.2)
    stream.step(0.1, gesture)
    assert not stream.step(0.2, gesture).pointer_events
    # 0.1 + 0.2 rounds above 0.3; all eligibility checks must agree.
    out = stream.step(0.3, gesture)
    assert E.GESTURE_STARTED in kinds(out)
    assert len(out.pointer_events) == 1


def test_reacquisition_boundary_resumes_point_pointer_and_held_events() -> None:
    stream = Stream(delay=0.2, grace=0.5)
    for time in (0, 0.1, 0.2):
        stream.step(time)
    stream.step(0.3, None)
    assert not stream.step(0.4).pointer_events
    assert not stream.step(0.5).pointer_events
    # Long cursor dropout adds a new gate at 0.4 + 0.2.
    out = stream.step(0.6)
    assert kinds(out) == [E.GESTURE_HELD]
    assert len(out.pointer_events) == 1


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


@pytest.mark.parametrize(
    "noise", [Gesture.FIST, Gesture.OPEN_PALM, Gesture.TWO_FINGER, Gesture.UNKNOWN]
)
def test_single_bad_label_does_not_freeze_cursor_or_change_point(noise: Gesture) -> None:
    stream = Stream(smoothing=True)
    for time in (0, 0.06, 0.12):
        stream.step(time)
    out = stream.step(
        0.15, noise, confidence=0.2 if noise is Gesture.UNKNOWN else 0.9, distance=0.6, x=0.51
    )
    assert out.stable_gestures[0].gesture is Gesture.POINT
    assert E.GESTURE_ENDED not in kinds(out)
    assert len(out.pointer_events) == 1
    assert len(stream.step(0.18, x=0.52).pointer_events) == 1


def test_cursor_noise_tolerance_is_bounded() -> None:
    stream = Stream()
    for time in (0, 0.06, 0.12):
        stream.step(time)
    assert stream.step(0.15, Gesture.UNKNOWN, confidence=0.2, distance=0.6).pointer_events
    assert not stream.step(0.20, Gesture.UNKNOWN, confidence=0.2, distance=0.6).pointer_events
    out = stream.step(0.28, Gesture.UNKNOWN, confidence=0.2, distance=0.6)
    assert out.stable_gestures[0].gesture is Gesture.UNKNOWN


def test_short_point_dropout_preserves_filter_and_resumes_without_delay() -> None:
    stream = Stream(delay=0.2, smoothing=True)
    for time in (0, 0.1, 0.2):
        stream.step(time)
    missing = stream.step(0.22, None)
    assert not missing.pointer_events
    returned = stream.step(0.25, x=0.51)
    assert returned.stable_gestures[0].gesture is Gesture.POINT
    assert len(returned.pointer_events) == 1
    assert 0.5 < returned.pointer_events[0].x < 0.51  # Filter was not reseeded.
    assert E.HAND_ACQUIRED not in kinds(returned)


def test_long_point_dropout_and_new_id_do_not_reuse_cursor_state() -> None:
    stream = Stream(delay=0.1, smoothing=True)
    for time in (0, 0.06, 0.12):
        stream.step(time)
    stream.step(0.14, None)
    assert not stream.step(0.23, x=0.9).pointer_events  # >80 ms, still within loss grace.
    stream.step(0.29, x=0.9)
    returned = stream.step(0.35, x=0.9)
    assert returned.pointer_events[0].x == 0.9
    assert not stream.step(0.38, hand_id="new", x=0.1).pointer_events


@pytest.mark.parametrize("rate", [30, 60])
def test_pinch_start_and_release_latency(rate: int) -> None:
    stream = Stream()
    for i in range(rate):
        stream.step(i / rate)
    started: float | None = None
    ended: float | None = None
    for i in range(rate, rate * 2):
        out = stream.step(i / rate, Gesture.PINCH, distance=0.24)
        if E.PINCH_STARTED in kinds(out):
            started = i / rate
    for i in range(rate * 2, rate * 3):
        out = stream.step(i / rate, Gesture.POINT, distance=0.36)
        if E.PINCH_ENDED in kinds(out):
            ended = i / rate
            assert kinds(out)[:3] == [E.GESTURE_ENDED, E.PINCH_ENDED, E.GESTURE_STARTED]
    assert started is not None and ended is not None
    assert 0.04 <= started - 1 <= 0.04 + 1 / rate
    assert 0.04 <= ended - 2 <= 0.04 + 1 / rate


def test_single_exit_spike_recovers_inside_hysteresis_band() -> None:
    stream = Stream(dwell=0.1)
    stream.step(0, Gesture.PINCH, distance=0.24)
    stream.step(0.05, Gesture.PINCH, distance=0.24)
    stream.step(0.07, Gesture.POINT, distance=0.36)
    for time in (0.09, 0.11, 0.13, 0.15):
        out = stream.step(time, Gesture.UNKNOWN, confidence=0, distance=0.30)
        assert out.stable_gestures[0].gesture is Gesture.PINCH
        assert E.PINCH_ENDED not in kinds(out)


def test_alternating_release_labels_cannot_keep_drawing_active() -> None:
    stream = Stream()
    stream.step(0, Gesture.PINCH, distance=0.1)
    stream.step(0.05, Gesture.PINCH, distance=0.1)
    stream.step(0.08, Gesture.UNKNOWN, confidence=0, distance=0.5)
    stream.step(0.10, Gesture.FIST, distance=0.5)
    out = stream.step(0.12, Gesture.POINT, distance=0.5)
    assert kinds(out) == [E.GESTURE_ENDED, E.PINCH_ENDED]
    assert out.stable_gestures[0].gesture is Gesture.UNKNOWN
    assert not out.pointer_events


def test_pinch_band_oscillation_and_single_frame_false_start() -> None:
    stream = Stream()
    for time in (0, 0.06, 0.12):
        stream.step(time)
    stream.step(0.14, Gesture.PINCH, distance=0.24)
    assert E.PINCH_STARTED not in kinds(stream.step(0.16, Gesture.POINT, distance=0.4))
    stream.step(0.20, Gesture.PINCH, distance=0.24)
    out = stream.step(0.25, Gesture.PINCH, distance=0.24)
    assert E.PINCH_STARTED in kinds(out)
    for i in range(20):
        out = stream.step(
            0.27 + i * 0.02, Gesture.UNKNOWN, confidence=0, distance=0.26 if i % 2 else 0.34
        )
        assert E.PINCH_ENDED not in kinds(out)


def test_pointer_filters_are_isolated_by_track_and_reset() -> None:
    processor = GestureTemporalProcessor(TemporalConfig(0, 0, 0.15, 0.1))

    def step(time: float, frame_id: int, ax: float, bx: float) -> TemporalOutput:
        hands = tuple(
            HandLandmarks(name, tuple(Landmark3D(x, 0.5) for _ in range(21)))
            for name, x in (("a", ax), ("b", bx))
        )
        return processor.process(
            HandFrame(FrameContext(frame_id, time, 640, 480), hands),
            tuple(GesturePrediction(h.hand_id, Gesture.POINT, 0.9) for h in hands),
        )

    step(0, 0, 0.1, 0.9)
    out = step(0.03, 1, 0.15, 0.9)
    assert 0.1 < out.pointer_events[0].x < 0.15
    assert out.pointer_events[1].x == 0.9
    processor.reset()
    assert step(0, 0, 0.8, 0.2).pointer_events[0].x == 0.8


def test_rejected_pinch_candidate_cannot_later_start_from_exit_band() -> None:
    stream = Stream()
    for time in (0, 0.06, 0.12):
        stream.step(time)
    stream.step(0.14, Gesture.PINCH, distance=0.24)
    stream.step(0.16, Gesture.POINT, distance=0.40)
    # A rejected one-frame candidate is not an accepted pinch. Its old latch
    # must not turn later UNKNOWN evidence in the exit band into a new stroke.
    for time in (0.19, 0.22, 0.25, 0.28):
        out = stream.step(time, Gesture.UNKNOWN, confidence=0, distance=0.30)
        assert E.PINCH_STARTED not in kinds(out)
        assert out.stable_gestures[0].gesture is not Gesture.PINCH
    stream.step(0.30, Gesture.PINCH, distance=0.24)
    assert E.PINCH_STARTED in kinds(stream.step(0.35, Gesture.PINCH, distance=0.24))


def test_irregular_stream_has_balanced_ordered_pinch_lifecycles() -> None:
    stream = Stream(delay=0.1)
    sequence: list[tuple[float, Gesture | None, float | None]] = [
        (0, Gesture.POINT, 0.5),
        (0.07, Gesture.POINT, 0.5),
        (0.13, Gesture.POINT, 0.5),
        (0.16, Gesture.PINCH, 0.2),
        (0.18, Gesture.PINCH, 0.2),
        (0.21, Gesture.PINCH, 0.2),
        (0.21, Gesture.PINCH, 0.2),  # Equal time: no duplicate transition.
        (0.24, Gesture.POINT, 0.36),
        (0.26, Gesture.UNKNOWN, 0.3),
        (0.29, None, None),
        (0.32, None, None),  # End once, no pointers.
        (0.34, Gesture.PINCH, 0.2),
        (0.39, Gesture.PINCH, 0.2),
        (0.45, Gesture.PINCH, 0.2),  # New stroke after reacquisition.
        (0.48, Gesture.UNKNOWN, 0.5),
        (0.51, Gesture.FIST, 0.5),
        (0.53, Gesture.POINT, 0.5),  # Alternating labels still release.
        (0.69, None, None),
        (0.80, None, None),
    ]
    active = False
    starts = ends = 0
    for time, gesture, distance in sequence:
        out = stream.step(time, gesture, distance=distance)
        events = out.gesture_events
        for index, event in enumerate(events):
            if event.kind is E.PINCH_STARTED:
                assert not active
                assert index > 0 and events[index - 1].kind is E.GESTURE_STARTED
                assert events[index - 1].gesture is Gesture.PINCH
                active = True
                starts += 1
            elif event.kind is E.PINCH_ENDED:
                assert active
                assert index > 0 and events[index - 1].kind is E.GESTURE_ENDED
                assert events[index - 1].gesture is Gesture.PINCH
                active = False
                ends += 1
            elif event.kind in (E.HAND_ACQUIRED, E.HAND_LOST):
                assert not active
        if gesture is None:
            assert not active and not out.pointer_events
        assert all(event.context == out.context for event in events)
    assert starts == ends == 2
