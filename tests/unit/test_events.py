from dataclasses import replace

import pytest

from gesture_control.core.events import (
    Command,
    CommandValue,
    GestureEvent,
    GestureEventType,
    PointerEvent,
    TemporalOutput,
)
from gesture_control.core.types import FrameContext, Gesture, StableGesture


def test_all_gesture_event_kinds_are_representable(context: FrameContext) -> None:
    for kind in GestureEventType:
        gesture = Gesture.NONE if kind is GestureEventType.HAND_LOST else Gesture.PINCH
        event = GestureEvent(context, "track-1", kind, gesture)
        assert event.context is context
        assert GestureEventType(kind.value) is kind


def test_pinch_and_hand_loss_semantics(context: FrameContext) -> None:
    with pytest.raises(ValueError, match="PINCH"):
        GestureEvent(context, "track-1", GestureEventType.PINCH_STARTED, Gesture.FIST)
    with pytest.raises(ValueError, match="NONE"):
        GestureEvent(context, "track-1", GestureEventType.HAND_LOST, Gesture.PINCH)
    with pytest.raises(ValueError, match="duration_s"):
        GestureEvent(context, "track-1", GestureEventType.GESTURE_HELD, Gesture.POINT, -1)


def test_pointer_coordinates_are_source_image_fractions(context: FrameContext) -> None:
    assert PointerEvent(context, "track-1", 0, 1).y == 1
    with pytest.raises(ValueError):
        PointerEvent(context, "track-1", 640, 480)
    with pytest.raises(ValueError):
        PointerEvent(context, "track-1", float("nan"), 0)


def test_temporal_batch_checks_context_and_state_timing(context: FrameContext) -> None:
    state = StableGesture("track-1", Gesture.POINT, 0.9, since_s=1)
    event = GestureEvent(context, "track-1", GestureEventType.GESTURE_STARTED, Gesture.POINT)
    output = TemporalOutput(context, (state,), (event,))
    assert output.gesture_events == (event,)
    with pytest.raises(ValueError, match="context"):
        replace(output, context=replace(context, frame_id=8))
    with pytest.raises(ValueError, match="unique"):
        replace(output, stable_gestures=(state, state))
    with pytest.raises(ValueError, match="after"):
        replace(output, stable_gestures=(replace(state, since_s=2),))


def test_command_parameters_are_detached_and_read_only(context: FrameContext) -> None:
    parameters: dict[str, CommandValue] = {"x": 0.5, "tool": "pen", "enabled": True, "value": None}
    command = Command(context, "canvas.move", parameters, hand_id="track-1")
    parameters["x"] = 0.8
    assert command.parameters["x"] == 0.5
    with pytest.raises(TypeError):
        command.parameters["x"] = 0.9  # type: ignore[index]
    assert dict(command.parameters)["tool"] == "pen"


def test_command_rejects_invalid_parameters(context: FrameContext) -> None:
    with pytest.raises(ValueError, match="finite"):
        Command(context, "custom.command", {"value": float("inf")})
    with pytest.raises(ValueError, match="JSON scalars"):
        Command(context, "custom.command", {"nested": []})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="nonempty"):
        Command(context, " ")
