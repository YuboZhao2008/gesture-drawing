from dataclasses import FrozenInstanceError, replace
from typing import cast

import pytest

from gesture_control.core.types import (
    FrameContext,
    Gesture,
    GesturePrediction,
    Handedness,
    HandFrame,
    HandLandmarks,
    Landmark3D,
    LandmarkIndex,
    NormalizedHand,
    StableGesture,
)


def test_gesture_vocabulary_and_round_trip() -> None:
    assert {gesture.name for gesture in Gesture} == {
        "NONE",
        "POINT",
        "PINCH",
        "OPEN_PALM",
        "FIST",
        "TWO_FINGER",
        "UNKNOWN",
    }
    assert Gesture("pinch") is Gesture.PINCH
    assert len({Gesture.NONE, Gesture.UNKNOWN}) == 2
    assert Handedness("left") is Handedness.LEFT
    assert [index.value for index in LandmarkIndex] == list(range(21))
    assert LandmarkIndex.INDEX_TIP.value == 8


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_landmarks_reject_nonfinite_coordinates(value: float) -> None:
    with pytest.raises(ValueError):
        Landmark3D(0, value, 0)


def test_landmarks_allow_offscreen_points_and_are_immutable() -> None:
    point = Landmark3D(-0.1, 1.1, -0.5)
    with pytest.raises(FrozenInstanceError):
        point.x = 0.0  # type: ignore[misc]


def test_hand_topology_and_tuple_snapshot(hand: HandLandmarks) -> None:
    with pytest.raises(ValueError, match="21"):
        replace(hand, landmarks=hand.landmarks[:-1])
    points = list(hand.landmarks)
    copied = replace(hand, landmarks=cast(tuple[Landmark3D, ...], points))
    points.clear()
    assert len(copied.landmarks) == 21
    assert copied.landmarks[LandmarkIndex.MIDDLE_MCP] == Landmark3D(0.5, 0.25)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan")])
def test_hand_confidence_validation(hand: HandLandmarks, confidence: float) -> None:
    with pytest.raises(ValueError):
        replace(hand, confidence=confidence)


def test_frame_supports_absence_and_distinct_tracks(
    context: FrameContext, hand: HandLandmarks
) -> None:
    assert HandFrame(context).hands == ()
    second = replace(hand, hand_id="track-2")
    assert len(HandFrame(context, (hand, second)).hands) == 2
    with pytest.raises(ValueError, match="unique"):
        HandFrame(context, (hand, hand))


@pytest.mark.parametrize(
    ("field", "value"),
    [("frame_id", -1), ("timestamp_s", -0.1), ("image_width", 0), ("image_height", True)],
)
def test_context_rejects_invalid_metadata(context: FrameContext, field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(context, **{field: value})  # type: ignore[arg-type]


def test_normalized_hand_retains_source_and_requires_origin_and_scale(hand: HandLandmarks) -> None:
    points = tuple(
        Landmark3D(0, -1 if index == LandmarkIndex.MIDDLE_MCP else 0) for index in LandmarkIndex
    )
    normalized = NormalizedHand(hand, points, scale=0.375)
    assert normalized.source is hand
    assert normalized.landmarks[LandmarkIndex.WRIST] == Landmark3D(0, 0)
    with pytest.raises(ValueError, match="scale"):
        replace(normalized, scale=0)
    with pytest.raises(ValueError, match="origin"):
        replace(normalized, landmarks=hand.landmarks)


def test_unknown_observation_is_distinct_from_inactive_temporal_state() -> None:
    prediction = GesturePrediction("track-1", Gesture.UNKNOWN, 0)
    inactive = StableGesture("track-1", Gesture.NONE, 0, since_s=1)
    assert prediction.gesture is not inactive.gesture
    with pytest.raises(ValueError, match="no-hand"):
        replace(prediction, gesture=Gesture.NONE)
    with pytest.raises(ValueError, match="since_s"):
        replace(inactive, since_s=-1)
    with pytest.raises(ValueError, match="hand_id"):
        replace(prediction, hand_id=" ")
    assert replace(prediction, pinch_distance=0.25).pinch_distance == 0.25
    with pytest.raises(ValueError, match="pinch_distance"):
        replace(prediction, pinch_distance=-0.1)
