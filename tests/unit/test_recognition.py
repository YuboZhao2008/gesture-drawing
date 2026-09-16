"""Synthetic anatomical chains; no camera, model files, or wall clock."""

from dataclasses import replace
from math import cos, sin

import pytest

from gesture_control.core.config import RecognitionConfig
from gesture_control.core.protocols import GestureRecognizer
from gesture_control.core.types import (
    FrameContext,
    Gesture,
    HandFrame,
    HandLandmarks,
    Landmark3D,
    NormalizedHand,
)
from gesture_control.recognition import (
    GeometricGestureRecognizer,
    GeometryConfig,
    finger_extension_score,
    joint_angle,
    landmark_distance,
)


def pose(extended: tuple[bool, bool, bool, bool], *, pinch: float | None = None) -> NormalizedHand:
    points = [
        Landmark3D(0, 0),
        Landmark3D(-0.5, -0.2),
        Landmark3D(-0.8, -0.4),
        Landmark3D(-1.1, -0.6),
        Landmark3D(-1.4, -0.8),
    ]
    for x, straight in zip((-0.4, 0, 0.35, 0.65), extended, strict=True):
        points.extend(
            (
                Landmark3D(x, -1),
                Landmark3D(x, -1.4),
                Landmark3D(x, -1.7 if straight else -1.1, 0 if straight else -0.1),
                Landmark3D(x, -2 if straight else -0.85, 0 if straight else -0.15),
            )
        )
    if pinch is not None:
        tip = points[8]
        points[4] = Landmark3D(tip.x + pinch, tip.y, tip.z)
    source = HandLandmarks("hand", tuple(points))
    return NormalizedHand(source, tuple(points), 1)


class FixedNormalizer:
    def __init__(self, normalized: NormalizedHand | None) -> None:
        self.normalized = normalized

    def normalize(self, hand: HandLandmarks, context: FrameContext) -> NormalizedHand | None:
        return self.normalized


@pytest.mark.parametrize(
    ("extended", "expected"),
    [
        ((True, False, False, False), Gesture.POINT),
        ((True, True, False, False), Gesture.TWO_FINGER),
        ((True, True, True, True), Gesture.OPEN_PALM),
        ((False, False, False, False), Gesture.FIST),
        ((False, True, False, True), Gesture.UNKNOWN),
    ],
)
def test_poses(extended: tuple[bool, bool, bool, bool], expected: Gesture) -> None:
    hand = pose(extended)
    recognizer = GeometricGestureRecognizer(FixedNormalizer(hand))
    result = recognizer.classify(hand)
    assert result.gesture is expected
    assert result.confidence == 0 if expected is Gesture.UNKNOWN else 0.7 < result.confidence < 1
    contract: GestureRecognizer = recognizer
    assert contract.recognize(HandFrame(FrameContext(0, 0, 640, 480), (hand.source,))) == (result,)


@pytest.mark.parametrize("distance", [0, 0.1, 0.25])
def test_pinch_priority_and_distance(distance: float) -> None:
    hand = pose((True, True, True, True), pinch=distance)
    result = GeometricGestureRecognizer(FixedNormalizer(hand)).classify(hand)
    assert result.gesture is Gesture.PINCH
    assert result.pinch_distance == pytest.approx(distance)


def test_confidence_and_threshold_configuration() -> None:
    hand = pose((True, False, False, False), pinch=0.2)
    recognizer = GeometricGestureRecognizer(FixedNormalizer(hand))
    close = recognizer.classify(pose((True, False, False, False), pinch=0.05))
    assert close.confidence > recognizer.classify(hand).confidence
    weak = replace(hand, source=replace(hand.source, confidence=0.5))
    assert recognizer.classify(weak).gesture is Gesture.UNKNOWN
    assert recognizer.classify(weak).confidence < 0.5
    strict = GeometricGestureRecognizer(
        FixedNormalizer(hand),
        RecognitionConfig(pinch_enter_threshold=0.1, pinch_exit_threshold=0.15),
    )
    assert strict.classify(hand).gesture is not Gesture.PINCH


@pytest.mark.parametrize("angle", [0.5, 1.5, 3.14])
@pytest.mark.parametrize("mirror", [-1, 1])
def test_rotation_mirroring_and_scale_invariance(angle: float, mirror: int) -> None:
    hand = pose((True, False, False, False))
    points = tuple(
        Landmark3D(
            2 * mirror * (p.x * cos(angle) - p.y * sin(angle)),
            2 * (p.x * sin(angle) + p.y * cos(angle)),
            2 * p.z,
        )
        for p in hand.landmarks
    )
    transformed = replace(hand, landmarks=points)
    recognizer = GeometricGestureRecognizer(FixedNormalizer(None))
    assert recognizer.classify(transformed).gesture is Gesture.POINT
    assert recognizer.classify(transformed).confidence == pytest.approx(
        recognizer.classify(hand).confidence
    )


def test_absence_and_normalization_failure(context: FrameContext, hand: HandLandmarks) -> None:
    recognizer = GeometricGestureRecognizer(FixedNormalizer(None))
    assert recognizer.recognize(HandFrame(context)) == ()
    (result,) = recognizer.recognize(HandFrame(context, (hand,)))
    assert result.gesture is Gesture.UNKNOWN
    assert result.confidence == 0
    assert result.pinch_distance is None


def test_collapsed_geometry_is_unknown() -> None:
    hand = pose((False, False, False, False))
    hand = replace(hand, landmarks=tuple(Landmark3D(0, 0) for _ in range(21)))
    result = GeometricGestureRecognizer(FixedNormalizer(None)).classify(hand)
    assert result.gesture is Gesture.UNKNOWN
    assert result.pinch_distance is None


def test_geometry_helpers() -> None:
    a, b, c, d = (Landmark3D(x, 0) for x in range(4))
    assert landmark_distance(a, d) == 3
    assert joint_angle(a, b, c) == 180
    assert joint_angle(a, a, a) == 0
    assert finger_extension_score(a, b, c, d) == 1
    assert finger_extension_score(a, a, a, a) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"folded_max": -1},
        {"extended_min": 0.5},
        {"confidence_floor": 1},
        {"confidence_ceiling": float("nan")},
        {"folded_max": True},
    ],
)
def test_invalid_geometry_config(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        GeometryConfig(**kwargs)
