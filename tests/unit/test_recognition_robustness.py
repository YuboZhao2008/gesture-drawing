"""Synthetic classification robustness, never detector-accuracy measurements."""

from dataclasses import replace
from math import cos, radians, sin

import pytest

from gesture_control.core.types import FrameContext, Gesture, HandFrame, Landmark3D
from gesture_control.recognition import GeometricGestureRecognizer, finger_extension_score
from gesture_control.training.augmentation import augment, source_observation
from gesture_control.training.evaluation import evaluate
from gesture_control.training.fixtures import synthetic_samples
from gesture_control.vision import PalmNormalizer


def test_augmented_corpus_acceptance_and_safe_rejection() -> None:
    report = evaluate(GeometricGestureRecognizer(PalmNormalizer()), variants=100, seed=1709)
    assert report["total"] == 1800
    assert isinstance(report["correct"], int) and report["correct"] >= 1760
    assert report["wrong_action_labels"] == 0
    assert report["corruption_rejection"] == {"total": 198, "rejected": 198}


@pytest.mark.parametrize("seed", [11, 97, 1234])
def test_unseen_seeds_never_turn_unsupported_pose_into_action(seed: int) -> None:
    recognizer = GeometricGestureRecognizer(PalmNormalizer())
    context = FrameContext(0, 0, 640, 480)
    for sample in synthetic_samples():
        if sample.label is Gesture.UNKNOWN:
            for index in range(30):
                hand = source_observation(augment(sample, seed=seed + index), context)
                (prediction,) = recognizer.recognize(HandFrame(context, (hand,)))
                assert prediction.gesture is Gesture.UNKNOWN


@pytest.mark.parametrize("curl", [45, 50, 55])
def test_mild_isolated_distal_curl_retains_extension(curl: float) -> None:
    angle = radians(curl)
    score = finger_extension_score(
        Landmark3D(0, 0),
        Landmark3D(0, -0.4),
        Landmark3D(0, -0.7),
        Landmark3D(0, -0.7 - 0.3 * cos(angle), -0.3 * sin(angle)),
    )
    assert score > 0.75


def test_one_straight_joint_cannot_hide_sharp_fold() -> None:
    score = finger_extension_score(
        Landmark3D(0, 0), Landmark3D(0, -0.4), Landmark3D(0.3, -0.4), Landmark3D(0.6, -0.4)
    )
    assert 0.55 < score < 0.75  # Ambiguous side bend, never an extended finger.


@pytest.mark.parametrize("joint", [2, 6, 10, 14, 18])
@pytest.mark.parametrize("distance", [0, 1e-6, 0.001])
def test_partially_collapsed_chains_reject_even_when_pinched(joint: int, distance: float) -> None:
    sample = next(sample for sample in synthetic_samples() if sample.label is Gesture.PINCH)
    context = FrameContext(0, 0, 640, 480)
    hand = source_observation(sample, context)
    points = list(hand.landmarks)
    anchor = points[joint - 1]
    points[joint] = Landmark3D(anchor.x + distance, anchor.y, anchor.z)
    hand = replace(hand, landmarks=tuple(points))
    (prediction,) = GeometricGestureRecognizer(PalmNormalizer()).recognize(
        HandFrame(context, (hand,))
    )
    assert prediction.gesture is Gesture.UNKNOWN
    assert prediction.confidence == 0
    assert prediction.pinch_distance is None


def test_anatomical_handedness_metadata_alone_does_not_change_prediction() -> None:
    from gesture_control.core.types import Handedness

    context = FrameContext(0, 0, 640, 480)
    recognizer = GeometricGestureRecognizer(PalmNormalizer())
    for sample in synthetic_samples():
        hand = source_observation(sample, context)
        (right,) = recognizer.recognize(HandFrame(context, (hand,)))
        (left,) = recognizer.recognize(
            HandFrame(context, (replace(hand, handedness=Handedness.LEFT),))
        )
        assert left == right


@pytest.mark.parametrize("curl", [80, 85, 90])
def test_half_bent_middle_finger_cannot_select_tool_under_noise(curl: float) -> None:
    from gesture_control.training.augmentation import AugmentationConfig

    sample = next(s for s in synthetic_samples() if s.label is Gesture.TWO_FINGER)
    points = list(sample.points)
    angle = radians(curl)
    points[12] = Landmark3D(0, -1.7 - 0.3 * cos(angle), -0.3 * sin(angle))
    sample = replace(sample, label=Gesture.UNKNOWN, points=tuple(points))
    context = FrameContext(0, 0, 640, 480)
    recognizer = GeometricGestureRecognizer(PalmNormalizer())
    perturbation = AugmentationConfig(noise_std=0.003, depth_std=0.006, joint_degrees=2)
    for seed in range(100):
        hand = source_observation(augment(sample, seed=seed, config=perturbation), context)
        (prediction,) = recognizer.recognize(HandFrame(context, (hand,)))
        assert prediction.gesture is Gesture.UNKNOWN


def test_all_unsupported_finger_combinations_reject_under_noise() -> None:
    from itertools import product

    folded = next(s for s in synthetic_samples() if s.label is Gesture.FIST)
    opened = next(s for s in synthetic_samples() if s.label is Gesture.OPEN_PALM)
    supported = {
        (True, False, False, False),
        (True, True, False, False),
        (False, False, False, False),
        (True, True, True, True),
    }
    context = FrameContext(0, 0, 640, 480)
    recognizer = GeometricGestureRecognizer(PalmNormalizer())
    for pattern in product((False, True), repeat=4):
        if pattern in supported:
            continue
        points = list(folded.points)
        for start, extended in zip((5, 9, 13, 17), pattern, strict=True):
            if extended:
                points[start : start + 4] = opened.points[start : start + 4]
        sample = replace(folded, label=Gesture.UNKNOWN, points=tuple(points))
        for seed in range(30):
            hand = source_observation(augment(sample, seed=seed), context)
            (prediction,) = recognizer.recognize(HandFrame(context, (hand,)))
            assert prediction.gesture is Gesture.UNKNOWN
