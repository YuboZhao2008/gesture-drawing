"""Offline robustness corpus and confusion metrics; never imported by inference."""

from dataclasses import replace

from gesture_control.core.protocols import GestureRecognizer
from gesture_control.core.types import FrameContext, Gesture, Handedness, HandFrame, Landmark3D

from .augmentation import augment, source_observation
from .fixtures import synthetic_samples


def evaluate(
    recognizer: GestureRecognizer, *, variants: int = 100, seed: int = 1709
) -> dict[str, object]:
    if isinstance(variants, bool) or not isinstance(variants, int) or variants < 1:
        raise ValueError("variants must be a positive integer")
    labels = tuple(label for label in Gesture if label is not Gesture.NONE)
    matrix = {label.value: {other.value: 0 for other in labels} for label in labels}
    wrong_actions = 0
    confidence_sums = {"correct": 0.0, "incorrect": 0.0}
    counts = {"correct": 0, "incorrect": 0}
    rejected_corruption = 0
    corruption_total = 0
    for group, sample in enumerate(synthetic_samples()):
        for variant in range(variants):
            frame_id = group * variants + variant
            context = FrameContext(frame_id, frame_id / 30, 640, 480)
            augmented = augment(sample, seed=seed + frame_id)
            hand = source_observation(
                augmented,
                context,
                scale=(0.08, 0.2, 0.34)[variant % 3],
                translation=((0.25, 0.65), (0.7, 0.85))[variant % 2],
                handedness=(Handedness.LEFT, Handedness.RIGHT)[variant % 2],
            )
            (prediction,) = recognizer.recognize(HandFrame(context, (hand,)))
            matrix[sample.label.value][prediction.gesture.value] += 1
            correct = prediction.gesture is sample.label
            key = "correct" if correct else "incorrect"
            counts[key] += 1
            confidence_sums[key] += prediction.confidence
            wrong_actions += int(not correct and prediction.gesture is not Gesture.UNKNOWN)
        # Corrupt observations have a distinct expected rejection label. They are
        # NOT counted as label-preserving augmentations or class recall examples.
        clean = source_observation(sample, context)
        corruptions = [replace(clean, confidence=0.3)]
        for joint in (2, 6, 10, 14, 18):
            points = list(clean.landmarks)
            anchor = points[joint - 1]
            points[joint] = anchor
            corruptions.append(replace(clean, landmarks=tuple(points)))
            points[joint] = Landmark3D(anchor.x + 1e-5, anchor.y, anchor.z)
            corruptions.append(replace(clean, landmarks=tuple(points)))
        for corrupt in corruptions:
            (prediction,) = recognizer.recognize(HandFrame(context, (corrupt,)))
            corruption_total += 1
            rejected_corruption += int(prediction.gesture is Gesture.UNKNOWN)
    per_class = {}
    for label in labels:
        name = label.value
        actual = sum(matrix[name].values())
        predicted = sum(matrix[other.value][name] for other in labels)
        true_positive = matrix[name][name]
        per_class[name] = {
            "support": actual,
            "correct": true_positive,
            "precision": true_positive / predicted if predicted else None,
            "recall": true_positive / actual if actual else None,
        }
    return {
        "total": sum(counts.values()),
        "correct": counts["correct"],
        "wrong_action_labels": wrong_actions,
        "confusion_matrix": matrix,
        "per_class": per_class,
        "mean_confidence": {
            key: confidence_sums[key] / count if count else None for key, count in counts.items()
        },
        "corruption_rejection": {"total": corruption_total, "rejected": rejected_corruption},
        "original_groups": len({sample.group_id for sample in synthetic_samples()}),
        "pose_templates": len(synthetic_samples()),
        "limitations": "Hand-authored regression groups only; confidence is a geometric "
        "margin, not a calibrated probability. No real-world accuracy claim.",
    }
