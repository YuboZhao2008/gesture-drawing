"""Deterministic offline perturbations, metadata and source-coordinate invariants."""

from dataclasses import replace
from math import dist

import pytest

from gesture_control.core.types import FrameContext, Gesture, Handedness, Landmark3D
from gesture_control.training.augmentation import (
    AugmentationConfig,
    LandmarkSample,
    augment,
    source_observation,
)
from gesture_control.training.fixtures import synthetic_samples
from gesture_control.vision import PalmNormalizer


def test_seed_is_deterministic_and_does_not_relabel() -> None:
    for sample in synthetic_samples():
        first = augment(sample, seed=42)
        assert first == augment(sample, seed=42)
        assert first != augment(sample, seed=43)
        assert (first.label, first.group_id, first.provenance, first.schema_version) == (
            sample.label,
            sample.group_id,
            sample.provenance,
            sample.schema_version,
        )
        assert first.points[0] == Landmark3D(0, 0, 0)


def test_identity_transform_and_rigid_transform_preserve_lengths() -> None:
    sample = synthetic_samples()[0]
    identity = AugmentationConfig(0, 0, 0, 0, 0)
    assert augment(sample, seed=1, config=identity) == sample
    rotated = augment(
        sample, seed=2, config=replace(identity, rotation_degrees=30, tilt_degrees=20)
    )
    for a, b in zip(sample.points, rotated.points, strict=True):
        assert dist((0, 0, 0), (a.x, a.y, a.z)) == pytest.approx(dist((0, 0, 0), (b.x, b.y, b.z)))


def test_joint_perturbation_keeps_bone_lengths() -> None:
    sample = synthetic_samples()[0]
    result = augment(sample, seed=13, config=AugmentationConfig(0, 0, 0, 0, 8))
    for tip in (4, 8, 12, 16, 20):
        a, b = sample.points[tip - 1 : tip + 1]
        c, d = result.points[tip - 1 : tip + 1]
        assert dist((a.x, a.y, a.z), (b.x, b.y, b.z)) == pytest.approx(
            dist((c.x, c.y, c.z), (d.x, d.y, d.z))
        )


@pytest.mark.parametrize("scale", [0.04, 0.2, 0.8])
@pytest.mark.parametrize("size", [(640, 480), (1280, 720), (480, 640)])
@pytest.mark.parametrize("handedness", [Handedness.LEFT, Handedness.RIGHT])
def test_source_conversion_and_canonical_normalization(
    scale: float, size: tuple[int, int], handedness: Handedness
) -> None:
    sample = synthetic_samples()[0]
    context = FrameContext(0, 0, *size)
    source = source_observation(
        sample, context, scale=scale, translation=(-0.3, 1.2), handedness=handedness
    )
    normalized = PalmNormalizer().normalize(source, context)
    assert normalized is not None
    for actual, expected in zip(normalized.landmarks, sample.points, strict=True):
        mirror = -1 if handedness is Handedness.LEFT else 1
        assert (actual.x, actual.y, actual.z) == pytest.approx(
            (mirror * expected.x, expected.y, expected.z)
        )
    assert normalized.source is source


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"label": Gesture.NONE},
        {"label": "point"},
        {"group_id": " "},
        {"provenance": ""},
        {"points": (Landmark3D(0, 0),)},
    ],
)
def test_invalid_samples_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(synthetic_samples()[0], **changes)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True, 90])
def test_unbounded_augmentation_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        AugmentationConfig(rotation_degrees=value)


def test_invalid_runtime_arguments() -> None:
    sample = synthetic_samples()[0]
    context = FrameContext(0, 0, 640, 480)
    with pytest.raises(ValueError):
        augment(sample, seed=True)
    with pytest.raises(ValueError):
        source_observation(sample, context, scale=0)
    with pytest.raises(ValueError):
        source_observation(sample, context, translation=(float("nan"), 0))
    with pytest.raises(ValueError):
        source_observation(sample, context, confidence=1.1)
    with pytest.raises(ValueError):
        LandmarkSample("bad", Gesture.POINT, (Landmark3D(1, 0),) * 21, "test")
