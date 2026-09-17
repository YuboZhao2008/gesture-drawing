"""Bounded, seedable augmentation of gesture-classification test/training data.

Offline only. This does not train or improve MediaPipe landmark detection.
"""

from dataclasses import dataclass, replace
from math import cos, isfinite, radians, sin
from random import Random

from gesture_control.core.types import (
    FrameContext,
    Gesture,
    Handedness,
    HandLandmarks,
    Landmark3D,
)


@dataclass(frozen=True, slots=True)
class LandmarkSample:
    """Version 1 synthetic pose in wrist-centered palm units.

    ``group_id`` identifies the original pose and must survive augmentation; any
    future learned evaluation must split these groups BEFORE making variants.
    Labels describe the intended pose, not a classifier's output.
    """

    group_id: str
    label: Gesture
    points: tuple[Landmark3D, ...]
    provenance: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != 1
        ):
            raise ValueError("unsupported sample schema version")
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("group_id must be nonempty")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("provenance must be nonempty")
        if not isinstance(self.label, Gesture) or self.label is Gesture.NONE:
            raise ValueError("a visible sample needs a gesture label")
        hand = HandLandmarks(self.group_id, self.points)
        object.__setattr__(self, "points", hand.landmarks)
        if hand.landmarks[0] != Landmark3D(0, 0, 0):
            raise ValueError("sample wrist must be at the origin")


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Small perturbations only; large transforms cannot promise pose meaning.

    Noise is Gaussian clipped at three standard deviations in palm units. Joint
    perturbation rotates a distal bone, preserving its length. Foreshortening is
    a mild orthographic hand tilt, not a simulated camera/detector model.
    """

    rotation_degrees: float = 20
    tilt_degrees: float = 15
    noise_std: float = 0.006
    depth_std: float = 0.012
    joint_degrees: float = 4

    def __post_init__(self) -> None:
        limits = {
            "rotation_degrees": 45,
            "tilt_degrees": 25,
            "noise_std": 0.015,
            "depth_std": 0.025,
            "joint_degrees": 8,
        }
        for name, maximum in limits.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0 <= value <= maximum
            ):
                raise ValueError(f"{name} must be finite and in [0, {maximum}]")


def augment(
    sample: LandmarkSample, *, seed: int, config: AugmentationConfig | None = None
) -> LandmarkSample:
    """Return a deterministic same-label variant without global random state.

    Use only interior, validated poses; a transform cannot guarantee that an
    arbitrary decision-boundary label remains meaningful. Pinch-contact fingers
    are perturbed together (global transform/noise only), preserving contact.
    """
    if not isinstance(sample, LandmarkSample):
        raise ValueError("sample must be LandmarkSample")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if config is not None and not isinstance(config, AugmentationConfig):
        raise ValueError("config must be AugmentationConfig")
    cfg = config or AugmentationConfig()
    rng = Random(seed)
    points = list(sample.points)
    for tip_index in (4, 8, 12, 16, 20):
        if sample.label is Gesture.PINCH and tip_index in (4, 8):
            continue
        angle = radians(rng.uniform(-cfg.joint_degrees, cfg.joint_degrees))
        tip, joint = points[tip_index], points[tip_index - 1]
        dy, dz = tip.y - joint.y, tip.z - joint.z
        points[tip_index] = Landmark3D(
            tip.x,
            joint.y + dy * cos(angle) - dz * sin(angle),
            joint.z + dy * sin(angle) + dz * cos(angle),
        )
    angle = radians(rng.uniform(-cfg.rotation_degrees, cfg.rotation_degrees))
    tilt = radians(rng.uniform(-cfg.tilt_degrees, cfg.tilt_degrees))

    def noise(std: float) -> float:
        return max(-3 * std, min(3 * std, rng.gauss(0, std)))

    transformed = [Landmark3D(0, 0, 0)]
    for point in points[1:]:
        x, y = (
            point.x * cos(angle) - point.y * sin(angle),
            point.x * sin(angle) + point.y * cos(angle),
        )
        transformed.append(
            Landmark3D(
                x + noise(cfg.noise_std),
                y * cos(tilt) - point.z * sin(tilt) + noise(cfg.noise_std),
                y * sin(tilt) + point.z * cos(tilt) + noise(cfg.depth_std),
            )
        )
    return replace(sample, points=tuple(transformed))


def source_observation(
    sample: LandmarkSample,
    context: FrameContext,
    *,
    scale: float = 0.2,
    translation: tuple[float, float] = (0.5, 0.75),
    handedness: Handedness = Handedness.RIGHT,
    confidence: float = 1,
) -> HandLandmarks:
    """Convert a sample to image units, including aspect ratio and mirroring.

    Whole-observation confidence can be degraded; the public observation format
    has no per-landmark confidence or missing-point field. Missing chains should
    be tested as corruption/rejection, not mislabeled label-preserving examples.
    """
    if not isinstance(sample, LandmarkSample) or not isinstance(context, FrameContext):
        raise ValueError("sample and context must use the public types")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("scale must be positive and finite")
    if len(translation) != 2 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
        for v in translation
    ):
        raise ValueError("translation must contain two finite numbers")
    if not isinstance(handedness, Handedness):
        raise ValueError("invalid handedness")
    mirror = -1 if handedness is Handedness.LEFT else 1
    aspect = context.image_width / context.image_height
    return HandLandmarks(
        sample.group_id,
        tuple(
            Landmark3D(
                translation[0] + mirror * scale * p.x,
                translation[1] + scale * p.y * aspect,
                scale * p.z,
            )
            for p in sample.points
        ),
        handedness,
        confidence,
    )
