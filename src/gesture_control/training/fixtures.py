"""Hand-authored synthetic chains, not recordings or measured human labels."""

from math import cos, radians, sin

from gesture_control.core.types import Gesture, Landmark3D

from .augmentation import LandmarkSample


def synthetic_samples() -> tuple[LandmarkSample, ...]:
    """Six classes with straight and comfortable distal-curl variants.

    Provenance: adapted from the repository's original test_recognition.pose
    construction. Segments retain positive anatomical order. Distal curl rotates
    only the final bone in depth and preserves its length. These few pose groups
    are a regression corpus, emphatically not an independent accuracy dataset.
    """
    patterns = {
        Gesture.POINT: (True, False, False, False),
        Gesture.TWO_FINGER: (True, True, False, False),
        Gesture.OPEN_PALM: (True, True, True, True),
        Gesture.FIST: (False, False, False, False),
        Gesture.PINCH: (True, False, False, False),
        Gesture.UNKNOWN: (False, True, False, True),
    }
    samples = []
    for label, extended in patterns.items():
        for curl in (0, 35, 55):
            points = [
                Landmark3D(0, 0),
                Landmark3D(-0.5, -0.2),
                Landmark3D(-0.8, -0.4),
                Landmark3D(-1.1, -0.6),
                Landmark3D(-1.4, -0.8),
            ]
            for x, straight in zip((-0.4, 0, 0.35, 0.65), extended, strict=True):
                angle = radians(curl)
                points.extend(
                    (
                        Landmark3D(x, -1),
                        Landmark3D(x, -1.4),
                        Landmark3D(x, -1.7 if straight else -1.1, 0 if straight else -0.1),
                        Landmark3D(
                            x,
                            -1.7 - 0.3 * cos(angle) if straight else -0.85,
                            -0.3 * sin(angle) if straight else -0.15,
                        ),
                    )
                )
            if label is Gesture.PINCH:
                # Move the whole thumb chain toward contact, using an articulated
                # thumb instead of stretching a single tip across the hand.
                tip = points[8]
                points[2] = Landmark3D(-0.8, -0.8, -0.05)
                points[3] = Landmark3D(-0.65, (tip.y - 0.8) / 2, tip.z / 2)
                points[4] = Landmark3D(tip.x + 0.08, tip.y, tip.z)
            samples.append(
                LandmarkSample(
                    f"synthetic-{label.value}",
                    label,
                    tuple(points),
                    "hand-authored chains derived from test_recognition.pose; "
                    f"distal curl {curl} degrees; no camera data",
                )
            )
    return tuple(samples)
