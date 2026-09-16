"""Canonical geometry, independent of camera and model dependencies."""

import math

from gesture_control.core.types import (
    FrameContext,
    HandLandmarks,
    Landmark3D,
    LandmarkIndex,
    NormalizedHand,
)


class PalmNormalizer:
    """Preserve rotation/handedness; wrist origin, 3D middle-MCP palm scale."""

    def normalize(self, hand: HandLandmarks, context: FrameContext) -> NormalizedHand | None:
        wrist = landmark(hand, LandmarkIndex.WRIST)
        aspect = context.image_height / context.image_width
        points = tuple(
            Landmark3D(p.x - wrist.x, (p.y - wrist.y) * aspect, p.z - wrist.z)
            for p in hand.landmarks
        )
        middle = points[LandmarkIndex.MIDDLE_MCP]
        scale = math.hypot(middle.x, middle.y, middle.z)
        if scale <= 1e-9:
            return None
        return NormalizedHand(
            hand, tuple(Landmark3D(p.x / scale, p.y / scale, p.z / scale) for p in points), scale
        )


def landmark(hand: HandLandmarks | NormalizedHand, index: LandmarkIndex) -> Landmark3D:
    """Look up one canonical joint, rejecting accidental negative/index integers."""
    if not isinstance(index, LandmarkIndex):
        raise ValueError("index must be LandmarkIndex")
    return hand.landmarks[index]


def index_fingertip(hand: HandLandmarks) -> Landmark3D:
    """Raw source-image fractions and image-width depth; never clipped or mirrored."""
    return landmark(hand, LandmarkIndex.INDEX_TIP)


def pointer_coordinates(hand: HandLandmarks) -> tuple[float, float]:
    """Normalized camera x/y fractions (may be off-image); temporal owns clipping."""
    tip = index_fingertip(hand)
    return tip.x, tip.y


def fingertip_pixels(hand: HandLandmarks, context: FrameContext) -> tuple[float, float]:
    """Continuous camera coordinates, using the contract's width/height divisors."""
    tip = index_fingertip(hand)
    return tip.x * context.image_width, tip.y * context.image_height


def hand_bounding_box(hand: HandLandmarks) -> tuple[float, float, float, float]:
    """Unclipped source fractions: left, top, right, bottom."""
    return (
        min(p.x for p in hand.landmarks),
        min(p.y for p in hand.landmarks),
        max(p.x for p in hand.landmarks),
        max(p.y for p in hand.landmarks),
    )


def palm_center(hand: HandLandmarks | NormalizedHand) -> Landmark3D:
    """Mean of wrist and four MCP joints, in the input coordinate system."""
    points = [
        landmark(hand, i)
        for i in (
            LandmarkIndex.WRIST,
            LandmarkIndex.INDEX_MCP,
            LandmarkIndex.MIDDLE_MCP,
            LandmarkIndex.RING_MCP,
            LandmarkIndex.PINKY_MCP,
        )
    ]
    return Landmark3D(
        sum(p.x for p in points) / 5, sum(p.y for p in points) / 5, sum(p.z for p in points) / 5
    )
