"""Immutable data exchanged by vision, recognition, and temporal processing."""

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from . import _validation as validate


class Gesture(StrEnum):
    NONE = "none"
    POINT = "point"
    PINCH = "pinch"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    TWO_FINGER = "two_finger"
    UNKNOWN = "unknown"


class Handedness(StrEnum):
    """Anatomical handedness, independent of preview mirroring."""

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class LandmarkIndex(IntEnum):
    """Canonical 21-point hand topology; backends must convert to this order."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass(frozen=True, slots=True)
class Landmark3D:
    """A finite point; its containing type defines the coordinate space."""

    x: float
    y: float
    z: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            validate.real(name, getattr(self, name))


def _landmarks(points: tuple[Landmark3D, ...]) -> tuple[Landmark3D, ...]:
    frozen = tuple(points)
    if len(frozen) != len(LandmarkIndex):
        raise ValueError("a hand must contain exactly 21 landmarks")
    if not all(isinstance(point, Landmark3D) for point in frozen):
        raise ValueError("landmarks must contain Landmark3D values")
    return frozen


@dataclass(frozen=True, slots=True)
class FrameContext:
    """Identity, stream-relative monotonic seconds, and source image size in pixels."""

    frame_id: int
    timestamp_s: float
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        validate.integer("frame_id", self.frame_id)
        validate.real("timestamp_s", self.timestamp_s, minimum=0)
        validate.integer("image_width", self.image_width, minimum=1)
        validate.integer("image_height", self.image_height, minimum=1)


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    """Image-space points for a tracked hand; see ARCHITECTURE.md for units."""

    hand_id: str
    landmarks: tuple[Landmark3D, ...]
    handedness: Handedness = Handedness.UNKNOWN
    confidence: float = 1.0

    def __post_init__(self) -> None:
        validate.identifier("hand_id", self.hand_id)
        validate.enum_value("handedness", self.handedness, Handedness)
        validate.real("confidence", self.confidence, minimum=0, maximum=1)
        object.__setattr__(self, "landmarks", _landmarks(self.landmarks))


@dataclass(frozen=True, slots=True)
class NormalizedHand:
    """Wrist-centered points in palm units, retaining their source observation.

    Normalization uses image-width units on all axes, then divides by scale.
    The wrist is the origin; scale is the wrist-to-middle-MCP distance.
    Rotation and anatomical handedness are preserved.
    """

    source: HandLandmarks
    landmarks: tuple[Landmark3D, ...]
    scale: float

    def __post_init__(self) -> None:
        if not isinstance(self.source, HandLandmarks):
            raise ValueError("source must be HandLandmarks")
        validate.positive("scale", self.scale)
        object.__setattr__(self, "landmarks", _landmarks(self.landmarks))
        wrist = self.landmarks[LandmarkIndex.WRIST]
        if any(abs(value) > 1e-9 for value in (wrist.x, wrist.y, wrist.z)):
            raise ValueError("normalized wrist must be at the origin")


@dataclass(frozen=True, slots=True)
class HandFrame:
    """One observation; an empty hands tuple is a valid no-hand frame."""

    context: FrameContext
    hands: tuple[HandLandmarks, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.context, FrameContext):
            raise ValueError("context must be FrameContext")
        object.__setattr__(self, "hands", tuple(self.hands))
        if not all(isinstance(hand, HandLandmarks) for hand in self.hands):
            raise ValueError("hands must contain HandLandmarks values")
        ids = [hand.hand_id for hand in self.hands]
        if len(ids) != len(set(ids)):
            raise ValueError("hand IDs must be unique within a frame")


@dataclass(frozen=True, slots=True)
class GesturePrediction:
    """Instantaneous classification, with optional pinch distance in palm units.

    Supplying the distance lets temporal processing apply pinch hysteresis without
    coupling to a particular recognizer or recomputing recognition features.
    """

    hand_id: str
    gesture: Gesture
    confidence: float
    pinch_distance: float | None = None

    def __post_init__(self) -> None:
        validate.identifier("hand_id", self.hand_id)
        validate.enum_value("gesture", self.gesture, Gesture)
        validate.real("confidence", self.confidence, minimum=0, maximum=1)
        if self.pinch_distance is not None:
            validate.real("pinch_distance", self.pinch_distance, minimum=0)
        if self.gesture is Gesture.NONE:
            raise ValueError("use an empty predictions tuple for a no-hand frame")


@dataclass(frozen=True, slots=True)
class StableGesture:
    """Current filtered state for a track; NONE means no active gesture."""

    hand_id: str
    gesture: Gesture
    confidence: float
    since_s: float

    def __post_init__(self) -> None:
        validate.identifier("hand_id", self.hand_id)
        validate.enum_value("gesture", self.gesture, Gesture)
        validate.real("confidence", self.confidence, minimum=0, maximum=1)
        validate.real("since_s", self.since_s, minimum=0)
