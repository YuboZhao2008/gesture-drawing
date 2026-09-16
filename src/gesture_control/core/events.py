"""Semantic temporal output and application-neutral command envelopes."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from . import _validation as validate
from .types import FrameContext, Gesture, StableGesture


class GestureEventType(StrEnum):
    GESTURE_STARTED = "gesture_started"
    GESTURE_HELD = "gesture_held"
    GESTURE_ENDED = "gesture_ended"
    PINCH_STARTED = "pinch_started"
    PINCH_ENDED = "pinch_ended"
    HAND_ACQUIRED = "hand_acquired"
    HAND_LOST = "hand_lost"


@dataclass(frozen=True, slots=True)
class GestureEvent:
    context: FrameContext
    hand_id: str
    kind: GestureEventType
    gesture: Gesture
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.context, FrameContext):
            raise ValueError("context must be FrameContext")
        validate.identifier("hand_id", self.hand_id)
        validate.enum_value("kind", self.kind, GestureEventType)
        validate.enum_value("gesture", self.gesture, Gesture)
        validate.real("duration_s", self.duration_s, minimum=0)
        if self.kind in (GestureEventType.PINCH_STARTED, GestureEventType.PINCH_ENDED):
            if self.gesture is not Gesture.PINCH:
                raise ValueError("pinch events must refer to PINCH")
        if self.kind is GestureEventType.HAND_LOST and self.gesture is not Gesture.NONE:
            raise ValueError("HAND_LOST must refer to NONE")


@dataclass(frozen=True, slots=True)
class PointerEvent:
    """Source-image position in [0, 1], top-left origin, y increasing down.

    Temporal tracking clips to image bounds; command mapping applies calibration
    and converts to workspace coordinates. This is not a desktop pixel position.
    """

    context: FrameContext
    hand_id: str
    x: float
    y: float

    def __post_init__(self) -> None:
        if not isinstance(self.context, FrameContext):
            raise ValueError("context must be FrameContext")
        validate.identifier("hand_id", self.hand_id)
        validate.real("x", self.x, minimum=0, maximum=1)
        validate.real("y", self.y, minimum=0, maximum=1)


@dataclass(frozen=True, slots=True)
class TemporalOutput:
    """Atomic output for one frame; gesture events are ordered before pointer events."""

    context: FrameContext
    stable_gestures: tuple[StableGesture, ...] = ()
    gesture_events: tuple[GestureEvent, ...] = ()
    pointer_events: tuple[PointerEvent, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.context, FrameContext):
            raise ValueError("context must be FrameContext")
        for name, expected in (
            ("stable_gestures", StableGesture),
            ("gesture_events", GestureEvent),
            ("pointer_events", PointerEvent),
        ):
            values = tuple(getattr(self, name))
            if not all(isinstance(value, expected) for value in values):
                raise ValueError(f"{name} contains an invalid value")
            object.__setattr__(self, name, values)
        ids = [state.hand_id for state in self.stable_gestures]
        if len(ids) != len(set(ids)):
            raise ValueError("stable gesture hand IDs must be unique")
        if any(state.since_s > self.context.timestamp_s for state in self.stable_gestures):
            raise ValueError("stable gestures cannot start after the current frame")
        events: tuple[GestureEvent | PointerEvent, ...] = (
            *self.gesture_events,
            *self.pointer_events,
        )
        for event in events:
            if event.context != self.context:
                raise ValueError("all events must share the output frame context")


type CommandValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class Command:
    """Extensible name plus flat JSON scalar parameters, copied into a read-only mapping.

    Mappers/adapters agree on command names and parameter schemas. Core deliberately
    does not prescribe drawing tools, mouse buttons, or application actions.
    """

    context: FrameContext
    name: str
    parameters: Mapping[str, CommandValue] = field(default_factory=dict)
    hand_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.context, FrameContext):
            raise ValueError("context must be FrameContext")
        validate.identifier("name", self.name)
        if self.hand_id is not None:
            validate.identifier("hand_id", self.hand_id)
        parameters = dict(self.parameters)
        for key, value in parameters.items():
            validate.identifier("parameter key", key)
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError("command parameters must be JSON scalars")
            if isinstance(value, float):
                validate.real(f"parameter {key}", value)
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
