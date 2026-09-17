"""Small prototype CommandMapper/OutputAdapter pair; all coordinates are explicit."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from gesture_control.core.events import Command, GestureEventType, TemporalOutput
from gesture_control.core.types import FrameContext, Gesture, GesturePrediction, HandFrame

from .calibration import Calibration

type Image = NDArray[np.uint8]


def canvas_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    if width < 1 or height < 1 or not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Finite coordinates and positive dimensions required")
    return round(min(1, max(0, x)) * (width - 1)), round(min(1, max(0, y)) * (height - 1))


@dataclass(slots=True)
class DrawingMapper:
    """PINCH operates the selected tool; a deliberate TWO_FINGER dwell toggles it.

    Tool selection uses current observations, never a retained stable gesture.
    The observer must call observe_evidence before map for gesture tool selection.
    Ordinary stroke mapping still subscribes only to PINCH companion events.
    """

    width: int = 640
    height: int = 480
    calibration: Calibration | None = None
    enabled: bool = True
    hand_id: str | None = None
    visible_ids: set[str] | None = None
    stroke_active: bool = False
    tool: str = field(default="pencil", init=False)
    tool_switch_progress: float = field(default=0.0, init=False)
    tool_switch_dwell_s: float = 0.65
    tool_switch_cooldown_s: float = 0.8
    tool_switch_release_s: float = 0.18
    tool_switch_min_confidence: float = 0.8
    _evidence_context: FrameContext | None = field(default=None, init=False, repr=False)
    _tool_owner: str | None = field(default=None, init=False, repr=False)
    _tool_since: float | None = field(default=None, init=False, repr=False)
    _release_since: float | None = field(default=None, init=False, repr=False)
    _cooldown_until: float = field(default=0.0, init=False, repr=False)
    _tool_armed: bool = field(default=True, init=False, repr=False)
    _pending_switch: str | None = field(default=None, init=False, repr=False)
    _tool_gestures: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("Canvas dimensions must be positive")
        for name in ("tool_switch_dwell_s", "tool_switch_cooldown_s", "tool_switch_release_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if (
            isinstance(self.tool_switch_min_confidence, bool)
            or not isinstance(self.tool_switch_min_confidence, (int, float))
            or not math.isfinite(self.tool_switch_min_confidence)
            or not 0 < self.tool_switch_min_confidence <= 1
        ):
            raise ValueError("tool_switch_min_confidence must be in (0, 1]")

    @property
    def tool_switch_cooldown_remaining_s(self) -> float:
        now = self._evidence_context.timestamp_s if self._evidence_context else 0.0
        return max(0.0, self._cooldown_until - now)

    def reset(self) -> None:
        """Cancel all interaction state, preserving the selected tool.

        Caller must also release its adapter. A fresh PINCH start is required.
        """
        self.hand_id = None
        self.stroke_active = False
        self._evidence_context = None
        self._tool_owner = None
        self._tool_since = None
        self._release_since = None
        self._cooldown_until = 0.0
        self._tool_armed = True
        self._pending_switch = None
        self._tool_gestures.clear()
        self.tool_switch_progress = 0.0

    def observe_evidence(
        self, frame: HandFrame, predictions: tuple[GesturePrediction, ...]
    ) -> None:
        """Record one frame's raw evidence; no adapter side effects occur here."""
        previous = self._evidence_context
        if previous is not None and (
            frame.context.frame_id <= previous.frame_id
            or frame.context.timestamp_s < previous.timestamp_s
        ):
            raise ValueError("Tool evidence must advance monotonically; reset first")
        by_id = {p.hand_id: p for p in predictions}
        if len(by_id) != len(predictions) or set(by_id) != {h.hand_id for h in frame.hands}:
            raise ValueError("expected exactly one prediction per observed hand")
        self._evidence_context = frame.context
        self.visible_ids = set(by_id)
        self._pending_switch = None
        self._tool_gestures = {p.hand_id for p in predictions if p.gesture is Gesture.TWO_FINGER}
        self.tool_switch_progress = 0.0
        now = frame.context.timestamp_s
        # A dropped/stalled frame cannot count as continuous deliberate evidence.
        if previous is not None and now - previous.timestamp_s > 0.15:
            self._tool_since = None
            self._release_since = None
        owner = self.hand_id or self._tool_owner
        hand = next((h for h in frame.hands if h.hand_id == owner), None)
        if hand is None:
            hand = frame.hands[0] if frame.hands else None
        if not self.enabled or hand is None:
            self._tool_since = None
            self._release_since = None
            self._tool_owner = None
            return
        if hand.hand_id != self._tool_owner:
            self._tool_since = None
            self._release_since = None
            self._tool_owner = hand.hand_id
        prediction = by_id[hand.hand_id]
        trusted = min(hand.confidence, prediction.confidence) >= self.tool_switch_min_confidence
        if not trusted or prediction.gesture is not Gesture.TWO_FINGER:
            self._tool_since = None
            if trusted and prediction.gesture not in (Gesture.UNKNOWN, Gesture.TWO_FINGER):
                if self._release_since is None:
                    self._release_since = now
                if now - self._release_since + 1e-12 >= self.tool_switch_release_s:
                    self._tool_armed = True
            else:
                self._release_since = None
            return
        self._release_since = None
        if not self._tool_armed or now + 1e-12 < self._cooldown_until:
            self._tool_since = None
            return
        if self._tool_since is None:
            self._tool_since = now
        self.tool_switch_progress = min(1.0, (now - self._tool_since) / self.tool_switch_dwell_s)
        if now - self._tool_since + 1e-12 >= self.tool_switch_dwell_s:
            self._pending_switch = hand.hand_id
            self._tool_armed = False
            self._tool_since = None
            self._cooldown_until = now + self.tool_switch_cooldown_s

    def _end(self, context: FrameContext, commands: list[Command]) -> None:
        if self.stroke_active:
            commands.append(Command(context, "stroke.end", hand_id=self.hand_id))
        self.hand_id = None
        self.stroke_active = False

    def map(self, output: TemporalOutput) -> tuple[Command, ...]:
        commands: list[Command] = []
        current_evidence = self._evidence_context == output.context
        tool_gestures = self._tool_gestures if current_evidence else set()
        if not self.enabled:
            self._end(output.context, commands)
            return tuple(commands)
        if self.visible_ids is not None and self.hand_id not in self.visible_ids:
            self._end(output.context, commands)
        if self.hand_id in tool_gestures:
            self._end(output.context, commands)
        if current_evidence and self._pending_switch is not None:
            self._end(output.context, commands)
            self.tool = "eraser" if self.tool == "pencil" else "pencil"
            commands.append(
                Command(output.context, "tool.select", {"tool": self.tool}, self._pending_switch)
            )
            self._pending_switch = None
            # A global tool transition is its own interaction. Do not consume
            # another hand's simultaneous pinch start or a stale pointer batch.
            return tuple(commands)
        lost = {e.hand_id for e in output.gesture_events if e.kind is GestureEventType.HAND_LOST}
        for event in output.gesture_events:
            if event.kind is GestureEventType.PINCH_STARTED:
                if (
                    self.hand_id is None
                    and event.hand_id not in tool_gestures
                    and event.hand_id not in lost
                    and (self.visible_ids is None or event.hand_id in self.visible_ids)
                ):
                    self.hand_id = event.hand_id
            elif event.hand_id == self.hand_id and (
                event.kind in (GestureEventType.PINCH_ENDED, GestureEventType.HAND_LOST)
                or (
                    event.kind is GestureEventType.GESTURE_STARTED
                    and event.gesture is not Gesture.PINCH
                )
            ):
                self._end(output.context, commands)
        # One cursor only; another hand must never contribute to an active stroke.
        pointers = [
            p
            for p in output.pointer_events
            if p.hand_id not in lost
            and p.hand_id not in tool_gestures
            and (self.visible_ids is None or p.hand_id in self.visible_ids)
        ]
        cursor_hand = self.hand_id or (pointers[0].hand_id if pointers else None)
        for pointer in pointers:
            if pointer.hand_id != cursor_hand:
                continue
            x, y = pointer.x, pointer.y
            if self.calibration is not None:
                try:
                    x, y = self.calibration.map(x, y)
                except ValueError:
                    self._end(output.context, commands)
                    continue
            px, py = canvas_pixel(x, y, self.width, self.height)
            parameters = {"x_px": px, "y_px": py}
            commands.append(Command(output.context, "pointer.move", parameters, pointer.hand_id))
            if pointer.hand_id == self.hand_id:
                name = "stroke.point" if self.stroke_active else "stroke.begin"
                commands.append(Command(output.context, name, parameters, pointer.hand_id))
                self.stroke_active = True
        return tuple(commands)


class DrawingCanvas:
    """OutputAdapter using integer canvas pixels (x_px/y_px), with track ownership."""

    def __init__(
        self, width: int = 640, height: int = 480, *, background: Image | None = None
    ) -> None:
        if width < 1 or height < 1:
            raise ValueError("Canvas dimensions must be positive")
        self.width, self.height = width, height
        if background is not None and (
            background.dtype != np.uint8 or background.shape != (height, width, 3)
        ):
            raise ValueError("Canvas background must be a matching uint8 BGR image")
        self._background: Image = (
            np.full((height, width, 3), (245, 247, 250), dtype=np.uint8)
            if background is None
            else background.copy()
        )
        # Ink coverage is separate from the background. Erasing removes coverage,
        # restoring the actual background; exported pixels remain ordinary BGR.
        self._ink: Image = np.zeros((height, width), dtype=np.uint8)
        self.pixels = self._background.copy()
        self.tool = "pencil"
        self.eraser_radius_px = max(4, round(min(width, height) * 0.03))
        self.cursor: tuple[int, int] | None = None
        self.pen_down = False
        self.hand_id: str | None = None
        self._previous: tuple[int, int] | None = None

    def release(self) -> None:
        self.pen_down = False
        self.hand_id = None
        self._previous = None

    def clear(self) -> None:
        self._ink.fill(0)
        self.pixels[:] = self._background
        self.release()

    def _mark(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """Update coverage and composite only the dirty rectangle, never a full frame."""
        erasing = self.tool == "eraser"
        radius = self.eraser_radius_px if erasing else 2
        value = 0 if erasing else 255
        if start == end:
            cv2.circle(self._ink, end, radius if erasing else 1, value, -1, cv2.LINE_AA)
        else:
            cv2.line(self._ink, start, end, value, radius * 2 if erasing else 3, cv2.LINE_AA)
        left = max(0, min(start[0], end[0]) - radius - 2)
        right = min(self.width, max(start[0], end[0]) + radius + 3)
        top = max(0, min(start[1], end[1]) - radius - 2)
        bottom = min(self.height, max(start[1], end[1]) + radius + 3)
        alpha = self._ink[top:bottom, left:right, None].astype(np.uint16)
        background = self._background[top:bottom, left:right].astype(np.uint16)
        ink_color = np.array((160, 72, 32), dtype=np.uint16)
        self.pixels[top:bottom, left:right] = (
            (background * (255 - alpha) + ink_color * alpha + 127) // 255
        ).astype(np.uint8)

    def execute(self, command: Command) -> None:
        if command.name in ("pointer.move", "stroke.begin", "stroke.point"):
            x, y = command.parameters.get("x_px"), command.parameters.get("y_px")
            if type(x) is not int or type(y) is not int:
                raise ValueError("Drawing commands require integer x_px/y_px canvas pixels")
            if command.hand_id is None:
                raise ValueError("Pointer/stroke commands require hand_id")
            point = (min(self.width - 1, max(0, x)), min(self.height - 1, max(0, y)))
            if self.pen_down and command.hand_id != self.hand_id:
                return
            if command.name == "pointer.move":
                self.cursor = point
            elif command.name == "stroke.begin":
                if not self.pen_down:
                    self.pen_down = True
                    self.hand_id = command.hand_id
                    self._previous = point
                    self.cursor = point
                    self._mark(point, point)
            elif self.pen_down:
                self._mark(self._previous or point, point)
                self._previous = point
                self.cursor = point
        elif command.name == "stroke.end":
            if command.hand_id is None or command.hand_id == self.hand_id:
                self.release()
        elif command.name == "canvas.clear":
            self.clear()
        elif command.name == "tool.select":
            tool = command.parameters.get("tool")
            if tool not in ("pencil", "eraser"):
                raise ValueError("tool.select requires pencil or eraser")
            self.release()
            self.tool = str(tool)
        else:
            raise ValueError(f"Unsupported canvas command: {command.name}")

    def render(self, size: tuple[int, int] | None = None) -> Image:
        """Scale only the preview; preserve export pixels and a readable cursor size."""
        width, height = size or (self.width, self.height)
        if width < 1 or height < 1:
            raise ValueError("Preview dimensions must be positive")
        image = (
            self.pixels.copy()
            if (width, height) == (self.width, self.height)
            else cast(Image, cv2.resize(self.pixels, (width, height), interpolation=cv2.INTER_AREA))
        )
        if self.cursor is not None:
            point = (
                round(self.cursor[0] * (width - 1) / max(1, self.width - 1)),
                round(self.cursor[1] * (height - 1) / max(1, self.height - 1)),
            )
            color = (
                (126, 78, 170)
                if self.tool == "eraser"
                else (160, 72, 32)
                if self.pen_down
                else (92, 100, 48)
            )
            if self.tool == "eraser":
                axes = (
                    max(3, round(self.eraser_radius_px * width / self.width)),
                    max(3, round(self.eraser_radius_px * height / self.height)),
                )
                cv2.ellipse(image, point, axes, 0, 0, 360, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.ellipse(image, point, axes, 0, 0, 360, color, 1, cv2.LINE_AA)
                return image
            cv2.circle(image, point, 10, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(image, point, 9, color, 1, cv2.LINE_AA)
            cv2.circle(image, point, 2, color, -1, cv2.LINE_AA)
            if not self.pen_down:
                cv2.drawMarker(image, point, color, cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
        return image
