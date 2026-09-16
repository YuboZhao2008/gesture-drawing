"""Small prototype CommandMapper/OutputAdapter pair; all coordinates are explicit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from gesture_control.core.events import Command, GestureEventType, TemporalOutput
from gesture_control.core.types import FrameContext, Gesture

from .calibration import Calibration

type Image = NDArray[np.uint8]


def canvas_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    if width < 1 or height < 1 or not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Finite coordinates and positive dimensions required")
    return round(min(1, max(0, x)) * (width - 1)), round(min(1, max(0, y)) * (height - 1))


@dataclass(slots=True)
class DrawingMapper:
    """Single active hand. Subscribe only to PINCH companion events, not raw labels."""

    width: int = 640
    height: int = 480
    calibration: Calibration | None = None
    enabled: bool = True
    hand_id: str | None = None
    visible_ids: set[str] | None = None
    stroke_active: bool = False

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("Canvas dimensions must be positive")

    def reset(self) -> None:
        """Caller must also release its adapter when resetting the stream/UI."""
        self.hand_id = None
        self.stroke_active = False

    def _end(self, context: FrameContext, commands: list[Command]) -> None:
        if self.stroke_active:
            commands.append(Command(context, "stroke.end", hand_id=self.hand_id))
        self.reset()

    def map(self, output: TemporalOutput) -> tuple[Command, ...]:
        commands: list[Command] = []
        if not self.enabled:
            self._end(output.context, commands)
            return tuple(commands)
        if self.visible_ids is not None and self.hand_id not in self.visible_ids:
            self._end(output.context, commands)
        lost = {e.hand_id for e in output.gesture_events if e.kind is GestureEventType.HAND_LOST}
        for event in output.gesture_events:
            if event.kind is GestureEventType.PINCH_STARTED:
                if (
                    self.hand_id is None
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
            if p.hand_id not in lost and (self.visible_ids is None or p.hand_id in self.visible_ids)
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

    def __init__(self, width: int = 640, height: int = 480) -> None:
        if width < 1 or height < 1:
            raise ValueError("Canvas dimensions must be positive")
        self.width, self.height = width, height
        self.pixels: Image = np.full((height, width, 3), (245, 247, 250), dtype=np.uint8)
        self.cursor: tuple[int, int] | None = None
        self.pen_down = False
        self.hand_id: str | None = None
        self._previous: tuple[int, int] | None = None

    def release(self) -> None:
        self.pen_down = False
        self.hand_id = None
        self._previous = None

    def clear(self) -> None:
        self.pixels[:] = (245, 247, 250)
        self.release()

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
                    cv2.circle(self.pixels, point, 1, (160, 72, 32), -1, cv2.LINE_AA)
            elif self.pen_down:
                cv2.line(self.pixels, self._previous or point, point, (160, 72, 32), 3, cv2.LINE_AA)
                self._previous = point
                self.cursor = point
        elif command.name == "stroke.end":
            if command.hand_id is None or command.hand_id == self.hand_id:
                self.release()
        elif command.name == "canvas.clear":
            self.clear()
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
            color = (160, 72, 32) if self.pen_down else (92, 100, 48)
            cv2.circle(image, point, 10, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(image, point, 9, color, 1, cv2.LINE_AA)
            cv2.circle(image, point, 2, color, -1, cv2.LINE_AA)
            if not self.pen_down:
                cv2.drawMarker(image, point, color, cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
        return image
