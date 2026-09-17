"""Thin UI observer for GestureControlApp; caller owns capture and scheduling."""

from __future__ import annotations

from collections import deque
from enum import StrEnum
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from gesture_control.core.events import TemporalOutput
from gesture_control.core.protocols import TemporalProcessor
from gesture_control.core.types import GesturePrediction, HandFrame

from .calibration import Calibration, CalibrationSession, Point
from .drawing import DrawingCanvas, DrawingMapper, Image
from .overlay import render_layout


class KeyAction(StrEnum):
    QUIT = "quit"
    CALIBRATE = "calibrate"
    CAPTURE = "capture"
    DIAGNOSTICS = "diagnostics"
    CLEAR = "clear"
    SAVE = "save"


def key_action(key: int) -> KeyAction | None:
    if key < 0:
        return None
    return {
        27: KeyAction.QUIT,
        ord("c"): KeyAction.CALIBRATE,
        32: KeyAction.CAPTURE,
        ord("d"): KeyAction.DIAGNOSTICS,
        ord("r"): KeyAction.CLEAR,
        ord("s"): KeyAction.SAVE,
    }.get(ord(chr(key & 0xFF).lower()))


class DemoUI:
    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 480,
        calibration_path: str | Path = "recordings/calibration.json",
        drawing_path: str | Path = "recordings/drawing.png",
    ) -> None:
        self.canvas = DrawingCanvas(width, height)
        self.mapper = DrawingMapper(width, height)
        self.calibration_path = Path(calibration_path)
        self.drawing_path = Path(drawing_path)
        self.session: CalibrationSession | None = None
        self.diagnostics = False
        self.feedback_error = False
        self.running = True
        self.status = "Uncalibrated: full camera workspace"
        self.frame: HandFrame | None = None
        self.predictions: tuple[GesturePrediction, ...] = ()
        self.output: TemporalOutput | None = None
        self.pointer: Point | None = None
        self.fps = 0.0
        self._frame_periods: deque[float] = deque(maxlen=30)
        self._last_present: float | None = None
        self._window_open = False
        self.window_name = "Gesture Studio"
        if self.calibration_path.exists():
            try:
                self.mapper.calibration = Calibration.load(self.calibration_path)
                self.status = "Calibrated: workspace loaded"
            except (OSError, ValueError) as error:
                self.status = f"Calibration not loaded: {error}"
                self.feedback_error = True

    def observe(
        self, frame: HandFrame, predictions: tuple[GesturePrediction, ...], output: TemporalOutput
    ) -> None:
        if output.context != frame.context:
            raise ValueError("UI observations must share a frame context")
        self.frame, self.predictions, self.output = frame, predictions, output
        visible = {hand.hand_id for hand in frame.hands}
        self.mapper.visible_ids = visible
        if self.mapper.hand_id is not None and self.mapper.hand_id not in visible:
            self.canvas.release()
        pointer = next((p for p in output.pointer_events if p.hand_id in visible), None)
        self.pointer = (pointer.x, pointer.y) if pointer else None
        if pointer is None:
            self.canvas.cursor = None
        if not visible:
            self.canvas.release()

    def wrap_temporal(self, processor: TemporalProcessor) -> ObservedTemporal:
        return ObservedTemporal(processor, self)

    def handle_key(self, key: int) -> None:
        action = key_action(key)
        if action is KeyAction.QUIT:
            self.running = False
            self.canvas.release()
            self.mapper.reset()
        elif action is KeyAction.DIAGNOSTICS:
            self.diagnostics = not self.diagnostics
        elif action is KeyAction.CLEAR:
            self.canvas.clear()
            self.mapper.reset()
            self.status = "Canvas cleared / Point to move, pinch to draw"
            self.feedback_error = False
        elif action is KeyAction.CALIBRATE:
            self.canvas.release()
            self.mapper.enabled = False
            self.mapper.reset()
            self.session = CalibrationSession()
            self.status = self.session.prompt
            self.feedback_error = False
        elif action is KeyAction.CAPTURE and self.session is not None:
            if self.pointer is None:
                self.status = "No pointer. Show a hand, then SPACE to capture"
                self.feedback_error = True
                return
            try:
                calibration = self.session.capture(self.pointer)
                self.feedback_error = False
                if calibration is not None:
                    calibration.save(self.calibration_path)
                    self.mapper.calibration = calibration
                    self.mapper.enabled = True
                    self.session = None
                    self.status = "Calibrated: pinch to draw"
                else:
                    self.status = self.session.prompt
            except OSError:
                self.status = "Save failed / SPACE to retry / C to restart"
                self.feedback_error = True
            except ValueError:
                self.status = "Corner rejected / Move, SPACE to retry / C to restart"
                self.feedback_error = True
        elif action is KeyAction.SAVE:
            try:
                self.drawing_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(self.drawing_path), self.canvas.pixels):
                    raise OSError("Image encoder failed")
                self.status = f"Drawing saved: {self.drawing_path.name}"
                self.feedback_error = False
            except (OSError, cv2.error) as error:
                self.status = f"Save failed: {error}"
                self.feedback_error = True

    def render(self, image: Image) -> Image:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Preview requires a uint8 BGR image")
        if self.frame is not None and image.shape[:2] != (
            self.frame.context.image_height,
            self.frame.context.image_width,
        ):
            raise ValueError("Preview dimensions must match FrameContext")
        corners = (
            tuple(self.session.points)
            if self.session is not None
            else (self.mapper.calibration.corners if self.mapper.calibration is not None else ())
        )
        # Display size never changes canvas pixels or calibration coordinates.
        scale = min(640 / self.canvas.width, 480 / self.canvas.height)
        canvas = self.canvas.render(
            (max(1, round(self.canvas.width * scale)), max(1, round(self.canvas.height * scale)))
        )
        return render_layout(
            image,
            canvas,
            self.frame,
            self.predictions,
            self.output,
            fps=self.fps,
            diagnostics=self.diagnostics,
            status=self.status,
            pen_down=self.canvas.pen_down,
            corners=corners,
            workspace_size=(self.canvas.width, self.canvas.height),
            calibration_step=len(self.session.points) if self.session is not None else None,
            calibrated=self.mapper.calibration is not None,
            calibration_error=self.feedback_error,
        )

    def present(self, image: Image) -> bool:
        """Display one frame, poll keys once, return False on ESC/window close."""
        if not self.running:
            return False
        now = perf_counter()
        if self._last_present is not None and now > self._last_present:
            # Average elapsed periods, not reciprocal FPS: bursty delivery must
            # not inflate the badge above the actual display throughput.
            self._frame_periods.append(now - self._last_present)
            self.fps = len(self._frame_periods) / sum(self._frame_periods)
        self._last_present = now
        rendered = self.render(image)
        if not self._window_open:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_open = True
            # HighGUI can restore a collapsed native window on Windows. Set a
            # useful initial client size once, preserving subsequent user resizing.
            cv2.resizeWindow(self.window_name, rendered.shape[1], rendered.shape[0])
        cv2.imshow(self.window_name, rendered)
        poll_key = getattr(cv2, "pollKey", None)
        self.handle_key(poll_key() if callable(poll_key) else cv2.waitKey(1))
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            visible = False  # Some backends remove the native window immediately.
        if not visible or not self.running:
            self.close()
        return self.running

    def close(self) -> None:
        self.running = False
        self.canvas.release()
        self.mapper.reset()
        if self._window_open:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
            self._window_open = False


class ObservedTemporal:
    """Protocol-preserving tap; no duplicate detection/classification/frame processing."""

    def __init__(self, processor: TemporalProcessor, ui: DemoUI) -> None:
        self.processor, self.ui = processor, ui

    def process(
        self, frame: HandFrame, predictions: tuple[GesturePrediction, ...]
    ) -> TemporalOutput:
        output = self.processor.process(frame, predictions)
        self.ui.observe(frame, predictions, output)
        return output

    def reset(self) -> None:
        self.processor.reset()
        self.ui.canvas.release()
        self.ui.canvas.cursor = None
        self.ui.mapper.reset()
        self.ui.pointer = None
        self.ui.frame = None
        self.ui.output = None
        self.ui.predictions = ()
