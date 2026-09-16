"""OpenCV capture. Acquiring a camera is explicit; failed reads are not no-hand frames."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from gesture_control.core.types import FrameContext


@dataclass(frozen=True, slots=True)
class CameraFrame:
    image: Any  # uint8 HxWx3 BGR ndarray; owned by this frame
    context: FrameContext


def validate_image(image: object, context: FrameContext | None = None) -> Any:
    import numpy as np

    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) <= 0
    ):
        raise ValueError("image must be a nonempty uint8 HxWx3 BGR ndarray")
    if context is not None and image.shape[:2] != (context.image_height, context.image_width):
        raise ValueError("image dimensions must match FrameContext")
    return image


class OpenCVCamera:
    """One stream per open(); resolution/FPS are requests, actual size is in context."""

    def __init__(
        self, index: int = 0, *, width: int = 640, height: int = 480, fps: float = 30.0
    ) -> None:
        for name, value, minimum in (
            ("index", index, 0),
            ("width", width, 1),
            ("height", height, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"invalid camera {name}")
        if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self._capture: Any = None
        self._start = 0.0
        self._frame_id = 0

    def open(self) -> OpenCVCamera:
        import cv2

        if self._capture is not None:
            return self
        capture = cv2.VideoCapture(self.index)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"could not open camera {self.index}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            self._start = time.monotonic()
        except BaseException:
            capture.release()
            raise
        self._capture = capture
        self._frame_id = 0
        return self

    def read(self) -> CameraFrame:
        if self._capture is None:
            raise RuntimeError("camera is not open")
        try:
            ok, image = self._capture.read()
            timestamp = time.monotonic() - self._start
            if not ok:
                raise RuntimeError(f"camera {self.index} failed to read a frame")
            image = validate_image(image)
            context = FrameContext(self._frame_id, timestamp, image.shape[1], image.shape[0])
        except BaseException:
            self.release()
            raise
        self._frame_id += 1
        return CameraFrame(image, context)

    def release(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    close = release

    def __enter__(self) -> OpenCVCamera:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
