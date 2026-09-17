"""OpenCV capture. Acquiring a camera is explicit; failed reads are not no-hand frames."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition, Thread
from types import TracebackType
from typing import Any

from gesture_control.core.types import FrameContext

LOGGER = logging.getLogger(__name__)


def _cleanup_after_failure(action: Callable[[], None]) -> None:
    """Preserve the active failure while still reporting failed native cleanup."""
    try:
        action()
    except Exception:
        LOGGER.exception("Camera cleanup failed")


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
        self,
        index: int = 0,
        *,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        latest_frame: bool = False,
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
        if not isinstance(latest_frame, bool):
            raise ValueError("latest_frame must be boolean")
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.latest_frame = latest_frame
        self._capture: Any = None
        self._start = 0.0
        self._frame_id = 0
        self.settings: dict[str, object] = {}
        self._condition = Condition()
        self._worker: Thread | None = None
        self._latest: CameraFrame | None = None
        self._error: BaseException | None = None
        self._cleanup_error: BaseException | None = None
        self._stopping = False
        self._ready = False

    @property
    def captured_frames(self) -> int:
        """Successful acquisitions, including frames superseded before consumption."""
        return self._frame_id

    @property
    def stream_time_s(self) -> float:
        """Current monotonic time on the FrameContext clock (not exposure time)."""
        return time.perf_counter() - self._start

    def open(self) -> OpenCVCamera:
        if self._worker is not None or self._capture is not None:
            return self
        if not self.latest_frame:
            self._open_capture()
            return self
        self._latest, self._error = None, None
        self._cleanup_error = None
        self._stopping = self._ready = False
        self._worker = Thread(target=self._capture_latest, name="gesture-camera")
        try:
            self._worker.start()
        except BaseException:
            self._worker = None
            raise
        try:
            with self._condition:
                self._condition.wait_for(lambda: self._ready)
                error = self._error
            if error is not None:
                raise error
        except BaseException:
            _cleanup_after_failure(self.release)
            raise
        return self

    def _open_capture(self) -> None:
        import cv2

        capture = cv2.VideoCapture(self.index)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"could not open camera {self.index}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            # Many backends ignore this; never infer support from get() alone.
            buffer_supported = capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.settings = {
                "backend": capture.getBackendName(),
                "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                "fps": capture.get(cv2.CAP_PROP_FPS),
                "buffer_size_accepted": bool(buffer_supported),
                "buffer_size_readback": capture.get(cv2.CAP_PROP_BUFFERSIZE),
                "latest_frame": self.latest_frame,
            }
            # perf_counter is monotonic and high resolution on Python 3.12
            # Windows, where monotonic() can quantize timestamps to ~16 ms.
            self._start = time.perf_counter()
        except BaseException:
            _cleanup_after_failure(capture.release)
            raise
        self._capture = capture
        self._frame_id = 0

    def _capture_latest(self) -> None:
        # All native operations, including open/release, belong to this thread.
        # Never release VideoCapture concurrently with a native read.
        try:
            self._open_capture()
            with self._condition:
                self._ready = True
                self._condition.notify_all()
            while True:
                with self._condition:
                    if self._stopping:
                        break
                frame = self._read_frame()
                with self._condition:
                    if self._stopping:
                        break
                    self._latest = frame  # one slot; superseded images are discarded
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
                self._latest = None
        finally:
            try:
                self._release_capture()
            except BaseException as error:
                with self._condition:
                    self._cleanup_error = error
                    if self._error is None:
                        self._error = error
                    self._latest = None
            finally:
                with self._condition:
                    self._ready = True
                    self._condition.notify_all()

    def read(self) -> CameraFrame:
        if self.latest_frame:
            if self._worker is None:
                raise RuntimeError("camera is not open")
            with self._condition:
                self._condition.wait_for(
                    lambda: self._latest is not None or self._error is not None or self._stopping
                )
                error = self._error
                frame, self._latest = self._latest, None
            if error is not None:
                _cleanup_after_failure(self.release)
                raise error
            if frame is None:
                raise RuntimeError("camera is not open")
            return frame
        return self._read_frame()

    def _read_frame(self) -> CameraFrame:
        if self._capture is None:
            raise RuntimeError("camera is not open")
        try:
            ok, image = self._capture.read()
            timestamp = time.perf_counter() - self._start
            if not ok:
                raise RuntimeError(f"camera {self.index} failed to read a frame")
            image = validate_image(image)
            context = FrameContext(self._frame_id, timestamp, image.shape[1], image.shape[0])
        except BaseException:
            _cleanup_after_failure(self._release_capture)
            raise
        self._frame_id += 1
        return CameraFrame(image, context)

    def release(self) -> None:
        worker = self._worker
        if worker is not None:
            with self._condition:
                self._stopping = True
                self._latest = None
                self._condition.notify_all()
            # Join before destroying state or reopening. Shutdown waits for the
            # backend's pending read; OpenCV has no portable USB read cancellation.
            worker.join()
            self._worker = None
            error, self._cleanup_error = self._cleanup_error, None
            if error is not None:
                raise error
        else:
            self._release_capture()

    def _release_capture(self) -> None:
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
        if exc is None:
            self.release()
        else:
            _cleanup_after_failure(self.release)
