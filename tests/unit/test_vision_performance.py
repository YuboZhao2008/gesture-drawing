"""Ownership, freshness and shutdown under a deliberately stalled consumer."""

from queue import Queue
from threading import Condition, Thread, get_ident
from typing import Any
from unittest.mock import MagicMock

import pytest

from gesture_control.vision import OpenCVCamera
from gesture_control.vision.benchmark import summarize_ms


class ControlledCapture:
    def __init__(self) -> None:
        self.frames: Queue[tuple[bool, Any]] = Queue()
        self.condition = Condition()
        self.reads = 0
        self.releases = 0
        self.native_threads = [get_ident()]

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: float) -> bool:
        self.native_threads.append(get_ident())
        return False  # unsupported properties must not prevent capture

    def get(self, prop: int) -> float:
        return 0.0

    def getBackendName(self) -> str:
        return "controlled"

    def read(self) -> tuple[bool, Any]:
        self.native_threads.append(get_ident())
        with self.condition:
            self.reads += 1
            self.condition.notify_all()
        return self.frames.get(timeout=3)

    def wait_for_read(self, count: int) -> None:
        with self.condition:
            assert self.condition.wait_for(lambda: self.reads >= count, timeout=2)

    def release(self) -> None:
        self.native_threads.append(get_ident())
        self.releases += 1


def test_latest_slot_ownership_freshness_and_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    captures: list[ControlledCapture] = []

    def create(index: int) -> ControlledCapture:
        capture = ControlledCapture()
        captures.append(capture)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", create)
    camera = OpenCVCamera(latest_frame=True)
    for _ in range(2):  # reopen starts a fresh stream, with no old slot or error
        camera.open()
        assert camera.open() is camera
        capture = captures[-1]
        try:
            for value in range(4):
                capture.frames.put((True, np.full((12, 16, 3), value, np.uint8)))
            capture.wait_for_read(5)  # first four have been published
            latest = camera.read()
            assert latest.context.frame_id == 3
            assert latest.image[0, 0, 0] == 3
            assert camera.captured_frames == 4
            latest.image[:] = 99  # UI may mutate its delivered frame safely
            capture.frames.put((True, np.full((12, 16, 3), 4, np.uint8)))
            capture.wait_for_read(6)
            next_frame = camera.read()
            assert next_frame.context.frame_id == 4
            assert next_frame.context.timestamp_s >= latest.context.timestamp_s
            assert next_frame.image[0, 0, 0] == 4
            assert latest.image[0, 0, 0] == 99
            assert camera.settings["buffer_size_accepted"] is False
        finally:
            closer = Thread(target=camera.close)
            closer.start()
            with camera._condition:
                assert camera._condition.wait_for(lambda: camera._stopping, timeout=2)
            # Shutdown must not call native release while read is in progress.
            assert capture.releases == 0
            capture.frames.put((True, np.zeros((12, 16, 3), np.uint8)))
            closer.join(timeout=2)
            assert not closer.is_alive()
        camera.close()
        assert capture.releases == 1
        assert len(set(capture.native_threads)) == 1
        assert capture.native_threads[0] != get_ident()
        assert camera._worker is None and camera._latest is None
        with pytest.raises(RuntimeError, match="not open"):
            camera.read()


@pytest.mark.parametrize("during_open", [True, False])
def test_latest_camera_errors_release_and_propagate(
    monkeypatch: pytest.MonkeyPatch, during_open: bool
) -> None:
    cv2 = pytest.importorskip("cv2")
    capture = MagicMock()
    capture.isOpened.return_value = not during_open
    capture.read.return_value = (False, None)
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: capture)
    camera = OpenCVCamera(latest_frame=True)
    with pytest.raises(RuntimeError, match="open camera" if during_open else "read a frame"):
        with camera:
            camera.read()
    camera.close()
    capture.release.assert_called_once()
    assert camera._worker is None


def test_latest_read_waits_for_new_frame_and_shutdown_wakes_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    capture = ControlledCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: capture)
    camera = OpenCVCamera(latest_frame=True).open()
    errors: list[BaseException] = []

    def consume() -> None:
        try:
            camera.read()
        except BaseException as error:
            errors.append(error)

    capture.frames.put((True, np.zeros((12, 16, 3), np.uint8)))
    capture.wait_for_read(2)
    assert camera.read().context.frame_id == 0
    reader = Thread(target=consume)
    reader.start()
    closer = Thread(target=camera.close)
    closer.start()
    reader.join(timeout=2)
    try:
        assert not reader.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    finally:
        capture.frames.put((False, None))
        closer.join(timeout=2)
    assert not closer.is_alive()
    assert capture.releases == 1


def test_latest_cleanup_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    cv2 = pytest.importorskip("cv2")
    capture = ControlledCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: capture)

    def fail_release() -> None:
        raise RuntimeError("release failed")

    monkeypatch.setattr(capture, "release", fail_release)
    camera = OpenCVCamera(latest_frame=True).open()
    errors: list[BaseException] = []

    def close() -> None:
        try:
            camera.close()
        except BaseException as error:
            errors.append(error)

    capture.wait_for_read(1)
    closer = Thread(target=close)
    closer.start()
    with camera._condition:
        assert camera._condition.wait_for(lambda: camera._stopping, timeout=2)
    np = pytest.importorskip("numpy")
    capture.frames.put((True, np.zeros((12, 16, 3), np.uint8)))
    closer.join(timeout=2)
    assert not closer.is_alive()
    assert len(errors) == 1 and str(errors[0]) == "release failed"
    assert camera._worker is None
    camera.close()


def test_timing_summary() -> None:
    result = summarize_ms([0.01, 0.02, 0.03])
    assert result == pytest.approx({"mean": 20, "p50": 20, "p95": 29, "max": 30})


@pytest.mark.parametrize("latest", [False, True])
@pytest.mark.parametrize("during_open", [False, True])
def test_camera_cleanup_preserves_primary_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    latest: bool,
    during_open: bool,
) -> None:
    cv2 = pytest.importorskip("cv2")
    capture = MagicMock()
    capture.isOpened.return_value = True
    primary = ValueError("original camera failure")
    if during_open:
        capture.set.side_effect = primary
    else:
        capture.read.side_effect = primary
    capture.release.side_effect = RuntimeError("native release failure")
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: capture)
    camera = OpenCVCamera(latest_frame=latest)
    with pytest.raises(ValueError) as caught:
        with camera:
            camera.read()
    assert caught.value is primary
    assert caplog.text.count("Camera cleanup failed") == 1
    assert "native release failure" in caplog.text
    capture.release.assert_called_once()
    assert camera._worker is None and camera._capture is None
    camera.close()


def test_camera_context_preserves_processing_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cv2 = pytest.importorskip("cv2")
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.release.side_effect = RuntimeError("native release failure")
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: capture)
    primary = ValueError("processing failed")
    with pytest.raises(ValueError) as caught:
        with OpenCVCamera():
            raise primary
    assert caught.value is primary
    assert "Camera cleanup failed" in caplog.text
    capture.release.assert_called_once()
