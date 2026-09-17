from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from gesture_control.core.performance import PerformanceMonitor
from gesture_control.core.types import FrameContext, Gesture, HandFrame
from gesture_control.profiling import instrument_pipeline
from gesture_control.ui.drawing import DrawingMapper

from .test_pipeline import Pipeline, source_hand


def test_instrumentation_preserves_pipeline_results_and_restores_hooks(tmp_path: Path) -> None:
    plain = Pipeline(tmp_path)
    measured = Pipeline(tmp_path)
    hand = source_hand(Gesture.PINCH)
    expected = plain.step(hand)
    monitor = PerformanceMonitor(warmup_frames=0)
    resize, mapping = cv2.resize, DrawingMapper.map
    with instrument_pipeline(
        monitor,
        camera=SimpleNamespace(read=lambda: None),
        detector=measured.app.detector,
        recognizer=measured.recognizer,
        temporal=measured.temporal,
        ui=measured.ui,
    ):
        with monitor.frame():
            assert measured.step(hand) == expected
            measured.ui.render(np.zeros((480, 640, 3), dtype=np.uint8))
        assert np.array_equal(measured.ui.canvas.pixels, plain.ui.canvas.pixels)
    report: dict[str, Any] = monitor.report()
    for stage in ("normalization", "recognition", "temporal", "mapping", "drawing", "render"):
        assert report["stages"][stage]["active_frames"] == 1
    assert report["stages"]["drawing"]["calls"] == 2
    assert cv2.resize is resize and DrawingMapper.map is mapping
    assert "recognize" not in vars(measured.recognizer)


def test_hooks_restore_after_processing_exception(tmp_path: Path) -> None:
    pipeline = Pipeline(tmp_path)
    resize, mapping = cv2.resize, DrawingMapper.map
    with (
        pytest.raises(RuntimeError, match="failed"),
        instrument_pipeline(
            PerformanceMonitor(),
            camera=SimpleNamespace(read=lambda: None),
            detector=pipeline.app.detector,
            recognizer=pipeline.recognizer,
            temporal=pipeline.temporal,
            ui=pipeline.ui,
        ),
    ):
        raise RuntimeError("failed")
    assert cv2.resize is resize and DrawingMapper.map is mapping
    assert "render" not in vars(pipeline.ui)


def test_profile_uses_camera_snapshot_and_counts_observations(tmp_path: Path) -> None:
    pipeline = Pipeline(tmp_path)
    monitor = PerformanceMonitor(warmup_frames=0)
    camera = SimpleNamespace(settings={"backend": "test", "latest_frame": True}, read=lambda: None)
    # There is deliberately no native camera property API on this sentinel.
    camera._capture = object()
    context = FrameContext(0, 0, 640, 480)
    frames = iter((HandFrame(context), HandFrame(context, (source_hand(Gesture.POINT),))))
    detector = SimpleNamespace(detect=lambda: next(frames))
    poll_key = cv2.pollKey
    with instrument_pipeline(
        monitor,
        camera=camera,
        detector=detector,
        recognizer=pipeline.recognizer,
        temporal=pipeline.temporal,
        ui=pipeline.ui,
    ):
        with monitor.frame():
            detector.detect()
            detector.detect()
        assert cv2.pollKey is not poll_key
    assert cv2.pollKey is poll_key
    assert monitor.metadata["camera"] == camera.settings
    assert monitor.metadata["camera"] is not camera.settings
    assert monitor.metadata["observations_including_warmup"] == {
        "frames": 2,
        "frames_with_hands": 1,
    }


def test_capture_profile_excludes_warmup_and_preserves_skipped_ids(tmp_path: Path) -> None:
    pipeline = Pipeline(tmp_path)
    monitor = PerformanceMonitor(warmup_frames=1)
    frames = [SimpleNamespace(context=FrameContext(i, i / 10, 640, 480)) for i in (0, 5, 7, 10)]
    pending = iter(frames)
    camera = SimpleNamespace(read=lambda: next(pending), stream_time_s=0.0)
    with instrument_pipeline(
        monitor,
        camera=camera,
        detector=pipeline.app.detector,
        recognizer=pipeline.recognizer,
        temporal=pipeline.temporal,
        ui=pipeline.ui,
    ):
        for frame in frames:
            camera.stream_time_s = frame.context.timestamp_s + 0.01
            with monitor.frame():
                assert camera.read() is frame
    report: Any = monitor.metadata["capture_delivery"]
    assert report["retained_deliveries"] == 3
    assert (report["first_frame_id"], report["last_frame_id"]) == (5, 10)
    assert report["skipped_frame_ids"] == 3
    assert report["acquisition_fps"] == pytest.approx(10)
    assert report["host_acquisition_to_delivery_ms"] == pytest.approx(
        {"mean": 10, "p50": 10, "p95": 10}
    )


def test_capture_profile_handles_empty_stream_and_bounds_memory() -> None:
    from gesture_control.profiling import _CaptureProfile

    profile = _CaptureProfile(warmup=0)
    assert profile.report()["acquisition_fps"] is None
    assert profile.report()["skipped_frame_ids"] == 0
    for i in range(9002):
        profile.observe(SimpleNamespace(context=FrameContext(i, i / 30, 640, 480)), object())
    assert profile.report()["retained_deliveries"] == 9000
    assert profile.report()["first_frame_id"] == 2
    assert profile.report()["host_acquisition_to_delivery_ms"] is None
