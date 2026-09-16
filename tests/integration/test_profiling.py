from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from gesture_control.core.performance import PerformanceMonitor
from gesture_control.core.types import Gesture
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
