from typing import Any

import pytest

from gesture_control.core.performance import PerformanceMonitor


class Clock:
    now = 0

    def __call__(self) -> int:
        return self.now

    def advance(self, milliseconds: int) -> None:
        self.now += milliseconds * 1_000_000


def test_nested_stages_repeated_calls_and_absent_stage_frames() -> None:
    clock = Clock()
    monitor = PerformanceMonitor(warmup_frames=0, clock=clock)
    with monitor.frame():
        with monitor.stage("render"):
            for _ in range(2):
                with monitor.stage("resize"):
                    clock.advance(2)
            clock.advance(6)
    clock.advance(5)  # Scheduler time between frames counts toward wall FPS.
    with monitor.frame():
        clock.advance(5)
    report: dict[str, Any] = monitor.report()
    assert report["effective_fps"] == 100
    assert report["stages"]["frame"]["mean_ms"] == 7.5
    assert report["stages"]["render"]["mean_ms"] == 5
    assert report["stages"]["resize"]["mean_ms"] == 2
    assert report["stages"]["resize"]["calls"] == 2
    assert report["stages"]["resize"]["active_frames"] == 1
    assert report["stages"]["resize"]["p95_ms"] == 4


def test_warmup_bounded_window_and_failure_do_not_contaminate_samples() -> None:
    clock = Clock()
    monitor = PerformanceMonitor(warmup_frames=1, max_samples=2, clock=clock)
    for duration in (100, 10, 20, 40):
        with monitor.frame():
            clock.advance(duration)
    with pytest.raises(RuntimeError, match="device"):
        with monitor.frame(), monitor.stage("capture"):
            clock.advance(50)
            raise RuntimeError("device")
    report: dict[str, Any] = monitor.report()
    assert report["measured_frames"] == 3
    assert report["retained_frames"] == 2
    assert report["failed_frames"] == 1
    assert report["stages"]["frame"]["mean_ms"] == 30
    assert report["frames_over_33_333_ms"] == 1
    assert "capture" not in report["stages"]
    assert report["effective_fps"] == pytest.approx(2 / 0.06)


def test_empty_report_and_invalid_scopes() -> None:
    monitor = PerformanceMonitor()
    assert monitor.report()["effective_fps"] is None
    with pytest.raises(ValueError):
        PerformanceMonitor(max_samples=0)
    with pytest.raises(ValueError):
        PerformanceMonitor(warmup_frames=-1)
    with pytest.raises(RuntimeError, match="overlap"), monitor.frame(), monitor.frame():
        pass
    with pytest.raises(ValueError), monitor.stage("frame"):
        pass
