"""Bounded, single-thread timing collection; no image data or per-frame IO."""

from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from math import ceil
from time import perf_counter_ns


class PerformanceMonitor:
    """Aggregate calls per frame; nested stage timings are inclusive, not additive.

    Percentiles use nearest rank over the retained window, including zero for stages
    absent from a frame. Startup/warmup and failed frames are excluded. Wall FPS
    includes time between retained frames; frame_ms measures the frame body only.
    """

    def __init__(
        self,
        *,
        warmup_frames: int = 30,
        max_samples: int = 9000,
        clock: Callable[[], int] = perf_counter_ns,
    ) -> None:
        if type(warmup_frames) is not int or warmup_frames < 0:
            raise ValueError("warmup_frames must be a nonnegative integer")
        if type(max_samples) is not int or max_samples < 1:
            raise ValueError("max_samples must be a positive integer")
        self.warmup_frames = warmup_frames
        self._clock = clock
        self._samples: deque[tuple[int, int, dict[str, int], dict[str, int]]] = deque(
            maxlen=max_samples
        )
        self._pending: dict[str, int] | None = None
        self._calls: dict[str, int] = {}
        self._attempts = 0
        self.measured_frames = 0
        self.failed_frames = 0
        self.metadata: dict[str, object] = {}

    @contextmanager
    def frame(self) -> Iterator[None]:
        if self._pending is not None:
            raise RuntimeError("frame scopes must not overlap")
        self._attempts += 1
        self._pending, self._calls = {}, {}
        start = self._clock()
        try:
            yield
        except BaseException:
            self.failed_frames += 1
            raise
        else:
            end = self._clock()
            if self._attempts > self.warmup_frames:
                self._pending["frame"] = end - start
                self._calls["frame"] = 1
                self._samples.append((start, end, self._pending, self._calls))
                self.measured_frames += 1
        finally:
            self._pending = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not name or name == "frame":
            raise ValueError("stage needs a nonempty name other than frame")
        start = self._clock()
        try:
            yield
        finally:
            elapsed = self._clock() - start
            if self._pending is not None:
                self._pending[name] = self._pending.get(name, 0) + elapsed
                self._calls[name] = self._calls.get(name, 0) + 1

    def report(self) -> dict[str, object]:
        samples = tuple(self._samples)
        stages: dict[str, object] = {}
        names = sorted({name for _, _, timings, _ in samples for name in timings})
        for name in names:
            values = sorted(timings.get(name, 0) / 1_000_000 for _, _, timings, _ in samples)
            stages[name] = {
                "mean_ms": sum(values) / len(values),
                "p50_ms": values[ceil(len(values) * 0.50) - 1],
                "p95_ms": values[ceil(len(values) * 0.95) - 1],
                "p99_ms": values[ceil(len(values) * 0.99) - 1],
                "max_ms": values[-1],
                "active_frames": sum(name in timings for _, _, timings, _ in samples),
                "calls": sum(calls.get(name, 0) for _, _, _, calls in samples),
            }
        elapsed = (samples[-1][1] - samples[0][0]) / 1_000_000_000 if samples else 0
        return {
            "schema_version": 1,
            "warmup_frames": self.warmup_frames,
            "measured_frames": self.measured_frames,
            "retained_frames": len(samples),
            "failed_frames": self.failed_frames,
            "effective_fps": len(samples) / elapsed if elapsed > 0 else None,
            "window_elapsed_s": elapsed,
            "frames_over_33_333_ms": sum(t["frame"] > 1_000_000_000 / 30 for _, _, t, _ in samples),
            "stages": stages,
            "metadata": dict(self.metadata),
        }
