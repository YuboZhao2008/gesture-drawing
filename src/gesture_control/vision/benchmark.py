"""Reproducible webcam timings without windows or recording any camera images."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from .camera import OpenCVCamera
from .debug import draw_hand_frame
from .detector import MediaPipeHandDetector


def summarize_ms(values: list[float]) -> dict[str, float]:
    """Mean and linearly interpolated percentiles; inputs are seconds."""
    ordered = sorted(value * 1000 for value in values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "mean": statistics.mean(ordered),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def benchmark(
    model: str | Path,
    *,
    index: int = 0,
    width: int = 640,
    height: int = 480,
    fps: float = 30,
    frames: int = 300,
    warmup: int = 30,
    latest_frame: bool = True,
    draw: bool = False,
    processing_delay_ms: float = 0,
) -> dict[str, object]:
    """Separate acquisition rate, achieved detection rate and inference capacity.

    Vision time runs from read request through detection. Acquisition-to-result
    age includes time in our single slot but excludes sensor/driver buffering;
    OpenCV does not supply portable exposure timestamps. No optical latency or
    accuracy claim can be inferred from these timings or the hand frame count.
    """
    if frames < 1 or warmup < 0:
        raise ValueError("frames must be positive and warmup nonnegative")
    if not math.isfinite(processing_delay_ms) or processing_delay_ms < 0:
        raise ValueError("processing_delay_ms must be finite and nonnegative")
    import cv2
    import mediapipe as mp  # type: ignore[import-untyped]

    reads: list[float] = []
    detects: list[float] = []
    vision: list[float] = []
    ages: list[float] = []
    drawing: list[float] = []
    loops: list[float] = []
    hand_frames = 0
    with (
        MediaPipeHandDetector(model) as detector,
        OpenCVCamera(
            index, width=width, height=height, fps=fps, latest_frame=latest_frame
        ) as camera,
    ):
        for _ in range(warmup):
            captured = camera.read()
            detector.detect(captured.image, captured.context)
        acquired_start = camera.captured_frames
        started = time.perf_counter()
        for _ in range(frames):
            begin = time.perf_counter()
            captured = camera.read()
            acquired = time.perf_counter()
            frame = detector.detect(captured.image, captured.context)
            detected = time.perf_counter()
            ages.append(camera.stream_time_s - captured.context.timestamp_s)
            if draw:
                draw_hand_frame(captured.image, frame)
            drawn = time.perf_counter()
            if processing_delay_ms:
                time.sleep(processing_delay_ms / 1000)
            loops.append(time.perf_counter() - begin)
            reads.append(acquired - begin)
            detects.append(detected - acquired)
            vision.append(detected - begin)
            drawing.append(drawn - detected)
            hand_frames += bool(frame.hands)
        elapsed = time.perf_counter() - started
        acquisitions = camera.captured_frames - acquired_start
        settings = dict(camera.settings)
        actual_size = [captured.context.image_width, captured.context.image_height]
    return {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "mediapipe": mp.__version__,
        "requested": {"width": width, "height": height, "fps": fps},
        "camera": settings,
        "actual_image_size": actual_size,
        "frames": frames,
        "warmup": warmup,
        "hand_frames": hand_frames,
        "draw": draw,
        "processing_delay_ms": processing_delay_ms,
        "capture_fps": acquisitions / elapsed,
        "detector_fps": frames / elapsed,
        "detector_capacity_fps": frames / sum(detects),
        "capture_wait_ms": summarize_ms(reads),
        "detector_latency_ms": summarize_ms(detects),
        "vision_frame_ms": summarize_ms(vision),
        "acquisition_to_result_ms": summarize_ms(ages),
        "drawing_ms": summarize_ms(drawing),
        "loop_ms": summarize_ms(loops),
        "latency_scope": "Host read completion to result; sensor/driver latency is unmeasured",
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/hand_landmarker.task"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--synchronous", action="store_true")
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--processing-delay-ms", type=float, default=0)
    parser.add_argument("--output", type=Path, help="Save JSON measurements, never camera images")
    args = parser.parse_args(argv)
    if args.frames < 1 or args.warmup < 0:
        parser.error("--frames must be positive and --warmup nonnegative")
    result = benchmark(
        args.model,
        index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        frames=args.frames,
        warmup=args.warmup,
        latest_frame=not args.synchronous,
        draw=args.draw,
        processing_delay_ms=args.processing_delay_ms,
    )
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
