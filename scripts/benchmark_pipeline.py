"""Synthetic landmark benchmark: real downstream stages, no camera/inference/display.

Run from an installed development checkout. Frame timestamps advance at 30 Hz;
execution is unpaced. Its FPS describes downstream capacity, never live camera FPS.
"""

import argparse
import json
import math
import platform
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import cv2
import numpy as np

from gesture_control.app import GestureControlApp
from gesture_control.core.config import AppConfig, WorkspaceConfig, load_config
from gesture_control.core.performance import PerformanceMonitor
from gesture_control.core.types import FrameContext, HandFrame, HandLandmarks, Landmark3D
from gesture_control.profiling import instrument_pipeline
from gesture_control.recognition import GeometricGestureRecognizer
from gesture_control.temporal import GestureTemporalProcessor
from gesture_control.ui.calibration import Calibration
from gesture_control.ui.demo import DemoUI
from gesture_control.vision import PalmNormalizer


class ReplayOnlyDetector:
    def detect(self, image: object, context: FrameContext) -> HandFrame:
        raise AssertionError("Synthetic landmarks must bypass detection")


def source_hand(pinched: bool, x: float) -> HandLandmarks:
    points = [(0.0, 0.0, 0.0), (-0.5, -0.2, 0), (-0.8, -0.4, 0), (-1.1, -0.6, 0), (-1.4, -0.8, 0)]
    for index, offset in enumerate((-0.4, 0, 0.35, 0.65)):
        extended = index == 0
        points.extend(
            [
                (offset, -1, 0),
                (offset, -1.4, 0),
                (offset, -1.7 if extended else -1.1, 0 if extended else -0.1),
                (offset, -2 if extended else -0.85, 0 if extended else -0.15),
            ]
        )
    if pinched:
        tip = points[8]
        points[4] = (tip[0] + 0.08, tip[1], tip[2])
    return HandLandmarks(
        "benchmark-hand",
        tuple(Landmark3D(x + a * 0.2, 0.8 + b * 0.2 / 0.75, c * 0.2) for a, b, c in points),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=960)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workspace", nargs=2, type=int, default=(1920, 1080))
    parser.add_argument("--diagnostics", choices=("on", "off"), default="on")
    parser.add_argument("--calibrated", action="store_true")
    parser.add_argument(
        "--frame-only", action="store_true", help="No stage hooks: overhead control"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.frames <= args.warmup:
        parser.error("frames must exceed a nonnegative warmup")
    config = load_config(args.config) if args.config else AppConfig()
    config = replace(config, workspace=WorkspaceConfig(*args.workspace))
    monitor = PerformanceMonitor(warmup_frames=args.warmup)
    recognizer = GeometricGestureRecognizer(PalmNormalizer(), config.recognition)
    temporal = GestureTemporalProcessor(config.temporal, config.recognition, config.smoothing)
    frames = [
        HandFrame(
            FrameContext(i, i / 30, 640, 480),
            (source_hand((i // 180) % 2 == 1, 0.5 + 0.12 * math.sin(i / 30)),),
        )
        for i in range(args.frames)
    ]
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    pointer_times: list[float] = []
    transitions: list[dict[str, object]] = []
    with TemporaryDirectory(prefix="gesture-benchmark-") as directory:
        ui = DemoUI(
            width=config.workspace.width,
            height=config.workspace.height,
            calibration_path=Path(directory) / "unused.json",
        )
        ui.diagnostics = args.diagnostics == "on"
        if args.calibrated:
            ui.mapper.calibration = Calibration(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))
        detector = ReplayOnlyDetector()
        app = GestureControlApp(
            detector=detector,
            recognizer=recognizer,
            temporal=ui.wrap_temporal(temporal),
            mapper=ui.mapper,
            adapter=ui.canvas,
        )
        hooks = (
            nullcontext()
            if args.frame_only
            else instrument_pipeline(
                monitor,
                camera=SimpleNamespace(read=lambda: None),
                detector=detector,
                recognizer=recognizer,
                temporal=temporal,
                ui=ui,
            )
        )
        try:
            with hooks:
                for frame in frames:
                    with monitor.frame():
                        app.process_landmarks(frame)
                        ui.render(image)
                    if ui.output is not None:
                        if ui.output.pointer_events:
                            pointer_times.append(frame.context.timestamp_s)
                        for event in ui.output.gesture_events:
                            if event.kind.value in ("gesture_started", "gesture_ended"):
                                transitions.append(
                                    {
                                        "frame": frame.context.frame_id,
                                        "timestamp_s": frame.context.timestamp_s,
                                        "kind": event.kind.value,
                                        "gesture": event.gesture.value,
                                    }
                                )
        finally:
            ui.close()
    report = monitor.report()
    report["scenario"] = {
        "synthetic_landmarks": True,
        "camera_inference_display": False,
        "timestamp_fps": 30,
        "workspace": args.workspace,
        "diagnostics": args.diagnostics,
        "calibrated": args.calibrated,
        "frame_only": args.frame_only,
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    report["behavior"] = {
        "first_pointer_s": pointer_times[0] if pointer_times else None,
        "max_pointer_gap_s": max(
            (b - a for a, b in zip(pointer_times, pointer_times[1:], strict=False)), default=0
        ),
        "transition_events": transitions,
        "note": "All frames include a visible synthetic hand; pose changes every 6 stream seconds.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{args.output}: {report['effective_fps']:.1f} downstream FPS")


if __name__ == "__main__":
    main()
