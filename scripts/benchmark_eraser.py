"""Deterministic canvas command benchmark; excludes vision, mapping and native UI."""

import argparse
import json
import math
import platform
from pathlib import Path

import cv2
import numpy as np

from gesture_control.core.events import Command
from gesture_control.core.performance import PerformanceMonitor
from gesture_control.core.types import FrameContext
from gesture_control.ui.drawing import DrawingCanvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=("pencil", "eraser"), default="pencil")
    parser.add_argument("--frames", type=int, default=960)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.frames <= args.warmup:
        parser.error("frames must exceed a nonnegative warmup")
    canvas = DrawingCanvas(1920, 1080)
    context = FrameContext(0, 0, 640, 480)
    # Seed actual pencil ink through the public adapter, outside timed work.
    for row in range(120, 961, 30):
        for name, x in (("stroke.begin", 120), ("stroke.point", 1800)):
            canvas.execute(Command(context, name, {"x_px": x, "y_px": row}, "seed"))
        canvas.execute(Command(context, "stroke.end", hand_id="seed"))
    if args.tool == "eraser":
        canvas.execute(Command(context, "tool.select", {"tool": "eraser"}))
    batches = []
    for i in range(args.frames):
        context = FrameContext(i + 1, (i + 1) / 30, 640, 480)
        x = round(960 + 760 * math.sin(i / 47))
        y = round(540 + 350 * math.sin(i / 73))
        commands = [Command(context, "pointer.move", {"x_px": x, "y_px": y}, "bench")]
        commands.append(
            Command(
                context,
                "stroke.begin" if i % 120 == 0 else "stroke.point",
                {"x_px": x, "y_px": y},
                "bench",
            )
        )
        if i % 120 == 119:
            commands.append(Command(context, "stroke.end", hand_id="bench"))
        batches.append(commands)
    monitor = PerformanceMonitor(warmup_frames=args.warmup)
    for commands in batches:
        with monitor.frame():
            with monitor.stage("canvas_commands"):
                for command in commands:
                    canvas.execute(command)
            with monitor.stage("canvas_preview"):
                canvas.render((640, 360))
    canvas.release()
    report = monitor.report()
    report["scenario"] = {
        "tool": args.tool,
        "workspace": [1920, 1080],
        "preview": [640, 360],
        "stroke_frames": 120,
        "synthetic_commands": True,
        "camera_inference_mapping_native_ui": False,
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output}: {report['effective_fps']:.1f} canvas frames/s")


if __name__ == "__main__":
    main()
