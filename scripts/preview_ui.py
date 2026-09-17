"""Reproducible UI screenshots using synthetic observations, without a camera/display.

Run: python scripts/preview_ui.py --output docs/images
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from gesture_control.core.events import Command, TemporalOutput
from gesture_control.core.types import (
    FrameContext,
    Gesture,
    GesturePrediction,
    Handedness,
    HandFrame,
    HandLandmarks,
    Landmark3D,
    StableGesture,
)
from gesture_control.ui.calibration import Calibration
from gesture_control.ui.demo import DemoUI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("recordings/ui-preview"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ui = DemoUI(calibration_path=args.output / "unused-calibration.json")
    context = FrameContext(0, 1.0, 640, 480)
    xy = (
        (0.49, 0.84),
        (0.35, 0.70),
        (0.30, 0.56),
        (0.37, 0.43),
        (0.45, 0.34),
        (0.45, 0.52),
        (0.43, 0.36),
        (0.43, 0.25),
        (0.46, 0.32),
        (0.53, 0.50),
        (0.53, 0.31),
        (0.52, 0.19),
        (0.51, 0.12),
        (0.61, 0.53),
        (0.63, 0.36),
        (0.64, 0.25),
        (0.65, 0.19),
        (0.68, 0.58),
        (0.72, 0.47),
        (0.76, 0.38),
        (0.78, 0.33),
    )
    hand = HandLandmarks("demo-hand", tuple(Landmark3D(x, y) for x, y in xy), Handedness.RIGHT)
    frame = HandFrame(context, (hand,))
    predictions = (GesturePrediction(hand.hand_id, Gesture.PINCH, 0.96),)
    output = TemporalOutput(
        context, stable_gestures=(StableGesture(hand.hand_id, Gesture.PINCH, 0.96, 0),)
    )
    ui.observe(frame, predictions, output)
    ui.fps = 52  # Illustrative fixture, not a performance measurement.
    ui.mapper.calibration = Calibration(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))
    ui.status = "Workspace ready"
    for i in range(240):
        t = i / 239
        x = round(100 + t * 440)
        y = round(250 + 65 * math.sin(t * 5 * math.pi) * math.sin(t * math.pi))
        ui.canvas.execute(
            Command(
                context,
                "stroke.begin" if i == 0 else "stroke.point",
                {"x_px": x, "y_px": y},
                hand.hand_id,
            )
        )
    camera = np.full((480, 640, 3), (43, 39, 33), dtype=np.uint8)
    cv2.putText(
        camera,
        "SYNTHETIC PREVIEW / NOT A CAMERA CAPTURE",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (134, 125, 112),
        1,
        cv2.LINE_AA,
    )

    def save(name: str) -> None:
        if not cv2.imwrite(str(args.output / f"ui-{name}.png"), ui.render(camera)):
            raise OSError("Could not save screenshot")

    save("normal")
    ui.handle_key(ord("d"))
    save("debug")
    ui.handle_key(ord("d"))
    ui.canvas.execute(Command(context, "stroke.end", hand_id=hand.hand_id))
    ui.mapper.tool = "eraser"  # Keep this direct adapter fixture consistent with mapping state.
    ui.canvas.execute(Command(context, "tool.select", {"tool": "eraser"}, hand.hand_id))
    for i, x in enumerate(range(290, 352, 4)):
        ui.canvas.execute(
            Command(
                context,
                "stroke.begin" if i == 0 else "stroke.point",
                {"x_px": x, "y_px": 250},
                hand.hand_id,
            )
        )
    save("eraser")
    ui.canvas.execute(Command(context, "stroke.end", hand_id=hand.hand_id))
    ui.handle_key(ord("c"))
    ui.pointer = (0.15, 0.15)
    ui.handle_key(32)
    save("calibration")
    ui.pointer = None
    ui.handle_key(32)
    save("error")
    ui.close()


if __name__ == "__main__":
    main()
