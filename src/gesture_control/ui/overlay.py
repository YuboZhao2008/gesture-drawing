"""OpenCV rendering over shared observations, independent of detector backends."""

from typing import cast

import cv2
import numpy as np

from gesture_control.core.events import TemporalOutput
from gesture_control.core.types import GesturePrediction, HandFrame, HandLandmarks, LandmarkIndex

from .drawing import Image

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
INK = (229, 234, 244)
MUTED = (157, 171, 190)
ACCENT = (176, 224, 67)


def label(
    image: Image,
    text: str,
    position: tuple[int, int],
    *,
    color: tuple[int, int, int] = INK,
    scale: float = 0.5,
) -> None:
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_hand(image: Image, hand: HandLandmarks) -> None:
    height, width = image.shape[:2]
    points = [
        (round(min(2, max(-1, p.x)) * (width - 1)), round(min(2, max(-1, p.y)) * (height - 1)))
        for p in hand.landmarks
    ]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, points[a], points[b], ACCENT, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, point, 3, INK, -1, cv2.LINE_AA)
    cv2.circle(image, points[LandmarkIndex.INDEX_TIP], 10, (80, 195, 255), 2, cv2.LINE_AA)
    xs, ys = zip(*points, strict=True)
    cv2.rectangle(image, (min(xs), min(ys)), (max(xs), max(ys)), MUTED, 1)


def diagnostic_lines(
    frame: HandFrame, predictions: tuple[GesturePrediction, ...], output: TemporalOutput, fps: float
) -> tuple[str, ...]:
    lines = [f"{fps:5.1f} FPS  |  {'TRACKING' if frame.hands else 'HAND LOST / SEARCHING'}"]
    for hand in frame.hands:
        raw = next((p for p in predictions if p.hand_id == hand.hand_id), None)
        stable = next((s for s in output.stable_gestures if s.hand_id == hand.hand_id), None)
        lines.append(f"{hand.hand_id[:18]}  /  {hand.handedness.value.upper()}")
        lines.append(f"RAW {raw.gesture.value.upper()}  {raw.confidence:.0%}" if raw else "RAW --")
        lines.append(
            f"STABLE {stable.gesture.value.upper()}  {stable.confidence:.0%}"
            if stable
            else "STABLE --"
        )
    return tuple(lines)


def render_layout(
    camera: Image,
    canvas: Image,
    frame: HandFrame | None,
    predictions: tuple[GesturePrediction, ...],
    output: TemporalOutput | None,
    *,
    fps: float,
    diagnostics: bool,
    status: str,
    pen_down: bool,
    corners: tuple[tuple[float, float], ...] = (),
    workspace_size: tuple[int, int] | None = None,
) -> Image:
    """Return a BGR composition without opening a window; preserve camera aspect ratio."""
    height, width = canvas.shape[:2]
    preview = camera.copy()
    if diagnostics and frame is not None:
        for hand in frame.hands:
            draw_hand(preview, hand)
    scale = min(width / preview.shape[1], height / preview.shape[0])
    resized = cast(
        Image,
        cv2.resize(
            preview,
            (max(1, round(preview.shape[1] * scale)), max(1, round(preview.shape[0] * scale))),
        ),
    )
    left: Image = np.full((height, width, 3), (29, 24, 20), dtype=np.uint8)
    ox, oy = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    for i, (x, y) in enumerate(corners):
        point = (round(x * (resized.shape[1] - 1)), round(y * (resized.shape[0] - 1)))
        cv2.circle(resized, point, 12, ACCENT, 2, cv2.LINE_AA)
        label(resized, str(i + 1), point, color=ACCENT)
    left[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
    panel: Image = np.full((height + 246, width * 2 + 36, 3), (29, 24, 20), dtype=np.uint8)
    label(panel, "GESTURE / STUDIO", (18, 30), color=ACCENT, scale=0.7)
    label(panel, "CAMERA + TRACKING", (18, 62), color=MUTED)
    label(panel, f"PENCIL / {'DRAWING' if pen_down else 'HOVER'}", (width + 30, 62), color=MUTED)
    panel[76 : 76 + height, 12 : 12 + width] = left
    panel[76 : 76 + height, width + 24 : width * 2 + 24] = canvas
    if diagnostics and frame is not None and output is not None:
        for i, line in enumerate(diagnostic_lines(frame, predictions, output, fps)[:5]):
            label(panel, line, (18, height + 102 + i * 23), color=MUTED)
    workspace_width, workspace_height = workspace_size or (width, height)
    label(
        panel,
        f"WORKSPACE {workspace_width} x {workspace_height}",
        (width + 30, height + 102),
        color=ACCENT,
    )
    for i in range(0, min(len(status), 144), 48):
        label(panel, status[i : i + 48], (width + 30, height + 127 + (i // 48) * 22))
    label(
        panel,
        "ESC Quit   C Calibrate   SPACE Capture   D Diagnostics   R Clear   S Save",
        (18, height + 230),
        color=MUTED,
        scale=0.48,
    )
    return panel
