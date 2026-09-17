"""OpenCV rendering over shared observations, independent of detector backends."""

from functools import lru_cache

import cv2
import numpy as np

from gesture_control.core.events import TemporalOutput
from gesture_control.core.types import GesturePrediction, HandFrame, HandLandmarks, LandmarkIndex

from .calibration import CORNERS
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
INK = (239, 237, 232)
MUTED = (154, 148, 138)
ACCENT = (185, 220, 103)
BACKGROUND = (22, 20, 17)
SURFACE = (32, 29, 25)
BORDER = (57, 51, 43)
WARNING = (99, 193, 242)
ERROR = (126, 133, 244)


def label(
    image: Image,
    text: str,
    position: tuple[int, int],
    *,
    color: tuple[int, int, int] = INK,
    scale: float = 0.5,
) -> None:
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_hand(image: Image, hand: HandLandmarks, *, diagnostics: bool = True) -> None:
    height, width = image.shape[:2]
    points = [
        (round(min(2, max(-1, p.x)) * (width - 1)), round(min(2, max(-1, p.y)) * (height - 1)))
        for p in hand.landmarks
    ]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, points[a], points[b], ACCENT, 1, cv2.LINE_AA)
    for index, point in enumerate(points):
        cv2.circle(image, point, 2, INK, -1, cv2.LINE_AA)
        if diagnostics:
            label(image, str(index), (point[0] + 5, point[1] - 5), scale=0.3)
    tip = points[LandmarkIndex.INDEX_TIP]
    cv2.circle(image, tip, 9, BACKGROUND, 3, cv2.LINE_AA)
    cv2.circle(image, tip, 8, ACCENT, 1, cv2.LINE_AA)
    if diagnostics:
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


def fit_label(
    image: Image,
    text: str,
    position: tuple[int, int],
    width: int,
    *,
    color: tuple[int, int, int] = MUTED,
    scale: float = 0.45,
) -> None:
    """Keep feedback inside its panel, including long filesystem errors."""
    while text and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width:
        text = text[:-4] + "..." if len(text) > 4 else ""
    label(image, text, position, color=color, scale=scale)


@lru_cache(maxsize=4)
def _chrome(width: int, height: int, camera_width: int) -> Image:
    """Immutable cached chrome. Dynamic renderers copy once and fill panel regions."""
    total = camera_width + width + 36
    panel: Image = np.full((height + 246, total, 3), BACKGROUND, dtype=np.uint8)
    cv2.rectangle(panel, (12, 68), (camera_width + 12, height + 100), SURFACE, -1)
    cv2.rectangle(panel, (camera_width + 24, 68), (total - 12, height + 100), SURFACE, -1)
    cv2.line(panel, (12, 57), (total - 12, 57), BORDER, 1)
    for x, pane_width in ((12, camera_width), (camera_width + 24, width)):
        cv2.rectangle(panel, (x, 68), (x + pane_width, height + 100), BORDER, 1)
    label(panel, "Gesture Control", (24, 36), scale=0.75)
    label(panel, "01 / CAMERA", (26, 89), color=MUTED, scale=0.4)
    label(panel, "02 / DRAWING CANVAS", (camera_width + 38, 89), color=INK, scale=0.4)
    cv2.line(panel, (24, height + 197), (total - 24, height + 197), BORDER, 1)
    label(
        panel,
        "C  Calibrate    R  Clear    S  Save    D  Debug    ESC  Exit",
        (24, height + 225),
        color=MUTED,
        scale=0.45,
    )
    panel.setflags(write=False)
    return panel


def essential_state(frame: HandFrame | None, output: TemporalOutput | None) -> tuple[str, str]:
    if frame is None or not frame.hands:
        return "NO HAND", "Show your hand to begin"
    state = (
        next((s for s in output.stable_gestures if s.hand_id == frame.hands[0].hand_id), None)
        if output
        else None
    )
    if state is None:
        return "ACQUIRING", "Hold your gesture steady"
    return state.gesture.value.replace("_", " ").upper(), f"Confidence {state.confidence:.0%}"


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
    calibration_step: int | None = None,
    calibrated: bool = False,
    calibration_error: bool = False,
) -> Image:
    """Render at presentation resolution; never mutate camera, canvas, or coordinates."""
    canvas_height, canvas_width = canvas.shape[:2]
    # Presentation has a readable minimum, independent of interaction resolution.
    # Tiny/narrow workspaces are letterboxed rather than shrinking the text chrome.
    width, height = max(640, canvas_width), max(240, canvas_height)
    camera_width = 400
    if calibration_step is not None and not 0 <= calibration_step < len(CORNERS):
        calibration_step = len(CORNERS) - 1
        calibration_error = True
        status = "Calibration interrupted / C to restart"
    panel = _chrome(width, height, camera_width).copy()
    total = panel.shape[1]
    tracking = frame is not None and bool(frame.hands)
    color = ACCENT if tracking else WARNING
    cv2.circle(panel, (total - 265, 29), 4, color, -1, cv2.LINE_AA)
    label(panel, "Tracking" if tracking else "Searching", (total - 251, 35), color=color)
    label(panel, f"{fps:.0f} FPS", (total - 101, 35), color=MUTED)

    # Resize before drawing the skeleton, avoiding a full source-resolution copy.
    scale = min(camera_width / camera.shape[1], height / camera.shape[0])
    rw, rh = max(1, round(camera.shape[1] * scale)), max(1, round(camera.shape[0] * scale))
    ox, oy = 12 + (camera_width - rw) // 2, 100 + (height - rh) // 2
    preview = panel[oy : oy + rh, ox : ox + rw]
    if (rh, rw) == camera.shape[:2]:
        preview[:] = camera
    else:
        cv2.resize(camera, (rw, rh), dst=preview, interpolation=cv2.INTER_AREA)
    if frame is not None:
        for hand in frame.hands:
            draw_hand(preview, hand, diagnostics=diagnostics)
    for i, (x, y) in enumerate(corners if diagnostics or calibration_step is not None else ()):
        point = (round(x * (rw - 1)), round(y * (rh - 1)))
        cv2.circle(preview, point, 6, ACCENT, -1, cv2.LINE_AA)
        label(preview, str(i + 1), (point[0] + 10, point[1] + 4), color=ACCENT, scale=0.4)
    if calibration_step is not None:
        x, y = ((0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85))[calibration_step]
        target = (round(x * rw), round(y * rh))
        cv2.circle(preview, target, 22, BACKGROUND, 5, cv2.LINE_AA)
        cv2.circle(preview, target, 21, WARNING, 2, cv2.LINE_AA)
        cv2.drawMarker(preview, target, WARNING, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        label(panel, "CALIBRATION", (camera_width - 110, 89), color=WARNING, scale=0.4)
    elif diagnostics:
        label(panel, "DEBUG", (camera_width - 62, 89), color=WARNING, scale=0.4)
    canvas_x = camera_width + 24 + (width - canvas_width) // 2
    canvas_y = 100 + (height - canvas_height) // 2
    panel[canvas_y : canvas_y + canvas_height, canvas_x : canvas_x + canvas_width] = canvas
    workspace = workspace_size or (canvas_width, canvas_height)
    label(panel, f"{workspace[0]} x {workspace[1]}", (total - 127, 89), color=MUTED, scale=0.4)

    if diagnostics and calibration_step is None and frame is not None and output is not None:
        lines = diagnostic_lines(frame, predictions, output, fps)[:4]
        card_height = min(rh, 16 + len(lines) * 20)
        card = preview[rh - card_height : rh, : min(rw, 345)]
        card[:] = SURFACE
        for i, line in enumerate(lines):
            fit_label(card, line, (10, 20 + i * 20), card.shape[1] - 20, scale=0.4)

    base = height + 100
    if calibration_step is not None:
        label(
            panel,
            f"{calibration_step + 1}/4  {CORNERS[calibration_step]}",
            (24, base + 34),
            color=WARNING,
            scale=0.65,
        )
        fit_label(
            panel,
            "Aim near target / SPACE to capture",
            (24, base + 64),
            camera_width - 20,
        )
        for i in range(4):
            cv2.circle(
                panel,
                (camera_width + 45 + i * 24, base + 29),
                5,
                ACCENT if i < calibration_step else WARNING if i == calibration_step else BORDER,
                -1,
                cv2.LINE_AA,
            )
        fit_label(
            panel,
            status if calibration_error else "Drawing paused / C to restart",
            (camera_width + 38, base + 64),
            width - 50,
            color=ERROR if calibration_error else MUTED,
        )
    else:
        gesture, confidence = essential_state(frame, output)
        label(panel, gesture, (24, base + 36), color=INK, scale=0.7)
        label(panel, confidence, (24, base + 65), color=MUTED, scale=0.45)
        cv2.circle(
            panel, (camera_width + 42, base + 29), 4, ACCENT if pen_down else MUTED, -1, cv2.LINE_AA
        )
        label(
            panel,
            "Pencil / Drawing" if pen_down else "Pencil / Hover",
            (camera_width + 56, base + 35),
            color=ACCENT if pen_down else INK,
            scale=0.5,
        )
        label(
            panel,
            "Calibrated" if calibrated else "Not calibrated",
            (total - 172, base + 35),
            color=ACCENT if calibrated else WARNING,
            scale=0.45,
        )
        fit_label(
            panel,
            status,
            (camera_width + 38, base + 65),
            width - 50,
            color=ERROR if calibration_error else MUTED,
        )
    return panel
