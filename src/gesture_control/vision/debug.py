"""Small in-place BGR overlays; callers copy the frame if preservation is needed."""

from typing import Any

from gesture_control.core.types import HandFrame

from .camera import validate_image
from .geometry import hand_bounding_box

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


def draw_hand_frame(image: object, frame: HandFrame) -> Any:
    """Draw joints, connections, box and index tip; return the same image."""
    import cv2

    canvas = validate_image(image, frame.context)
    width, height = frame.context.image_width, frame.context.image_height

    # Clamp only rendering coordinates, never observations; also protects OpenCV ints.
    def pixel(x: float, y: float) -> tuple[int, int]:
        return round(min(max(x, 0), 1) * (width - 1)), round(min(max(y, 0), 1) * (height - 1))

    for hand in frame.hands:
        points = [pixel(p.x, p.y) for p in hand.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, points[a], points[b], (80, 200, 80), 2)
        for p in points:
            cv2.circle(canvas, p, 3, (0, 220, 255), -1)
        left, top, right, bottom = hand_bounding_box(hand)
        cv2.rectangle(canvas, pixel(left, top), pixel(right, bottom), (255, 180, 0), 1)
        cv2.circle(canvas, points[8], 7, (255, 0, 255), 2)
    return canvas
