"""Camera, MediaPipe detection and canonical geometry (optional vision extra)."""

from .camera import CameraFrame, OpenCVCamera
from .debug import HAND_CONNECTIONS, draw_hand_frame
from .detector import MediaPipeHandDetector
from .geometry import (
    PalmNormalizer,
    fingertip_pixels,
    hand_bounding_box,
    index_fingertip,
    landmark,
    palm_center,
    pointer_coordinates,
)

__all__ = [
    "CameraFrame",
    "OpenCVCamera",
    "MediaPipeHandDetector",
    "PalmNormalizer",
    "landmark",
    "index_fingertip",
    "pointer_coordinates",
    "fingertip_pixels",
    "hand_bounding_box",
    "palm_center",
    "HAND_CONNECTIONS",
    "draw_hand_frame",
]
