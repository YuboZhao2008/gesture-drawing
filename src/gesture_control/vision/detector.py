"""Synchronous MediaPipe Tasks VIDEO adapter for one unmirrored camera stream."""

from __future__ import annotations

import math
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import uuid4

from gesture_control.core.types import (
    FrameContext,
    Handedness,
    HandFrame,
    HandLandmarks,
    Landmark3D,
)

from .camera import validate_image


class MediaPipeHandDetector:
    """Initialize once. Owns no camera, application commands, or gesture state.

    Tasks labels are preserved for unmirrored input; mirrored input swaps anatomical
    labels. Coordinates always describe the supplied image. Confidence is the
    handedness score: Tasks does not expose per-result detection/tracking scores.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        input_is_mirrored: bool = False,
        max_wrist_distance: float = 0.25,
        max_gap_s: float = 0.25,
    ) -> None:
        for value in (min_detection_confidence, min_presence_confidence, min_tracking_confidence):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("confidence thresholds must be in [0, 1]")
        for value in (max_wrist_distance, max_gap_s):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("continuity limits must be finite and positive")
        if not isinstance(input_is_mirrored, bool):
            raise ValueError("input_is_mirrored must be boolean")
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Hand Landmarker model not found: {path}; see vision README")
        import mediapipe as mp  # type: ignore[import-untyped]
        from mediapipe.tasks.python import (  # type: ignore[import-untyped]
            BaseOptions,
            vision,
        )

        self._mp: Any = mp
        self._backend: Any = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path.resolve())),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )
        self._mirrored = input_is_mirrored
        self._max_distance, self._max_gap = max_wrist_distance, max_gap_s
        self._previous: HandLandmarks | None = None
        self._context: FrameContext | None = None
        self._timestamp_ms = -1
        self._sequence = 0
        self._prefix = uuid4().hex[:12]

    def detect(self, image: object, context: FrameContext) -> HandFrame:
        import cv2

        if self._backend is None:
            raise RuntimeError("detector is closed")
        bgr = validate_image(image, context)
        previous_context = self._context
        if previous_context is not None and (
            context.frame_id <= previous_context.frame_id
            or context.timestamp_s < previous_context.timestamp_s
        ):
            raise ValueError("frame IDs must increase and timestamps must not decrease")
        # VIDEO requires strictly increasing integer milliseconds, unlike the core.
        timestamp_ms = max(int(context.timestamp_s * 1000), self._timestamp_ms + 1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._backend.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        self._timestamp_ms = timestamp_ms
        self._context = context
        if not result.hand_landmarks:
            self._previous = None
            return HandFrame(context)
        try:
            points = tuple(
                Landmark3D(float(p.x), float(p.y), float(p.z)) for p in result.hand_landmarks[0]
            )
            categories = result.handedness[0] if result.handedness else []
            category = max(categories, key=lambda c: c.score) if categories else None
            label = category.category_name.lower() if category is not None else "unknown"
            handedness = Handedness(label) if label in ("left", "right") else Handedness.UNKNOWN
            if self._mirrored and handedness is not Handedness.UNKNOWN:
                handedness = Handedness.RIGHT if handedness is Handedness.LEFT else Handedness.LEFT
            confidence = float(category.score) if category is not None else 0.0
            # Validate the backend payload before using it for continuity.
            hand = HandLandmarks("pending", points, handedness, confidence)
        except (ValueError, TypeError, AttributeError, IndexError) as error:
            self._previous = None
            raise ValueError("malformed MediaPipe hand result") from error
        prior = self._previous
        continuous = prior is not None and previous_context is not None
        if prior is not None and previous_context is not None:
            a, b = prior.landmarks[0], points[0]
            continuous = (
                context.timestamp_s - previous_context.timestamp_s <= self._max_gap
                and (context.image_width, context.image_height)
                == (previous_context.image_width, previous_context.image_height)
                and math.hypot(a.x - b.x, (a.y - b.y) * context.image_height / context.image_width)
                <= self._max_distance
                and (
                    prior.handedness == handedness
                    or Handedness.UNKNOWN in (prior.handedness, handedness)
                )
            )
        if not continuous:
            self._sequence += 1
        hand_id = (
            prior.hand_id
            if continuous and prior is not None
            else (f"vision-{self._prefix}-{self._sequence}")
        )
        hand = HandLandmarks(hand_id, hand.landmarks, hand.handedness, hand.confidence)
        self._previous = hand
        return HandFrame(context, (hand,))

    def close(self) -> None:
        backend, self._backend = self._backend, None
        self._previous = None
        if backend is not None:
            backend.close()

    release = close

    def __enter__(self) -> MediaPipeHandDetector:
        if self._backend is None:
            raise RuntimeError("detector is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
