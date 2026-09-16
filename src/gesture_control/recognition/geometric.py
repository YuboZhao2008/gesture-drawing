"""Deterministic geometric recognition; persistence belongs to temporal processing."""

from dataclasses import dataclass
from math import isfinite

from gesture_control.core.config import RecognitionConfig
from gesture_control.core.protocols import LandmarkNormalizer
from gesture_control.core.types import Gesture, GesturePrediction, HandFrame, NormalizedHand

from .geometry import finger_extension_score, landmark_distance


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    """Extension thresholds; the gap leaves ambiguous poses unknown."""

    folded_max: float = 0.55
    extended_min: float = 0.75
    confidence_floor: float = 0.75
    confidence_ceiling: float = 0.98

    def __post_init__(self) -> None:
        for value in (
            self.folded_max,
            self.extended_min,
            self.confidence_floor,
            self.confidence_ceiling,
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("geometry thresholds must be numbers")
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError("geometry thresholds must be finite and in [0, 1]")
        if not 0 < self.folded_max < self.extended_min < 1:
            raise ValueError("require 0 < folded_max < extended_min < 1")
        if self.confidence_floor >= self.confidence_ceiling:
            raise ValueError("confidence_floor must be less than confidence_ceiling")


class GeometricGestureRecognizer:
    """Inject the vision-owned canonical normalizer to implement GestureRecognizer.

    ``classify`` also accepts already-normalized observations directly. Pinch has
    priority. Thumb posture is unrestricted for fist/point/two finger; open palm
    requires a reasonably straight thumb as well as four fingers.
    """

    def __init__(
        self,
        normalizer: LandmarkNormalizer,
        config: RecognitionConfig | None = None,
        geometry: GeometryConfig | None = None,
    ) -> None:
        self.normalizer = normalizer
        self.config = config or RecognitionConfig()
        self.geometry = geometry or GeometryConfig()

    def recognize(self, frame: HandFrame) -> tuple[GesturePrediction, ...]:
        predictions = []
        for hand in frame.hands:
            normalized = self.normalizer.normalize(hand, frame.context)
            if normalized is None:
                predictions.append(GesturePrediction(hand.hand_id, Gesture.UNKNOWN, 0))
            else:
                if normalized.source != hand:
                    raise ValueError("normalizer must retain the source observation")
                predictions.append(self.classify(normalized))
        return tuple(predictions)

    def classify(self, hand: NormalizedHand) -> GesturePrediction:
        """Return one instantaneous prediction, retaining normalized pinch distance."""
        points = hand.landmarks
        distance = landmark_distance(points[4], points[8])
        cfg = self.geometry
        gesture = Gesture.UNKNOWN
        quality = 0.0
        chains = [points[start : start + 4] for start in (1, 5, 9, 13, 17)]
        # Collapsed chains are invalid geometry, not convincing folded fingers.
        valid = all(
            landmark_distance(a, b) > 1e-12
            for chain in chains
            for a, b in zip(chain, chain[1:], strict=False)
        )
        if not valid:
            return GesturePrediction(hand.source.hand_id, gesture, 0)
        scores = [finger_extension_score(*chain) for chain in chains]
        if distance <= self.config.pinch_enter_threshold:
            gesture = Gesture.PINCH
            quality = 1 - distance / self.config.pinch_enter_threshold
        else:
            patterns = (
                (Gesture.POINT, (True, False, False, False)),
                (Gesture.TWO_FINGER, (True, True, False, False)),
                (Gesture.FIST, (False, False, False, False)),
                (Gesture.OPEN_PALM, (True, True, True, True)),
            )
            for label, pattern in patterns:
                margins = [
                    (score - cfg.extended_min) / (1 - cfg.extended_min)
                    if extended
                    else (cfg.folded_max - score) / cfg.folded_max
                    for score, extended in zip(scores[1:], pattern, strict=True)
                ]
                if label is Gesture.OPEN_PALM:
                    margins.append((scores[0] - cfg.extended_min) / (1 - cfg.extended_min))
                if min(margins) >= 0:
                    gesture, quality = label, min(margins)
                    break
        confidence = (
            0.0
            if gesture is Gesture.UNKNOWN
            else (cfg.confidence_floor + (cfg.confidence_ceiling - cfg.confidence_floor) * quality)
            * hand.source.confidence
        )
        if confidence < self.config.min_confidence:
            gesture = Gesture.UNKNOWN
        return GesturePrediction(hand.source.hand_id, gesture, confidence, distance)
