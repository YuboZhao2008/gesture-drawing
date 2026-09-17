"""Per-track, timestamp-driven gesture dwell and safe pinch lifecycle handling."""

from collections import deque
from dataclasses import dataclass, field
from math import isfinite

from gesture_control.core.config import RecognitionConfig, SmoothingConfig, TemporalConfig
from gesture_control.core.events import GestureEvent, GestureEventType, PointerEvent, TemporalOutput
from gesture_control.core.types import (
    FrameContext,
    Gesture,
    GesturePrediction,
    HandFrame,
    HandLandmarks,
    StableGesture,
)

from .filtering import OneEuroFilter, PointerFilterConfig
from .motion import MotionConfig


@dataclass(frozen=True, slots=True)
class HistoryConfig:
    """Time window for accepted-confidence averaging; cap only bounds memory."""

    window_s: float = 0.3
    max_samples: int = 256

    def __post_init__(self) -> None:
        if (
            isinstance(self.window_s, bool)
            or not isinstance(self.window_s, (int, float))
            or not isfinite(self.window_s)
            or self.window_s <= 0
        ):
            raise ValueError("window_s must be positive and finite")
        if (
            isinstance(self.max_samples, bool)
            or not isinstance(self.max_samples, int)
            or self.max_samples < 1
        ):
            raise ValueError("max_samples must be a positive integer")


@dataclass(slots=True)
class _Track:
    stable: StableGesture
    last_seen: float
    eligible_at: float
    last_held: float
    candidate: Gesture = Gesture.NONE
    candidate_since: float = 0.0
    missing: bool = False
    pinched: bool = False
    pointer: tuple[float, float] | None = None
    pointer_time: float = 0.0
    pointer_filter: OneEuroFilter | None = None
    pointer_evidence_at: float | None = None
    pinch_release_since: float | None = None
    history: deque[tuple[float, Gesture, float]] = field(default_factory=deque)


class GestureTemporalProcessor:
    """Continuous candidate dwell with confidence gating and distance hysteresis.

    Initial acquisition and stroke reacquisition gate starts for the configured
    delay; dwell runs concurrently. Short POINT dropouts preserve cursor continuity.
    History averages confidence only; pinch uses shorter start/release dwell caps.
    Missing observations immediately end pinch and suppress holds/pointers; other
    gestures expire at loss grace. Unknown/low confidence must dwell before ending
    an active gesture. Long inter-observation gaps expire even a returning track.
    """

    def __init__(
        self,
        config: TemporalConfig | None = None,
        recognition: RecognitionConfig | None = None,
        smoothing: SmoothingConfig | None = None,
        history: HistoryConfig | None = None,
        *,
        pointer_filter: PointerFilterConfig | None = None,
        motion: MotionConfig | None = None,
    ) -> None:
        self.config = config or TemporalConfig()
        self.recognition = recognition or RecognitionConfig()
        self.smoothing = smoothing or SmoothingConfig()
        self.history_config = history or HistoryConfig()
        self.pointer_config = pointer_filter or PointerFilterConfig(
            adaptive=self.smoothing.adaptive,
            min_cutoff=self.smoothing.min_cutoff,
            beta=self.smoothing.beta,
            derivative_cutoff=self.smoothing.derivative_cutoff,
        )
        self.motion = motion or MotionConfig()
        self._tracks: dict[str, _Track] = {}
        self._context: FrameContext | None = None

    def reset(self) -> None:
        """Forget stream state; callers must separately release adapter resources."""
        self._tracks.clear()
        self._context = None

    def process(
        self, frame: HandFrame, predictions: tuple[GesturePrediction, ...]
    ) -> TemporalOutput:
        context = frame.context
        self._validate(frame, predictions)
        self._context = context
        now = context.timestamp_s
        observed = {hand.hand_id: hand for hand in frame.hands}
        events: list[GestureEvent] = []
        pointers: list[PointerEvent] = []
        for hand_id, previous in tuple(self._tracks.items()):
            absent = hand_id not in observed
            gap = now - previous.last_seen
            expired = (absent and gap >= self.config.hand_loss_grace_s) or (
                not absent
                and self.config.hand_loss_grace_s > 0
                and gap > self.config.hand_loss_grace_s
            )
            if absent or expired:
                self._missing(previous, context, events)
            if expired:
                self._transition(previous, Gesture.NONE, 0, context, events)
                events.append(
                    GestureEvent(context, hand_id, GestureEventType.HAND_LOST, Gesture.NONE)
                )
                del self._tracks[hand_id]
        by_id = {prediction.hand_id: prediction for prediction in predictions}
        for hand in frame.hands:
            track = self._tracks.get(hand.hand_id)
            if track is None:
                track = _Track(
                    StableGesture(hand.hand_id, Gesture.NONE, 0, now),
                    now,
                    now + self.config.reacquisition_delay_s,
                    now,
                )
                self._tracks[hand.hand_id] = track
                events.append(
                    GestureEvent(
                        context, hand.hand_id, GestureEventType.HAND_ACQUIRED, Gesture.NONE
                    )
                )
            if track.missing:
                short_gap = now - track.last_seen <= self.motion.pointer_dropout_grace_s
                if not short_gap:
                    track.pointer = None
                    track.pointer_filter = None
                # Cursor-only continuity is safe; a released stroke still has to
                # pass full reacquisition protection before it can restart.
                if not (short_gap and track.stable.gesture is Gesture.POINT):
                    track.eligible_at = now + self.config.reacquisition_delay_s
                track.missing = False
            track.last_seen = now
            gesture, confidence = self._evidence(track, hand, by_id[hand.hand_id])
            self._advance(track, gesture, confidence, context, events)
            if self._pointer_ready(track, hand, gesture, by_id[hand.hand_id], now):
                pointers.append(self._pointer(track, hand, context))
        return TemporalOutput(
            context, tuple(t.stable for t in self._tracks.values()), tuple(events), tuple(pointers)
        )

    def _validate(self, frame: HandFrame, predictions: tuple[GesturePrediction, ...]) -> None:
        ids = [prediction.hand_id for prediction in predictions]
        if len(ids) != len(set(ids)) or set(ids) != {h.hand_id for h in frame.hands}:
            raise ValueError("expected exactly one prediction per observed hand")
        if self._context is not None and (
            frame.context.frame_id <= self._context.frame_id
            or frame.context.timestamp_s < self._context.timestamp_s
        ):
            raise ValueError("frame IDs must increase and timestamps cannot decrease; reset first")

    def _missing(self, track: _Track, context: FrameContext, events: list[GestureEvent]) -> None:
        track.missing = True
        track.pinched = False
        track.candidate = Gesture.NONE
        track.candidate_since = context.timestamp_s
        track.history.clear()
        track.pinch_release_since = None
        if track.stable.gesture is Gesture.PINCH:
            self._transition(track, Gesture.NONE, 0, context, events)

    def _evidence(
        self, track: _Track, hand: HandLandmarks, prediction: GesturePrediction
    ) -> tuple[Gesture, float]:
        cfg = self.recognition
        if hand.confidence < cfg.min_confidence:
            track.pinched = False
            return Gesture.UNKNOWN, 0
        distance = prediction.pinch_distance
        if track.pinched and distance is not None and distance < cfg.pinch_exit_threshold:
            # Distance remains useful when the instantaneous non-pinch pose is unknown.
            return Gesture.PINCH, min(
                hand.confidence, max(track.stable.confidence, cfg.min_confidence)
            )
        # Only an accepted pinch may survive an exit spike during release dwell.
        # A rejected candidate must enter again before it can start a stroke.
        if track.stable.gesture is not Gesture.PINCH:
            track.pinched = False
        if prediction.confidence < cfg.min_confidence:
            return Gesture.UNKNOWN, prediction.confidence
        if prediction.gesture is Gesture.PINCH:
            if distance is not None and distance > cfg.pinch_enter_threshold:
                return Gesture.UNKNOWN, 0
            track.pinched = True
        return prediction.gesture, prediction.confidence

    def _pointer_ready(
        self,
        track: _Track,
        hand: HandLandmarks,
        gesture: Gesture,
        prediction: GesturePrediction,
        now: float,
    ) -> bool:
        if now + 1e-12 < track.eligible_at or hand.confidence < self.recognition.min_confidence:
            return False
        stable = track.stable.gesture
        if stable not in (Gesture.POINT, Gesture.PINCH):
            return False
        if gesture == stable:
            track.pointer_evidence_at = now
            return True
        # Only cursor motion tolerates classification noise. Never extend drawing
        # from evidence that may be an opening pinch or an untrusted fingertip.
        return (
            stable is Gesture.POINT
            and track.pointer_evidence_at is not None
            and (
                prediction.confidence >= self.recognition.min_confidence
                or prediction.pinch_distance is not None
            )
            and now - track.pointer_evidence_at <= self.motion.pointer_noise_grace_s
        )

    def _advance(
        self,
        track: _Track,
        gesture: Gesture,
        confidence: float,
        context: FrameContext,
        events: list[GestureEvent],
    ) -> None:
        now = context.timestamp_s
        track.history.append((now, gesture, confidence))
        while track.history and (
            track.history[0][0] < now - self.history_config.window_s
            or len(track.history) > self.history_config.max_samples
        ):
            track.history.popleft()
        if gesture != track.candidate:
            track.candidate = gesture
            track.candidate_since = now
        dwell = self.config.gesture_dwell_s
        if gesture is Gesture.PINCH:
            dwell = min(dwell, self.motion.pinch_start_s)
        elif track.stable.gesture is Gesture.PINCH and gesture is Gesture.POINT:
            dwell = min(dwell, self.motion.pinch_release_s)
        ready = now - track.candidate_since + 1e-12 >= dwell
        # Release evidence may alternate UNKNOWN/POINT/FIST: its timer must not
        # restart with each label, otherwise a genuinely opened pinch can stick.
        if track.stable.gesture is Gesture.PINCH and gesture is not Gesture.PINCH:
            if track.pinch_release_since is None:
                track.pinch_release_since = now
            if now - track.pinch_release_since + 1e-12 >= min(
                self.config.gesture_dwell_s, self.motion.pinch_release_s
            ) and not (ready and gesture is Gesture.POINT):
                self._transition(track, Gesture.UNKNOWN, 0, context, events)
        else:
            track.pinch_release_since = None
        if ready and now + 1e-12 >= track.eligible_at:
            samples = [
                c for t, g, c in track.history if g == gesture and t >= track.candidate_since
            ]
            self._transition(track, gesture, sum(samples) / len(samples), context, events)
        if (
            track.stable.gesture not in (Gesture.NONE, Gesture.UNKNOWN)
            and gesture == track.stable.gesture
            and now + 1e-12 >= track.eligible_at
            and now - track.last_held + 1e-12 >= self.config.held_interval_s
        ):
            events.append(
                GestureEvent(
                    context,
                    track.stable.hand_id,
                    GestureEventType.GESTURE_HELD,
                    track.stable.gesture,
                    now - track.stable.since_s,
                )
            )
            track.last_held = now

    def _transition(
        self,
        track: _Track,
        gesture: Gesture,
        confidence: float,
        context: FrameContext,
        events: list[GestureEvent],
    ) -> None:
        old = track.stable
        if old.gesture == gesture:
            track.stable = StableGesture(old.hand_id, gesture, confidence, old.since_s)
            return
        now = context.timestamp_s
        if old.gesture not in (Gesture.NONE, Gesture.UNKNOWN):
            events.append(
                GestureEvent(
                    context,
                    old.hand_id,
                    GestureEventType.GESTURE_ENDED,
                    old.gesture,
                    now - old.since_s,
                )
            )
            if old.gesture is Gesture.PINCH:
                events.append(
                    GestureEvent(
                        context,
                        old.hand_id,
                        GestureEventType.PINCH_ENDED,
                        old.gesture,
                        now - old.since_s,
                    )
                )
        track.stable = StableGesture(old.hand_id, gesture, confidence, now)
        if gesture is not Gesture.PINCH:
            track.pinched = False
        track.last_held = now
        if gesture not in (Gesture.NONE, Gesture.UNKNOWN):
            events.append(
                GestureEvent(context, old.hand_id, GestureEventType.GESTURE_STARTED, gesture)
            )
            if gesture is Gesture.PINCH:
                events.append(
                    GestureEvent(context, old.hand_id, GestureEventType.PINCH_STARTED, gesture)
                )

    def _pointer(self, track: _Track, hand: HandLandmarks, context: FrameContext) -> PointerEvent:
        tip = hand.landmarks[8]
        x, y = max(0.0, min(1.0, tip.x)), max(0.0, min(1.0, tip.y))
        if self.smoothing.enabled and self.pointer_config.adaptive:
            if track.pointer_filter is None:
                track.pointer_filter = OneEuroFilter(self.pointer_config)
            x, y = track.pointer_filter.filter(x, y, context.timestamp_s)
        elif track.pointer is not None and self.smoothing.enabled:
            # Core alpha at a 30 Hz reference rate, preserving the time constant.
            alpha = 1 - (1 - self.smoothing.alpha) ** (
                30 * (context.timestamp_s - track.pointer_time)
            )
            x = track.pointer[0] + alpha * (x - track.pointer[0])
            y = track.pointer[1] + alpha * (y - track.pointer[1])
        track.pointer = (x, y)
        track.pointer_time = context.timestamp_s
        return PointerEvent(context, hand.hand_id, x, y)
