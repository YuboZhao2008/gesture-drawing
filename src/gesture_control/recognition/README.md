# Geometric recognition and temporal integration

These implementations use only the standard library. Core contracts are unchanged.
The vision component must supply a `LandmarkNormalizer` implementing the canonical
aspect-corrected, wrist-centered palm-unit normalization in `ARCHITECTURE.md`.

```python
from gesture_control.core.config import AppConfig
from gesture_control.recognition import GeometricGestureRecognizer, GeometryConfig
from gesture_control.temporal import GestureTemporalProcessor, HistoryConfig

config = AppConfig()
# normalizer is supplied by the vision integration agent.
recognizer = GeometricGestureRecognizer(normalizer, config.recognition)
temporal = GestureTemporalProcessor(config.temporal, config.recognition, config.smoothing)
predictions = recognizer.recognize(frame)
output = temporal.process(frame, predictions)
```

Pass the same recognition settings to both stages. Direct normalized inputs can use
`recognizer.classify(normalized_hand)`. Public geometry helpers are
`landmark_distance(a, b)`, `joint_angle(a, joint, b)` (degrees), and
`finger_extension_score(mcp, pip, dip, tip)` ([0, 1]).

## Recognition

Pinch takes priority when thumb/index tip distance is at or below the enter
threshold. Other poses use the minimum of normalized chain reach and PIP/DIP
straightness. POINT requires index extension and three folded fingers; TWO_FINGER
requires index/middle extension and ring/pinky folding. FIST requires the four
primary fingers folded. OPEN_PALM additionally requires a straight thumb. Thumb
posture is otherwise unrestricted. Unmatched, ambiguous, or unusable geometry is
UNKNOWN. Empty frames produce no predictions, never UNKNOWN or NONE predictions.

Confidence interpolates between the configurable floor and ceiling according to
the weakest pose margin (or pinch closeness), then multiplies detector confidence.
Below the shared confidence gate the label becomes UNKNOWN while retaining the
score. Invalid chains or failed normalization produce zero confidence and no
pinch distance. The recognizer is stateless; the temporal stage owns hysteresis.

| Setting | Default |
| --- | --- |
| Existing `RecognitionConfig.min_confidence` | 0.7 |
| Existing pinch enter / exit, palm units | 0.25 / 0.35 |
| New `GeometryConfig.folded_max` / `extended_min` | 0.55 / 0.75 |
| New confidence floor / ceiling | 0.75 / 0.98 |
| Existing dwell / reacquisition delay | 0.12 s / 0.2 s |
| Existing hand loss grace / held interval | 0.15 s / 0.1 s |
| New `HistoryConfig.window_s` / `max_samples` | 0.3 s / 256 |

New settings are constructor-level options, not new AppConfig/TOML sections.

## Temporal semantics

- Each track has its own candidate, continuous dwell timer, pinch latch, rolling
  prediction history, stable state, and pointer filter. The rolling window averages
  confidence for the current candidate; it does not vote away contradictory input.
  One contradictory frame resets candidate dwell but leaves the accepted gesture
  intact. Stable timestamps record acceptance, not the first candidate observation.
- Initial acquisition emits HAND_ACQUIRED immediately. Starts wait for both dwell
  and acquisition delay, which run concurrently. Missing observations clear evidence;
  every return from absence gets a fresh delay and dwell, even within loss grace.
- The pinch latch enters at distance <= enter and exits at distance >= exit.
  Inside the band, valid distance plus sufficient detector confidence preserves
  pinch despite an UNKNOWN instantaneous pose. Missing/unusable distance falls back
  to confidence-gated labels. Label-only classifiers work but lack distance hysteresis.
- Replacements end the old gesture before starting the new. Generic pinch starts
  and ends precede their PINCH_STARTED/PINCH_ENDED companions. Pinch holds use
  GESTURE_HELD(PINCH). Subscribe to one transition representation to avoid duplicate
  commands. UNKNOWN is a visible but non-actionable stable state: accepting it ends
  any prior action but emits no UNKNOWN start/hold/end events.
- The first absent frame immediately ends active pinch. Other active gestures
  end at loss grace, before HAND_LOST(NONE); expired tracks are removed. No holds or
  pointers are emitted while absent. An observation gap longer than a positive
  loss grace also expires a track, even when it returns in that frame. Zero grace
  disables inferred-gap loss and releases immediately on an explicitly absent frame.
- Pointer events use the source-image index fingertip, clipped to [0, 1]. They are
  emitted every eligible POINT/PINCH frame with matching trusted evidence,
  independently of held-event cadence. The optional exponential filter interprets
  core alpha at 30 Hz and adjusts by elapsed seconds at other rates. Gesture events
  must be consumed before pointer events in each output batch.
- Strictly increasing frame IDs and nondecreasing timestamps are enforced before
  mutation. `reset()` clears all state for replay/new streams and emits no events.
  The caller must release adapter resources separately on reset or shutdown.

## Limits

Synthetic tests establish deterministic behavior, not measured camera accuracy.
Occlusion, inaccurate depth, unusual anatomy, and edge-on poses need real-world
threshold tuning. Thumb/index contact in a fist can win pinch priority. Open palm
currently requires all five fingers rather than inferring an occluded finger.
Deliberate dropout release may split a stroke. Short contradictory visible evidence
can retain a stable gesture until dwell expires, but cannot emit matching pointers.
The caller must continue feeding frames (including empty ones) for timeout events;
there is no background timer. Stable track IDs must come from vision. Configure
positive loss grace above expected frame gaps for slow or highly irregular streams.
