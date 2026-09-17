# Geometric recognition and temporal integration

These implementations use only the standard library. Core contracts are unchanged.
The vision component must supply a `LandmarkNormalizer` implementing the canonical
aspect-corrected, wrist-centered palm-unit normalization in `ARCHITECTURE.md`.

```python
from gesture_control.core.config import AppConfig
from gesture_control.recognition import GeometricGestureRecognizer, GeometryConfig
from gesture_control.temporal import GestureTemporalProcessor, HistoryConfig

from gesture_control.vision import PalmNormalizer

config = AppConfig()
normalizer = PalmNormalizer()
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
threshold. Other poses combine normalized chain reach and PIP/DIP straightness.
The angle score averages the two joints but can improve the weaker angle by at
most 18 degrees, tolerating a modest isolated distal curl without masking a sharp
bend. Reach still constrains folded-back fingers. POINT requires index extension
and three folded fingers; TWO_FINGER requires index/middle extension and ring/pinky folding. FIST requires the four
primary fingers folded. OPEN_PALM additionally requires a straight thumb. Thumb
posture is otherwise unrestricted. Unmatched, ambiguous, or unusable geometry is
UNKNOWN. Empty frames produce no predictions, never UNKNOWN or NONE predictions.

Confidence interpolates between the configurable floor and ceiling according to
the weakest pose margin (or pinch closeness), then multiplies detector confidence.
Below the shared confidence gate the label becomes UNKNOWN while retaining the
score. Duplicate joints, bone lengths at or below 5% of the longest bone in that
finger, or failed normalization produce zero confidence and no pinch distance. The recognizer is stateless; the temporal stage owns hysteresis.

| Setting | Default |
| --- | --- |
| Existing `RecognitionConfig.min_confidence` | 0.7 |
| Existing pinch enter / exit, palm units | 0.25 / 0.35 |
| New `GeometryConfig.folded_max` / `extended_min` | 0.55 / 0.75 |
| New confidence floor / ceiling | 0.75 / 0.98 |
| Existing dwell / reacquisition delay | 0.12 s / 0.2 s |
| Existing hand loss grace / held interval | 0.15 s / 0.1 s |
| New `HistoryConfig.window_s` / `max_samples` | 0.3 s / 256 |
| `PointerFilterConfig.min_cutoff` / `beta` / `derivative_cutoff` | 1.5 Hz / 20 / 1 Hz |
| `MotionConfig.pinch_start_s` / `pinch_release_s` | 0.04 s / 0.04 s |
| `MotionConfig.pointer_noise_grace_s` / `pointer_dropout_grace_s` | 0.06 s / 0.08 s |

Geometry, history, and motion timing options remain constructor-level settings.
Filter selection and tuning also live in `AppConfig.smoothing` / `[smoothing]`:

```toml
[smoothing]
enabled = true
adaptive = true
min_cutoff = 1.5
beta = 20.0
derivative_cutoff = 1.0
alpha = 0.35 # Used only when adaptive = false (legacy exponential filter).
```

Cutoffs must be positive finite numbers, beta nonnegative and finite, and adaptive
a boolean. `enabled = false` bypasses both filters. Existing alpha-only files
remain valid; set `adaptive = false` to select the alpha-based behavior explicitly.

## Temporal semantics

- Each track has its own candidate, continuous dwell timer, pinch latch, rolling
  prediction history, stable state, and pointer filter. The rolling window averages
  confidence for the current candidate; it does not vote away contradictory input.
  One contradictory frame resets candidate dwell but leaves the accepted gesture
  intact. Stable timestamps record acceptance, not the first candidate observation.
- Initial acquisition emits HAND_ACQUIRED immediately. Starts wait for both dwell
  and acquisition delay, which run concurrently. Missing observations clear evidence;
  stroke reacquisition gets a fresh delay and dwell, even within loss grace. An
  already accepted POINT returning within 80 ms of its last observation resumes
  immediately with its previous filter state. Longer gaps reset the filter and gate
  renewed activity. New track IDs always get independent state and acquisition delay.
- The pinch latch enters at distance <= enter and begins release at distance >= exit.
  Inside the band, valid distance plus sufficient detector confidence preserves
  pinch despite an UNKNOWN instantaneous pose. Missing/unusable distance falls back
  to confidence-gated labels. Label-only classifiers work but lack distance hysteresis.
  A single exit spike does not clear the latch: returning to the band during release
  dwell preserves an already accepted hold. An unaccepted pinch candidate loses its
  latch on exit and must cross the enter threshold again; stale candidate evidence
  cannot start a stroke from the exit band. Pinch starts and releases use a 40 ms
  dwell cap (or the core dwell if shorter). Continuous non-pinch evidence releases drawing even when
  its labels alternate. Other gesture replacements retain core dwell protection.
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
  emitted every eligible POINT/PINCH frame independently of held-event cadence.
  Established POINT tolerates up to 60 ms of classification noise when detector
  confidence remains sufficient and pose confidence or normalized geometry remains
  available. PINCH requires matching evidence; uncertain release never extends a stroke.
  The default One Euro filter increases cutoff with speed, without buffering frames.
  `PointerFilterConfig(adaptive=False)` selects the original exponential filter,
  using core alpha at 30 Hz and adjusting by elapsed seconds at other rates.
  `SmoothingConfig.enabled=False` bypasses either filter. Core alpha applies only
  to the legacy filter. Gesture events
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
can retain a stable gesture until dwell expires; only bounded POINT cursor continuity
is allowed during uncertain evidence, never additional drawing.
The caller must continue feeding frames (including empty ones) for timeout events;
there is no background timer. Stable track IDs must come from vision. Configure
positive loss grace above expected frame gaps for slow or highly irregular streams.

## Motion tuning and measurements

Existing constructors and `process`/`reset` signatures remain compatible. By default,
the processor derives its filter from the supplied `SmoothingConfig`, so loaded
TOML/JSON settings reach the filter without UI or runner changes. An explicit
`pointer_filter=` overrides adaptive/cutoff/beta settings; the shared `enabled`
switch and legacy `alpha` remain effective. Constructor overrides are keyword-only:

```python
from gesture_control.temporal import MotionConfig, PointerFilterConfig

temporal = GestureTemporalProcessor(
    config.temporal,
    config.recognition,
    config.smoothing,
    pointer_filter=PointerFilterConfig(min_cutoff=1.5, beta=20, derivative_cutoff=1),
    motion=MotionConfig(pinch_start_s=0.04, pinch_release_s=0.04),
)
```

The [One Euro algorithm](https://gery.casiez.net/1euro/) filters the raw derivative
and uses `min_cutoff + beta * speed` for position filtering. This implementation
uses one cutoff from 2D speed for both axes; coordinates remain source-image
fractions. Lower minimum cutoff reduces stationary jitter; higher beta reduces
motion lag. Repeated timestamps retain filter state; reset clears position and speed.

Inspection identified pointer suppression on contradictory labels and filter reset
plus 200 ms reacquisition after even one missing frame as sources of cursor stalls.
The existing 300 ms history only averages confidence; it does not buffer pointers.
Recognition confidence (0.7), pinch distances (0.25/0.35), ordinary dwell (120 ms),
and new-track acquisition (200 ms) are retained. There is no camera recording here
to measure the contribution of actual landmark noise or tracking ID churn.

Reproduce the deterministic comparison with `python tests/unit/test_motion.py`.
It uses seeded uniform +/-0.006 noise, 30/60 Hz and repeating irregular intervals
(13, 27, 18, 40, 22 ms). Pixel units below multiply horizontal source-image error
by 1920; these are synthetic filter measurements, not end-to-end camera latency.
Stationary/slow scores discard the first second. Fast motion is 0.8 widths/s;
slow motion is 0.05 widths/s. Lag is signed mean tracking error divided by speed.

| Metric (old EMA → adaptive) | 30 Hz | 60 Hz | Irregular |
| --- | --- | --- | --- |
| Stationary RMS error, pixels | 2.72 → 2.36 | 1.72 → 1.58 | 2.16 → 1.89 |
| Slow motion RMS error, pixels | 6.24 → 5.89 | 6.76 → 5.58 | 6.76 → 6.10 |
| Fast motion RMS error, pixels | 93.67 → 13.44 | 107.21 → 16.30 | 99.09 → 14.93 |
| Fast motion estimated lag, ms | 61.0 → 8.3 | 69.8 → 10.1 | 64.5 → 9.2 |
| Reversal RMS error, pixels | 75.90 → 36.00 | 83.10 → 34.96 | 76.60 → 33.17 |

Stationary raw RMS error is 6.88/6.36/6.43 pixels respectively. Slow-motion lag is
53–58 ms after filtering, versus 59–68 ms before. A step reaches 90% within 50 ms
of its first observed sample at all tested cadences, without overshoot. Pinch
start/release now accepts after 40 ms of evidence, sampled at the next frame:
66.7 ms at 30 Hz or 50 ms at 60 Hz, versus 133.3 ms before. Initial/reacquired
strokes still respect acquisition protection. Hand absence ends a stroke immediately.

## Classification robustness evaluation

See [offline augmentation and evidence](../training/README.md) for the seedable
synthetic corpus, original-vs-improved comparison, confusion metrics, rejection
tests, limitations, and the decision to keep geometry instead of training a model
on insufficiently diverse fixtures. Augmentation never runs in live inference.
