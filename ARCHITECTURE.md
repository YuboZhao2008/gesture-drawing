# Architecture and shared contracts

This repository implements vision, geometric recognition, temporal processing, and
an internal drawing UI on top of shared contracts. Keep the core independent of camera backends, array libraries,
model frameworks, GUI frameworks, and operating-system input APIs.

## Module boundaries and ownership

| Module | Responsibility | Boundary |
| --- | --- | --- |
| `core/types.py` | Architecture/core: observation and classification value objects | Standard library only |
| `core/events.py` | Architecture/core: semantic events and command envelopes | Imports core types only |
| `core/config.py` | Architecture/core: validated settings and explicit file loading | No component construction or global configuration |
| `core/protocols.py` | Architecture/core: shared structural interfaces | Imports core values; never concrete stages |
| `app.py` | Architecture/core: dependency composition and frame dispatch | Depends on core protocols, not implementations |
| `vision/` | Vision agent: capture, backend conversion, tracking IDs, normalization | Implements `HandDetector` and `LandmarkNormalizer` |
| `recognition/` | Recognition/training agent: instantaneous geometric or learned classification | Implements `GestureRecognizer`; may receive a normalizer |
| `training/` | Recognition/training agent: offline datasets, feature extraction, fitting, evaluation | Reuses recognition feature code; no training in the frame loop |
| `temporal/` | Temporal agent: dwell, hysteresis, loss/reacquisition, sequences, pointer smoothing | Implements `TemporalProcessor` |
| `control/` | Control agent: gesture-to-intent policy, calibration transforms, workspace mapping | Implements `CommandMapper` |
| `control/adapters/` | Control agent: canvas, desktop input, application integrations | Implements `OutputAdapter`; owns external side effects |
| `ui/` | UI agent: previews, controls, calibration interaction | Composes/observes components; does not redefine core types |
| `tests/replay/` | Replay infrastructure agent: recording schema, fixtures, deterministic regression runner | Enters through `process_landmarks`; no codec or runner exists yet |

Changes to public types, units, or protocol signatures require coordination with
their consumers. Concrete stages should receive collaborators by injection, not
import another stage's concrete class. Core imports never point into these stages.
`scripts/` is reserved for future entry points and development utilities.

## Data flow and public interfaces

```text
Caller: source image + FrameContext
  HandDetector.detect(image, context) -> HandFrame
  GestureRecognizer.recognize(frame) -> tuple[GesturePrediction, ...]
      LandmarkNormalizer.normalize(hand, context) -> NormalizedHand | None
  TemporalProcessor.process(frame, predictions) -> TemporalOutput
  CommandMapper.map(output) -> tuple[Command, ...]
  OutputAdapter.execute(command) -> None                 (in command order)
```

Import interfaces from `gesture_control.core.protocols`, observation types from
`gesture_control.core.types`, events/commands from `gesture_control.core.events`,
and settings from `gesture_control.core.config`. These are structural `Protocol`s:
implementations need not inherit them. Static checking verifies signature compatibility;
the app also checks cross-stage frame identity and prediction coverage at runtime.

`GestureControlApp` accepts the five primary dependencies as keyword arguments.
`process_frame` runs one synchronous pass and returns dispatched commands.
`process_landmarks` bypasses detection while keeping the same downstream path. No
event bus, background worker, camera loop, mock production backend, or service locator
is introduced inside the core app. The image argument is deliberately `object`: the camera source
and detector agree on representation, color order, and buffer lifetime at composition
time. A detector must validate that object before interpreting it.

The caller serializes calls and owns resource acquisition, shutdown, clock selection,
and error recovery. Exceptions propagate. Output execution is ordered, not atomic:
an adapter failure may follow successful commands and advanced temporal state. Do
not automatically retry an entire frame. Future desktop adapters must own held-input
cleanup on shutdown/disconnect; the core currently creates no such input state.

## Shared values, coordinates, and time

All public value dataclasses are frozen and slotted. Sequence fields are copied to
tuples. `Command.parameters` is copied into a read-only mapping of flat JSON scalars;
use `dict(command.parameters)` when serializing it. Core does not define a recording
format or promise generic dataclass/pickle serialization. Enum string values are the
shared vocabulary for future codecs.

| Type | Meaning / invariant |
| --- | --- |
| `Landmark3D` | Finite x/y/z values; the containing hand specifies units |
| `LandmarkIndex` | Canonical 21-joint topology, wrist=0 through pinky tip=20 |
| `HandLandmarks` | Exactly 21 image-space points, track ID, anatomical handedness, confidence in [0, 1] |
| `NormalizedHand` | Source observation plus wrist-centered, isotropically scaled points and positive scale |
| `FrameContext` | Nonnegative frame ID and timestamp in seconds, positive source width/height in pixels |
| `HandFrame` | Context and zero or more hands with distinct track IDs |
| `GesturePrediction` | One instantaneous gesture/confidence per observed hand; optional normalized pinch distance |
| `StableGesture` | Per-track filtered state and its start timestamp; NONE denotes inactivity |
| `GestureEvent` | A transition/hold event, hand ID, context, gesture, and elapsed duration in seconds |
| `PointerEvent` | Smoothed source-image x/y fractions in [0, 1]; no pixel or device action |
| `TemporalOutput` | One context, current stable states, ordered gesture events, and ordered pointer events |
| `Command` | Context, extensible command name, scalar parameters, optional originating hand ID |

Image coordinates have top-left origin: x increases right and is divided by source
width; y increases down and is divided by source height. z is wrist-relative depth
in image-width units, negative toward the camera. Detector adapters must convert
backend conventions to this contract. Off-image landmarks are allowed; detection
must not silently clamp observations. Handedness is anatomical and independent of
mirroring. Preview mirroring and interaction mirroring must be handled explicitly.

Canonical normalization first converts y to image-width units using height/width,
subtracts the wrist on all axes, and divides by the 3D wrist-to-middle-MCP distance
in those units. That distance is `NormalizedHand.scale`; the result uses palm units.
Rotation and handedness are preserved. A normalizer returns `None` when the scale
is degenerate; the recognizer retains the hand ID and returns UNKNOWN with zero
confidence. Any additional feature rotation or handedness canonicalization belongs
to recognition and must be identical during training and inference.

`GesturePrediction.pinch_distance`, when provided, is the 3D thumb-tip to index-tip
distance in canonical palm units. Temporal processing can use it for enter/exit
hysteresis without recomputing features. An implementation that does not provide it
can still emit pinch labels, but cannot claim distance-based hysteresis. Recognition
produces instantaneous labels and measurements; temporal processing owns persistence.

Temporal pointer tracking clips its output to source-image bounds. Command mapping
then applies calibration, optional interaction mirroring, and workspace conversion.
An uncalibrated pixel mapping is x*(width-1), y*(height-1); adapters and command schemas
must specify rounding and any desktop origin offsets. The prototype UI persists
calibration locally; desktop origins and multi-monitor discovery remain future work.

Within a stream, frame IDs strictly increase and timestamps never decrease; dropped
frame IDs are allowed. `timestamp_s` is elapsed time from a monotonic stream clock,
not wall time. Temporal processing must use frame timestamps, not execution speed or
assumed FPS. Replay preserves those times. The caller must reset temporal state via
`TemporalProcessor.reset()` before a new stream or backward seek and coordinate any
mapper/adapter state cleanup. This stateless app skeleton does not detect stream changes.

## Absence, gestures, and event semantics

The common gestures are NONE, POINT, PINCH, OPEN_PALM, FIST, TWO_FINGER, and UNKNOWN.
An absent hand is represented by no observation and no prediction. A visible hand
that cannot be classified produces UNKNOWN. A prediction cannot be NONE. Temporal
state may retain a missing track during its grace period and then represent an
inactive track with `StableGesture(..., gesture=NONE)` or retire that state.

The detector owns stable track IDs during continuous observation; handedness is not
an identity. Temporal processing owns loss and reacquisition timing. A new unrelated
hand must not inherit an active track's gesture state. Exactly one prediction is
required for every observed ID, even when confidence is low or normalization fails.
Empty frames must reach temporal processing to permit release/loss events.

Event kinds are GESTURE_STARTED, GESTURE_HELD, GESTURE_ENDED, PINCH_STARTED,
PINCH_ENDED, HAND_ACQUIRED, and HAND_LOST. The temporal implementation must document
and test its transition ordering against these conventions:

- A started event begins an accepted stable gesture; held events report elapsed
  accepted-state time at the configured interval; ended events identify the old gesture.
- End the old gesture before starting a replacement. Emit acquisition before any
  gesture start; end any active gesture before HAND_LOST. HAND_LOST carries NONE.
- PINCH_STARTED/PINCH_ENDED are convenience companion events to the corresponding
  generic pinch transitions. Emit the generic transition first, then its companion.
  A mapper must subscribe to one representation for an action to avoid double firing.
- Gesture events in a batch precede pointer events. Every event and command retains
  the input context; stable-state start times cannot be in the future. A loss event
  may refer to a track absent from the current frame.

Events describe hand semantics. They must never encode a drawing tool, mouse button,
key binding, or application-specific action. Mappers and adapters agree on command
names (for example, an eventual `canvas.*` namespace), parameter units, and schemas.
The prototype `pointer.move`, `stroke.begin`, `stroke.point`, `stroke.end`, and
`canvas.clear` catalog lives in [UI integration](src/gesture_control/ui/README.md).
Unsupported adapter commands must raise.

## Configuration and extension rules

`AppConfig` contains immutable camera, recognition, temporal, smoothing, and workspace
sections plus debug mode. Construct it directly, use `AppConfig.from_mapping`, or
call `load_config` with a TOML/JSON path. Partial sections use defaults; unknown keys,
wrong types (including booleans passed as numbers), nonfinite values, and invalid
bounds fail immediately. File and parse errors propagate. YAML and environment
overrides are intentionally absent; no parser dependency is needed.

Timing settings use seconds. Confidence values are [0, 1]. Pinch enter must be less
than exit, both in palm units; the recognition section shares these thresholds with
the temporal implementation as needed. Reacquisition delay gates renewed activity,
loss grace tolerates brief missing observations, and dwell accepts a gesture after
continuous evidence. Temporal work must specify their exact transition interactions.
Smoothing alpha is the weight of the newest pointer observation in (0, 1]. Workspace
dimensions are positive pixels; they do not discover the actual display.

Future stages receive only the settings and collaborators they need. Add heavyweight
dependencies as optional extras when a concrete backend requires them. Share feature
code between inference and training, keep device effects inside adapters, and keep
test doubles inside tests. There are no global mutable settings or singleton stages.

Replay infrastructure should version its future schema, preserve frame contexts and
hand IDs, and test recordings at variable playback speeds against the same events.
Generated recordings, datasets, and model artifacts must remain untracked; intentional
small test fixtures should live under `tests/` with provenance and schema versions.

## Validation

Install all extras with `.\.venv\Scripts\python.exe -m pip install -e ".[dev,vision,ui]"`,
then run the venv validation commands in [README.md](README.md#testing).
Current tests verify value invariants, configuration loading, protocol-compatible test
doubles, no-hand flow, replay entry, cross-stage identity, ordered dispatch, and error
propagation. They do not establish real-time performance or recognition accuracy.

## Integrated runtime

`python -m gesture_control` enters `runner.py`, constructs one camera and one detector,
injects `PalmNormalizer` into recognition, and composes the existing app with
`GestureTemporalProcessor`, `ObservedTemporal`, `DrawingMapper`, and `DrawingCanvas`.
The UI observer receives each exact temporal batch before mapping so it can mark
visible IDs and release drawing on the first absent frame. Temporal also ends pinch
immediately on absence, with generic end before its companion; mapping consumes only
pinch companions and emits one stroke end. Gesture transitions precede pointers.

The runner owns resource cleanup through independently guarded callbacks, including
partial startup, capture errors, presentation errors, and keyboard interruption.
Cleanup failures are logged while later resources are still closed. No desktop input
is synthesized. UI rendering may shrink the preview of the canvas while preserving
its configured pixel resolution and source-to-workspace calibration transform.
Calibration persistence and internal drawing are implemented by `ui/`; the general
`control/` package remains an extension point for future adapters.
