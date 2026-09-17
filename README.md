# Gesture Control

A Python webcam prototype that turns hand landmarks into stable gestures and an
interactive drawing canvas. Point to move the cursor, pinch to draw, and calibrate
a comfortable working area. Includes a live skeleton, raw/stable gesture diagnostics,
four-corner calibration, and PNG export.

![Gesture Control interface, synthetic preview](docs/images/ui-normal.png)

Dark OpenCV interface with a persistent skeleton, compact gesture/status footer,
and optional advanced diagnostics. The screenshot uses synthetic observations;
its displayed FPS is illustrative. See [calibration preview](docs/images/ui-calibration.png)
and [debug preview](docs/images/ui-debug.png).

## Architecture

```text
OpenCVCamera -> MediaPipeHandDetector -> GeometricGestureRecognizer
                                       (PalmNormalizer for classification)
            -> GestureTemporalProcessor -> DrawingMapper -> DrawingCanvas
                                                           -> UI presentation
```

`GestureControlApp` runs processing stages synchronously. A camera worker keeps
one latest frame available without building a queue; rendering and key handling
stay on the main thread. `runner.py` owns construction, controls, and shutdown.
`ObservedTemporal` lets the UI observe the same frame/predictions without repeating
recognition. Core contracts use only the
standard library; vision and UI dependencies are optional extras.

Frame timestamps drive dwell, pinch hysteresis, hand loss, and pointer smoothing.
Track IDs and frame contexts propagate through the pipeline. No hand means empty
observations and predictions; UNKNOWN means a visible, unrecognized hand.
Palm-normalized coordinates are used only for classification. Pointer coordinates
come from the source-image index fingertip, then map through calibration to canvas
pixels. Preview resizing never changes interaction coordinates.

See [shared contracts](ARCHITECTURE.md), [vision setup and APIs](src/gesture_control/vision/README.md),
[recognition and temporal semantics](src/gesture_control/recognition/README.md),
and [UI integration and command schemas](src/gesture_control/ui/README.md).

## Installation

Requires Python 3.12+, a webcam, and a desktop display for the interactive UI.
From the repository root in PowerShell, create a venv if one does not exist:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,vision,ui]"
Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile models/hand_landmarker.task
```

Both extras use **opencv-contrib-python**, which satisfies MediaPipe and provides
desktop `cv2`. Do not also install `opencv-python` or a headless OpenCV distribution
in this venv: they share module files. The model is downloaded explicitly, kept
locally, and ignored by Git. The app never downloads it automatically.
On POSIX, use `.venv/bin/python` for the commands below.

## Launch

```powershell
.\.venv\Scripts\python.exe -m gesture_control
```

Keep the Gesture Studio window focused for keyboard controls. Capture and preview
are unmirrored. The default camera is index 0, requested at 640x480 and 30 FPS;
actual capture size and timing come from the camera. The drawing surface defaults
to 1920x1080, displayed at a smaller size without changing export resolution.

```powershell
.\.venv\Scripts\python.exe -m gesture_control --config configs/default.toml
.\.venv\Scripts\python.exe -m gesture_control --help
```

`--model`, `--calibration`, and `--drawing` select file paths (relative to the current
directory unless absolute). `--config` explicitly loads TOML/JSON overrides; without
it the app uses built-in defaults matching `configs/default.toml`. Change
`camera.index` for another webcam and `workspace.width/height` for canvas resolution.
Unknown keys, invalid types, nonfinite values, and inconsistent thresholds fail early.
Defaults are starting points for tuning, not measured accuracy recommendations.

Normal mode keeps the skeleton visible and hides advanced diagnostics. Use D to
toggle them, `--diagnostics` to start with them, or `--no-diagnostics` to override
`debug = true` in a configuration file. `--camera-width` / `--camera-height` request
capture dimensions; unsupported requests may negotiate a different mode.
`--synchronous-camera` selects the original capture path for comparison or device
troubleshooting. The default latest-frame worker favors fresh frames under load;
its frame IDs can skip, and it does not make the camera's sensor run faster.

ESC, window close, Ctrl+C, camera failure, and exceptions trigger cleanup. Active
drawing, mapper state, temporal state, camera, detector, and UI windows each receive
a cleanup attempt. Failures are logged and do not block remaining cleanup or replace
the original processing exception. Startup/runtime errors return a nonzero exit code.

## Controls and gestures

| Key | Action |
| --- | --- |
| ESC | Release and quit |
| C | Start/restart calibration; disable drawing |
| SPACE | Capture the current pointing position for calibration |
| D | Toggle advanced diagnostics, bounds, and landmark indices |
| R | Clear canvas and release the stroke |
| S | Save PNG to `recordings/drawing.png` (overwrites that path) |

| Gesture / observation | Behavior |
| --- | --- |
| POINT (index extended, other fingers folded) | Move cursor without drawing |
| PINCH (thumb/index tips close) | Begin once, continue while active, end once on release |
| OPEN_PALM | Neutral; stop an accepted drawing gesture |
| FIST / TWO_FINGER | Recognized for diagnostics; no drawing action |
| UNKNOWN | Non-actionable; ends prior gesture after dwell |
| Hand missing | Release drawing immediately, even during tracking loss grace |

Gesture starts normally require 0.12 s dwell and 0.2 s acquisition delay, running
concurrently. PINCH start/release use a 40 ms dwell cap, accepted on the next
observed frame; initial/new-track acquisition still applies. Established POINT
can tolerate up to 60 ms of visible classification noise. Uncertain pinch release
never adds drawing points. Missing hands release pinch immediately. Reappearance
starts a new stroke after acceptance; no line connects across disappearance. Clear and
calibration require a fresh pinch start, so release and pinch again before drawing.

The default adaptive One Euro filter suppresses small stationary jitter and
responds faster during movement. Tune `[smoothing]` in `configs/default.toml`:
lower `min_cutoff` for more stationary smoothing, or increase `beta` for less
motion lag. Cutoffs use Hz and speed uses source-image fractions per second.
Set `adaptive = false` for the legacy `alpha` filter, or `enabled = false` to
bypass filtering. See the [deterministic motion comparison](src/gesture_control/recognition/README.md#motion-tuning-and-measurements).

## Calibration

Press C, make POINT, then place the fingertip at each comfortable source-image
corner: **top left, top right, bottom right, bottom left**. Hold still and press
SPACE at each corner. These are positions in the unmirrored camera image, not
coordinates of the preview window. A projective transform maps source-image
fractions to workspace fractions, then the mapper clamps and rounds to canvas pixels.

Calibration rejects crossed, repeated, tiny, or incorrectly ordered corners. Press
C to restart. Drawing stays disabled until calibration completes. Successful
calibration saves to `recordings/calibration.json` and reloads on the next launch.
Without calibration, the whole image maps to the canvas. Recalibrate after changing
camera pose, resolution/aspect ratio, crop, or orientation; saved calibration does
not automatically detect those changes.

## Testing

All automated tests use synthetic frames, geometry, or mocked backends; no physical
webcam or display is required. Install all extras before running the complete suite.
The [CI workflow](.github/workflows/ci.yml) runs these five checks on Windows and
Linux with Python 3.12; it needs no downloaded model or camera access.

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_tmp
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m pip check
```

Integration tests cover real normalization/recognition/temporal processing, source
pointer flow, ordered pinch commands, safe hand loss, UNKNOWN versus absence,
calibration, and lifecycle failures including partial startup and failed cleanup.
`process_landmarks(HandFrame)` supports deterministic landmark input; a recording
codec/replay CLI is not implemented.

Optional hardware checks (the first opens a window; the second still processes
camera frames and renders in memory):

```powershell
.\.venv\Scripts\python.exe -m gesture_control --max-frames 30
.\.venv\Scripts\python.exe -m gesture_control --headless --max-frames 30
```

For manual acceptance, verify skeleton acquisition, POINT cursor movement, PINCH
start/continuation/release, disappearance and reappearance without a connecting line,
four-corner calibration, D/R/S controls, and ESC exit. If camera 0 cannot open, check
Windows camera permissions, close other camera apps, and select the correct index.

## Performance measurement

The [integrated polish report](docs/POLISH_RELEASE.md) records before/after trials,
validation, and the distinction between live no-hand timing and synthetic motion.
The [final integration audit](docs/FINAL_INTEGRATION.md) records the subsequent
capture-mode comparison, current validation, and remaining hardware acceptance.
The [initial architecture review](PERFORMANCE.md) explains the measured bottlenecks.
To collect stage timings locally without saving webcam images:

```powershell
.\.venv\Scripts\python.exe -m gesture_control --max-frames 960 --profile-warmup 60 --profile recordings/performance.json
.\.venv\Scripts\python.exe scripts/benchmark_pipeline.py --calibrated --diagnostics off --output recordings/synthetic-performance.json
```

Profiling is opt-in. Reports include capture, inference, recognition, temporal,
mapping, drawing, rendering, and native event-pump timings. Nested stage durations
must not be added together. The synthetic benchmark excludes the camera, inference,
and native window; its FPS is processing capacity, not achieved webcam FPS.
The `metadata.capture_delivery` section reports acquisition FPS from frame IDs and
timestamps, skipped IDs between retained deliveries, and host acquisition-to-delivery
age. It excludes capture warmup and retains at most 9000 deliveries. This host age
excludes sensor/driver buffering and is not optical motion-to-photon latency.

## Multi-agent development

Start one lead agent with [.agent/START_LEAD.md](.agent/START_LEAD.md).
[AGENTS.md](AGENTS.md) defines repository policy; [.agent/README.md](.agent/README.md)
explains delegation, isolated worktrees, recovery, and validated local integration.
The lead owns the final result and coordinates specialists directly.

## Project structure

| Path | Purpose |
| --- | --- |
| `src/gesture_control/core/` | Validated values, configuration, protocols, performance metrics |
| `src/gesture_control/vision/` | Camera capture, MediaPipe adapter, palm normalization |
| `src/gesture_control/recognition/` | Geometric gesture classification |
| `src/gesture_control/temporal/` | Dwell, pinch hysteresis, hand loss, One Euro filtering |
| `src/gesture_control/ui/` | Calibration, drawing, camera overlay and controls |
| `src/gesture_control/runner.py` | CLI, composition and resource lifecycle |
| `tests/` | Synthetic unit and integration regressions |
| `scripts/` | Deterministic benchmarks and synthetic UI previews |
| `configs/`, `docs/` | Example settings, screenshots and validation reports |
| `models/`, `recordings/`, `datasets/` | Ignored local assets; placeholders are tracked |

## Limitations

- One hand and one owned stroke at a time; no desktop mouse/keyboard automation.
- Geometric thresholds are not a trained classifier or a measured accuracy claim.
  Occlusion, depth errors, edge-on poses, lighting, and individual anatomy affect results.
  Thumb/index contact can take pinch priority even in a fist.
- Track continuity is conservative: a missing frame, handedness flip, fast motion,
  or long frame gap can split a track/stroke. Slow capture may need a larger
  `temporal.hand_loss_grace_s`; no background timer advances a stalled stream.
- The camera currently negotiates 30 FPS on the tested device. Driver delivery,
  exposure, inference, and scheduling can still create uneven frames. Shutdown
  waits for an in-progress native camera read; a hung driver can delay release.
- No eraser, undo, brush dynamics, training pipeline, or packaged recording/replay UI.
- Calibration and PNG exports are local generated files. `models/`, `datasets/`, and
  `recordings/` keep placeholders only. The three specialist `*-worktree/` directories
  are real Git worktrees and remain ignored and intact.
