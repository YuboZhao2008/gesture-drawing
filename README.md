# Gesture Control

A Python webcam prototype that turns hand landmarks into stable gestures and an
interactive drawing canvas. Point to move the cursor, pinch to draw, and calibrate
a comfortable working area. Includes a live skeleton, raw/stable gesture diagnostics,
four-corner calibration, and PNG export.

## Architecture

```text
OpenCVCamera -> MediaPipeHandDetector -> GeometricGestureRecognizer
                                       (PalmNormalizer for classification)
            -> GestureTemporalProcessor -> DrawingMapper -> DrawingCanvas
                                                           -> UI presentation
```

`GestureControlApp` runs the stages synchronously. `runner.py` owns capture,
construction, controls, and shutdown. `ObservedTemporal` lets the UI observe the
same frame/predictions without repeating recognition. Core contracts use only the
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
| D | Toggle diagnostics and skeleton |
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

Gesture starts normally require 0.12 s dwell and 0.2 s acquisition delay. Visible
release also uses dwell; contradictory evidence stops pointer updates while that
transition settles. Missing hands release pinch immediately. Reappearance starts a
new stroke after acceptance; no line connects across disappearance. Clear and
calibration require a fresh pinch start, so release and pinch again before drawing.

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

## Limitations

- One hand and one owned stroke at a time; no desktop mouse/keyboard automation.
- Geometric thresholds are not a trained classifier or a measured accuracy claim.
  Occlusion, depth errors, edge-on poses, lighting, and individual anatomy affect results.
  Thumb/index contact can take pinch priority even in a fist.
- Track continuity is conservative: a missing frame, handedness flip, fast motion,
  or long frame gap can split a track/stroke. Slow capture may need a larger
  `temporal.hand_loss_grace_s`; no background timer advances a stalled stream.
- No eraser, undo, brush dynamics, training pipeline, or packaged recording/replay UI.
- Calibration and PNG exports are local generated files. `models/`, `datasets/`, and
  `recordings/` keep placeholders only. The three specialist `*-worktree/` directories
  are real Git worktrees and remain ignored and intact.
