# Gesture Studio prototype

Install `python -m pip install -e ".[dev,ui]"`. UI runtime dependencies are NumPy
and desktop OpenCV (`opencv-python`, not the headless build). Core remains unchanged.

The single OpenCV window contains an aspect-preserving camera preview on the left,
an internal pencil canvas on the right, and diagnostics/controls below. The overlay
shows shared hand landmarks, canonical skeleton, bounding box, index tip, raw/stable
gesture confidence, anatomical handedness, tracking state, measured presentation FPS,
workspace size, tool state, and calibration status. The demo controls one stroke/hand
at a time; the text panel prioritizes the first observed hand.

## Final integration

Use the existing application composition. Supply the actual detector, recognizer,
temporal processor, and caller-owned capture loop; this UI creates no camera:

```python
from gesture_control.app import GestureControlApp
from gesture_control.ui.demo import DemoUI

ui = DemoUI(width=640, height=480)
app = GestureControlApp(
    detector=detector,
    recognizer=recognizer,
    temporal=ui.wrap_temporal(temporal),
    mapper=ui.mapper,
    adapter=ui.canvas,
)
try:
    for image, context in frames:  # caller supplies uint8 BGR and FrameContext
        app.process_frame(image, context)
        if not ui.present(image):
            break
finally:
    ui.close()
    # Release caller-owned capture resources here (or use their context manager).
```

`ObservedTemporal` forwards the shared protocol and observes the exact frame,
predictions, and output, without repeating recognition. Empty frames must continue
through the app. `observe` also releases drawing immediately on observed hand loss,
even during temporal grace; the mapper then emits the matching end command. After
loss, clear, calibration, or reset, a fresh PINCH_STARTED is required to draw.
Call the wrapped processor's `reset()` before a new stream, and always `close()` in
`finally`, including on capture failure. Do not dispatch commands twice.

`DemoUI.render(image)` returns an image without a GUI; `present(image)` displays it,
polls keys once, and returns whether to continue. `handle_key(code)` is independently
testable. There is no constructor loop or module-level application runner.

## Commands and units

`DrawingMapper` implements `CommandMapper`; `DrawingCanvas` implements `OutputAdapter`.
They can also be constructed separately with matching width/height.

| Command | Parameters | Meaning |
| --- | --- | --- |
| `pointer.move` | integer `x_px`, `y_px`; required `hand_id` | Move cursor only |
| `stroke.begin` | integer `x_px`, `y_px`; required `hand_id` | Start at this current pointer |
| `stroke.point` | integer `x_px`, `y_px`; required `hand_id` | Extend owned stroke |
| `stroke.end` | none; owning `hand_id`, or None for cleanup | Release, idempotently |
| `canvas.clear` | none | Erase and release |

Input PointerEvent x/y are **source-image normalized fractions**. Calibration maps
these to workspace normalized fractions; the mapper then clamps to [0,1] and rounds
`x*(width-1), y*(height-1)` into canvas pixels. Preview resize/letterboxing never enters
interaction mapping. Palm-normalized landmarks are never used. No preview or input
mirroring is applied: use the same camera orientation for detection and display.

PINCH_STARTED arms a stroke; the first valid pointer in that or a later frame emits
stroke.begin. Pointer updates while armed emit stroke.point, interpolated with an
antialiased straight pencil line. PINCH_ENDED/HAND_LOST end an active stroke once.
Generic pinch start/end events do not duplicate their companion events. POINT and
OPEN_PALM do not start strokes. Another hand cannot move or end an owned stroke.
Calling adapter.clear() externally must be paired with mapper.reset(); UI R does both.

## Calibration and controls

| Key | Action |
| --- | --- |
| ESC | Release and quit |
| C | Start/restart calibration; drawing is disabled |
| SPACE | Capture current temporal pointer at the prompted corner |
| D | Toggle diagnostic details and skeleton |
| R | Clear canvas and release |
| S | Save canvas PNG without cursor |

Capture **TOP LEFT -> TOP RIGHT -> BOTTOM RIGHT -> BOTTOM LEFT** in the unmirrored
camera preview. Move the pointing fingertip to each comfortable workspace corner,
hold it still, and press SPACE. A four-point projective homography maps those corners
to (0,0), (1,0), (1,1), (0,1). Validation rejects nonfinite/out-of-image coordinates,
repeated points, wrong ordering, concavity, crossing edges, and tiny workspace area.
Invalid fourth captures retain the first three points; press C to restart them all.

Calibration persists atomically to `recordings/calibration.json` (version 1) and
reloads at construction. The previous valid transform remains available if a new
calibration fails, but drawing remains disabled until calibration completes.
Default drawing export is `recordings/drawing.png`. Both paths are configurable;
generated files remain ignored. Recalibrate after changing camera pose, aspect ratio,
crop, or mirroring. Calibration is intentionally independent of preview window size.

No live camera/display validation is claimed by unit tests. Tests exercise raster
rendering, geometry, persistence, commands, ownership, state transitions, safety,
and controls without opening a window. No eraser, undo, or brush dynamics are included.
