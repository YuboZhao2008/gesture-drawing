# Gesture Studio prototype

Install `python -m pip install -e ".[dev,ui]"`. UI runtime dependencies are NumPy
and desktop OpenCV (`opencv-contrib-python`, shared with vision; not the headless build). Core remains unchanged.

The single OpenCV window contains a secondary, aspect-preserving camera preview on
the left, a larger pencil canvas on the right, and diagnostics/controls below. The overlay
shows shared hand landmarks, canonical skeleton, bounding box, index tip, raw/stable
gesture confidence, anatomical handedness, tracking state, measured presentation FPS,
workspace size, tool state, and calibration status. The demo controls one stroke/hand
at a time; the text panel prioritizes the first observed hand.

Normal mode starts with only essential status: tracking/FPS in the header, the stable
gesture and confidence in the footer, and pencil/calibration state beside it. The
skeleton and fingertip remain visible. D adds the raw gesture, track ID, handedness,
landmark indices, and bounds in a compact camera overlay. No additional GUI package
is needed. The light drawing paper sits inside restrained dark panel borders.

Calibration displays the requested corner, 1/4 through 4/4 progress, captured corner
markers, and an amber target. Targets suggest the corner direction: SPACE captures
the actual temporal pointer at a comfortable position, not the target's fixed position.
Errors appear in coral; successful calibration returns to drawing with a green
Calibrated status. C restarts the flow.

The camera pane is 400 pixels wide; the canvas pane uses a minimum 640x240 presentation
area, with aspect-preserving preview scaling and letterboxing. This keeps feedback readable
even for a 1x1 or very narrow canvas; export and interaction dimensions remain unchanged.
Debug cards are hidden during calibration so they cannot cover lower-corner targets.
Fourth-point geometry and save failures leave the first three captures intact. A save
failure offers SPACE to retry the current fourth corner; C restarts all captures.
An externally restored session with too many points displays restart guidance safely.

## Final integration

Launch the integrated camera application with `.\.venv\Scripts\python.exe -m gesture_control`.
`runner.py` owns its capture loop and failure-safe resource cleanup. For embedding,
use the existing application composition with your own stages; this UI creates no camera:

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
testable. There is no constructor loop; the package runner owns scheduling.
The event pump uses `cv2.pollKey()` when available, falling back to `waitKey(1)` on
older builds. Capture paces the real loop; the UI adds no intentional key-poll delay.
Large drawing canvases are resized only for presentation (up to 640x480),
while pointer mapping and saved PNGs retain the configured canvas resolution.

## Commands and units

`DrawingMapper` implements `CommandMapper`; `DrawingCanvas` implements `OutputAdapter`.
They can also be constructed separately with matching width/height.

| Command | Parameters | Meaning |
| --- | --- | --- |
| `pointer.move` | integer `x_px`, `y_px`; required `hand_id` | Move cursor only |
| `stroke.begin` | integer `x_px`, `y_px`; required `hand_id` | Start selected Pencil/Eraser at this pointer |
| `stroke.point` | integer `x_px`, `y_px`; required `hand_id` | Extend owned drawing/erasing stroke |
| `stroke.end` | none; owning `hand_id`, or None for cleanup | Release, idempotently |
| `tool.select` | `tool`: `pencil` or `eraser` | Release and select the active tool |
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

`DemoUI.observe` calls `DrawingMapper.observe_evidence(frame, predictions)` before
mapping the matching temporal batch. Standalone users must do the same to enable
gesture tool selection. Hold TWO_FINGER for 0.65 seconds with both observation and
prediction confidence at least 0.80 to toggle tools. A gap over 0.15 seconds, missing
hand, changed ID, or uncertain pose interrupts the hold. A held pose toggles only
once; rearming requires 0.18 seconds of a confident other recognized gesture and
a 0.8-second switch cooldown. Tool gestures never start strokes, and selection
emits `stroke.end` before `tool.select` when a stroke was active.

The tool palette shows Pencil/Eraser, progress, and selection feedback. PINCH draws
or erases; POINT moves the corresponding cursor. Calibration is identical for both.
Clear, calibration, and stream reset preserve tool selection but release strokes
and cancel pending holds. The eraser radius is 3% of the smaller canvas dimension,
rounded to pixels with a four-pixel minimum; its preview outline scales with the canvas.
It removes a separate ink-coverage mask and restores the actual copied background,
including custom nonuniform BGR backgrounds. Dirty-region compositing bounds update
cost. PNG export remains the composited BGR image; transparent export and undo are
not implemented, and no retained-stroke rewrite was needed.

## Calibration and controls

| Key | Action |
| --- | --- |
| ESC | Release and quit |
| C | Start/restart calibration; drawing is disabled |
| SPACE | Capture current temporal pointer at the prompted corner |
| D | Toggle advanced diagnostics, bounds, and landmark indices |
| R | Clear canvas and release |
| S | Save canvas PNG without cursor |

Capture **TOP LEFT -> TOP RIGHT -> BOTTOM RIGHT -> BOTTOM LEFT** in the unmirrored
camera preview. Move the pointing fingertip to each comfortable workspace corner,
hold it still, and press SPACE. A four-point projective homography maps those corners
to (0,0), (1,0), (1,1), (0,1). Validation rejects nonfinite/out-of-image coordinates,
repeated points, wrong ordering, concavity, crossing edges, and tiny workspace area.
Repeated captures are rejected immediately without advancing to the next corner.
Invalid fourth captures retain the first three points; move and press SPACE to retry
the fourth corner, or press C to restart them all.

Calibration persists atomically to `recordings/calibration.json` (version 1) and
reloads at construction. The previous valid transform remains available if a new
calibration fails, but drawing remains disabled until calibration completes.
Default drawing export is `recordings/drawing.png`. Both paths are configurable;
generated files remain ignored. Recalibrate after changing camera pose, aspect ratio,
crop, or mirroring. Calibration is intentionally independent of preview window size.

No live camera/display validation is claimed by unit tests. Tests exercise raster
rendering, geometry, persistence, commands, ownership, state transitions, safety,
and controls without opening a window. Undo and brush dynamics are not included.

## Visual verification and performance

`python scripts/preview_ui.py --output docs/images` regenerates clearly labeled
synthetic screenshots of normal, debug, calibration, and error states. The fixture's
52 FPS label is illustrative, not measured camera performance.

Static layout elements are cached by presentation size (four entries maximum).
Camera skeletons render after preview scaling; the persistent canvas is resized
directly into a display buffer before adding a fixed-size, high-contrast cursor.
No preview operation changes stroke coordinates, saved pixels, or export resolution.
Existing antialiased stroke interpolation is preserved.
The native window receives a useful initial size once; subsequent user resizing
is preserved. The FPS badge measures frames over elapsed time across the latest
30 intervals, avoiding inflated values when camera delivery arrives in bursts.

Local render-only benchmark against dd16f99, synthetic 640x480 camera with one hand,
20 warmup frames and median of five batches of 100 frames:

| Canvas | Normal before / after | Debug before / after |
| --- | --- | --- |
| 640x480 | 5.72 / 1.14 ms | 5.87 / 1.42 ms |
| 1920x1080 | 6.62 / 2.07 ms | 6.60 / 2.27 ms |

These measurements exclude capture, inference, and native window presentation and
are machine-dependent. The specialist's initial camera attempt failed to return
a frame. After integration, the live camera/window, D/R/S, C/SPACE missing-pointer
feedback, and Escape cleanup were checked successfully. Physical hand interaction
still needs manual acceptance; see the [release report](../../../docs/POLISH_RELEASE.md).
