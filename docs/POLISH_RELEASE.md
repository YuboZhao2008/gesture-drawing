# Gesture Control polish and performance release

This is the original release record. See the [final integration audit](FINAL_INTEGRATION.md)
for subsequent validation and capture-mode trials; the measurements below retain
their original conditions and source revisions.

Validated on September 16, 2026. The integrated application has a redesigned dark
OpenCV interface, faster rendering/event polling, latest-frame capture, and adaptive
pointer smoothing. No dependencies were added, and recognition geometry and public
source-coordinate contracts were preserved.

## Launch

From PowerShell:

```powershell
# From the repository root:
.\.venv\Scripts\python.exe -m gesture_control
```

Normal mode keeps the skeleton visible. D toggles advanced diagnostics. C starts
calibration, SPACE captures a corner, R clears, S saves, and ESC exits. Saved user
calibration and drawings were not overwritten during validation.

## Live before/after measurements

Five alternating trials per version, each with 960 processed frames, 60 warmup
frames discarded, and 900 retained. All ten runs completed successfully. Values
below are medians across the five run-level measurements, not pooled percentiles.

| Metric | Original integrated prototype | Final integrated runtime |
| --- | ---: | ---: |
| Effective FPS | 20.33 | 29.03 |
| FPS range across trials | 19.92–20.47 | 28.94–29.31 |
| Mean frame duration | 49.19 ms | 34.44 ms |
| Frame duration p95 | 67.47 ms | 60.46 ms |
| Mean camera read/wait | 13.24 ms | 7.00 ms |
| Mean detection including inference | 22.06 ms | 21.89 ms |
| Mean UI rendering | 9.84 ms | 3.39 ms |
| Mean native key/event pump | 3.26 ms | 1.32 ms |

This is about 43% more processed frames per second and 66% less rendering time.
Inference cost was essentially unchanged. Occasional long frames remain: the
camera and driver still deliver unevenly, and the new pipeline is not a guarantee
of perfectly even 30 FPS.

Conditions: camera 0, MSMF, negotiated 640×480 at 30 FPS, CPU XNNPACK,
1920×1080 drawing surface, native OpenCV windows, the same model and root virtual
environment. Python 3.12.10, OpenCV 4.14.0, MediaPipe 0.10.35, NumPy 2.5.3.
Each version used its default UI mode. All 9,600 input frames in this comparison
had **no detected hands**. These measurements establish live capture/inference/UI
costs, not moving-hand accuracy, gesture feel, or optical motion-to-photon latency.
No webcam images were recorded.

Baseline source was archived from `dd16f99ff585027ef09a71f1d9b3b3bd8332e94b`
without changing any real worktree. It received the same timing hooks and a
benchmark-only initial window resize so a collapsed native window could not
distort the comparison. Final runtime was
`10bc59935cbce29bcb31df93e79776616d9177dd`; subsequent edits were documentation,
a profiling docstring clarification, and stronger camera-setting assertions.
Profiling is opt-in and adds overhead to both sides. Nested stages are inclusive;
do not add their durations together.

Raw timing reports, source-file hashes, environment metadata, and aggregate
statistics are saved in [performance/2026-09-16](performance/2026-09-16).

## Capture defaults and remaining jitter

Three subsequent trials of the final runtime with `--synchronous-camera` measured
27.02 median FPS (26.44–27.11), 37.00 ms mean frame duration, and 56.13 ms median
frame p95. These were follow-up runs, not interleaved pairs with latest mode.
Latest mode improves throughput and freshness here, but does not win every frame
variability metric. Synchronous mode remains available for troubleshooting.

The vision specialist's separate slow-consumer probe measured a backend sample-age
proxy decreasing from 97.90 to 74.59 ms mean, and 153.46 to 104.95 ms p95, with
a 60 ms simulated consumer delay. Its timestamp was not independently calibrated
to sensor exposure, so this is **not an optical latency measurement**. See the
[vision report](../src/gesture_control/vision/PERFORMANCE.md) for methodology,
negotiated resolutions, repeated camera images, and backend comparisons.

The default retains 640×480/30 requests and original model thresholds. One worker
owns native camera open/read/release and replaces one unconsumed frame slot; IDs
may skip while timestamps still identify acquisition time. RGB conversion storage
is reused without changing source coordinates. Shutdown waits for a native read
to return; a hung device driver can still delay close.

## Deterministic motion and drawing

Five alternating synthetic runs per version used moving POINT/PINCH landmarks,
real recognition/normalization, calibration, drawing, and rendering. Each retained
900 frames after 60 warmup frames, with diagnostics off and a 1920×1080 canvas.
They exclude the webcam, MediaPipe inference, native windows, and event pumping.

| Synthetic downstream metric | Before | After |
| --- | ---: | ---: |
| Mean complete downstream frame | 9.26 ms | 2.68 ms |
| Frame p95 | 13.52 ms | 3.54 ms |
| Mean rendering | 8.83 ms | 2.37 ms |
| Processing capacity | 107.87 frames/s | 372.74 frames/s |

Those capacity values are not live webcam FPS. Smaller submillisecond stage
differences are not claimed as independent causal improvements.

The seeded trajectory comparison was rerun at 30 Hz, 60 Hz, and irregular frame
intervals. At 30 Hz, stationary RMS error improved from 2.72 to 2.36 equivalent
pixels at a 1920-pixel source width; fast-motion estimated filter lag decreased
from 60.97 to 8.31 ms. Slow-motion estimated lag remains about 53–58 ms across the
tested cadences. These are synthetic filter metrics, not end-to-end camera latency.
The default One Euro filter uses minimum cutoff 1.5 Hz, beta 20, derivative cutoff
1 Hz. Settings are exposed in `[smoothing]`, with the legacy alpha filter and
unfiltered mode still available.

PINCH start/release evidence now has a 40 ms dwell cap, accepted at the next frame;
initial/new-track acquisition protection remains. A rejected one-frame pinch
candidate no longer leaves a latch that can later start drawing from the exit
band. Hand absence immediately ends drawing and never connects strokes across
disappearance. Brief POINT noise tolerance affects cursor continuity only.

Tests retain empty observations/predictions for no hand, UNKNOWN only for visible
unrecognized hands, source coordinates for pointers and palm coordinates only for
classification, timestamp-driven behavior, hand IDs, ordered gesture transitions
before pointers, generic PINCH lifecycle before companion events, and balanced
stroke begin/end events. The existing camera setup assertion was strengthened to
check all four property requests explicitly.

## UI changes

The camera and canvas have clear panel headings, readable status and controls,
tracking/pen/calibration feedback, and a clean normal mode. The skeleton remains
visible while landmark indices, bounds, and extra diagnostic text are optional.
Calibration shows the current corner and progress, disables drawing, rejects a
missing pointer, and supports retrying a failed fourth-corner save.

Static interface elements are cached, the persistent drawing surface is resized
directly for preview, and cursor/stroke antialiasing is retained. Presentation size
never changes saved pixels or calibration coordinates. The first window receives
a useful size without overriding later user resizing. The FPS badge counts frames
over elapsed time rather than averaging instantaneous reciprocal rates.

`pollKey` is used where supported, with one `waitKey(1)` fallback otherwise.
A separate synthetic native-window probe measured approximately 15.32 ms for
`waitKey(1)` versus 0.26 ms for `pollKey` under its paced workload; full application
event-pump values are in the live table above.

The [normal](images/ui-normal.png), [debug](images/ui-debug.png), and
[calibration](images/ui-calibration.png) screenshots are clearly labeled synthetic
previews. Their illustrative 52 FPS badge is not a benchmark result.

## Validation and live checks

All commands used the root project's virtual environment and returned exit code 0:

| Command | Result |
| --- | --- |
| `.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_tmp` | 292 passed, no skips |
| `.\.venv\Scripts\python.exe -m ruff check .` | All checks passed |
| `.\.venv\Scripts\python.exe -m ruff format --check .` | 60 files already formatted |
| `.\.venv\Scripts\python.exe -m mypy` | No issues in 51 source files |
| `.\.venv\Scripts\python.exe -m pip check` | No broken requirements |

The live keyboard check processed 3,356 frames, showed the correctly sized normal
window and NO HAND state, toggled D, cleared with R, saved a 1920×1080 PNG with S
to a scratch path, and showed C/SPACE calibration guidance and missing-pointer
feedback. ESC exited with code 0. A separate unprofiled run processed 2,880 frames
and exited with code 0 through the title-bar Close button. A subsequent headless
30-frame run reopened and released the camera successfully. Keys and native
controls were operated through desktop automation, not a person's physical input.

One attempted reopen while the prior UI still owned the webcam failed to read;
the sequential reopen after confirmed shutdown succeeded. Alt+F4 did not close
that window through automation; the actual Close button did. MediaPipe emitted
nonfatal feedback/projection messages and, on longer runs, an uploader warning.

**Still manual:** sustained skeleton alignment, POINT movement and perceived lag,
PINCH drawing/release, removing and returning a hand mid-stroke, and completing
four-corner calibration with a real hand. Their deterministic behavior is covered
by tests, but no physical hand was present in these final live runs. Single-hand
tracking, heuristic recognition, possible stroke splits on tracking loss, and no
desktop input automation, undo, or eraser remain limitations.

## Integration record

The existing agents were coordinated through their VS Code terminals. Fresh finish
branches were based on the current main `2c51c0c`, which already contained the first
smoothing commit on top of `dd16f99`. Their final changes were merged on main:

| Role | Final specialist commit |
| --- | --- |
| A: performance review/instrumentation | `55767bc2957dd7ba025ff98437c8bb8050cb4526` |
| B: vision performance | `f97d01cf99d3ea2d55088f30c20532bd03b22f74` |
| C: interaction safety/configuration | `069ca1fdd48f18fea5f1d6f2d3ae90ee177db43f` |
| D: UI polish | `ac15d101a9b34a3e2418203b4a3417f52c877045` |

Integration reconciled runner options, profiling around nonblocking polling,
worker-safe camera metadata, debug configuration, documentation, and final window
and FPS regressions. All real worktrees were preserved. No remote push or release
publication was performed.
