# Performance review and benchmark procedure

This is the initial architect's review. The changes it recommends have since been
integrated; see [the release measurements and validation](docs/POLISH_RELEASE.md)
for the final implementation, repeated comparisons, and remaining manual checks.
The subsequent [final integration audit](docs/FINAL_INTEGRATION.md) includes
alternating capture-mode trials and delivery telemetry from the integrated UI.

Review target: `dd16f99ff585027ef09a71f1d9b3b3bd8332e94b` on Windows, Python
3.12.10, OpenCV 4.14.0, MediaPipe 0.10.35, NumPy 2.5.3. Camera backend reported
MSMF, 640x480, 30 FPS, buffer size 1; OpenCV reported 22 threads. A reported
buffer size does not establish sensor-to-application frame age.
Model SHA256: `FBC2A30080C3C557093B5DDFC334698132EB341044CCEE322CCF8BCF3607CDE1`.

Only opt-in instrumentation, a synthetic benchmark, tests, and this report were
added. Specialist implementations were not edited. While this review was running,
the worktree was externally switched at 17:02:19 local time to
`agent/perf-review-finish`, based on `2c51c0c` (new temporal smoothing). All four live
runs and the first four synthetic timing runs below completed before that switch.
The separate behavioral comparison explicitly loads the baseline temporal source
and compares it with the new implementation; do not mix those two revisions.

## Measurements

Each live run captured 330 frames, discarded the first 30, and retained 300.
No camera images were saved. These were short, uncontrolled live scenes: the first
visible run and both headless retained samples had no detected hands; the second
visible run had a hand on only 6/300 frames. They establish capture/inference/UI
costs, not sustained hand-tracking accuracy or motion-to-photon latency.

| Stage, mean milliseconds | Headless 1 | Visible 1 | Visible 2 | Headless 2 |
| --- | ---: | ---: | ---: | ---: |
| Camera read | 14.678 | 11.511 | 5.387 | 11.180 |
| Detection including inference | 13.676 | 17.944 | 10.134 | 20.520 |
| MediaPipe inference alone (nested) | 13.374 | 17.587 | 9.862 | 20.127 |
| Rendering | 7.644 | 8.337 | 6.546 | 9.290 |
| imshow | N/A | 0.365 | 0.278 | N/A |
| waitKey(1) | N/A | 8.774 | 10.745 | N/A |
| Total frame | 36.203 | 47.253 | 33.375 | 41.248 |
| Total frame p95 | 53.832 | 63.445 | 46.839 | 55.791 |
| Effective FPS | 27.62 | 21.16 | 29.96 | 24.24 |

Visible run 1 waitKey p95 was 15.916 ms, max 89.261 ms; run 2 p95 was 14.691 ms,
max 16.545 ms. Capture maxima across these runs were 42-62 ms. These costs are
serial in the current loop. The difference between run totals is not a controlled
estimate of window overhead: scene, inference, scheduling, and load varied.

The deterministic downstream benchmark uses real recognition, normalization,
temporal processing, calibration, drawing, and rendering with synthetic moving
POINT/PINCH landmarks. It excludes camera, MediaPipe, windows, and event pumping.
960 frames, 60 warmup, 900 retained; timestamps advance at 30 Hz without wall pacing.

| Stage | Mean ms | p95 ms |
| --- | ---: | ---: |
| Recognition, including normalization | 0.157 | 0.214 |
| Normalization (nested) | 0.091 | 0.126 |
| Temporal | 0.052 | 0.071 |
| Mapping including calibration | 0.068 | 0.094 |
| Calibration (nested) | 0.046 | 0.063 |
| Drawing command execution, summed per frame | 0.014 | 0.033 |
| Rendering | 9.381 | 11.587 |
| Canvas copy/cursor rendering (nested) | 1.426 | 1.895 |
| Two resizes (nested) | 1.373 | 1.761 |
| Two background buffer fills (nested) | 4.645 | 5.831 |
| Skeleton (nested) | 0.279 | 0.401 |
| Text rasterization, static and diagnostic (nested) | 0.315 | 0.441 |
| Total downstream frame | 9.836 | 12.149 |

Timings are inclusive: do not sum nested stages. Missing stages contribute zero to
per-frame means; active frame and call counts are included in JSON. Calibration and
drawing ran on 880/900 frames because temporal transitions suppress pointer output.

Exploratory A/B results (one run each, not statistically controlled): diagnostics off
reduced total from 9.836 to 9.165 ms; a 640x480 workspace reduced it to 8.113 ms.
The smaller workspace also changes layout aspect/size, so that is not a pure canvas
copy comparison. Full stage instrumentation versus frame-only timing was 9.836 vs
9.319 ms, an observed 0.517 ms difference including run variance. Treat submillisecond
stage results as approximate; run repeated alternating trials before claiming gains.

Raw local reports are under ignored `recordings/perf-*.json`. They are not committed.

## Ranked findings and assignments

1. **Interaction gating and smoothing cause perceived lag independently of CPU.**
   Baseline temporal processing emits a pointer only when instantaneous evidence
   equals the stable POINT/PINCH state. UI hides the cursor when no pointer arrives.
   At 30 Hz, the synthetic POINT-to-PINCH change takes 133.3 ms to accept and creates
   a 166.7 ms gap between pointer samples. Initial acquisition is gated for 200 ms;
   dwell runs concurrently, not additively. Default EMA alpha=0.35 at 30 Hz implies
   a 77.4 ms time constant, approximately 178 ms continuous-time 90% step response,
   and 61.9 ms steady ramp lag at 30 Hz. These are filter calculations, not measured
   sensor-to-screen latency. Recognition/Temporal should separate cursor continuity
   from stroke permission, shorten pinch-specific dwell, and evaluate adaptive
   smoothing while preserving immediate missing-hand stroke release.

   **Already arriving:** `2c51c0c` adds adaptive filtering, 40 ms pinch dwell caps,
   and bounded cursor tolerances. The same synthetic sequence now accepts pinch
   in 66.7 ms, with a 66.7 ms maximum pointer gap; initial acquisition remains 200 ms.
   Validate this work rather than reimplementing it. Its adaptive defaults are not
   configured by core alpha alone; the live runner needs an explicit, documented way
   to select/tune the new filter if configuration experiments are required.

2. **Blocking capture, synchronous inference, and HighGUI share one loop.**
   Camera reads and inference block event handling; waitKey adds another measured
   9-11 ms on average. There is no application-owned latest-frame policy. Timestamping
   happens after read, so queued sensor age cannot be measured from FrameContext.
   Vision should expose actual backend/capture telemetry, evaluate a single-slot
   latest-frame capture worker with overwrite counts and bounded shutdown, and
   preserve chronological frame contexts. Do not assume setting a buffer property
   guarantees freshness. UI should benchmark pollKey where supported, retaining key
   processing/window close behavior and avoiding a busy loop. Do not remove event
   pumping or replace it with waitKey(0).

3. **Full UI composition wastes several milliseconds every frame.**
   UI should cache static backgrounds/text and reuse layout buffers, render the
   cursor in the display-sized layer where possible, and avoid resizing when source
   and destination sizes match. Preserve the 1920x1080 drawing/export resolution.
   The canvas is copied (6.22 MB) and downsampled every frame; preview is copied and
   resized; new left/panel buffers are filled each frame. This is a better target
   than redesigning graphics or removing diagnostics, which saved only ~0.67 ms here.

4. **MediaPipe inference is the dominant computation and varies by run.**
   Measured means range 9.9-20.1 ms, with live p95 up to 26.2 ms. Vision should use
   a fixed recorded input when comparing resolution, backend, or confidence settings.
   VIDEO mode, one hand, and one-time model construction are already used. BGR-to-RGB
   conversion averaged only 0.055-0.084 ms; bypassing color correctness is not useful.
   Evaluate thread scheduling/oversubscription (22 OpenCV threads were reported)
   experimentally; do not impose a global thread count without a controlled trial.

5. **Tracking identity loss can turn a one-frame miss into a long interruption.**
   The baseline detector clears its previous hand on any empty result. On return it
   allocates a new ID, so new-track gating applies even if temporal logic supports
   short same-ID dropouts. Vision should test bounded identity continuity. A reused
   ID must never resume an old stroke across absence or migrate to another hand.

6. **Recognition arithmetic, drawing primitives, and calibration are low priority.**
   There is one normalization per observed hand and one pipeline/render pass per
   frame. No repeated model construction or per-frame calibration solve was found.
   The homography is solved once; per-pointer mapping is tiny. Calibration can still
   amplify spatial noise when the calibrated region is small or perspective gain is
   high. Keep all smoothing/coordinate units explicit when tuning it.

## Targets and integration risks

Initial engineering targets, not claimed measurements: sustained 30 camera/interaction
updates per second; frame p95 <=40 ms and p99 <=50 ms; cursor motion-to-photon median
<=80 ms and p95 <=120 ms; accepted pinch start/release p95 <=150 ms. Release on observed
hand loss must remain immediate. Revisit acquisition gating separately from continuous
motion. Treat 60 FPS as a later target requiring verified camera support and a 16.7 ms
budget, not merely a config change.

Capture workers introduce ownership, shutdown, and stale-frame risks. Keep queue
capacity one, preserve timestamps of consumed frames, report drops, and keep GUI work
on its owning thread. Skipped observations can cross loss/reacquisition thresholds.
Do not confuse `read()` return time with exposure time. UI canvas caches must invalidate
on clear, resize, strokes, calibration, and status changes without changing export pixels.
Continuing pointer events during gesture ambiguity must not accidentally extend drawing:
the current mapper emits stroke points whenever its pinch state remains active.
Per-frame PNG saves and calibration file writes happen synchronously on key actions;
measure those spikes separately from steady-state performance.

## Reusable instrumentation

`PerformanceMonitor` is standard-library-only, bounded to the latest 9000 successful
post-warmup frames, single-threaded, and writes nothing per frame. It reports mean,
nearest-rank p50/p95/p99/max, stage call counts, slow-frame counts, and wall effective
FPS. Startup is excluded. Failed frames are counted but excluded from distributions.
The runner writes a report only on successful completion; programmatic callers can
inspect the monitor after exceptions.

`profiling.instrument_pipeline` installs scoped wrappers and restores them on exit,
including exceptions. It modifies no specialist source. OpenCV, NumPy, calibration,
and slotted mapper hooks are temporarily process-wide: do not use concurrently with
another pipeline. It records source file hashes, backend properties, and thread count.
The inference substage optionally accesses the current backend's private method;
total detection timing remains the public boundary to preserve in later backends.

## Exact post-improvement benchmark procedure

1. Use an isolated checkout at a recorded SHA (`git rev-parse HEAD` and
   `git status --short`). Record model SHA256, config, calibration, dependency versions,
   camera backend/settings, power mode, screen refresh, and CPU. Close competing camera
   apps and stop builds/tests during measurement. Use the same scene, lighting, model,
   and hand sequence for comparisons. Never benchmark across a concurrent checkout.
2. Install `python -m pip install -e ".[dev,vision,ui]"`; run all tests, Ruff lint/format,
   mypy, and `pip check`. Use module imports to verify no resource opens at import time.
3. Run the following with the actual model path. Each run retains 900 frames after
   60 warmup frames. The normal run displays a window; do not press ESC early unless
   testing shutdown. Headless still renders but does not pump HighGUI events.

   ```powershell
   python -m gesture_control --model models/hand_landmarker.task --max-frames 960 --profile-warmup 60 --profile recordings/perf-window.json
   python -m gesture_control --model models/hand_landmarker.task --headless --max-frames 960 --profile-warmup 60 --profile recordings/perf-headless.json
   python scripts/benchmark_pipeline.py --calibrated --output recordings/perf-landmarks.json
   python scripts/benchmark_pipeline.py --calibrated --diagnostics off --output recordings/perf-no-diagnostics.json
   python scripts/benchmark_pipeline.py --calibrated --workspace 640 480 --output recordings/perf-small-workspace.json
   python scripts/benchmark_pipeline.py --calibrated --frame-only --output recordings/perf-frame-only.json
   ```

4. Repeat each comparison five times in alternating A/B order; preserve separately
   named output files. Run separate live scenarios: no hand, stationary POINT, smooth
   moving POINT, sustained PINCH drawing, repeated pinch transitions, disappearance/
   return, calibrated movement. Reuse a fixed recorded image sequence for inference
   A/B tests once the vision harness supports it. Synthetic downstream FPS is not live FPS.
5. Compare distributions and pointer sample gaps, not just the smoothed on-screen FPS.
   For deterministic landmarks, compare transition acceptance, pointer RMS noise at
   rest, moving-pointer lag, false starts/ends, stroke continuity, and missing-hand
   release at 15/30/60 Hz timestamps plus irregular intervals and dropped observations.
   Reject speedups that create ghost strokes or connect across absence.
6. Measure physical motion-to-photon separately: film the moving hand and displayed
   cursor together at 240 FPS (4.17 ms/video frame), score at least 20 starts/stops and
   pinch transitions, and report median/p95 plus camera/display conditions. FrameContext
   timestamps and imshow return time alone cannot measure exposure-to-display latency.
7. Repeat ESC mid-stroke, camera disconnect, unavailable webcam, window close, save,
   and calibration capture. Verify prompt cleanup, no held stroke, and that the camera
   reopens. Record event-pump gaps/worker shutdown bounds if introducing concurrency.
