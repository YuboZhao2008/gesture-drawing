# Final integration audit

Historical September 16 audit. See the [September 17 stability release](STABILITY_RELEASE.md)
for the current validated product and UI.

Validated September 16, 2026, using the root project's `.venv\Scripts\python.exe`.
The starting main was `2182536f5761a056e5e8c038e838fbda34f29c80`, clean. It already
contained all specialist commits and the earlier integration fixes. The original
working prototype was `dd16f99ff585027ef09a71f1d9b3b3bd8332e94b`.

Work took place on `agent/polish-integration`, created from current main. No reset,
history rewrite, specialist branch edit, agent delegation, worktree removal, or
remote push was performed. All four finish commits were verified as ancestors:

| Specialist | Integrated commit |
| --- | --- |
| A: instrumentation | `55767bc2957dd7ba025ff98437c8bb8050cb4526` |
| B: capture | `f97d01cf99d3ea2d55088f30c20532bd03b22f74` |
| C: temporal | `069ca1fdd48f18fea5f1d6f2d3ae90ee177db43f` |
| D: UI | `ac15d101a9b34a3e2418203b4a3417f52c877045` |

No specialist implementation was removed in this audit. The existing integration's
clean diagnostics default overrides B's original diagnostics-on default. Its runner
already reconciles camera width/height, latest-frame selection, debug overrides,
profile output, and warmup through one application argument parser. Re-merging
these ancestor commits would make no changes, so no redundant merges were created.

## Final changes and defaults

Added opt-in capture-delivery telemetry to the existing profiler: acquisition FPS
from delivered IDs and acquisition timestamps, skipped IDs, and host delivery age.
It excludes capture warmup and keeps at most 9000 contexts, with no image storage.
Skipped counts exclude acquisitions before the first retained delivery and after
the last; they count superseded application frames, not missing sensor exposures.
Delivery samples can include a frame whose downstream processing subsequently
fails; completed-frame timing separately excludes failed frames. Hooks restore on
normal exit and exceptions. This instrumentation is never installed when profiling
is off; production capture and GUI threading are unchanged.

Added integration coverage for profile/warmup CLI forwarding and JSON output,
bounded delivery telemetry, skipped frame IDs with acquisition timestamps,
adaptive filtering, immediate hand-loss release, and a new hand ID starting an
isolated stroke. Existing tests retain no-hand versus UNKNOWN semantics,
palm-normalized classification, source-image pointers, calibration mapping,
ordered pinch lifecycle, constructor overrides, legacy EMA, and filter bypass.

Final defaults remain **640x480 requested at 30 FPS, latest-frame capture,
diagnostics off, and adaptive One Euro (1.5 Hz minimum cutoff, beta 20, 1 Hz
derivative cutoff)**. `--synchronous-camera` retains the synchronous option.
The new measurements show equivalent throughput in the usual UI workload, with
no meaningful latest-frame penalty. Prior loaded-UI and slow-consumer evidence
supports retaining the worker when processing falls behind. This is a conservative
choice, not a claim that latest-frame always improves frame regularity.

The UI retains cached static layout, direct presentation-size canvas rendering,
same-size resize avoidance, readable minimum panel sizes and letterboxing,
calibration retry/recovery, a useful initial window size, and elapsed-time FPS.
No further rendering optimization was justified by the approximately 2 ms cost.
No second smoothing layer was added. ESC, C, SPACE, D, R, and S remain supported.

## Integrated camera and UI measurements

Six standard native-window runs alternated synchronous/latest three times. Every
run used camera 0, MSMF, 640x480/30 request and negotiation, the same MediaPipe
detector/model and thresholds, a 1920x1080 uncalibrated canvas, diagnostics off,
60 warmup frames, and 900 retained frames. No artificial consumer delay was added.
OpenCV reported that setting buffer size 1 was unsupported, despite readback 1.
The model SHA256 was
`FBC2A30080C3C557093B5DDFC334698132EB341044CCEE322CCF8BCF3607CDE1`.

Hand presence was uncontrolled: synchronous trials detected hands in 736, 0, and
398 of 960 input frames; latest trials in 14, 0, and 0. These observations are not
ground-truth gesture labels. Brief test execution overlapped trial 3, so neither
mixed-scene aggregate nor trial 3 is a controlled causal timing comparison.
All raw reports are preserved, including these limitations.

The second pair had **no detected hands in either run** and no concurrent tests.
Its matched UI measurements are shown below. Each cell is mean / median / p95,
in milliseconds. Percentiles are per-run nearest-rank values, not pooled.

| Stage | Synchronous | Latest-frame |
| --- | ---: | ---: |
| Capture wait | 19.53 / 19.38 / 37.41 | 19.59 / 21.14 / 38.58 |
| Detection including inference | 10.58 / 9.98 / 16.17 | 10.48 / 10.02 / 13.61 |
| Whole frame, including capture and presentation | 33.35 / 32.67 / 52.98 | 33.35 / 34.64 / 53.42 |
| UI render | 1.98 / 1.86 / 2.61 | 1.98 / 1.88 / 2.59 |
| Native pollKey | 0.78 / 0.73 / 1.06 | 0.81 / 0.74 / 1.14 |

That pair measured **29.984 versus 29.981 processed/presented FPS**, with
**29.998 versus 29.998 host acquisition FPS**, and **0 versus 0 skipped IDs**.
Presentation rate counts completed native presentation calls, not verified monitor
scanout. Native camera timestamps follow host acquisition; neither negotiated FPS
nor host acquisition FPS independently measures exposure cadence.

Across all three trials per mode, median processed FPS was 29.984 synchronous and
29.983 latest. Ranges were 29.984-30.025 and 29.981-29.998 respectively. No standard
trial skipped IDs. Camera delivery remained uneven, despite near-30 throughput.
The matched pair's host acquisition-to-delivery mean / median / p95 was
0.043 / 0.044 / 0.063 ms synchronous and 4.060 / 0.107 / 12.658 ms latest.
This metric starts after native read returns, so it cannot reveal synchronous
sensor/driver backlog and must not be used as an optical latency comparison.

Two separate exploratory native-window probes read CAP_PROP_POS_MSEC on the
camera-owning thread and subtract it from the host clock at delivery:

| Backend sample-age proxy | Synchronous | Latest-frame |
| --- | ---: | ---: |
| Mean | 65.36 ms | 49.12 ms |
| Median | 56.71 ms | 46.98 ms |
| p95 | 121.60 ms | 67.42 ms |
| Processed FPS | 28.56 | 29.98 |

Both used the same 60/900 warmup/sample counts and UI configuration, but detected
hands in 7 versus 630 input frames. Querying the extra native property also adds
work, especially on the synchronous thread. These are supporting observations,
not a controlled causal estimate. Sample timestamps were nondecreasing; their
alignment to exposure and the host clock was not independently calibrated.
**Sample-age proxy is not optical motion-to-photon latency.** The specialist's
separate 60 ms slow-consumer probe remains historical evidence: mean 97.90 to
74.59 ms, p95 153.46 to 104.95 ms.

## Event polling and previous baseline

A separate latest-frame run selected the production waitKey(1) fallback with the
same configuration and 60/900 samples. It measured 28.59 FPS, 43 skipped frame IDs,
and event polling mean / median / p95 of 2.98 / 1.29 / 13.65 ms. Whole-frame p95
was 59.22 ms. It detected hands in 2/960 frames. The normal pollKey trial's much
lower event-poll cost supports retaining pollKey with the existing availability
fallback. GUI calls remain on the runner thread. Keyboard and window-close
branches are tested with native APIs mocked; see live-check limitations below.

The earlier repeated comparison against `dd16f99` is retained in
[POLISH_RELEASE.md](POLISH_RELEASE.md): 20.33 to 29.03 FPS, mean frame 49.19 to
34.44 ms, frame p95 67.47 to 60.46 ms, render 9.84 to 3.39 ms, and mean event
poll 3.26 to 1.32 ms. Those are historical same-session before/after measurements.
They must not be combined with this audit's current timings as a new controlled
baseline comparison. No new physical smoothness or gesture accuracy claim is made.

The seeded filter benchmark was rerun. At 60 Hz, stationary RMS error is
1.718 to 1.585 equivalent pixels at a 1920-pixel reference width; estimated fast
motion lag is 69.8 to 10.1 ms. Irregular timing gives 2.161 to 1.892 pixels and
64.5 to 9.2 ms. Slow motion retains about 53-58 ms estimated lag. The synthetic
POINT-to-PINCH pointer gap remains 66.7 ms versus the original 166.7 ms, with
200 ms initial acquisition protection. These are synthetic temporal metrics.

## Validation and hardware checks

The full required suite passed after the final edits:

| Command using `.\.venv\Scripts\python.exe` | Result |
| --- | --- |
| `-m pytest --basetemp=.pytest_tmp` | 296 passed |
| `-m ruff check .` | All checks passed |
| `-m ruff format --check .` | 64 files already formatted |
| `-m mypy` | No issues in 51 source files |
| `-m pip check` | No broken requirements |

The initial sandboxed targeted test setup could not recreate `.pytest_tmp`;
rerunning with filesystem approval passed without changing tests. Four regression
tests were added to the starting main's 292 tests.

This audit completed **8,700 camera frames**: nine 960-frame profiled native-window
runs, one unprofiled 30-frame native-window run, and a subsequent unprofiled
30-frame headless reopen. Every process exited 0; MediaPipe initialized, native
windows were enumerated, rendering/presentation returned successfully, and the
camera reopened after cleanup. No webcam images, user calibration, or drawings
were saved or overwritten. Nonfatal MediaPipe feedback/projection warnings remain.

The computer-use helper failed twice during state capture with
`foreground window did not report a process id`. Therefore this audit does **not**
claim fresh screenshot inspection, native ESC/Close-button testing, or D/R/S/C/SPACE
interaction. Earlier successful native key and Close-button checks are documented
separately in the original release report; automated regression tests still pass.
No controlled physical hand interaction was available to this engineer. Unlabelled
hand detections during some runs do not validate skeleton alignment or gestures.

**Manual acceptance remains:** skeleton alignment, POINT feel, PINCH drawing and
release, disappearance/reappearance safety, completing four-corner calibration,
and repeating native keyboard/Close-button checks. Known limitations include
single-hand heuristic recognition, tracking-loss stroke splits, uneven camera
delivery, driver-dependent read/shutdown delays, slow-motion filter lag, and no
desktop input automation, undo, or eraser.

## Reproduction and evidence

Launch normally from the repository root:

```powershell
.\.venv\Scripts\python.exe -m gesture_control
```

For each standard mode, alternate three runs with distinct output names:

```powershell
.\.venv\Scripts\python.exe -m gesture_control --no-diagnostics --max-frames 960 --profile-warmup 60 --profile recordings/sync.json --synchronous-camera
.\.venv\Scripts\python.exe -m gesture_control --no-diagnostics --max-frames 960 --profile-warmup 60 --profile recordings/latest.json
```

[Raw reports and probe harnesses](performance/2026-09-16-final) include source
hashes, dependency versions, full stage distributions, and delivery metadata.
Run the optional `final_sample_probe.py` or `final_waitkey.py` harness with the
same application CLI arguments; the sample probe assumes 60 warmup deliveries.
These harnesses do not introduce a second production argument parser. No network
access or image recording is needed. Stop other tests and workloads and control
the scene for a new causal comparison.

The final integration commit is titled `Polish gesture control performance and interface`;
local main is fast-forwarded only after the checks above. Its exact resulting SHA
and clean status are reported by the release engineer after the Git operation.
