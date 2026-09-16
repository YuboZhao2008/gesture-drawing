# Vision performance pass — 2026-09-16

The change reduces stale-frame delay under a slow consumer, but **does not
demonstrate stable 30+ processed FPS or faster hand tracking on this webcam**.
The device negotiated 30 FPS for every 30/60 FPS request. All measured detector
frames contained **no hands**. Gesture accuracy, moving-hand smoothness, and
optical camera-to-landmark latency remain unmeasured.

Baseline: `2c51c0c` (its vision implementation is unchanged from the initial
`dd16f99` baseline). Final branch: `agent/vision-performance-finish`. No root
checkout edits or merges were made. Machine-readable measurements are in
[performance-results.json](performance-results.json).

## Method and results

Windows, camera 0, MSMF, CPU XNNPACK, Python 3.12.10, OpenCV 4.14.0, MediaPipe
0.10.35. Final comparisons used the **root project's .venv Python**, with
`PYTHONPATH` set to this worktree's `src`. Baseline camera/detector classes were
loaded from `git show 2c51c0c:...` into temporary modules; the surrounding pipeline
and measurement code were identical. Baseline and candidate runs alternated at
each setting. Each had 30 warmup frames followed by 240 measured frames. Model
SHA256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.
No camera images were saved. These are short hardware samples, not long-run
stability guarantees; inference differences of a few tenths of a millisecond
should not be treated as a causal speedup/regression.

| Requested resolution/FPS | Actual resolution/FPS | Capture FPS before → after | Processed detector FPS before → after | Mean detector ms before → after | Mean vision frame ms before → after |
|---|---|---:|---:|---:|---:|
| 640×480 / 30 | 640×480 / 30 | 30.06 → 30.06 | 30.06 → 28.63 | 10.82 → 11.14 | 33.26 → 34.93 |
| 640×480 / 60 | 640×480 / 30 | 30.07 → 30.06 | 30.07 → 28.63 | 11.26 → 10.96 | 33.25 → 34.93 |
| 960×540 / 60 | **800×600** / 30 | 30.05 → 30.06 | 30.05 → 28.63 | 12.17 → 12.51 | 33.27 → 34.93 |
| 1280×720 / 60 | 1280×720 / 30 | 30.06 → 30.03 | 30.06 → 28.60 | 12.74 → 13.16 | 33.26 → 34.96 |

Capture FPS counts all successful acquisitions; processed FPS counts completed
detector calls per wall time. Detector capacity at 640×480/30 was 92.38 → 89.76
calls/s, the inverse of mean detector-call time, **not** achieved tracking FPS.
Vision frame time runs from requesting a frame through receiving landmarks,
including camera wait. At 640×480/30 its p95 was 66.36 → 69.61 ms. This device's
bursty delivery remains a limiting factor; latest mode deliberately skips
superseded frames rather than processing a backlog to inflate throughput.

Full integrated app, headless **with rendering**, 640×480/30, same warmup/sample
count and existing diagnostics enabled:

| Metric | Before | After |
|---|---:|---:|
| Application FPS (interval between read requests) | 30.10 | 28.68 |
| Mean detector latency | 11.64 ms | 11.69 ms |
| Mean UI rendering | 7.31 ms | 7.33 ms |
| Mean camera wait | 14.17 ms | 15.74 ms |

No native display window was opened in these runs, so display/vsync costs are
excluded. Two earlier integrated attempts failed at the first webcam read; the
exclusive-camera final pass above succeeded for both implementations.

## Freshness evidence and tradeoff

`CAP_PROP_BUFFERSIZE=1` returned **False** under MSMF, despite get() returning 1.
An initial 500 ms consumer stall was followed by several reads as short as 2–4
ms, motivating a continuously draining single-slot worker. The setter/readback
distinction agrees with [OpenCV's backend-dependent property documentation](https://docs.opencv.org/4.13.0/d4/d15/group__videoio__flags__base.html).

A separate controlled hardware probe compared synchronous and latest-slot
capture at 640×480/30. Each consumed 110 frames, with a 60 ms sleep before each
of the final 80 reads; no detector ran in this probe. `CAP_PROP_POS_MSEC` on this
MSMF device returned increasing sample timestamps numerically aligned with the
host performance-counter clock. We calculated
`perf_counter_at_read_return * 1000 - sample_timestamp_ms` as a **sample-age
proxy**. The similar unstalled means (60.11 vs 59.00 ms) support comparing it
within this session. The sensor's actual exposure instant/clock calibration was
not independently verified, so this is not an optical latency measurement or a
portable guarantee about CAP_PROP_POS_MSEC.

| Delayed-consumer metric | Synchronous | Latest slot |
|---|---:|---:|
| Mean sample-age proxy | 97.90 ms | 74.59 ms |
| p95 sample-age proxy | 153.46 ms | 104.95 ms |
| Maximum sample-age proxy | 167.34 ms | 113.11 ms |
| Acquisitions by last consumed frame | 110 | 176 |
| Consecutive byte-identical delivered frames, all 110 samples | 39 | 22 |

That is a 23.30 ms mean and 48.51 ms p95 improvement under this controlled stall.
It justifies latest mode for freshness, at the measured cost of about 1.4 fewer
processed frames/s in the unstalled application. The production worker stores
only one unconsumed frame, keeps draining while UI/inference run, and never
replays a consumed slot. Backend/sensor buffering and repeated camera images
still exist; the app cannot guarantee zero physical frame age.

Host acquisition-to-result age in the regular benchmark is a different metric:
at 640×480/30 it was 10.87 → 15.36 ms. It excludes all time before the native read
returns, so it **cannot** expose synchronous driver backlog. The baseline also
used a coarser Windows monotonic clock. Do not substitute this host metric for
the sample-age proxy or claim either is optical end-to-end latency.

DirectShow was also tested, with 30 warmup + 150 measured frames in each mode:
14.91 FPS for both synchronous and latest capture, versus MSMF's 30.20 and 28.96
processed FPS in that comparison. No backend override was introduced.

## Implementation and retained behavior

- Request a one-frame backend buffer and expose negotiated settings/support.
- Add opt-in `OpenCVCamera(latest_frame=True)`; the integrated runner enables it.
  One owning thread opens, reads and releases the device; a condition protects
  one replaceable slot. Reads wait for new data, errors propagate, and close joins
  before reuse. No unbounded frame queue, detached daemon, or native release/read
  race. Shutdown waits for a pending native read; a stuck driver can delay it
  because OpenCV has no portable USB read-cancel operation.
- Keep acquisition-based elapsed monotonic timestamps, with `perf_counter` to
  avoid roughly 16 ms clock quantization on Python 3.12 Windows. Acquired frame
  IDs can skip on delivery but are never renumbered. Core FrameContext is unchanged.
- Reuse one contiguous BGR→RGB destination at a fixed resolution. No extra source
  copy, resizing or crop. MediaPipe's immutable image copy remains required.
- Keep one initialized VIDEO detector, one hand, and the original 0.5 detection,
  presence and tracking thresholds. VIDEO already uses tracking to avoid palm
  detection when possible ([MediaPipe guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)).
  No confidence tuning, hand-ID algorithm, no-hand handling, PalmNormalizer,
  recognition heuristic, or source-coordinate convention changed.
- Debug rendering remains outside vision inference. `--no-diagnostics` selects
  the existing UI switch; D still toggles it. UI layout is unchanged. UI copies
  and preview resizes remain in UI scope; measured rendering cost is reported
  above. No-hand runs do not establish the cost of drawing a full skeleton.
- Add a standalone benchmark with actual dimensions/FPS, warmup exclusion,
  capture FPS, detector FPS/capacity, latency and vision frame-time distributions.

## Recommended runtime and integration points

Use **640×480, requested 30 FPS, latest-frame capture** on this webcam. It preserves
the previous input resolution and has the lowest measured detector cost. This
pass did not measure gesture accuracy, so it does not establish an accuracy
optimum. Keep 1280×720 available for smaller/distant hands and validate accuracy
with the intended user. 960×540 was not supported here. Request 60 FPS on a
different camera only after confirming both negotiated and measured rates; this
hardware provided no 45–60 FPS evidence. `--synchronous-camera` retains the
comparison/fallback path with its freshness tradeoff.

Runner merge points for agent A: added keyword arguments `camera_width`,
`camera_height`, `latest_frame`, `diagnostics`; forward width/height/latest to
`OpenCVCamera`, assign `ui.diagnostics`, and add corresponding CLI flags
`--camera-width`, `--camera-height`, `--synchronous-camera`, `--no-diagnostics`.
No profiling hooks or changes to the frame-processing loop were added. A can
consume `camera.settings`, `camera.captured_frames`, and `camera.stream_time_s`
from its profiling hooks. These do not change the hand-detector/core protocols.

Reproduce candidate measurements from this worktree:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& ../.venv/Scripts/python.exe -m gesture_control.vision.benchmark --width 640 --height 480 --fps 30 --frames 240 --warmup 30
& ../.venv/Scripts/python.exe -m gesture_control.vision.benchmark --width 960 --height 540 --fps 60 --frames 240 --warmup 30
& ../.venv/Scripts/python.exe -m gesture_control.vision.benchmark --width 1280 --height 720 --fps 60 --frames 240 --warmup 30
& ../.venv/Scripts/python.exe -m gesture_control.vision.benchmark --synchronous --processing-delay-ms 60
& ../.venv/Scripts/python.exe -m gesture_control.vision.benchmark --processing-delay-ms 60
```

The last two commands report portable **host** timings, not the MSMF sample-age
proxy above. Local evidence scripts and raw probe samples are retained under the
ignored worktree `recordings/` directory (`final_comparison.py`,
`freshness_probe.py` and their JSON outputs); summarized evidence is committed.

Validation: **221 repository tests passed** with root Python and worktree source;
Ruff check/format and strict mypy passed. Added tests cover stalled-consumer
skipping, independent image ownership, waiting for new data, reader wakeup during
close, same-thread native cleanup, reopen, startup/read/cleanup failures, and RGB
buffer reuse/resolution changes without mutating prior backend images. Webcam
resources were released after all probes.
