# Stable portfolio prototype — September 17, 2026

This pass started from clean main at
`ca6d49298ce81ff43307eba4298d33c0a9b39de9`. The reported baseline was independently
confirmed: 296 tests and all five validation gates passed. The existing architecture,
geometric thresholds, adaptive One Euro filter, capture defaults, and drawing
semantics were retained.

## Product changes

- The camera preview is now secondary: a 400-pixel pane beside the 640-pixel canvas
  preview. Aspect ratios, source coordinates, 1920x1080 default export resolution,
  cached chrome, skeleton, controls, and diagnostics remain intact. All four
  synthetic README screenshots were regenerated and visually inspected.
- Repeated calibration captures are rejected immediately. Invalid fourth corners
  retain the first three points and offer SPACE retry; the previous valid transform
  and saved file remain intact until successful completion.
- Gesture acceptance, first pointer, and held events agree at floating-point
  acquisition boundaries. For example, acquisition at 0.1 seconds with a 0.2-second
  delay now emits the accepted PINCH and its pointer together at 0.3 seconds.
- Camera startup/read errors remain the primary exception when native cleanup also
  fails. Cleanup errors are logged, native release is attempted once, and ordinary
  cleanup-only failures still propagate. Regression coverage includes synchronous
  capture, latest-frame capture, and context-manager unwinding.
- Invalid camera dimensions fail before constructing UI or detector resources.
- A GitHub Actions workflow configures all five gates on Windows and Linux with
  Python 3.12. It requires neither a webcam nor a downloaded model. Hosted execution
  has not been performed because this work was not pushed. Action references follow
  the official [checkout](https://github.com/actions/checkout) and
  [setup-python](https://github.com/actions/setup-python) usage documentation.

## Ownership, review and integration

The lead used GPT-6 Astra with HIGH reasoning and owned shared files, runtime fixes,
review, integration, and validation. Three native GPT-6 Astra specialists worked in
isolated checkouts; historical worktrees were preserved.

| Specialist | Reasoning | Final candidate / reviewed SHA | Outcome |
| --- | --- | --- | --- |
| Interaction reliability | HIGH | `0c12bced3c2db933578dc413dab4b01527940fd6` | Reviewed and merged |
| UI / calibration | MEDIUM | `9afedc04ed5042171700ae934ca7fc2f52fec005` | Revised, reviewed and merged |
| Independent adversarial review | HIGH | Runtime review: `66191668c5f9aef52301623b8d746e2e6cd94b22` | Camera finding reproduced, then verified fixed |

The first UI candidate, `c1cc4219444db3608c97495c76527f94cb081104`, was withheld
from approval after the lead found an old integration-test width expectation.
One targeted revision updated preview geometry and strengthened canvas-position
verification while retaining cursor/export assertions. No specialist work was
rejected. The interaction specialist required no revision.

Lead runtime changes are in `2a35622`; interaction integration is `6619166` and
UI integration is `73cf0a1846933f029fabe614186cd9b339e8f681`. The latter is the
product-code SHA used for the final measurements below. Subsequent release work
updates documentation and evidence only. The local orchestration checkpoint records
full merge SHAs, final review, validation, and final main promotion without embedding
a self-referential final commit SHA here. Specialist branches remain frozen and intact.

## Validation

All commands used the project-local `.venv/Scripts/python.exe` on Windows.
Worktree checks selected and verified their own source imports without changing the
shared environment. The lead independently inspected candidate diffs and reran
70 interaction tests and 87 UI/integration tests, then repeated affected tests after
each sequential merge. The independent reviewer passed 108 targeted tests on the
combined runtime/interaction candidate.

| Final combined command | Result |
| --- | --- |
| `-m pytest --basetemp=.pytest_tmp` | 318 passed; no skips |
| `-m ruff check .` | Passed |
| `-m ruff format --check .` | 75 files already formatted |
| `-m mypy` | Passed; 51 source files |
| `-m pip check` | No broken requirements |

The 22 additional cases cover timing boundaries, calibration recovery and persistence,
camera dual-failure cleanup, pre-resource dimension validation, preview geometry,
and PNG export without cursor pixels. No checks were weakened. A sandbox cache-write
warning during targeted tests disappeared in the authorized full validation run.

## Comparable synthetic performance

Three baseline and three final sequential runs used the same interpreter, dependencies,
1920x1080 workspace, calibration, diagnostics off, and deterministic moving POINT/PINCH
landmarks: 960 frames, 60 warmup, 900 retained, timestamps at 30 Hz, no wall pacing.
No other test or benchmark process ran during these measurements. Camera, inference,
native windows, and event polling are excluded. The UI pane size is intentionally
different; these are whole-product workload comparisons, not isolated algorithm timings.
A greater-than-10% median regression was the investigation threshold.

| Metric | Baseline | Final |
| --- | ---: | ---: |
| Downstream capacity, three runs (FPS) | 515.68 / 517.80 / 513.58 | 500.37 / 510.17 / 512.39 |
| Median capacity (FPS) | 515.68 | 510.17 |
| Median of per-run frame means (ms) | 1.934 | 1.955 |
| Median of per-run frame p95 (ms) | 2.288 | 2.350 |
| Median of per-run render means (ms) | 1.688 | 1.724 |

The observed capacity difference is -1.1%; no speedup is claimed. The small cost is
accepted for the clearer layout and remains well below the camera frame budget.
The interaction-only candidate separately measured 524.5 / 513.3 / 512.3 FPS.
Transition events were identical in the baseline/final synthetic scenario, with
first pointer at 0.2 seconds and maximum pointer gap of 0.100 seconds across both
transition directions. This is not a physical latency measurement.

The seeded filter comparison was rerun unchanged. At 60 Hz, stationary RMS error is
1.718 pixels with EMA versus 1.585 with adaptive filtering; estimated fast-motion lag
is 69.8 versus 10.1 ms at a 1920-pixel reference width. Slow-motion filter lag remains
about 53–58 ms. These synthetic results do not establish physical gesture accuracy.

## Device checks and remaining manual acceptance

The lead exclusively owned camera 0. A final native-window run processed 3,600 frames
and exited 0. MediaPipe initialized and a `Gesture Studio` window was enumerated.
The next headless run reopened the camera, processed 330 frames and exited 0; a
subsequent native-window run again reopened it, processed 330 frames and exited 0.
Total: **4,260 actual camera frames**. The two short profiled runs excluded 30 warmup
frames and retained 300; both detected zero hands in all 330 observations.

| Final short device run | Headless with rendering | Native window |
| --- | ---: | ---: |
| Processed FPS | 30.057 | 30.040 |
| Whole-frame mean / p95 (ms) | 33.266 / 50.544 | 33.285 / 49.221 |
| Render mean (ms) | 2.265 | 2.227 |
| Host acquisition FPS | 30.045 | 30.042 |
| Skipped frame IDs | 0 | 0 |

Both used MSMF, requested/negotiated 640x480 at 30 FPS, latest-frame capture,
diagnostics off, and an uncalibrated workspace. These short uncontrolled runs are
fresh smoke/performance observations, not causal before/after measurements. Native
presentation returns do not verify monitor scanout; physical motion-to-photon latency
was not measured. No webcam images or user calibration/drawings were saved or overwritten.

Computer-use window activation/state capture failed with `foreground window did not
report a process id`; recovery found no remaining window after the finite run ended.
Therefore native ESC, Close, D/R/S/C/SPACE input and live visual inspection are not
claimed in this pass. Synthetic control/state tests pass. Physical skeleton alignment,
POINT feel, PINCH start/hold/release, disappearance safety, and four-corner calibration
still require a person at the camera. Backend feedback/projection warnings and a
nonfatal Clearcut uploader failure appeared during the long run.

Known limitations remain single-hand heuristic recognition, conservative track/stroke
splitting, uneven camera delivery, slow-motion filter lag, native-driver shutdown
delays if a read hangs, and no undo/eraser or desktop input automation. The repository
does not declare a license; no license choice was made on the owner's behalf.

## Reproduction

Launch from the repository root:

```powershell
.\.venv\Scripts\python.exe -m gesture_control
```

Synthetic benchmark and finite hardware check:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_pipeline.py --calibrated --diagnostics off --output recordings/synthetic-performance.json
.\.venv\Scripts\python.exe -m gesture_control --max-frames 330 --profile-warmup 30 --profile recordings/performance.json
```

[Raw final/baseline reports and filter metrics](performance/2026-09-17-finish)
include dependency versions, source hashes, timing distributions, and sample counts.
The model remained unchanged; no dependency was added. Nothing was pushed remotely.
