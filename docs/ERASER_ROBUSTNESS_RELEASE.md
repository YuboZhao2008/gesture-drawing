# Eraser and recognition quality pass — September 17, 2026

Baseline: `43465e83721060323f9cb9df671dfb6dbaf20e00`. The final implementation
was measured at `4b83f0d9651525ddb633eacb941f3afe54cbb204`; subsequent release
documentation and synthetic screenshots do not change the runtime. The task
checkpoint records the final validated main SHA without a self-referential commit.

## Interaction and canvas

Hold TWO_FINGER (index and middle extended) for 650 ms to toggle Pencil/Eraser.
PINCH operates the selected tool; POINT moves its cursor. FIST was rejected for
selection because thumb/index contact can overlap the existing pinch vocabulary.
Both hand and classification confidence must reach 0.80. Missing/changed hands,
uncertain classification, and frame gaps over 150 ms interrupt selection evidence.
A held pose toggles once. Rearming requires 180 ms of another confident recognized
gesture, plus an 800 ms cooldown. Persistent release guidance prevents a misleading
hold instruction while disarmed. Tool selection terminates strokes before changing
tools and suppresses drawing in that command batch.

The eraser removes coverage from a separate ink mask, restoring the copied canvas
background, including nonuniform backgrounds. Dirty rectangles bound compositing
work. Radius is 3% of the smaller canvas dimension, minimum four pixels: 32 px at
1920×1080. Its display outline follows preview scaling. Existing source-coordinate
calibration, stroke ownership, immediate hand-loss release, and fresh-stroke rules
apply to both tools. Clear/reset preserve selection but cancel interaction state.

The original canvas had no retained strokes or undo. A small mask extension was
enough; a retained-object rewrite would introduce unnecessary scope. Export remains
composited BGR PNG; transparent export, undo, and brush dynamics remain unimplemented.
The One Euro filter, temporal processor, camera and MediaPipe model were unchanged.

## Recognition and augmentation

The weakest joint alone previously rejected moderate isolated distal curl.
The new score averages joint angles with an 18-degree cap on improvement of the
weaker joint, retaining the existing chain-reach constraint and UNKNOWN gap.
Duplicate joints and bone lengths at or below 5% of the longest bone in the same
finger reject the whole observation before pinch evaluation, preventing unstable
partial geometry from creating actionable labels or trusted pinch distances.

Seeded offline augmentation covers planar rotation, rigid 3D tilt, clipped position
and depth noise, bone-length-preserving distal perturbation, source translation,
scale, mirroring and aspect ratios. Corruption and whole-observation confidence
degradation are separate rejection cases. The public observation format does not
provide missing-point/per-landmark confidence fields. No detector training occurred:
this is augmentation of **gesture-classification test/training data**.

The versioned sample metadata preserves provenance and original group IDs. The
corpus has six authored source groups and 18 derived templates, not independent
people or camera recordings. At seed 1709, 100 variants per template:

| Result | Original geometry | Improved geometry |
| --- | ---: | ---: |
| Intended label | 1,479/1,800 (82.17%) | 1,771/1,800 (98.39%) |
| Wrong actionable label | 0 | 0 |
| Corruption correctly rejected | 149/198 | 198/198 |

All remaining classification errors were UNKNOWN rejections. Another 300 noisy
half-bent middle-finger cases and 360 unsupported finger combinations remain UNKNOWN.
These are targeted synthetic regression results, not real-world accuracy.
[Full confusion matrix and precision/recall](performance/2026-09-17-quality/recognition-evaluation.json)
and [methodology](../src/gesture_control/training/README.md) accompany the comparison.

Original geometry, improved geometry, and a RandomForest/small-MLP hybrid were
compared architecturally. ML was rejected without fitting: six correlated synthetic
groups cannot establish independent generalization or meaningful held-out benefit.
No learned model, dependency, or invented ML accuracy number was introduced. A future
study must split original groups/subjects before augmentation and evaluate rejection,
per-class outcomes and inference time on independently labeled real observations.

## Performance and hardware

Three baseline and three final identical downstream trials: 960 frames, 60 warmup,
900 retained, synthetic moving POINT/PINCH at 30 Hz timestamps, calibrated 1920×1080
workspace, diagnostics off. No competing test/benchmark jobs ran during measurements.
Windows 11, Python 3.12.10, NumPy 2.5.3, OpenCV 4.14.0.94, MediaPipe 0.10.35;
22 OpenCV threads. Values below are medians of the three reports.

| Synthetic downstream metric | Baseline | Final |
| --- | ---: | ---: |
| Capacity FPS | 504.45 | 483.02 |
| Mean frame ms | 1.977 | 2.065 |
| Frame p95 ms | 2.389 | 2.522 |
| Recognition mean ms | 0.0831 | 0.0881 |
| Temporal mean ms | 0.0263 | 0.0270 |
| Mapping mean ms | 0.0250 | 0.0266 |
| Drawing mean ms | 0.0059 | 0.0199 |
| Rendering mean ms | 1.7465 | 1.8050 |

Frame cost increased about 4.4% (0.088 ms), below the predeclared 10% investigation
threshold. This modest cost buys real background-preserving erasure and tool feedback.
Offline augmentation is absent from the runtime path. These FPS values are processing
capacity, not webcam FPS or physical latency; nested stage timings must not be added.

A separate three-repeat canvas workload uses preseeded pencil ink, continuous
120-frame strokes, 1920×1080 pixels and a 640×360 preview. Mean canvas command cost:
baseline pencil 0.0089 ms, final pencil 0.0426 ms, final eraser 0.2410 ms. Including
preview, frame means were 0.718, 0.762 and 0.942 ms respectively. Eraser has no baseline
equivalent. There is no claimed rendering speedup.

Exclusive camera-0 checks completed and released:

| Run | Frames / warmup | Effective FPS | Frame p95 ms | Frames with hands |
| --- | ---: | ---: | ---: | ---: |
| Baseline native window | 330 / 30 | 29.956 | 54.795 | 128/330 |
| Final native window | 330 / 30 | 29.934 | 52.257 | 0/330 |
| Final headless | 330 / 30 | 29.940 | 52.573 | 0/330 |

All three exited successfully, used MSMF 640×480/30 FPS latest-frame capture, and
reported zero skipped IDs in retained deliveries. Camera reopen succeeded between
runs. Final live scenes contained no detected hands; scene differences preclude
causal live latency/recognition comparisons. Native window processing and cleanup
completed, but no human gesture, keypress or monitor-scanout acceptance is claimed.
MediaPipe emitted its existing feedback/projection warnings. No webcam images,
user calibration, or user drawings were saved or overwritten.

[Raw reports](performance/2026-09-17-quality) retain source hashes and workload details.

## Orchestration and validation

| Specialist | Model / reasoning | Final specialist SHA |
| --- | --- | --- |
| Recognition / augmentation | GPT-6 Astra HIGH | `c3d27afff388ac3b7576cc26c21189fbd33283ff` |
| Eraser / interaction | GPT-6 Astra HIGH | `814d7df3076154e94c548acb5ae15c3d5e55e047` |
| UI | GPT-6 Astra MEDIUM | `0f820fb6c8b8ac0a1dd9defcbf561e97fa171ee0` |
| Independent review | GPT-6 Astra HIGH | Report-only; fixed candidate reviews recorded in checkpoint |

The lead inspected actual diffs, independently tested candidates, and merged frozen
SHAs sequentially. Eraser and UI each had one targeted revision cycle for selection
progress/rearm feedback; UI also corrected a variable-name typing conflict. The lead
added real DemoUI end-to-end tests at 15/30/60 Hz, corrected an old loss fixture that
reused a frame ID, and added a missing test annotation. Recognition needed no revision.
No specialist implementation was rejected; ML was rejected as an architecture option.

Final validation: 444 tests, no skips; Ruff lint and formatting pass; strict mypy
passes for 60 source files; pip check reports no broken requirements. All five gates
are rerun on the final committed tree before main promotion. Synthetic normal,
eraser and calibration previews were visually inspected; all presentation variants
also have layout bounds tests. No dependencies or shared core contracts changed.

## Manual acceptance still required

1. With a real hand, compare POINT smoothness and PINCH start/hold/release to the
   previous version; try both hands, comfortable curls, tilt and different distances.
2. Hold TWO_FINGER to select Eraser; hold beyond cooldown to confirm no repeat.
   Release, draw/erase continuously, then deliberately return to Pencil. Try a brief
   accidental TWO_FINGER pose and verify it does not switch.
3. Hide/reintroduce the hand while erasing and move to a distant position. Confirm
   immediate release and no connecting erased segment; repeat with another hand.
4. Calibrate four corners and verify identical pencil/eraser placement. Test clear,
   calibration restart, PNG save, diagnostics, ESC and native window close.
5. Try mild occlusion, edge-on poses and poorer lighting. Recognition can safely
   reject uncertain observations; fist thumb/index contact can still count as pinch.
   Physical motion-to-photon latency requires a separate high-speed-camera test.

Launch from the repository root:

```powershell
.\.venv\Scripts\python.exe -m gesture_control
```

The release is local only. Existing registered worktrees are preserved; no remote push
was performed. The final SHA and clean-main verification live in the task checkpoint.
