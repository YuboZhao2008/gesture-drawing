# Offline classification robustness data

This package augments **gesture-classification test/training data**. It does not
train, fine-tune, or improve the MediaPipe landmark detector. It is not imported
by the live inference path and adds no dependency.

`LandmarkSample` is a validated, immutable version-1 in-memory sample format with
21 finite wrist-centered landmarks, an intended gesture label, original group ID,
and provenance. `synthetic_samples()` derives 18 templates from **six** original
hand-authored chain groups adapted from the existing recognition tests. Distal
curl variants keep their original group ID. They are not camera recordings,
independently measured human annotations, or 18 independent hand identities.

`augment(sample, seed=...)` uses a local random generator, bounded planar rotation,
rigid 3D tilt, clipped Gaussian position/depth noise, and small distal-joint
rotations that preserve that bone's length. Pinch thumb/index distal joints are
not independently rotated. `source_observation` provides translation, isotropic
scale, anatomical mirroring, aspect conversion, and whole-observation confidence.
The public hand format has no missing-joint or per-landmark confidence field.
Duplicate/near-duplicate joints and degraded whole-hand confidence are separate
**corruption/rejection** scenarios; they are not called label-preserving variants.

Use augmentation only on manually justified interior poses. Bounded noise cannot
guarantee semantic label preservation for arbitrary poses sitting on a decision
boundary. The tests explicitly distinguish comfortable distal curls from a
half-bent middle finger, and reject all unsupported finger combinations under
noise. Large perspective warps, unbounded noise, and invented occluded coordinates
are excluded. Global 3D tilt tests orientation and mild orthographic foreshortening;
it does not simulate a calibrated perspective camera.

## Reproduce the comparison

From the candidate checkout, use its source path/interpreter. Supply an unchanged
checkout of baseline commit `43465e83721060323f9cb9df671dfb6dbaf20e00`:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python scripts/evaluate_recognition.py --baseline-root C:/path/to/baseline --seed 1709 --variants 100 --output recordings/recognition-evaluation.json
```

The baseline recognition package is loaded under an isolated module name from
that checkout. Both recognizers receive the exact same deterministic observations
and canonical normalizer. Output includes a complete confusion matrix, per-class
precision/recall, false action counts, corruption rejection, and confidence
summaries. Confidence is a geometric margin, **not a calibrated probability**.
No timing benchmark or model fitting is performed by this script.

Seed 1709 with 100 variants per template gives 1,800 observations:

| Intended class | Baseline correct / 300 | Improved correct / 300 |
| --- | ---: | ---: |
| POINT | 203 | 296 |
| TWO_FINGER | 195 | 296 |
| OPEN_PALM | 181 | 279 |
| FIST | 300 | 300 |
| PINCH | 300 | 300 |
| UNKNOWN | 300 | 300 |
| Total | 1,479 / 1,800 (82.17%) | 1,771 / 1,800 (98.39%) |

All classification errors were UNKNOWN rejections: both versions produced zero
wrong actionable labels on this corpus. Actionable-class precision is 1.0 for
both versions; recall is the count above divided by 300. UNKNOWN precision changes
from 0.4831 to 0.9119, with recall 1.0 for both. Separately, corruption rejection
improves from 149/198 to 198/198. These are synthetic regression results selected
to probe known failure modes, **not measured real-world recognition accuracy**.
The additional tests cover 300 mildly noisy half-bent middle-finger poses and
360 noisy unsupported finger combinations; all remain UNKNOWN.

## Architecture decision

| Approach | Evidence / decision |
| --- | --- |
| Original geometry | Fast, inspectable and already invariant to rigid transforms; rejects modest isolated distal curl and accepts some nearly collapsed chains. Baseline retained for comparison through an unchanged checkout. |
| Improved geometry | Keeps the architecture, rejection gap, confidence gate and geometric pinch priority. Uses bounded angle averaging constrained by reach, and rejects severely imbalanced bone lengths. Accepted based on the reproducible regression evidence above. |
| RandomForest or small MLP / geometric pinch hybrid | Considered but **not trained or merged**. Six authored pose groups provide no independent anatomy/camera/subject diversity. Learning their augmentations could memorize the generator; a split by augmented rows would leak originals and falsely imply generalization. No demonstrated benefit justifies a model artifact, inference dependency or additional rejection calibration. |

Before any future learned classifier, collect independently labeled real poses,
version the stored dataset, split subjects/original group IDs **before** augmentation,
then report held-out confusion, per-class precision/recall, rejection performance,
and inference timing. Retain geometric PINCH if it is superior. No ML accuracy
number is claimed here. Physical edge-on poses, inaccurate depth, partial occlusion,
and thumb/index contact inside a fist still need human acceptance testing.
