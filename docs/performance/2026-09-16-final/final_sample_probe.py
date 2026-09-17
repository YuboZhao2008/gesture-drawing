"""MSMF sample-age probe around the production runner, with native reads on their owner.

Fixed 60-delivery warmup. Forward ordinary runner options, including --profile.
The backend sample timestamp is uncalibrated; this is not optical latency.
"""

import json
import sys
from collections import deque
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter

import cv2

from gesture_control import runner, vision
from gesture_control.vision.camera import CameraFrame


@dataclass(frozen=True, slots=True)
class SampleFrame(CameraFrame):
    sample_ms: float


class SampleCamera(vision.OpenCVCamera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deliveries = 0
        self.samples = deque(maxlen=9000)
        instances.append(self)

    def _read_frame(self):
        frame = super()._read_frame()
        # This executes on the capture owner, including in worker mode.
        sample_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC)
        return SampleFrame(frame.image, frame.context, sample_ms)

    def read(self):
        frame = super().read()
        self.deliveries += 1
        if self.deliveries > 60:
            self.samples.append((frame.sample_ms, perf_counter() * 1000 - frame.sample_ms))
        return frame


instances = []
vision.OpenCVCamera = SampleCamera
code = runner.main(sys.argv[1:])
if code == 0:
    path = Path(sys.argv[sys.argv.index("--profile") + 1])
    report = json.loads(path.read_text(encoding="utf-8"))
    samples = list(instances[0].samples)
    ages = sorted(age for _, age in samples)
    report["metadata"]["backend_sample_age_probe"] = {
        "method": "perf_counter at delivery minus CAP_PROP_POS_MSEC; sample clock/exposure "
        "not independently calibrated. Additional native get on capture owner per acquisition.",
        "samples": len(samples),
        "first_backend_ms": samples[0][0],
        "last_backend_ms": samples[-1][0],
        "nondecreasing": all(b[0] >= a[0] for a, b in zip(samples, samples[1:], strict=False)),
        "mean_ms": sum(ages) / len(ages),
        "p50_ms": ages[ceil(len(ages) * 0.5) - 1],
        "p95_ms": ages[ceil(len(ages) * 0.95) - 1],
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
raise SystemExit(code)
