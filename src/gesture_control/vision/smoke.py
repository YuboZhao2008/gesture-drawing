"""Optional real model/camera smoke test: python -m gesture_control.vision.smoke."""

import argparse
import time

from . import MediaPipeHandDetector, OpenCVCamera, PalmNormalizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--blank", action="store_true", help="test model without a webcam")
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")
    normalizer = PalmNormalizer()
    observed = 0
    with MediaPipeHandDetector(args.model) as detector:
        start = time.monotonic()
        if args.blank:
            import numpy as np

            from gesture_control.core.types import FrameContext

            image = np.zeros((480, 640, 3), dtype=np.uint8)
            for i in range(args.frames):
                frame = detector.detect(image, FrameContext(i, i / 30, 640, 480))
                assert not frame.hands, "unexpected hand on blank image"
        else:
            with OpenCVCamera(args.camera) as camera:
                for _ in range(args.frames):
                    captured = camera.read()
                    frame = detector.detect(captured.image, captured.context)
                    for hand in frame.hands:
                        assert len(hand.landmarks) == 21
                        normalizer.normalize(hand, frame.context)
                        observed += 1
        elapsed = time.monotonic() - start
    print(f"{args.frames} frames processed in {elapsed:.3f}s; hand observations: {observed}")


if __name__ == "__main__":
    main()
