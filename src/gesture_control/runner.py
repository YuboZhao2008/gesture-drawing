"""Live composition and resource ownership; the core remains backend independent."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path

from .app import GestureControlApp
from .core.config import AppConfig, load_config

LOGGER = logging.getLogger(__name__)


def _cleanup(name: str, action: Callable[[], object]) -> None:
    """Attempt every cleanup without masking the original processing failure."""
    try:
        action()
    except Exception:
        LOGGER.exception("Cleanup failed: %s", name)


def run(
    config: AppConfig,
    *,
    model_path: str | Path = "models/hand_landmarker.task",
    calibration_path: str | Path = "recordings/calibration.json",
    drawing_path: str | Path = "recordings/drawing.png",
    max_frames: int | None = None,
    headless: bool = False,
) -> int:
    """Process a camera stream once; return frame count. Errors propagate after cleanup.

    Headless mode still renders each frame, but opens no windows. It requires a
    finite frame limit and is intended for hardware smoke checks, not interaction.
    """
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    if headless and max_frames is None:
        raise ValueError("headless mode requires max_frames")

    # Optional dependencies are loaded only after CLI argument parsing.
    import cv2

    from .recognition import GeometricGestureRecognizer
    from .temporal import GestureTemporalProcessor
    from .ui.demo import DemoUI
    from .vision import MediaPipeHandDetector, OpenCVCamera, PalmNormalizer

    with ExitStack() as resources:
        resources.callback(_cleanup, "UI windows", cv2.destroyAllWindows)
        ui = DemoUI(
            width=config.workspace.width,
            height=config.workspace.height,
            calibration_path=calibration_path,
            drawing_path=drawing_path,
        )
        resources.callback(_cleanup, "UI", ui.close)
        detector = MediaPipeHandDetector(
            model_path, min_detection_confidence=config.camera.min_detection_confidence
        )
        resources.callback(_cleanup, "detector", detector.close)
        camera = OpenCVCamera(index=config.camera.index, fps=config.camera.target_fps)
        resources.callback(_cleanup, "camera", camera.close)
        temporal = GestureTemporalProcessor(config.temporal, config.recognition, config.smoothing)
        resources.callback(_cleanup, "temporal state", temporal.reset)
        resources.callback(_cleanup, "drawing mapper", ui.mapper.reset)
        resources.callback(_cleanup, "active stroke", ui.canvas.release)
        app = GestureControlApp(
            detector=detector,
            recognizer=GeometricGestureRecognizer(PalmNormalizer(), config.recognition),
            temporal=ui.wrap_temporal(temporal),
            mapper=ui.mapper,
            adapter=ui.canvas,
        )
        # Skeleton and diagnostics are on by default so acquisition is visible.
        camera.open()
        count = 0
        while max_frames is None or count < max_frames:
            captured = camera.read()
            app.process_frame(captured.image, captured.context)
            count += 1
            if headless:
                ui.render(captured.image)
            elif not ui.present(captured.image):
                break
        LOGGER.info("Processed %d frames", count)
        return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gesture Studio: point to move, pinch to draw")
    parser.add_argument(
        "--config", type=Path, help="AppConfig TOML/JSON; otherwise built-in defaults"
    )
    parser.add_argument("--model", type=Path, default=Path("models/hand_landmarker.task"))
    parser.add_argument("--calibration", type=Path, default=Path("recordings/calibration.json"))
    parser.add_argument("--drawing", type=Path, default=Path("recordings/drawing.png"))
    parser.add_argument("--max-frames", type=int, help="Exit after N frames (hardware smoke check)")
    parser.add_argument(
        "--headless", action="store_true", help="Render without windows; needs frame limit"
    )
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.headless and args.max_frames is None:
        parser.error("--headless requires --max-frames")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_config(args.config) if args.config else AppConfig()
        if config.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        run(
            config,
            model_path=args.model,
            calibration_path=args.calibration,
            drawing_path=args.drawing,
            max_frames=args.max_frames,
            headless=args.headless,
        )
    except KeyboardInterrupt:
        LOGGER.info("Interrupted; resources closed")
        return 130
    except Exception as error:
        LOGGER.error("%s", error)
        LOGGER.debug("Application failure", exc_info=True)
        return 1
    return 0
