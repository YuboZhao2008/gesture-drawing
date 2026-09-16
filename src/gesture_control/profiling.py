"""Opt-in integration instrumentation, installed/restored only for a benchmark run.

Specialist source files stay untouched. Native OpenCV and calibration hooks are
process-wide during this scope: use only with the current single-thread runner.
Private backend inference detail is optional; detection timing is always available.
"""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from functools import wraps
from hashlib import sha256
from pathlib import Path
from typing import Any

from .core.performance import PerformanceMonitor


@contextmanager
def instrument_pipeline(
    monitor: PerformanceMonitor,
    *,
    camera: Any,
    detector: Any,
    recognizer: Any,
    temporal: Any,
    ui: Any,
) -> Iterator[None]:
    import cv2
    import numpy as np

    from .ui import demo, overlay
    from .ui.calibration import Calibration

    package = Path(__file__).parent
    source_hashes = {
        str(path.relative_to(package)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    monitor.metadata["source_sha256"] = source_hashes
    monitor.metadata["opencv_threads"] = cv2.getNumThreads()
    settings = getattr(camera, "settings", None)
    capture = getattr(camera, "_capture", None)
    if isinstance(settings, dict):
        # The capture worker owns native access; consume its startup snapshot.
        monitor.metadata["camera"] = dict(settings)
    elif capture is not None:
        monitor.metadata["camera"] = {
            "backend": capture.getBackendName(),
            "reported_fps": capture.get(cv2.CAP_PROP_FPS),
            "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "reported_buffer_size": capture.get(cv2.CAP_PROP_BUFFERSIZE),
        }
    with ExitStack() as restore:

        def hook(owner: Any, attribute: str, stage: str) -> None:
            original = getattr(owner, attribute)
            own_attribute = attribute in vars(owner)

            @wraps(original)
            def timed(*args: Any, **kwargs: Any) -> Any:
                with monitor.stage(stage):
                    result = original(*args, **kwargs)
                if stage == "detection":
                    observations["frames"] += 1
                    observations["frames_with_hands"] += bool(result.hands)
                return result

            setattr(owner, attribute, timed)
            if own_attribute:
                restore.callback(setattr, owner, attribute, original)
            else:
                restore.callback(delattr, owner, attribute)

        observations = {"frames": 0, "frames_with_hands": 0}
        monitor.metadata["observations_including_warmup"] = observations
        for owner, attribute, stage in (
            (camera, "read", "capture"),
            (detector, "detect", "detection"),
            (recognizer, "recognize", "recognition"),
            (recognizer.normalizer, "normalize", "normalization"),
            (temporal, "process", "temporal"),
            (ui, "observe", "observation"),
            (ui.mapper, "map", "mapping"),
            (ui.canvas, "execute", "drawing"),
            (ui.canvas, "render", "canvas_render"),
            (ui, "render", "render"),
            (ui, "present", "present"),
            (ui, "handle_key", "key_handler"),
            (demo, "render_layout", "layout"),
            (Calibration, "map", "calibration"),
            (overlay, "draw_hand", "skeleton"),
            (overlay, "label", "text_raster"),
            (cv2, "resize", "resize"),
            (cv2, "cvtColor", "color_conversion"),
            (cv2, "imshow", "imshow"),
            (cv2, "waitKey", "wait_key"),
            (cv2, "getWindowProperty", "window_poll"),
            (np, "full", "buffer_fill"),
        ):
            # Slotted mappers have no instance dictionary; patch their class instead.
            hook(owner if hasattr(owner, "__dict__") else type(owner), attribute, stage)
        if callable(getattr(cv2, "pollKey", None)):
            hook(cv2, "pollKey", "poll_key")
        backend = getattr(detector, "_backend", None)
        if backend is not None and hasattr(backend, "__dict__"):
            hook(backend, "detect_for_video", "inference")
        yield
