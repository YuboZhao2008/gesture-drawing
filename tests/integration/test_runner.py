"""Exercise the production lifecycle with synthetic capture; no webcam/window needed."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from gesture_control import runner, temporal, vision
from gesture_control.core.config import AppConfig, TemporalConfig, WorkspaceConfig
from gesture_control.core.performance import PerformanceMonitor
from gesture_control.core.types import FrameContext, Gesture, HandFrame
from gesture_control.ui import demo
from gesture_control.ui.drawing import Image
from gesture_control.vision.camera import CameraFrame

from .test_pipeline import source_hand


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "detector_init",
        "camera_open",
        "camera_read",
        "detect",
        "present",
        "interrupt",
        "cleanup",
        "present_cleanup",
    ],
)
def test_lifecycle_attempts_all_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str | None,
) -> None:
    calls: list[str] = []
    instances: list[demo.DemoUI] = []

    class UI(demo.DemoUI):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(width=301, height=201, calibration_path=tmp_path / "calibration.json")
            instances.append(self)

        def present(self, image: Image) -> bool:
            calls.append("present")
            assert self.canvas.pen_down
            self.render(image)
            if failure in ("present", "present_cleanup"):
                raise RuntimeError(f"{failure} failed")
            if failure == "interrupt":
                raise KeyboardInterrupt
            self.handle_key(27)
            return self.running

        def close(self) -> None:
            calls.append("ui_close")
            super().close()

    class Detector:
        def __init__(self, model_path: object, **kwargs: object) -> None:
            calls.append("detector_init")
            if failure == "detector_init":
                raise RuntimeError("detector_init failed")

        def detect(self, image: object, context: FrameContext) -> HandFrame:
            calls.append("detect")
            if failure == "detect":
                raise RuntimeError("detect failed")
            return HandFrame(context, (source_hand(Gesture.PINCH),))

        def close(self) -> None:
            calls.append("detector_close")
            if failure in ("cleanup", "present_cleanup"):
                raise RuntimeError("detector cleanup failed")

    class Camera:
        def __init__(self, **kwargs: object) -> None:
            pass

        def open(self) -> None:
            calls.append("camera_open")
            if failure == "camera_open":
                raise RuntimeError("camera_open failed")

        def read(self) -> CameraFrame:
            calls.append("camera_read")
            if failure == "camera_read":
                raise RuntimeError("camera_read failed")
            return CameraFrame(
                np.zeros((480, 640, 3), dtype=np.uint8), FrameContext(0, 0, 640, 480)
            )

        def close(self) -> None:
            calls.append("camera_close")
            if failure in ("cleanup", "present_cleanup"):
                raise RuntimeError("camera cleanup failed")

    class Temporal(temporal.GestureTemporalProcessor):
        def reset(self) -> None:
            calls.append("temporal_reset")
            assert not instances[0].canvas.pen_down
            assert instances[0].mapper.hand_id is None
            super().reset()
            if failure in ("cleanup", "present_cleanup"):
                raise RuntimeError("temporal cleanup failed")

    monkeypatch.setattr(demo, "DemoUI", UI)
    monkeypatch.setattr(vision, "MediaPipeHandDetector", Detector)
    monkeypatch.setattr(vision, "OpenCVCamera", Camera)
    monkeypatch.setattr(temporal, "GestureTemporalProcessor", Temporal)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: calls.append("windows_close"))
    config = AppConfig(temporal=TemporalConfig(gesture_dwell_s=0, reacquisition_delay_s=0))
    if failure in (None, "cleanup"):
        assert runner.run(config) == 1
    elif failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            runner.run(config)
    else:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            runner.run(config)
    assert calls.count("detector_init") == 1
    if failure == "detector_init":
        assert calls[-2:] == ["ui_close", "windows_close"]
    else:
        assert calls[-5:] == [
            "temporal_reset",
            "camera_close",
            "detector_close",
            "ui_close",
            "windows_close",
        ]
    assert not instances[0].canvas.pen_down and not instances[0].running
    if failure in ("cleanup", "present_cleanup"):
        assert caplog.text.count("Cleanup failed:") == 3


@pytest.mark.parametrize(
    ("debug", "diagnostics", "expected"),
    [(False, None, False), (True, None, True), (True, False, False), (False, True, True)],
)
def test_finite_headless_runner_processes_and_renders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    debug: bool,
    diagnostics: bool | None,
    expected: bool,
) -> None:
    from .test_pipeline import UnusedDetector

    class Detector(UnusedDetector):
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def detect(self, image: object, context: FrameContext) -> HandFrame:
            return HandFrame(context)

        def close(self) -> None:
            pass

    class Camera:
        def __init__(self, **kwargs: object) -> None:
            self.count = 0

        def open(self) -> None:
            pass

        def read(self) -> CameraFrame:
            context = FrameContext(self.count, self.count * 0.05, 640, 480)
            self.count += 1
            return CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), context)

        def close(self) -> None:
            pass

    def unexpected_window(*args: object, **kwargs: object) -> None:
        raise AssertionError("Headless mode must not open a window")

    class UI(demo.DemoUI):
        def render(self, image: Image) -> Image:
            assert self.diagnostics is expected
            return super().render(image)

    monkeypatch.setattr(demo, "DemoUI", UI)
    monkeypatch.setattr(vision, "MediaPipeHandDetector", Detector)
    monkeypatch.setattr(vision, "OpenCVCamera", Camera)
    monkeypatch.setattr(cv2, "namedWindow", unexpected_window)
    assert (
        runner.run(
            AppConfig(workspace=WorkspaceConfig(640, 480), debug=debug),
            headless=True,
            max_frames=3,
            diagnostics=diagnostics,
            calibration_path=tmp_path / "none.json",
        )
        == 3
    )


def test_cli_help_and_validation(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as help_exit:
        runner.main(["--help"])
    assert help_exit.value.code == 0
    assert "--model" in capsys.readouterr().out
    for args in (["--headless"], ["--max-frames", "0"]):
        with pytest.raises(SystemExit) as error:
            runner.main(args)
        assert error.value.code == 2


def test_cli_reports_startup_failure(tmp_path: Path) -> None:
    assert runner.main(["--config", str(tmp_path / "missing.toml")]) == 1


@pytest.mark.parametrize(
    ("options", "expected"), [([], None), (["--diagnostics"], True), (["--no-diagnostics"], False)]
)
def test_cli_forwards_capture_and_diagnostic_options(
    monkeypatch: pytest.MonkeyPatch,
    options: list[str],
    expected: bool | None,
) -> None:
    received: dict[str, object] = {}

    def fake_run(config: AppConfig, **kwargs: object) -> int:
        received.update(kwargs)
        return 3

    monkeypatch.setattr(runner, "run", fake_run)
    assert (
        runner.main(
            [
                "--headless",
                "--max-frames",
                "3",
                "--camera-width",
                "800",
                "--camera-height",
                "600",
                "--synchronous-camera",
                *options,
            ]
        )
        == 0
    )
    assert received["diagnostics"] is expected
    assert received["camera_width"] == 800 and received["camera_height"] == 600
    assert received["latest_frame"] is False
    assert received["headless"] is True and received["max_frames"] == 3
    assert received["performance"] is None


def test_cli_forwards_profile_warmup_and_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "profiles" / "run.json"

    def fake_run(config: AppConfig, **kwargs: object) -> int:
        monitor = kwargs["performance"]
        assert isinstance(monitor, PerformanceMonitor)
        assert monitor.warmup_frames == 2
        assert kwargs["latest_frame"] is True
        for _ in range(3):
            with monitor.frame():
                pass
        return 3

    monkeypatch.setattr(runner, "run", fake_run)
    assert runner.main(["--profile", str(path), "--profile-warmup", "2"]) == 0
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["retained_frames"] == 1
    assert report["warmup_frames"] == 2
    assert report["environment"]["config"]["smoothing"]["adaptive"] is True
