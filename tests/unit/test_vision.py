from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from gesture_control.core.protocols import HandDetector, LandmarkNormalizer
from gesture_control.core.types import (
    FrameContext,
    Handedness,
    HandLandmarks,
    Landmark3D,
    LandmarkIndex,
)
from gesture_control.vision import (
    MediaPipeHandDetector,
    OpenCVCamera,
    PalmNormalizer,
    draw_hand_frame,
    fingertip_pixels,
    hand_bounding_box,
    index_fingertip,
    landmark,
    palm_center,
    pointer_coordinates,
)


def test_normalization_invariants(context: FrameContext) -> None:
    hand = HandLandmarks("a", tuple(Landmark3D(i * 0.02, i * 0.03, -i * 0.01) for i in range(21)))
    normalizer: LandmarkNormalizer = PalmNormalizer()
    baseline = normalizer.normalize(hand, context)
    assert baseline is not None
    for factor, shift in ((1, 0.4), (3, 0), (0.2, -0.7)):
        moved = replace(
            hand,
            landmarks=tuple(
                Landmark3D(p.x * factor + shift, p.y * factor + shift, p.z * factor + shift)
                for p in hand.landmarks
            ),
        )
        actual = normalizer.normalize(moved, context)
        assert actual is not None
        assert actual.source is moved
        assert actual.scale == pytest.approx(baseline.scale * factor)
        for a, b in zip(actual.landmarks, baseline.landmarks, strict=True):
            assert (a.x, a.y, a.z) == pytest.approx((b.x, b.y, b.z))
    middle = baseline.landmarks[LandmarkIndex.MIDDLE_MCP]
    assert middle.y / middle.x == pytest.approx(1.5 * 0.75)
    assert baseline.landmarks[0] == Landmark3D(0, 0, 0)


def test_handedness_preserved(hand: HandLandmarks, context: FrameContext) -> None:
    for side in Handedness:
        source = replace(hand, handedness=side)
        result = PalmNormalizer().normalize(source, context)
        assert result is not None and result.source.handedness is side
    mirrored = replace(hand, landmarks=tuple(Landmark3D(-p.x, p.y, p.z) for p in hand.landmarks))
    left = PalmNormalizer().normalize(mirrored, context)
    right = PalmNormalizer().normalize(hand, context)
    assert left is not None and right is not None
    assert left.scale == right.scale


def test_degenerate_and_helpers(hand: HandLandmarks, context: FrameContext) -> None:
    assert (
        PalmNormalizer().normalize(replace(hand, landmarks=(Landmark3D(0, 0),) * 21), context)
        is None
    )
    assert index_fingertip(hand) == landmark(hand, LandmarkIndex.INDEX_TIP)
    assert pointer_coordinates(hand) == (0.5, 0.75)
    assert fingertip_pixels(hand, context) == (320, 360)
    assert hand_bounding_box(hand) == (0.5, 0.25, 0.5, 0.75)
    assert palm_center(hand) == Landmark3D(0.5, 0.65)
    with pytest.raises(ValueError):
        landmark(hand, -1)  # type: ignore[arg-type]


@pytest.fixture
def image() -> Any:
    np = pytest.importorskip("numpy")
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch, image: Any) -> MagicMock:
    cv2 = pytest.importorskip("cv2")
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, image)
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: capture)
    return capture


def test_camera_failure(capture: MagicMock) -> None:
    capture.isOpened.return_value = False
    with pytest.raises(RuntimeError, match="open camera"):
        OpenCVCamera().open()
    capture.release.assert_called_once()


def test_camera_lifecycle(capture: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([100.0, 100.1, 100.3])
    monkeypatch.setattr("gesture_control.vision.camera.time.monotonic", lambda: next(ticks))
    with OpenCVCamera(width=1280, height=720, fps=60) as camera:
        assert camera.open() is camera
        first, second = camera.read(), camera.read()
        assert first.context.frame_id == 0
        assert first.context.timestamp_s == pytest.approx(0.1)
        assert (first.context.image_width, first.context.image_height) == (640, 480)
        assert second.context.frame_id == 1
        assert second.context.timestamp_s == pytest.approx(0.3)
    camera.release()
    capture.release.assert_called_once()
    assert capture.set.call_count == 3
    with pytest.raises(RuntimeError, match="not open"):
        camera.read()


def test_read_and_setup_failure_release(capture: MagicMock) -> None:
    capture.set.side_effect = RuntimeError("setup")
    with pytest.raises(RuntimeError, match="setup"):
        OpenCVCamera().open()
    capture.release.assert_called_once()
    capture.reset_mock(side_effect=True)
    capture.read.return_value = (False, None)
    camera = OpenCVCamera().open()
    with pytest.raises(RuntimeError, match="read"):
        camera.read()
    camera.close()
    capture.release.assert_called_once()


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    vision = pytest.importorskip("mediapipe.tasks.python.vision")
    backend = MagicMock()
    backend.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[], handedness=[])
    monkeypatch.setattr(vision.HandLandmarker, "create_from_options", lambda options: backend)
    return backend


def result(offset: float = 0, label: str = "Right") -> Any:
    return SimpleNamespace(
        hand_landmarks=[
            [
                SimpleNamespace(x=0.1 + offset + i * 0.01, y=0.2 + i * 0.01, z=-i * 0.01)
                for i in range(21)
            ]
        ],
        handedness=[[SimpleNamespace(category_name=label, score=0.92)]],
    )


def make_detector(tmp_path: Path, **kwargs: Any) -> MediaPipeHandDetector:
    model = tmp_path / "test.task"
    model.touch()
    return MediaPipeHandDetector(model, **kwargs)


def test_empty_and_lifecycle(
    backend: MagicMock, tmp_path: Path, image: Any, context: FrameContext
) -> None:
    with make_detector(tmp_path) as concrete:
        detector: HandDetector = concrete
        frame = detector.detect(image, context)
        assert frame.context is context and frame.hands == ()
        detector.detect(image, replace(context, frame_id=8))
    concrete.release()
    backend.close.assert_called_once()
    assert [c.args[1] for c in backend.detect_for_video.call_args_list] == [1500, 1501]
    with pytest.raises(RuntimeError, match="closed"):
        concrete.detect(image, context)


def test_detection_continuity(backend: MagicMock, tmp_path: Path, image: Any) -> None:
    backend.detect_for_video.return_value = result()
    with make_detector(tmp_path) as detector:

        def detect(i: int) -> HandLandmarks:
            return detector.detect(image, FrameContext(i, i / 30, 640, 480)).hands[0]

        first = detect(0)
        assert len(first.landmarks) == 21
        assert first.handedness is Handedness.RIGHT and first.confidence == 0.92
        assert first.landmarks[8] == Landmark3D(0.18, 0.28, -0.08)
        backend.detect_for_video.return_value = result(0.01)
        assert detect(1).hand_id == first.hand_id
        backend.detect_for_video.return_value = result(0.6)
        jumped = detect(2)
        assert jumped.hand_id != first.hand_id
        backend.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[], handedness=[])
        assert detector.detect(image, FrameContext(3, 0.1, 640, 480)).hands == ()
        backend.detect_for_video.return_value = result(0.6)
        assert detect(4).hand_id != jumped.hand_id
        with pytest.raises(ValueError, match="increase"):
            detect(4)


def test_mirrored_labels_and_offscreen(
    backend: MagicMock, tmp_path: Path, image: Any, context: FrameContext
) -> None:
    backend.detect_for_video.return_value = result(1, "Left")
    with make_detector(tmp_path, input_is_mirrored=True) as detector:
        hand = detector.detect(image, context).hands[0]
        assert hand.handedness is Handedness.RIGHT
        assert hand.landmarks[0].x > 1


@pytest.mark.parametrize("malformation", ["short", "nan", "missing_z"])
def test_malformed_results(
    backend: MagicMock, tmp_path: Path, image: Any, context: FrameContext, malformation: str
) -> None:
    payload = result()
    if malformation == "short":
        payload.hand_landmarks[0].pop()
    elif malformation == "nan":
        payload.hand_landmarks[0][0].x = float("nan")
    else:
        del payload.hand_landmarks[0][0].z
    backend.detect_for_video.return_value = payload
    with make_detector(tmp_path) as detector, pytest.raises(ValueError, match="malformed"):
        detector.detect(image, context)


def test_image_validation_and_debug(
    backend: MagicMock, tmp_path: Path, image: Any, context: FrameContext
) -> None:
    backend.detect_for_video.return_value = result()
    with make_detector(tmp_path) as detector:
        invalid_images: tuple[object, ...] = (
            None,
            [],
            image[:, :, 0],
            image.astype("float32"),
            image[:10],
        )
        for invalid in invalid_images:
            with pytest.raises(ValueError):
                detector.detect(invalid, context)
        backend.detect_for_video.assert_not_called()
        frame = detector.detect(image, context)
        assert draw_hand_frame(image, frame) is image
        assert image.any()


def test_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model not found"):
        MediaPipeHandDetector(tmp_path / "missing.task")


def test_geometry_preserves_mirror_and_depth(context: FrameContext) -> None:
    source = HandLandmarks(
        "right",
        tuple(Landmark3D(i * 0.01, i * 0.02, -i * 0.03) for i in range(21)),
        Handedness.RIGHT,
    )
    mirrored = replace(
        source,
        handedness=Handedness.LEFT,
        landmarks=tuple(Landmark3D(1 - p.x, p.y, p.z) for p in source.landmarks),
    )
    right = PalmNormalizer().normalize(source, context)
    left = PalmNormalizer().normalize(mirrored, context)
    assert right is not None and left is not None
    for a, b in zip(right.landmarks, left.landmarks, strict=True):
        assert (a.x, a.y, a.z) == pytest.approx((-b.x, b.y, b.z))


@pytest.mark.parametrize("change", ["gap", "handedness", "resolution"])
def test_continuity_breaks(backend: MagicMock, tmp_path: Path, image: Any, change: str) -> None:
    backend.detect_for_video.return_value = result()
    with make_detector(tmp_path) as detector:
        first = detector.detect(image, FrameContext(0, 0, 640, 480)).hands[0]
        context = FrameContext(1, 0.03, 640, 480)
        if change == "gap":
            context = replace(context, timestamp_s=1)
        elif change == "handedness":
            backend.detect_for_video.return_value = result(label="Left")
        else:
            image = image[:240]
            context = replace(context, image_height=240)
        assert detector.detect(image, context).hands[0].hand_id != first.hand_id


def test_backend_color_conversion(
    backend: MagicMock, tmp_path: Path, image: Any, context: FrameContext
) -> None:
    image[0, 0] = [1, 2, 3]
    with make_detector(tmp_path) as detector:
        detector.detect(image, context)
    passed = backend.detect_for_video.call_args.args[0].numpy_view()
    assert passed[0, 0].tolist() == [3, 2, 1]
    assert image[0, 0].tolist() == [1, 2, 3]
