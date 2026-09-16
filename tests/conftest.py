import pytest

from gesture_control.core.types import FrameContext, HandLandmarks, Landmark3D, LandmarkIndex


@pytest.fixture
def context() -> FrameContext:
    return FrameContext(frame_id=7, timestamp_s=1.5, image_width=640, image_height=480)


@pytest.fixture
def hand() -> HandLandmarks:
    points = [Landmark3D(0.5, 0.75) for _ in LandmarkIndex]
    points[LandmarkIndex.MIDDLE_MCP] = Landmark3D(0.5, 0.25)
    return HandLandmarks(hand_id="track-1", landmarks=tuple(points))
