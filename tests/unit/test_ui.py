"""UI logic and raster rendering tests; no camera or display is opened."""

from pathlib import Path

import numpy as np
import pytest

from gesture_control.app import GestureControlApp
from gesture_control.core.events import Command, GestureEvent, PointerEvent, TemporalOutput
from gesture_control.core.events import GestureEventType as Kind
from gesture_control.core.protocols import CommandMapper, OutputAdapter
from gesture_control.core.types import (
    FrameContext,
    Gesture,
    GesturePrediction,
    HandFrame,
    HandLandmarks,
    StableGesture,
)
from gesture_control.ui.calibration import UNIT_CORNERS, Calibration, CalibrationSession
from gesture_control.ui.demo import DemoUI, KeyAction, key_action
from gesture_control.ui.drawing import DrawingCanvas, DrawingMapper, canvas_pixel
from gesture_control.ui.overlay import diagnostic_lines


def event(context: FrameContext, kind: Kind, hand_id: str = "track-1") -> GestureEvent:
    gesture = Gesture.NONE if kind is Kind.HAND_LOST else Gesture.PINCH
    return GestureEvent(context, hand_id, kind, gesture)


def output(
    context: FrameContext,
    kinds: tuple[Kind, ...] = (),
    point: tuple[float, float] | None = (0.5, 0.5),
) -> TemporalOutput:
    return TemporalOutput(
        context,
        gesture_events=tuple(event(context, k) for k in kinds),
        pointer_events=() if point is None else (PointerEvent(context, "track-1", *point),),
    )


def dispatch(mapper: DrawingMapper, canvas: DrawingCanvas, batch: TemporalOutput) -> list[str]:
    commands = mapper.map(batch)
    for command in commands:
        canvas.execute(command)
    return [command.name for command in commands]


@pytest.mark.parametrize(
    "corners",
    [
        (),
        ((0, 0),) * 4,
        ((0, 0), (1, 1), (1, 0), (0, 1)),
        ((0, 0), (1, 0), (0.1, 0.1), (0, 1)),
        ((0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)),
        ((0, 0), (1, 0), (1, float("nan")), (0, 1)),
        ((0, 0), (2, 0), (1, 1), (0, 1)),
        ((1, 0), (1, 1), (0, 1), (0, 0)),
    ],
)
def test_invalid_calibration(corners: tuple[tuple[float, float], ...]) -> None:
    with pytest.raises(ValueError):
        Calibration(corners)


def test_perspective_and_persistence(tmp_path: Path) -> None:
    corners = ((0.2, 0.1), (0.9, 0.2), (0.8, 0.9), (0.1, 0.8))
    calibration = Calibration(corners)
    for source, target in zip(corners, UNIT_CORNERS, strict=True):
        assert calibration.map(*source) == pytest.approx(target)
    assert calibration.map(0.5, 0.5) == pytest.approx((0.5, 0.5))
    path = tmp_path / "nested" / "calibration.json"
    calibration.save(path)
    loaded = Calibration.load(path)
    assert loaded == calibration
    assert loaded.map(0.5, 0.5) == pytest.approx((0.5, 0.5))


@pytest.mark.parametrize(
    "document", ["[]", "{}", '{"version":2,"corners":[]}', '{"version":1,"corners":[[true,0]]}']
)
def test_bad_serialization(tmp_path: Path, document: str) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(document)
    with pytest.raises(ValueError):
        Calibration.load(path)


def test_calibration_flow() -> None:
    session = CalibrationSession()
    assert "TOP LEFT" in session.prompt
    for point in UNIT_CORNERS[:3]:
        assert session.capture(point) is None
    assert "BOTTOM LEFT" in session.prompt
    assert session.capture(UNIT_CORNERS[-1]) == Calibration(UNIT_CORNERS)


def test_mapping_clamps_after_calibration(context: FrameContext) -> None:
    mapper = DrawingMapper(101, 201, Calibration(((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))))
    commands = mapper.map(output(context, point=(0, 1)))
    assert commands[0].parameters == {"x_px": 0, "y_px": 200}
    assert commands[0].hand_id == "track-1"
    assert commands[0].context == context
    assert canvas_pixel(-10, 10, 11, 21) == (0, 20)
    assert canvas_pixel(0.5, 0.5, 11, 21) == (5, 10)
    with pytest.raises(ValueError):
        canvas_pixel(float("inf"), 0, 10, 10)


def test_strokes_start_at_current_pointer_and_end_once(context: FrameContext) -> None:
    mapper, canvas = DrawingMapper(101, 101), DrawingCanvas(101, 101)
    baseline = canvas.pixels.copy()
    assert dispatch(mapper, canvas, output(context, point=(0, 0))) == ["pointer.move"]
    assert np.array_equal(baseline, canvas.pixels)
    assert (
        dispatch(
            mapper, canvas, output(context, (Kind.GESTURE_STARTED, Kind.PINCH_STARTED), point=None)
        )
        == []
    )
    assert not canvas.pen_down
    assert dispatch(mapper, canvas, output(context, point=(0.8, 0.8))) == [
        "pointer.move",
        "stroke.begin",
    ]
    assert np.array_equal(canvas.pixels[0:70, 0:70], baseline[0:70, 0:70])
    assert dispatch(mapper, canvas, output(context, (Kind.GESTURE_HELD,), (0.9, 0.8))) == [
        "pointer.move",
        "stroke.point",
    ]
    assert not np.array_equal(canvas.pixels[80, 85], baseline[80, 85])
    names = dispatch(mapper, canvas, output(context, (Kind.GESTURE_ENDED, Kind.PINCH_ENDED)))
    assert names.count("stroke.end") == 1
    assert not canvas.pen_down
    assert "stroke.end" not in dispatch(mapper, canvas, output(context, (Kind.PINCH_ENDED,)))


def test_loss_never_connects_strokes(context: FrameContext) -> None:
    mapper, canvas = DrawingMapper(101, 101), DrawingCanvas(101, 101)
    dispatch(mapper, canvas, output(context, (Kind.PINCH_STARTED,), (0.1, 0.1)))
    assert dispatch(mapper, canvas, output(context, (Kind.HAND_LOST,))) == ["stroke.end"]
    dispatch(mapper, canvas, output(context, (Kind.GESTURE_HELD,), (0.9, 0.9)))
    assert not canvas.pen_down
    dispatch(mapper, canvas, output(context, (Kind.PINCH_STARTED,), (0.9, 0.9)))
    assert tuple(canvas.pixels[50, 50]) == (245, 247, 250)


def test_other_hand_cannot_draw_or_release(context: FrameContext) -> None:
    mapper, canvas = DrawingMapper(), DrawingCanvas()
    dispatch(mapper, canvas, output(context, (Kind.PINCH_STARTED,)))
    canvas.execute(Command(context, "stroke.end", hand_id="other"))
    assert canvas.pen_down
    pixels = canvas.pixels.copy()
    canvas.execute(Command(context, "stroke.point", {"x_px": 0, "y_px": 0}, "other"))
    assert np.array_equal(pixels, canvas.pixels)
    commands = mapper.map(
        TemporalOutput(context, gesture_events=(event(context, Kind.HAND_LOST, "other"),))
    )
    assert not commands


def test_open_palm_is_neutral(context: FrameContext) -> None:
    mapper, canvas = DrawingMapper(), DrawingCanvas()
    dispatch(mapper, canvas, output(context, (Kind.PINCH_STARTED,)))
    batch = TemporalOutput(
        context,
        gesture_events=(GestureEvent(context, "track-1", Kind.GESTURE_STARTED, Gesture.OPEN_PALM),),
    )
    assert dispatch(mapper, canvas, batch) == ["stroke.end"]
    assert not canvas.pen_down


@pytest.mark.parametrize(
    ("key", "action"),
    [
        (27, KeyAction.QUIT),
        (ord("C"), KeyAction.CALIBRATE),
        (32, KeyAction.CAPTURE),
        (ord("d"), KeyAction.DIAGNOSTICS),
        (ord("r"), KeyAction.CLEAR),
        (ord("s"), KeyAction.SAVE),
        (-1, None),
        (ord("x"), None),
    ],
)
def test_keys(key: int, action: KeyAction | None) -> None:
    assert key_action(key) is action


def test_ui_observation_render_and_cleanup(
    tmp_path: Path, context: FrameContext, hand: HandLandmarks
) -> None:
    ui = DemoUI(calibration_path=tmp_path / "calibration.json")
    predictions = (GesturePrediction(hand.hand_id, Gesture.PINCH, 0.95),)
    frame = HandFrame(context, (hand,))
    batch = output(context, (Kind.PINCH_STARTED,))
    ui.observe(frame, predictions, batch)
    dispatch(ui.mapper, ui.canvas, batch)
    assert ui.canvas.pen_down
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = ui.render(image)
    assert rendered.shape == (726, 1316, 3)
    assert not image.any()
    ui.observe(HandFrame(context), (), TemporalOutput(context))
    assert not ui.canvas.pen_down
    assert [c.name for c in ui.mapper.map(TemporalOutput(context))] == ["stroke.end"]
    ui.handle_key(ord("c"))
    assert not ui.mapper.enabled
    for point in UNIT_CORNERS:
        ui.pointer = point
        ui.handle_key(32)
    assert ui.mapper.enabled and ui.session is None
    assert Calibration.load(tmp_path / "calibration.json") == Calibration(UNIT_CORNERS)
    ui.handle_key(27)
    ui.close()
    ui.close()
    assert not ui.running and not ui.canvas.pen_down


def test_overlay_text(context: FrameContext, hand: HandLandmarks) -> None:
    lines = diagnostic_lines(
        HandFrame(context, (hand,)),
        (GesturePrediction(hand.hand_id, Gesture.POINT, 0.9),),
        TemporalOutput(
            context, stable_gestures=(StableGesture(hand.hand_id, Gesture.POINT, 0.8, 0),)
        ),
        30,
    )
    assert "TRACKING" in lines[0]
    assert "RAW POINT  90%" in lines
    assert "STABLE POINT  80%" in lines
    assert "HAND LOST" in diagnostic_lines(HandFrame(context), (), TemporalOutput(context), 0)[0]


def test_adapter_contract_and_invalid_commands(context: FrameContext) -> None:
    mapper: CommandMapper = DrawingMapper()
    adapter: OutputAdapter = DrawingCanvas()
    for command in mapper.map(output(context)):
        adapter.execute(command)
    with pytest.raises(ValueError):
        adapter.execute(Command(context, "unknown"))
    with pytest.raises(ValueError):
        adapter.execute(Command(context, "pointer.move", {"x_px": 0.5, "y_px": 2}, "track-1"))


def test_real_app_bridge(tmp_path: Path, context: FrameContext, hand: HandLandmarks) -> None:
    class Stages:
        def detect(self, image: object, context: FrameContext) -> HandFrame:
            return HandFrame(context, (hand,))

        def recognize(self, frame: HandFrame) -> tuple[GesturePrediction, ...]:
            return (GesturePrediction(hand.hand_id, Gesture.PINCH, 0.99),)

        def process(
            self, frame: HandFrame, predictions: tuple[GesturePrediction, ...]
        ) -> TemporalOutput:
            return output(frame.context, (Kind.GESTURE_STARTED, Kind.PINCH_STARTED))

        def reset(self) -> None:
            pass

    stages = Stages()
    ui = DemoUI(calibration_path=tmp_path / "calibration.json")
    observed = ui.wrap_temporal(stages)
    app = GestureControlApp(
        detector=stages, recognizer=stages, temporal=observed, mapper=ui.mapper, adapter=ui.canvas
    )
    commands = app.process_frame(object(), context)
    assert [c.name for c in commands] == ["pointer.move", "stroke.begin"]
    assert ui.canvas.pen_down and ui.canvas.cursor == (320, 240)
    assert ui.frame is not None and ui.output is not None
    observed.reset()
    assert not ui.canvas.pen_down and ui.mapper.hand_id is None


def test_true_projective_mapping() -> None:
    corners = ((0.3, 0.1), (0.7, 0.1), (0.9, 0.9), (0.1, 0.9))
    calibration = Calibration(corners)
    for point, target in zip(corners, UNIT_CORNERS, strict=True):
        assert calibration.map(*point) == pytest.approx(target)
    assert calibration.map(0.5, 0.5) == pytest.approx((0.5, 2 / 3))
