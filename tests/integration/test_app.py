"""Composition tests only: no test double is a production gesture implementation."""

from dataclasses import dataclass, field, replace

import pytest

from gesture_control.app import GestureControlApp
from gesture_control.core.events import Command, GestureEvent, GestureEventType, TemporalOutput
from gesture_control.core.protocols import LandmarkNormalizer
from gesture_control.core.types import (
    FrameContext,
    Gesture,
    GesturePrediction,
    HandFrame,
    HandLandmarks,
    NormalizedHand,
)


@dataclass
class PipelineDouble:
    frame: HandFrame
    predictions: tuple[GesturePrediction, ...]
    output: TemporalOutput
    commands: tuple[Command, ...]
    calls: list[str] = field(default_factory=list)
    executed: list[Command] = field(default_factory=list)

    def detect(self, image: object, context: FrameContext) -> HandFrame:
        self.calls.append("detect")
        return self.frame

    def recognize(self, frame: HandFrame) -> tuple[GesturePrediction, ...]:
        assert frame is self.frame
        self.calls.append("recognize")
        return self.predictions

    def process(
        self, frame: HandFrame, predictions: tuple[GesturePrediction, ...]
    ) -> TemporalOutput:
        assert frame is self.frame
        assert predictions is self.predictions
        self.calls.append("process")
        return self.output

    def reset(self) -> None:
        self.calls.clear()

    def map(self, output: TemporalOutput) -> tuple[Command, ...]:
        assert output is self.output
        self.calls.append("map")
        return self.commands

    def execute(self, command: Command) -> None:
        self.calls.append("execute")
        self.executed.append(command)

    def app(self) -> GestureControlApp:
        # Strict type checking verifies structural compatibility with all five protocols.
        return GestureControlApp(
            detector=self, recognizer=self, temporal=self, mapper=self, adapter=self
        )


@pytest.fixture
def pipeline(context: FrameContext, hand: HandLandmarks) -> PipelineDouble:
    return PipelineDouble(
        frame=HandFrame(context, (hand,)),
        predictions=(GesturePrediction(hand.hand_id, Gesture.POINT, 0.9),),
        output=TemporalOutput(context),
        commands=(Command(context, "test.first"), Command(context, "test.second")),
    )


def test_dependency_flow_and_command_order(pipeline: PipelineDouble) -> None:
    commands = pipeline.app().process_frame(object(), pipeline.frame.context)
    assert commands == pipeline.commands
    assert pipeline.executed == list(commands)
    assert pipeline.calls == ["detect", "recognize", "process", "map", "execute", "execute"]


def test_recorded_landmarks_bypass_detection(pipeline: PipelineDouble) -> None:
    pipeline.app().process_landmarks(pipeline.frame)
    assert pipeline.calls == ["recognize", "process", "map", "execute", "execute"]


def test_no_hand_frame_reaches_temporal_and_dispatches_release(context: FrameContext) -> None:
    lost = GestureEvent(context, "previous-track", GestureEventType.HAND_LOST, Gesture.NONE)
    release = Command(context, "test.release", hand_id="previous-track")
    pipeline = PipelineDouble(
        HandFrame(context), (), TemporalOutput(context, gesture_events=(lost,)), (release,)
    )
    assert pipeline.app().process_frame(object(), context) == (release,)
    assert "process" in pipeline.calls
    assert pipeline.executed == [release]


@pytest.mark.parametrize("bad_ids", [(), ("other-track",), ("track-1", "track-1")])
def test_invalid_recognizer_contract_stops_before_temporal(
    pipeline: PipelineDouble, bad_ids: tuple[str, ...]
) -> None:
    pipeline.predictions = tuple(
        GesturePrediction(hand_id, Gesture.POINT, 1) for hand_id in bad_ids
    )
    with pytest.raises(ValueError, match="exactly one"):
        pipeline.app().process_landmarks(pipeline.frame)
    assert pipeline.calls == ["recognize"]


def test_detector_context_mismatch_stops_pipeline(pipeline: PipelineDouble) -> None:
    with pytest.raises(ValueError, match="detector"):
        pipeline.app().process_frame(object(), replace(pipeline.frame.context, frame_id=8))
    assert pipeline.calls == ["detect"]


def test_temporal_context_mismatch_stops_before_mapping(pipeline: PipelineDouble) -> None:
    pipeline.output = TemporalOutput(replace(pipeline.frame.context, frame_id=8))
    with pytest.raises(ValueError, match="temporal"):
        pipeline.app().process_landmarks(pipeline.frame)
    assert pipeline.calls == ["recognize", "process"]


def test_all_command_contexts_checked_before_any_side_effect(pipeline: PipelineDouble) -> None:
    invalid = replace(pipeline.commands[1], context=replace(pipeline.frame.context, frame_id=8))
    pipeline.commands = (pipeline.commands[0], invalid)
    with pytest.raises(ValueError, match="mapper"):
        pipeline.app().process_landmarks(pipeline.frame)
    assert pipeline.executed == []


def test_adapter_failure_propagates_without_retry(pipeline: PipelineDouble) -> None:
    class FailingAdapter:
        def execute(self, command: Command) -> None:
            pipeline.executed.append(command)
            raise RuntimeError("device disconnected")

    app = pipeline.app()
    app.adapter = FailingAdapter()
    with pytest.raises(RuntimeError, match="disconnected"):
        app.process_landmarks(pipeline.frame)
    assert pipeline.executed == [pipeline.commands[0]]


def test_normalizer_protocol_supports_unusable_geometry(
    context: FrameContext, hand: HandLandmarks
) -> None:
    class DegenerateNormalizer:
        def normalize(self, hand: HandLandmarks, context: FrameContext) -> NormalizedHand | None:
            return None

    normalizer: LandmarkNormalizer = DegenerateNormalizer()
    assert normalizer.normalize(hand, context) is None
