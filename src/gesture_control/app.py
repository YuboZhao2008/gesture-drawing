"""Synchronous, single-frame composition without camera or device implementations."""

from .core.events import Command
from .core.protocols import (
    CommandMapper,
    GestureRecognizer,
    HandDetector,
    OutputAdapter,
    TemporalProcessor,
)
from .core.types import FrameContext, HandFrame


class GestureControlApp:
    """Caller supplies frames, serializes calls, and owns component lifecycles.

    Errors propagate. Command execution is ordered but not transactional: earlier
    commands may already have executed if a later command fails. Do not blindly retry.
    """

    def __init__(
        self,
        *,
        detector: HandDetector,
        recognizer: GestureRecognizer,
        temporal: TemporalProcessor,
        mapper: CommandMapper,
        adapter: OutputAdapter,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.temporal = temporal
        self.mapper = mapper
        self.adapter = adapter

    def process_frame(self, image: object, context: FrameContext) -> tuple[Command, ...]:
        """Detect and process a supplied image; return successfully dispatched commands."""
        frame = self.detector.detect(image, context)
        if frame.context != context:
            raise ValueError("detector must preserve the supplied frame context")
        return self.process_landmarks(frame)

    def process_landmarks(self, frame: HandFrame) -> tuple[Command, ...]:
        """Enter after detection for recorded landmarks; uses the same downstream path."""
        predictions = self.recognizer.recognize(frame)
        ids = [prediction.hand_id for prediction in predictions]
        if len(ids) != len(set(ids)) or set(ids) != {hand.hand_id for hand in frame.hands}:
            raise ValueError("recognizer must return exactly one prediction per observed hand")
        output = self.temporal.process(frame, predictions)
        if output.context != frame.context:
            raise ValueError("temporal processor must preserve the frame context")
        commands = self.mapper.map(output)
        if any(command.context != frame.context for command in commands):
            raise ValueError("mapper must preserve the frame context on all commands")
        for command in commands:
            self.adapter.execute(command)
        return commands
