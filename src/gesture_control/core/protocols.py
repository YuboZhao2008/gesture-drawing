"""Structural interfaces; implementations need not inherit from these protocols."""

from typing import Protocol

from .events import Command, TemporalOutput
from .types import FrameContext, GesturePrediction, HandFrame, HandLandmarks, NormalizedHand


class HandDetector(Protocol):
    def detect(self, image: object, context: FrameContext) -> HandFrame:
        """Convert a backend-owned image into canonical tracked hand observations.

        Keep the supplied context. Image representation is agreed at composition
        time; implementations validate it without imposing an array library on core.
        """
        ...


class LandmarkNormalizer(Protocol):
    def normalize(self, hand: HandLandmarks, context: FrameContext) -> NormalizedHand | None:
        """Return canonical palm-unit coordinates, or None for degenerate geometry.

        Recognition owns orchestration of this helper, which must not drop a track:
        normalization failure becomes an UNKNOWN prediction with confidence zero.
        """
        ...


class GestureRecognizer(Protocol):
    def recognize(self, frame: HandFrame) -> tuple[GesturePrediction, ...]:
        """Return exactly one prediction per observed hand; empty for no hands.

        Use UNKNOWN for an unclassifiable hand. Normalize through LandmarkNormalizer
        as needed; never apply dwell, smoothing, or application-specific actions here.
        """
        ...


class TemporalProcessor(Protocol):
    def process(
        self, frame: HandFrame, predictions: tuple[GesturePrediction, ...]
    ) -> TemporalOutput:
        """Advance track state, including on empty frames, using frame time only."""
        ...

    def reset(self) -> None:
        """Clear all track/timing state before a new stream or a replay seek."""
        ...


class CommandMapper(Protocol):
    def map(self, output: TemporalOutput) -> tuple[Command, ...]:
        """Translate semantic output to ordered commands with the same context."""
        ...


class OutputAdapter(Protocol):
    def execute(self, command: Command) -> None:
        """Apply one command; unsupported commands and execution errors must raise."""
        ...
