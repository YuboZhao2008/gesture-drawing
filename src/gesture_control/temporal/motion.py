"""Interaction-specific timing options without changing shared core configuration."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class MotionConfig:
    """Fast pinch dwell caps and bounded cursor-only interruption tolerance.

    Core gesture dwell remains the upper bound (including explicit zero dwell).
    These tolerances never bridge a missing-hand stroke or bypass new-track gates.
    """

    pinch_start_s: float = 0.04
    pinch_release_s: float = 0.04
    pointer_noise_grace_s: float = 0.06
    pointer_dropout_grace_s: float = 0.08

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
