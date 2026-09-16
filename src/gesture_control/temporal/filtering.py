"""Timestamp-driven pointer filters in source-image fraction units.

One Euro follows Casiez, Roussel and Vogel (CHI 2012):
https://gery.casiez.net/1euro/ . No prediction/extrapolation or frame buffering.
"""

from dataclasses import dataclass
from math import hypot, isfinite, pi


@dataclass(frozen=True, slots=True)
class PointerFilterConfig:
    """Cutoffs in Hz; beta scales speed measured in image fractions/second.

    Set adaptive=False to retain the original core SmoothingConfig.alpha filter.
    SmoothingConfig.enabled=False still bypasses either filter.
    """

    min_cutoff: float = 1.5
    beta: float = 20.0
    derivative_cutoff: float = 1.0
    adaptive: bool = True

    def __post_init__(self) -> None:
        for name in ("min_cutoff", "beta", "derivative_cutoff"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not isfinite(value) or value < 0 or (name != "beta" and value == 0):
                raise ValueError(f"invalid {name}")
        if not isinstance(self.adaptive, bool):
            raise ValueError("adaptive must be a boolean")


class OneEuroFilter:
    """Two-dimensional One Euro with a shared, rotation-invariant speed cutoff."""

    def __init__(self, config: PointerFilterConfig | None = None) -> None:
        self.config = config or PointerFilterConfig()
        self.reset()

    def reset(self) -> None:
        self._time: float | None = None
        self._raw = (0.0, 0.0)
        self._value = (0.0, 0.0)
        self._velocity = (0.0, 0.0)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        return 1.0 / (1.0 + 1.0 / (2.0 * pi * cutoff * dt))

    def filter(self, x: float, y: float, timestamp_s: float) -> tuple[float, float]:
        """Filter a sample; equal timestamps retain state without a derivative spike."""
        if not all(isfinite(v) for v in (x, y, timestamp_s)) or timestamp_s < 0:
            raise ValueError("coordinates/time must be finite and time nonnegative")
        if self._time is None:
            self._time, self._raw, self._value = timestamp_s, (x, y), (x, y)
            return self._value
        dt = timestamp_s - self._time
        if dt < 0:
            raise ValueError("timestamps cannot decrease; reset first")
        if dt == 0:
            return self._value
        derivative_alpha = self._alpha(self.config.derivative_cutoff, dt)
        velocity = tuple(
            previous + derivative_alpha * ((raw - old) / dt - previous)
            for raw, old, previous in zip((x, y), self._raw, self._velocity, strict=True)
        )
        self._velocity = (velocity[0], velocity[1])
        cutoff = self.config.min_cutoff + self.config.beta * hypot(*self._velocity)
        alpha = self._alpha(cutoff, dt)
        self._value = (
            self._value[0] + alpha * (x - self._value[0]),
            self._value[1] + alpha * (y - self._value[1]),
        )
        self._time, self._raw = timestamp_s, (x, y)
        return self._value
