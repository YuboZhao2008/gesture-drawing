"""Reproducible synthetic comparison: python tests/unit/test_motion.py.

Errors are image fractions (1920x multiplication in the report is illustrative).
The baseline exactly reproduces the previous alpha=0.35, 30 Hz referenced EMA.
Random(17) noise is uniform +/-0.006, independently sampled per trajectory.
"""

import json
import math
import random
from collections.abc import Callable
from statistics import fmean

import pytest

from gesture_control.temporal import MotionConfig, OneEuroFilter, PointerFilterConfig


def times(rate: int) -> list[float]:
    if rate:
        return [i / rate for i in range(3 * rate + 1)]
    result = [0.0]
    intervals = (0.013, 0.027, 0.018, 0.04, 0.022)
    while result[-1] < 3:
        result.append(result[-1] + intervals[(len(result) - 1) % len(intervals)])
    return result


def compare(
    rate: int,
    trajectory: Callable[[float], float],
    window: tuple[float, float],
    noise: float = 0.006,
) -> dict[str, float]:
    rng = random.Random(17)
    adaptive = OneEuroFilter()
    previous: float | None = None
    last_time = 0.0
    errors: dict[str, list[float]] = {"raw": [], "ema": [], "adaptive": []}
    for time in times(rate):
        truth = trajectory(time)
        raw = truth + rng.uniform(-noise, noise)
        alpha = 1 - 0.65 ** (30 * (time - last_time))
        ema = raw if previous is None else previous + alpha * (raw - previous)
        filtered = adaptive.filter(raw, 0.5, time)[0]
        if window[0] < time <= window[1]:
            for name, value in (("raw", raw), ("ema", ema), ("adaptive", filtered)):
                errors[name].append(value - truth)
        previous, last_time = ema, time
    result = {
        f"{name}_rms_px": math.sqrt(fmean(e * e for e in values)) * 1920
        for name, values in errors.items()
    }
    result.update({f"{name}_bias": fmean(values) for name, values in errors.items()})
    return result


def metrics(rate: int) -> dict[str, dict[str, float]]:
    result = {
        "stationary": compare(rate, lambda t: 0.5, (1, 3)),
        "slow": compare(rate, lambda t: 0.2 + 0.05 * t, (1, 3)),
        "fast": compare(rate, lambda t: 0.1 + 0.8 * min(0.9, max(0, t - 1)), (1.4, 1.9)),
        "reversal": compare(rate, lambda t: 0.2 + 0.8 * max(0, 0.5 - abs(t - 1.5)), (1.5, 1.9)),
    }
    for name, speed in (("slow", 0.05), ("fast", 0.8)):
        for method in ("ema", "adaptive"):
            result[name][f"{method}_lag_ms"] = -result[name][f"{method}_bias"] / speed * 1000
    return result


@pytest.mark.parametrize("rate", [30, 60, 0])
def test_jitter_and_trajectory_error_budgets(rate: int) -> None:
    report = metrics(rate)
    stationary = report["stationary"]
    assert stationary["adaptive_rms_px"] < stationary["raw_rms_px"] * 0.45
    assert stationary["adaptive_rms_px"] < stationary["ema_rms_px"]
    assert report["slow"]["adaptive_rms_px"] < report["slow"]["ema_rms_px"]
    assert report["fast"]["adaptive_rms_px"] < report["fast"]["ema_rms_px"] * 0.35
    assert report["fast"]["adaptive_lag_ms"] < 25
    assert report["reversal"]["adaptive_rms_px"] < report["reversal"]["ema_rms_px"]


@pytest.mark.parametrize("rate", [30, 60, 0])
def test_step_response_has_no_overshoot_or_buffering(rate: int) -> None:
    filter_ = OneEuroFilter()
    samples = [(t, filter_.filter(0.2 if t < 1 else 0.8, 0.5, t)[0]) for t in times(rate)]
    first = next(t for t, _ in samples if t >= 1)
    reached = next(t for t, x in samples if t >= 1 and x >= 0.74)
    assert reached - first <= 0.05
    assert all(0.2 <= x <= 0.8 for _, x in samples)


def test_filter_reset_equal_times_and_validation() -> None:
    filter_ = OneEuroFilter()
    assert filter_.filter(0.2, 0.3, 1) == (0.2, 0.3)
    assert filter_.filter(0.9, 0.8, 1) == (0.2, 0.3)
    with pytest.raises(ValueError):
        filter_.filter(0, 0, 0)
    with pytest.raises(ValueError):
        filter_.filter(float("nan"), 0, 2)
    filter_.reset()
    assert filter_.filter(0.9, 0.8, 0) == (0.9, 0.8)


@pytest.mark.parametrize("field", ["min_cutoff", "beta", "derivative_cutoff"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_filter_settings(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        PointerFilterConfig(**{field: value})  # type: ignore[arg-type]


def test_timing_config_and_filter_endpoints() -> None:
    with pytest.raises(ValueError):
        MotionConfig(pointer_dropout_grace_s=-1)
    with pytest.raises(ValueError):
        PointerFilterConfig(min_cutoff=0)
    with pytest.raises(ValueError):
        PointerFilterConfig(derivative_cutoff=0)
    assert PointerFilterConfig(beta=0).beta == 0


if __name__ == "__main__":
    print(json.dumps({str(rate or "irregular"): metrics(rate) for rate in (30, 60, 0)}, indent=2))
