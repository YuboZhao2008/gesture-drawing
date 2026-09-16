import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gesture_control.core.config import AppConfig, CameraConfig, SmoothingConfig, load_config
from gesture_control.core.types import (
    FrameContext,
    Gesture,
    GesturePrediction,
    HandFrame,
    HandLandmarks,
    Landmark3D,
)
from gesture_control.temporal import GestureTemporalProcessor, OneEuroFilter, PointerFilterConfig


def test_checked_in_defaults_match_dataclass_defaults() -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / "default.toml"
    assert load_config(path) == AppConfig()


def test_partial_configuration_keeps_unmentioned_defaults() -> None:
    config = AppConfig.from_mapping({"camera": {"target_fps": 60}, "debug": True})
    assert config.camera == CameraConfig(target_fps=60)
    assert config.workspace == AppConfig().workspace
    assert config.debug


def test_json_round_trip(tmp_path: Path) -> None:
    expected = replace(AppConfig(), camera=CameraConfig(index=2), debug=True)
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(asdict(expected)), encoding="utf-8")
    assert load_config(path) == expected


def test_toml_partial_override(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("[workspace]\nwidth = 2560\nheight = 1440\n", encoding="utf-8")
    assert load_config(path).workspace.width == 2560


@pytest.mark.parametrize(
    "settings",
    [
        {"typo": True},
        {"camera": {"target_fpz": 30}},
        {"camera": []},
        {"camera": {"index": True}},
        {"camera": {"index": -1}},
        {"camera": {"target_fps": "30"}},
        {"camera": {"target_fps": 0}},
        {"camera": {"target_fps": float("inf")}},
        {"camera": {"min_detection_confidence": 1.1}},
        {"recognition": {"min_confidence": -0.1}},
        {"recognition": {"min_confidence": float("nan")}},
        {"recognition": {"pinch_enter_threshold": 0}},
        {"recognition": {"pinch_enter_threshold": 0.35}},
        {"recognition": {"pinch_exit_threshold": 0.1}},
        {"temporal": {"gesture_dwell_s": -1}},
        {"temporal": {"reacquisition_delay_s": -1}},
        {"temporal": {"hand_loss_grace_s": -1}},
        {"temporal": {"held_interval_s": 0}},
        {"smoothing": {"enabled": "true"}},
        {"smoothing": {"alpha": 0}},
        {"smoothing": {"alpha": 1.1}},
        {"workspace": {"width": 0}},
        {"workspace": {"height": 1080.5}},
        {"debug": 1},
    ],
)
def test_invalid_settings_fail_early(settings: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AppConfig.from_mapping(settings)


def test_zero_delays_and_confidence_endpoints_are_valid() -> None:
    settings = AppConfig.from_mapping(
        {
            "camera": {"min_detection_confidence": 0},
            "recognition": {"min_confidence": 1},
            "temporal": {"gesture_dwell_s": 0, "reacquisition_delay_s": 0, "hand_loss_grace_s": 0},
            "smoothing": {"alpha": 1},
        }
    )
    assert settings.temporal.gesture_dwell_s == 0


def test_loading_does_not_hide_io_or_parse_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="table"):
        load_config(path)
    with pytest.raises(ValueError, match=".toml or .json"):
        load_config(tmp_path / "settings.yaml")


def test_direct_construction_is_validated() -> None:
    with pytest.raises(ValueError):
        CameraConfig(target_fps=-30)
    with pytest.raises(ValueError, match="CameraConfig"):
        AppConfig(camera={})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("adaptive", 1),
        ("adaptive", "false"),
        ("min_cutoff", 0),
        ("min_cutoff", True),
        ("min_cutoff", "1.5"),
        ("min_cutoff", float("nan")),
        ("min_cutoff", float("inf")),
        ("beta", -1),
        ("beta", True),
        ("beta", float("nan")),
        ("beta", float("inf")),
        ("derivative_cutoff", 0),
        ("derivative_cutoff", -1),
        ("derivative_cutoff", True),
        ("derivative_cutoff", float("inf")),
    ],
)
def test_smoothing_parameters_rejected_before_component_construction(
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="smoothing"):
        AppConfig.from_mapping({"smoothing": {name: value}})
    with pytest.raises(ValueError, match="smoothing"):
        SmoothingConfig(**{name: value})  # type: ignore[arg-type]


def pointer_step(processor: GestureTemporalProcessor, frame_id: int, x: float) -> float:
    hand = HandLandmarks("config-test", tuple(Landmark3D(x, 0.5) for _ in range(21)))
    output = processor.process(
        HandFrame(FrameContext(frame_id, frame_id / 30, 640, 480), (hand,)),
        (GesturePrediction(hand.hand_id, Gesture.POINT, 0.9),),
    )
    return output.pointer_events[0].x


@pytest.mark.parametrize("suffix", [".toml", ".json"])
@pytest.mark.parametrize("adaptive", [True, False])
def test_loaded_smoothing_controls_processor(tmp_path: Path, suffix: str, adaptive: bool) -> None:
    settings = {
        "smoothing": {
            "adaptive": adaptive,
            "min_cutoff": 2.5,
            "beta": 0,
            "derivative_cutoff": 3.0,
            "alpha": 0.25,
        },
        "temporal": {"gesture_dwell_s": 0, "reacquisition_delay_s": 0},
    }
    path = tmp_path / f"motion{suffix}"
    if suffix == ".json":
        path.write_text(json.dumps(settings), encoding="utf-8")
    else:
        path.write_text(
            f"[smoothing]\nadaptive = {str(adaptive).lower()}\nmin_cutoff = 2.5\n"
            "beta = 0\nderivative_cutoff = 3.0\nalpha = 0.25\n"
            "[temporal]\ngesture_dwell_s = 0\nreacquisition_delay_s = 0\n",
            encoding="utf-8",
        )
    config = load_config(path)
    processor = GestureTemporalProcessor(config.temporal, smoothing=config.smoothing)
    expected_config = PointerFilterConfig(2.5, 0, 3.0, adaptive)
    assert processor.pointer_config == expected_config
    assert pointer_step(processor, 0, 0) == 0
    reference = OneEuroFilter(expected_config)
    reference.filter(0, 0.5, 0)
    expected = reference.filter(1, 0.5, 1 / 30)[0] if adaptive else 0.25
    assert pointer_step(processor, 1, 1) == pytest.approx(expected)


def test_explicit_filter_override_and_disabled_smoothing() -> None:
    config = AppConfig.from_mapping(
        {
            "temporal": {"gesture_dwell_s": 0, "reacquisition_delay_s": 0},
            "smoothing": {"alpha": 0.2, "min_cutoff": 8, "beta": 50},
        }
    )
    override = PointerFilterConfig(adaptive=False)
    processor = GestureTemporalProcessor(
        config.temporal, smoothing=config.smoothing, pointer_filter=override
    )
    assert processor.pointer_config is override
    pointer_step(processor, 0, 0)
    assert pointer_step(processor, 1, 1) == pytest.approx(0.2)
    disabled = GestureTemporalProcessor(
        config.temporal, smoothing=replace(config.smoothing, enabled=False), pointer_filter=override
    )
    pointer_step(disabled, 0, 0)
    assert pointer_step(disabled, 1, 1) == 1


def test_existing_alpha_and_positional_configuration_remain_valid() -> None:
    assert SmoothingConfig(False, 0.5).enabled is False
    config = AppConfig.from_mapping({"smoothing": {"alpha": 0.2}})
    assert config.smoothing.alpha == 0.2
    assert config.smoothing.adaptive is True
    assert GestureTemporalProcessor().pointer_config == PointerFilterConfig()
