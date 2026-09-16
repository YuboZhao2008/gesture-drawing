import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gesture_control.core.config import AppConfig, CameraConfig, load_config


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
