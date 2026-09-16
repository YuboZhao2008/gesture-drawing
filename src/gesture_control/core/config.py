"""Validated, immutable configuration with explicit TOML/JSON loading."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from . import _validation as validate


@dataclass(frozen=True, slots=True)
class CameraConfig:
    index: int = 0
    target_fps: float = 30.0
    min_detection_confidence: float = 0.5

    def __post_init__(self) -> None:
        validate.integer("camera.index", self.index)
        validate.positive("camera.target_fps", self.target_fps)
        validate.real(
            "camera.min_detection_confidence", self.min_detection_confidence, minimum=0, maximum=1
        )


@dataclass(frozen=True, slots=True)
class RecognitionConfig:
    min_confidence: float = 0.7
    pinch_enter_threshold: float = 0.25
    pinch_exit_threshold: float = 0.35

    def __post_init__(self) -> None:
        validate.real("recognition.min_confidence", self.min_confidence, minimum=0, maximum=1)
        validate.positive("recognition.pinch_enter_threshold", self.pinch_enter_threshold)
        validate.positive("recognition.pinch_exit_threshold", self.pinch_exit_threshold)
        if self.pinch_enter_threshold >= self.pinch_exit_threshold:
            raise ValueError("pinch_enter_threshold must be less than pinch_exit_threshold")


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    gesture_dwell_s: float = 0.12
    reacquisition_delay_s: float = 0.2
    hand_loss_grace_s: float = 0.15
    held_interval_s: float = 0.1

    def __post_init__(self) -> None:
        validate.real("temporal.gesture_dwell_s", self.gesture_dwell_s, minimum=0)
        validate.real("temporal.reacquisition_delay_s", self.reacquisition_delay_s, minimum=0)
        validate.real("temporal.hand_loss_grace_s", self.hand_loss_grace_s, minimum=0)
        validate.positive("temporal.held_interval_s", self.held_interval_s)


@dataclass(frozen=True, slots=True)
class SmoothingConfig:
    enabled: bool = True
    alpha: float = 0.35

    def __post_init__(self) -> None:
        validate.boolean("smoothing.enabled", self.enabled)
        validate.real("smoothing.alpha", self.alpha, minimum=0, maximum=1)
        if self.alpha == 0:
            raise ValueError("smoothing.alpha must be > 0")


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    width: int = 1920
    height: int = 1080

    def __post_init__(self) -> None:
        validate.integer("workspace.width", self.width, minimum=1)
        validate.integer("workspace.height", self.height, minimum=1)


def _section[T](section_type: type[T], value: object, name: str) -> T:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a table with string keys")
    try:
        return section_type(**value)
    except TypeError as error:
        raise ValueError(f"invalid {name} configuration: {error}") from error


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    debug: bool = False

    def __post_init__(self) -> None:
        for name, expected in (
            ("camera", CameraConfig),
            ("recognition", RecognitionConfig),
            ("temporal", TemporalConfig),
            ("smoothing", SmoothingConfig),
            ("workspace", WorkspaceConfig),
        ):
            if not isinstance(getattr(self, name), expected):
                raise ValueError(f"{name} must be {expected.__name__}")
        validate.boolean("debug", self.debug)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> AppConfig:
        """Load partial settings over defaults; reject misspellings and wrong types."""
        if not isinstance(data, Mapping):
            raise ValueError("configuration must be a table")
        allowed = {"camera", "recognition", "temporal", "smoothing", "workspace", "debug"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown configuration keys: {', '.join(sorted(map(str, unknown)))}")
        debug = data.get("debug", False)
        if not isinstance(debug, bool):
            raise ValueError("debug must be a boolean")
        return cls(
            camera=_section(CameraConfig, data.get("camera", {}), "camera"),
            recognition=_section(RecognitionConfig, data.get("recognition", {}), "recognition"),
            temporal=_section(TemporalConfig, data.get("temporal", {}), "temporal"),
            smoothing=_section(SmoothingConfig, data.get("smoothing", {}), "smoothing"),
            workspace=_section(WorkspaceConfig, data.get("workspace", {}), "workspace"),
            debug=debug,
        )


def load_config(path: str | Path) -> AppConfig:
    """Load an explicit .toml or .json path. IO and parse errors propagate."""
    config_path = Path(path)
    match config_path.suffix.lower():
        case ".toml":
            with config_path.open("rb") as stream:
                data = tomllib.load(stream)
        case ".json":
            with config_path.open(encoding="utf-8") as stream:
                data = json.load(stream)
        case _:
            raise ValueError("configuration path must end in .toml or .json")
    return AppConfig.from_mapping(data)
