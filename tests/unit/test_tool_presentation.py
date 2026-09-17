"""Presentation checks for tool state and deliberate switch feedback."""

from pathlib import Path

import numpy as np
import pytest

from gesture_control.core.events import Command, TemporalOutput
from gesture_control.core.types import FrameContext, GesturePrediction, HandFrame
from gesture_control.ui import overlay
from gesture_control.ui.demo import DemoUI
from gesture_control.ui.drawing import Image


@pytest.mark.parametrize(
    "tool,active,expected",
    [
        ("pencil", False, "Pencil / Hover"),
        ("pencil", True, "Pencil / Drawing"),
        ("eraser", False, "Eraser / Hover"),
        ("eraser", True, "Eraser / Erasing"),
    ],
)
def test_tool_palette_and_activity_labels(
    tool: str, active: bool, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels: list[tuple[str, tuple[int, int, int]]] = []

    def capture(
        image: Image,
        text: str,
        position: tuple[int, int],
        *,
        color: tuple[int, int, int] = overlay.INK,
        scale: float = 0.5,
    ) -> None:
        labels.append((text, color))

    monkeypatch.setattr(overlay, "label", capture)
    overlay.render_layout(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.uint8),
        None,
        (),
        None,
        fps=30,
        diagnostics=False,
        status="Ready",
        pen_down=active,
        tool=tool,
    )
    active_color = overlay.ERASER if tool == "eraser" else overlay.ACCENT
    assert (tool.title(), active_color) in labels
    assert ("Pencil" if tool == "eraser" else "Eraser", overlay.MUTED) in labels
    assert (expected, active_color if active else overlay.INK) in labels
    assert any(f"Pinch to {'erase' if tool == 'eraser' else 'draw'}" in text for text, _ in labels)


@pytest.mark.parametrize(
    "tool,progress,cooldown,hint",
    [
        ("pencil", 0.5, 0, "Hold two fingers for Eraser / 50%"),
        ("eraser", 0.25, 0, "Hold two fingers for Pencil / 25%"),
        ("eraser", 0, 0.8, "Eraser selected / Lower two fingers before switching again"),
    ],
)
def test_tool_transition_feedback(
    tool: str, progress: float, cooldown: float, hint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels: list[str] = []

    def capture(
        image: Image,
        text: str,
        position: tuple[int, int],
        *,
        color: tuple[int, int, int] = overlay.INK,
        scale: float = 0.5,
    ) -> None:
        labels.append(text)

    monkeypatch.setattr(overlay, "label", capture)
    overlay.render_layout(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.uint8),
        None,
        (),
        None,
        fps=30,
        diagnostics=False,
        status="Ready",
        pen_down=False,
        tool=tool,
        tool_switch_progress=progress,
        tool_switch_cooldown_s=cooldown,
    )
    assert hint in labels


def test_clear_and_calibration_preserve_visible_tool(tmp_path: Path, context: FrameContext) -> None:
    ui = DemoUI(calibration_path=tmp_path / "none.json")
    ui.mapper.tool = "eraser"
    ui.canvas.execute(Command(context, "tool.select", {"tool": "eraser"}))
    ui.handle_key(ord("r"))
    assert ui.mapper.tool == ui.canvas.tool == "eraser"
    assert not ui.canvas.pen_down
    ui.handle_key(ord("c"))
    assert ui.mapper.tool == ui.canvas.tool == "eraser"
    assert not ui.mapper.enabled
    rendered = ui.render(np.zeros((480, 640, 3), dtype=np.uint8))
    assert rendered.shape == (726, 1076, 3)


def test_observer_forwards_exact_evidence(
    tmp_path: Path, context: FrameContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gesture_control.ui.drawing import DrawingMapper

    evidence: list[tuple[HandFrame, tuple[GesturePrediction, ...]]] = []

    def observe(
        self: DrawingMapper, frame: HandFrame, predictions: tuple[GesturePrediction, ...]
    ) -> None:
        evidence.append((frame, predictions))

    monkeypatch.setattr(DrawingMapper, "observe_evidence", observe)
    ui = DemoUI(calibration_path=tmp_path / "none.json")
    frame = HandFrame(context)
    predictions: tuple[GesturePrediction, ...] = ()
    ui.observe(frame, predictions, TemporalOutput(context))
    assert evidence == [(frame, predictions)]
    assert evidence[0][0] is frame and evidence[0][1] is predictions


@pytest.mark.parametrize("progress,cooldown", [(0.5, 0.0), (0.0, 0.8), (0.0, 0.0)])
@pytest.mark.parametrize("calibrating", [False, True])
def test_tool_feedback_fits_small_workspace_and_preserves_calibration(
    progress: float,
    cooldown: float,
    calibrating: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2

    boxes: list[tuple[int, int, int, int, str]] = []

    def capture(
        image: Image,
        text: str,
        position: tuple[int, int],
        *,
        color: tuple[int, int, int] = overlay.INK,
        scale: float = 0.5,
    ) -> None:
        if not text or image.shape[1] < 1076:
            return
        (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x, y = position
        assert x >= 0 and x + width < image.shape[1]
        assert y - height >= 0 and y + baseline < image.shape[0]
        boxes.append((x, y - height, x + width, y + baseline, text))

    monkeypatch.setattr(overlay, "label", capture)
    overlay._chrome.cache_clear()
    overlay.render_layout(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((1, 1, 3), dtype=np.uint8),
        None,
        (),
        None,
        fps=30,
        diagnostics=False,
        status="Saved: " + "very long path " * 30,
        pen_down=False,
        tool="eraser",
        tool_switch_progress=progress,
        tool_switch_cooldown_s=cooldown,
        calibration_step=0 if calibrating else None,
    )
    for index, first in enumerate(boxes):
        for second in boxes[index + 1 :]:
            assert not (
                max(first[0], second[0]) < min(first[2], second[2])
                and max(first[1], second[1]) < min(first[3], second[3])
            ), (first[4], second[4])
    texts = [box[4] for box in boxes]
    if calibrating:
        assert "Drawing paused / C to restart" in texts
        assert not any("two fingers" in text for text in texts)
    else:
        assert any("two fingers" in text for text in texts)
    # Avoid leaving test-rendered chrome in the shared cache.
    overlay._chrome.cache_clear()


def test_tool_release_hint_persists_after_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    labels: list[str] = []

    def capture(
        image: Image,
        text: str,
        position: tuple[int, int],
        *,
        color: tuple[int, int, int] = overlay.INK,
        scale: float = 0.5,
    ) -> None:
        labels.append(text)

    monkeypatch.setattr(overlay, "label", capture)
    overlay.render_layout(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.uint8),
        None,
        (),
        None,
        fps=30,
        diagnostics=False,
        status="Ready",
        pen_down=False,
        tool="eraser",
        tool_switch_release_required=True,
    )
    assert "Lower two fingers before switching again" in labels
    assert not any("Hold two fingers to switch" in text for text in labels)
