"""Four image-fraction corners mapped to the unit canvas; no display required."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

type Point = tuple[float, float]
CORNERS = ("TOP LEFT", "TOP RIGHT", "BOTTOM RIGHT", "BOTTOM LEFT")
UNIT_CORNERS: tuple[Point, ...] = ((0, 0), (1, 0), (1, 1), (0, 1))


def validate_points(points: tuple[Point, ...]) -> None:
    if len(points) != 4:
        raise ValueError("Capture four corners: top left, top right, bottom right, bottom left")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for p in points for v in p):
        raise ValueError("Corners must be finite image fractions in [0, 1]")
    crosses = []
    for i in range(4):
        a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if min(crosses) < 0.005:
        raise ValueError("Corners must form a convex clockwise workspace with separated edges")
    tl, tr, br, bl = points
    if not (tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]):
        raise ValueError("Use top left, top right, bottom right, bottom left order")
    area = (
        sum(
            points[i][0] * points[(i + 1) % 4][1] - points[(i + 1) % 4][0] * points[i][1]
            for i in range(4)
        )
        / 2
    )
    if area < 0.02:
        raise ValueError("Workspace must cover at least 2% of the camera image")


@dataclass(frozen=True, slots=True)
class Calibration:
    corners: tuple[Point, ...]
    _matrix: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        points = tuple((float(x), float(y)) for x, y in self.corners)
        validate_points(points)
        object.__setattr__(self, "corners", points)
        rows: list[tuple[float, ...]] = []
        targets: list[float] = []
        for (x, y), (u, v) in zip(points, UNIT_CORNERS, strict=True):
            rows.extend(((x, y, 1, 0, 0, 0, -u * x, -u * y), (0, 0, 0, x, y, 1, -v * x, -v * y)))
            targets.extend((u, v))
        matrix = np.append(np.linalg.solve(rows, targets), 1).reshape(3, 3)
        matrix.setflags(write=False)
        object.__setattr__(self, "_matrix", matrix)

    def map(self, x: float, y: float) -> Point:
        """Return unclipped canvas fractions; the output adapter clips to its surface."""
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Pointer must be finite")
        mapped = self._matrix @ np.array((x, y, 1))
        if abs(mapped[2]) < 1e-9:
            raise ValueError("Pointer lies on the perspective horizon")
        return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps({"version": 1, "corners": self.corners}), encoding="utf-8")
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {"version", "corners"}:
            raise ValueError("Invalid calibration document")
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("Unsupported calibration version")
        corners = data["corners"]
        if not isinstance(corners, list) or any(
            not isinstance(p, list) or len(p) != 2 or any(type(v) not in (int, float) for v in p)
            for p in corners
        ):
            raise ValueError("Invalid calibration corners")
        return cls(tuple((p[0], p[1]) for p in corners))


@dataclass(slots=True)
class CalibrationSession:
    points: list[Point] = field(default_factory=list)

    @property
    def prompt(self) -> str:
        return f"Aim at {CORNERS[len(self.points)]} - SPACE to capture"

    def capture(self, point: Point) -> Calibration | None:
        candidate = (*self.points, point)
        if len(candidate) == 4:
            result = Calibration(candidate)
            return result
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in point):
            raise ValueError("Pointer must be inside the camera image")
        self.points.append(point)
        return None
