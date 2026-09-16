# Gesture Control

A Python foundation for a real-time hand gesture control platform. The planned
system turns camera observations into stable gestures and application commands,
supporting pointer tracking, drawing, calibration, and deterministic landmark replay.

```text
Camera → hand landmarks → normalized landmarks → gesture predictions
       → temporal filtering / sequences → commands → output adapters
```

**Current status:** architecture and contracts only. The package provides validated
data types, configuration loading, structural interfaces, and synchronous frame
orchestration. Camera backends, classifiers, temporal state machines, desktop input,
training, and UI are extension points; none are implemented yet. Runtime dependencies
are limited to the Python standard library. There is no camera demo or CLI yet.

## Development

Requires Python 3.12 or newer. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

On POSIX systems, activate with `source .venv/bin/activate`. If activation is
unavailable, call the environment's Python executable directly.

## Configuration and composition

```python
from gesture_control.core.config import load_config

config = load_config("configs/default.toml")
```

TOML and JSON files may override any subset of defaults. Unknown keys, invalid types,
nonfinite numbers, and inconsistent thresholds raise errors. Loading has no implicit
environment overrides or device side effects. Defaults are starting points that need
calibration, not measured operating recommendations.

Implementations will be composed through these interfaces:

```python
from gesture_control.app import GestureControlApp

# detector, recognizer, temporal, mapper, and adapter are supplied implementations.
app = GestureControlApp(
    detector=detector,
    recognizer=recognizer,
    temporal=temporal,
    mapper=mapper,
    adapter=adapter,
)
# Supply an image and FrameContext to app.process_frame(image, context).
# Supply a recorded HandFrame to app.process_landmarks(frame).
```

The caller owns capture, scheduling, component construction, and resource cleanup.
Configuration sections should be passed to the components that need them.

See [ARCHITECTURE.md](ARCHITECTURE.md) for public contracts, units, invariants, and
ownership boundaries. Unit and composition tests live in `tests/unit` and
`tests/integration`; `tests/replay` is reserved for future replay infrastructure.
`datasets/`, `recordings/`, and `models/` retain directory placeholders only; their
generated contents are ignored by Git.
