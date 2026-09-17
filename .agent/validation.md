# Repository validation

Read [README.md](../README.md), [ARCHITECTURE.md](../ARCHITECTURE.md), and
`pyproject.toml` for current commands and contracts. Record command, working
directory, tested SHA, exit code, result summary, and evidence location. Do not
change product code merely to hide an environment failure.

## Environment and targeted checks

Use Windows Python `.\.venv\Scripts\python.exe` relative to the checkout. Python 3.12+ and the
`dev`, `vision`, and `ui` extras are needed for the complete suite. Follow the root
README installation instructions when setup is necessary and permitted.

Each writing worktree should have its own environment installed from that checkout.
Git worktrees do not copy the ignored `.venv`. Do not run an editable install for
one branch into another agent's shared venv: imports may silently test the wrong
source. Verify `gesture_control.__file__` resolves into the assigned checkout before
trusting its tests. If reusing an interpreter for read-only checks, explicitly select
the intended source path and verify imports; serialize any environment mutations.

Run the relevant test module(s) first, for example:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_motion.py --basetemp=.pytest_tmp_motion
```

Use checkout-specific temporary directories. Pytest clears its `--basetemp` directory;
inspect an existing path before using it and never point it at unknown user data or
a registered worktree. Never run concurrent pytest jobs with the same base directory.

## Primary integration checks

Run from the integration checkout. Run full tests after integration, and verify all
five gates on the final combined tree:

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_tmp
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m pip check
```

Inspect each exit code; a successful last command does not prove earlier commands
passed. Run `git diff --check` and inspect staged/unstaged changes before committing.
Do not weaken tests, suppress typing/lint failures without cause, reduce coverage,
or alter benchmark workloads to obtain green output. Identify pre-existing failures
separately. Missing tools, skips, blocked hardware, and incomplete checks must be
reported; they cannot be labeled passed. Check new dependencies for necessity and
compatibility; `pip check` checks installed dependency consistency, not security.

## Hardware and human acceptance

Automated tests use synthetic inputs/mocks and require no physical camera/display.
Device checks are separate and only relevant when the change needs them. Coordinate
one exclusive camera owner/window in the checkpoint and confirm release afterward.
Both commands below use the physical camera; headless does not mean synthetic:

```powershell
.\.venv\Scripts\python.exe -m gesture_control --max-frames 30
.\.venv\Scripts\python.exe -m gesture_control --headless --max-frames 30
```

Use root README manual acceptance for POINT/PINCH, hand loss, calibration, controls,
and shutdown where relevant. Distinguish an automated device smoke check from a
human physically performing gestures. Never claim that synthetic tests verify
physical accuracy or motion-to-photon latency.

## Performance work

Read [PERFORMANCE.md](../PERFORMANCE.md) and the reports linked from the root README.
Measure the baseline before modifying a hot path. Record before/after SHAs, workload,
Python/dependencies, hardware, frame count, warmup, camera mode, resolution,
calibration, diagnostics, and repeats. Use the same workload and environment;
avoid concurrent benchmarks or competing camera users. Define an acceptable
regression threshold for the objective before comparing results.

The existing deterministic downstream benchmark can be run at each revision with
distinct output paths (create the permitted output directory first):

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_pipeline.py --calibrated --diagnostics off --output recordings/synthetic-performance.json
```

Preserve baseline evidence before using the same output filename on a later run.
Report throughput separately from latency, median/p95 where useful, variability,
and quality/resource tradeoffs. The synthetic benchmark excludes camera, inference,
and native window costs; its FPS is capacity, not achieved webcam FPS. Host frame
age is not sensor-to-display physical latency. Retain comparable evidence and tune
or reject regressions. Documentation-only harness changes need no device test or
benchmark unless they make a new performance claim.
