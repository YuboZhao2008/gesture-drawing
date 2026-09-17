"""Reproducible synthetic gesture-classification robustness evaluation.

Use --baseline-root with an unchanged checkout of the baseline revision to run
both implementations on the exact same generated observations. No timing claims
or training are made. Camera/detector accuracy cannot be inferred from this corpus.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from gesture_control.recognition import GeometricGestureRecognizer
from gesture_control.training.evaluation import evaluate
from gesture_control.vision import PalmNormalizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--variants", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1709)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.variants < 1:
        parser.error("variants must be positive")
    report = {
        "scope": "synthetic gesture-classification test data; NOT detector accuracy",
        "schema_version": 1,
        "seed": args.seed,
        "variants_per_group": args.variants,
        "improved_geometry": evaluate(
            GeometricGestureRecognizer(PalmNormalizer()), variants=args.variants, seed=args.seed
        ),
    }
    if args.baseline_root:
        directory = args.baseline_root.resolve() / "src" / "gesture_control" / "recognition"
        name = "_gesture_evaluation_baseline"
        spec = importlib.util.spec_from_file_location(
            name, directory / "__init__.py", submodule_search_locations=[str(directory)]
        )
        if spec is None or spec.loader is None:
            parser.error("cannot import baseline recognition package")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        report["baseline_geometry"] = evaluate(
            module.GeometricGestureRecognizer(PalmNormalizer()),
            variants=args.variants,
            seed=args.seed,
        )
        report["baseline_root"] = str(args.baseline_root.resolve())
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
