"""
Command-line entry point: evaluate a model's predictions against a
dataset and write a `report.json` summarising performance.

Example
-------
.. code-block:: bash

    python -m reacthuman.cli \\
        --dataset path/to/dataset_furniture_test \\
        --predictions path/to/claude_predictions.jsonl \\
        --model-name claude-opus-4.7 \\
        --out report_claude.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .io import load_dataset, load_predictions
from .metrics import aggregate_results, evaluate_scene


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reacthuman",
        description="ReactHuman benchmark — evaluation CLI",
    )
    p.add_argument(
        "--dataset", required=True, type=Path,
        help="Root directory containing one <scene_id>/spec.json per scene.",
    )
    p.add_argument(
        "--predictions", required=True, type=Path,
        help="JSON-Lines file of model predictions.",
    )
    p.add_argument(
        "--model-name", required=True, type=str,
        help="Identifier echoed into the report.",
    )
    p.add_argument(
        "--out", required=True, type=Path,
        help="Destination path for the JSON report.",
    )
    p.add_argument(
        "--depth-lookup", required=False, type=Path, default=None,
        help=("Optional JSON file mapping scene_id -> depth at predicted pixel, "
              "needed if predictions store pixel coordinates without depth."),
    )
    p.add_argument("--version", action="version", version=f"reacthuman {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    specs = load_dataset(args.dataset)
    preds = load_predictions(args.predictions)

    depth_lookup = {}
    if args.depth_lookup is not None:
        with open(args.depth_lookup) as f:
            depth_lookup = json.load(f)

    pred_by_id = {p.scene_id: p for p in preds}
    results = []
    for spec in specs:
        if spec.scene_id not in pred_by_id:
            continue
        results.append(
            evaluate_scene(
                spec,
                pred_by_id[spec.scene_id],
                depth_at_predicted_pixel=depth_lookup.get(spec.scene_id),
            )
        )

    if not results:
        print("error: no overlapping scene_ids between dataset and predictions",
              file=sys.stderr)
        return 1

    report = aggregate_results(args.model_name, results)
    report.to_json(args.out)
    print(f"Wrote {args.out} ({report.n_scenes} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
