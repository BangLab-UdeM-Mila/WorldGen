#!/usr/bin/env python3
"""
Track 1 Evaluation Script — Semantic Action Classification
============================================================
Evaluates model predictions against ground-truth action labels from the
ReactHuman benchmark.

Input format (model predictions file)
--------------------------------------
One prediction per line in JSON format:
  {"scene_id": "abc12345", "predicted_action": "EXECUTE_CATCH"}
  {"scene_id": "def67890", "predicted_action": "TRIGGER_DODGE"}
  ...

Or as a JSON array:
  [{"scene_id": "abc12345", "predicted_action": "EXECUTE_CATCH"}, ...]

Usage
-----
  # Full evaluation
  python evaluate_track1.py manifest.json predictions.jsonl

  # Evaluate from a dataset root (uses manifest.json inside it)
  python evaluate_track1.py dataset/ predictions.jsonl

  # JSON output for downstream processing
  python evaluate_track1.py manifest.json predictions.jsonl --json

  # Per-task breakdown table
  python evaluate_track1.py manifest.json predictions.jsonl --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


VALID_ACTIONS = {"EXECUTE_CATCH", "TRIGGER_DODGE", "BRACE_FOR_IMPACT"}


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> dict[str, dict]:
    """Return {scene_id: entry} from manifest.json."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = data.get("scenes", data) if isinstance(data, dict) else data
    result = {}
    for entry in scenes:
        sid = entry.get("scene_id")
        if sid:
            result[sid] = entry
    return result


def load_predictions(pred_path: Path) -> dict[str, str]:
    """Return {scene_id: predicted_action} from a JSONL or JSON array file."""
    text = pred_path.read_text(encoding="utf-8").strip()
    preds: dict[str, str] = {}

    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    for rec in records:
        sid    = rec.get("scene_id")
        action = rec.get("predicted_action") or rec.get("prediction") or rec.get("action")
        if sid and action:
            preds[sid] = action.strip().upper()

    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    gt_map: dict[str, dict],
    pred_map: dict[str, str],
) -> dict:
    """Compute all Track 1 metrics."""
    matched_ids = set(gt_map) & set(pred_map)
    missing     = set(gt_map) - set(pred_map)
    extra       = set(pred_map) - set(gt_map)

    if not matched_ids:
        return {
            "error": "No scene IDs matched between manifest and predictions.",
            "manifest_scenes": len(gt_map),
            "predicted_scenes": len(pred_map),
            "matched": 0,
        }

    # ── Overall accuracy ──────────────────────────────────────────────────────
    correct  = sum(1 for sid in matched_ids
                   if gt_map[sid]["ground_truth_action"] == pred_map[sid])
    total    = len(matched_ids)
    accuracy = correct / total if total else 0.0

    # ── Per-task accuracy ─────────────────────────────────────────────────────
    task_correct: Counter = Counter()
    task_total:   Counter = Counter()
    for sid in matched_ids:
        task = gt_map[sid].get("task_type", "unknown")
        task_total[task]  += 1
        if gt_map[sid]["ground_truth_action"] == pred_map[sid]:
            task_correct[task] += 1
    task_accuracy = {
        task: round(task_correct[task] / task_total[task], 4)
        for task in task_total
    }

    # ── Per-action accuracy ───────────────────────────────────────────────────
    action_correct: Counter = Counter()
    action_total:   Counter = Counter()
    for sid in matched_ids:
        gt = gt_map[sid]["ground_truth_action"]
        action_total[gt]  += 1
        if gt == pred_map[sid]:
            action_correct[gt] += 1
    action_accuracy = {
        a: round(action_correct[a] / action_total[a], 4)
        for a in action_total
    }

    # ── Confusion matrix ──────────────────────────────────────────────────────
    confusion: dict[str, Counter] = defaultdict(Counter)
    for sid in matched_ids:
        gt   = gt_map[sid]["ground_truth_action"]
        pred = pred_map[sid]
        confusion[gt][pred] += 1

    # ── Adversarial metrics ───────────────────────────────────────────────────
    adv_ids     = [sid for sid in matched_ids if gt_map[sid].get("adversarial")]
    adv_total_n = len(adv_ids)
    adv_correct_n = sum(1 for sid in adv_ids
                        if gt_map[sid]["ground_truth_action"] == pred_map[sid])
    adv_fooled_n  = adv_total_n - adv_correct_n
    adv_metrics = {
        "adv_total":   adv_total_n,
        "adv_correct": adv_correct_n,
        "adv_fooled":  adv_fooled_n,
        "adv_correct_rate": round(adv_correct_n / adv_total_n, 4) if adv_total_n else None,
        "adv_fooled_rate":  round(adv_fooled_n  / adv_total_n, 4) if adv_total_n else None,
    }

    return {
        "overall_accuracy": round(accuracy, 4),
        "matched_scenes":   total,
        "correct_scenes":   correct,
        "missing_scenes":   len(missing),
        "extra_predictions": len(extra),
        "per_task_accuracy": task_accuracy,
        "per_action_accuracy": action_accuracy,
        "confusion_matrix":  {gt: dict(preds) for gt, preds in confusion.items()},
        "adversarial":        adv_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def render_report(metrics: dict, verbose: bool = False) -> str:
    if "error" in metrics:
        return f"[error] {metrics['error']}\n"

    lines = [
        "# Track 1 Evaluation Report",
        f"\n**Overall Accuracy:** {metrics['overall_accuracy']*100:.2f}%  "
        f"({metrics['correct_scenes']} / {metrics['matched_scenes']} scenes)\n",
    ]

    if metrics["missing_scenes"]:
        lines.append(f"⚠ Missing predictions for {metrics['missing_scenes']} scenes.")
    if metrics["extra_predictions"]:
        lines.append(f"⚠ {metrics['extra_predictions']} extra predictions (not in manifest).")

    # ── Per-action ────────────────────────────────────────────────────────────
    lines += ["\n## Per-action Accuracy\n", "| Action | Accuracy |", "|---|---|"]
    for action in sorted(metrics["per_action_accuracy"]):
        acc = metrics["per_action_accuracy"][action]
        lines.append(f"| `{action}` | {acc*100:.1f}% |")

    # ── Adversarial ───────────────────────────────────────────────────────────
    adv = metrics["adversarial"]
    if adv["adv_total"]:
        lines += [
            "\n## Adversarial Scenes\n",
            f"| Metric | Value |",
            "|---|---|",
            f"| Total adversarial | {adv['adv_total']} |",
            f"| adv_correct | {adv['adv_correct']} ({adv['adv_correct_rate']*100:.1f}%) |",
            f"| adv_fooled  | {adv['adv_fooled']}  ({adv['adv_fooled_rate']*100:.1f}%) |",
        ]

    # ── Per-task ──────────────────────────────────────────────────────────────
    if verbose:
        lines += ["\n## Per-task Accuracy\n",
                  "| Task type | Accuracy |", "|---|---|"]
        for task in sorted(metrics["per_task_accuracy"]):
            acc = metrics["per_task_accuracy"][task]
            lines.append(f"| `{task}` | {acc*100:.1f}% |")

        # ── Confusion matrix ──────────────────────────────────────────────────
        conf      = metrics["confusion_matrix"]
        all_acts  = sorted({a for row in conf.values() for a in row} | set(conf.keys()))
        lines += ["\n## Confusion Matrix (rows=GT, cols=Predicted)\n"]
        header = "| GT \\ Pred | " + " | ".join(f"`{a}`" for a in all_acts) + " |"
        sep    = "|---|" + "---|" * len(all_acts)
        lines += [header, sep]
        for gt in all_acts:
            row = conf.get(gt, {})
            cells = " | ".join(str(row.get(pred, 0)) for pred in all_acts)
            lines.append(f"| `{gt}` | {cells} |")

    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Track 1 evaluation (semantic action)")
    p.add_argument("manifest",    type=Path,
                   help="manifest.json file or dataset root directory.")
    p.add_argument("predictions", type=Path,
                   help="Model predictions file (JSONL or JSON array).")
    p.add_argument("--json",    action="store_true",
                   help="Output raw JSON metrics instead of a human-readable report.")
    p.add_argument("--verbose", action="store_true",
                   help="Show per-task accuracy and confusion matrix.")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Write report to this file (default: stdout).")
    args = p.parse_args()

    # Resolve manifest path
    manifest_path = args.manifest
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    if not manifest_path.exists():
        print(f"[error] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    if not args.predictions.exists():
        print(f"[error] Predictions file not found: {args.predictions}", file=sys.stderr)
        sys.exit(1)

    gt_map   = load_manifest(manifest_path)
    pred_map = load_predictions(args.predictions)
    metrics  = compute_metrics(gt_map, pred_map)

    if args.json:
        output_text = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    else:
        output_text = render_report(metrics, verbose=args.verbose)

    if args.output:
        args.output.write_text(output_text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(output_text, end="")


if __name__ == "__main__":
    main()
