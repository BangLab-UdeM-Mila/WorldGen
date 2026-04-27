#!/usr/bin/env python3
"""
Dataset Manifest + Stats Generator
====================================
Scans a generated dataset directory and produces:
  manifest.json  — one entry per scene with all ground-truth fields
  stats.md       — human-readable statistics table

Usage
-----
  python generate_manifest.py dataset/ --output dataset/
  python generate_manifest.py dataset/ --output dataset/ --json-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────

def collect_scenes(root: Path) -> list[dict]:
    """Walk root recursively for metadata.json files and return their content."""
    scenes = []
    for meta_file in sorted(root.rglob("metadata.json")):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [warn] cannot read {meta_file}: {e}", file=sys.stderr)
            continue

        # Resolve video paths relative to the root
        scene_dir = meta_file.parent
        videos = {}
        for vname in ("video_observer", "video_closeup", "video_overhead"):
            vpath = scene_dir / f"{vname}.mp4"
            videos[vname] = str(vpath.relative_to(root)) if vpath.exists() else None

        entry = {
            "scene_id":             data.get("scene_id", scene_dir.name),
            "seed":                 data.get("seed"),
            "task_type":            data.get("task_type"),
            "object":               data.get("object"),
            "room":                 data.get("room"),
            "adversarial":          data.get("adversarial", False),
            "ground_truth_action":  data.get("ground_truth_action"),
            "safety_label":         data.get("safety_label"),
            "time_to_floor_s":      data.get("time_to_floor_s"),
            "interception_point_3d": data.get("interception_point_3d"),
            "videos":               videos,
            "metadata_path":        str(meta_file.relative_to(root)),
        }
        scenes.append(entry)

    return scenes


def build_stats(scenes: list[dict]) -> str:
    if not scenes:
        return "No scenes found.\n"

    total = len(scenes)
    lines = [
        "# ReactHuman Dataset Statistics",
        f"\n**Total scenes:** {total}\n",
    ]

    # ── By task type ──────────────────────────────────────────────────────────
    task_counts: Counter = Counter(s["task_type"] for s in scenes)
    lines += ["## Scenes per task type\n", "| Task type | Count | % |", "|---|---|---|"]
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        lines.append(f"| `{task}` | {count} | {pct:.1f}% |")

    # ── By ground-truth action ────────────────────────────────────────────────
    action_counts: Counter = Counter(s["ground_truth_action"] for s in scenes)
    lines += ["\n## Ground-truth action distribution\n",
              "| Action | Count | % |", "|---|---|---|"]
    for action, count in sorted(action_counts.items()):
        pct = count / total * 100
        lines.append(f"| `{action}` | {count} | {pct:.1f}% |")

    # ── Adversarial fraction ──────────────────────────────────────────────────
    adv_total = sum(1 for s in scenes if s.get("adversarial"))
    adv_pct   = adv_total / total * 100 if total else 0
    lines += [f"\n**Adversarial scenes:** {adv_total} / {total} ({adv_pct:.1f}%)\n"]

    # ── Per task × action breakdown ───────────────────────────────────────────
    task_action: dict[str, Counter] = defaultdict(Counter)
    for s in scenes:
        task_action[s["task_type"]][s["ground_truth_action"]] += 1

    all_actions = ["EXECUTE_CATCH", "TRIGGER_DODGE", "BRACE_FOR_IMPACT"]
    lines += ["\n## Task × action breakdown\n",
              "| Task type | CATCH | DODGE | BRACE | Total |",
              "|---|---|---|---|---|"]
    for task in sorted(task_action.keys()):
        ac = task_action[task]
        catch = ac.get("EXECUTE_CATCH", 0)
        dodge = ac.get("TRIGGER_DODGE", 0)
        brace = ac.get("BRACE_FOR_IMPACT", 0)
        tot   = catch + dodge + brace
        lines.append(f"| `{task}` | {catch} | {dodge} | {brace} | {tot} |")

    # ── TTF statistics ────────────────────────────────────────────────────────
    ttfs = [s["time_to_floor_s"] for s in scenes if s.get("time_to_floor_s") is not None]
    if ttfs:
        import statistics
        lines += [
            "\n## Time-to-floor statistics (seconds)\n",
            f"- Count with TTF: {len(ttfs)} / {total}",
            f"- Min:    {min(ttfs):.3f} s",
            f"- Max:    {max(ttfs):.3f} s",
            f"- Mean:   {statistics.mean(ttfs):.3f} s",
            f"- Median: {statistics.median(ttfs):.3f} s",
        ]

    # ── Room distribution ─────────────────────────────────────────────────────
    room_counts: Counter = Counter(s["room"] for s in scenes if s.get("room"))
    if room_counts:
        lines += ["\n## Room distribution\n", "| Room | Count |", "|---|---|"]
        for room, count in sorted(room_counts.items()):
            lines.append(f"| {room} | {count} |")

    # ── Video completeness ────────────────────────────────────────────────────
    complete = sum(
        1 for s in scenes
        if all(v is not None for v in (s.get("videos") or {}).values())
    )
    lines += [f"\n**Scenes with all 3 videos:** {complete} / {total}\n"]

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate dataset manifest + stats")
    p.add_argument("dataset_root", type=Path,
                   help="Root directory of the generated dataset.")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output directory (default: same as dataset_root).")
    p.add_argument("--json-only", action="store_true",
                   help="Write manifest.json only, skip stats.md.")
    args = p.parse_args()

    root   = args.dataset_root.resolve()
    outdir = (args.output or root).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"[error] Dataset root not found: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {root} …")
    scenes = collect_scenes(root)
    print(f"Found {len(scenes)} scenes.")

    # Write manifest.json
    manifest = {
        "dataset_root": str(root),
        "total_scenes": len(scenes),
        "scenes": scenes,
    }
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"Wrote {manifest_path}")

    if not args.json_only:
        stats_md    = build_stats(scenes)
        stats_path  = outdir / "stats.md"
        stats_path.write_text(stats_md, encoding="utf-8")
        print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
