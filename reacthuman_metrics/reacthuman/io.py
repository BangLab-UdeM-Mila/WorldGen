"""
IO helpers that bridge the on-disk `spec.json` format produced by
`genesis_scene_generation` to the typed :mod:`reacthuman.schemas`
classes used throughout the metric stack.

Keeping IO isolated here means that changes to the scene generator's
output schema cost a single function edit, not a search-and-replace
across the codebase.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from .schemas import (
    ActionPrimitive,
    DifficultyVector,
    GroundTruth,
    ModelPrediction,
    SafetyLabel,
    SceneSpec,
    TaskFamily,
)


# --------------------------------------------------------------------------- #
# Scene spec loader
# --------------------------------------------------------------------------- #
def load_scene_spec(spec_path: str | Path) -> SceneSpec:
    """Parse a single `spec.json` file emitted by the scene generator.

    The expected on-disk format is::

        {
          "scene_id": "<uuid>",
          "task_family": "object_drop",
          "difficulty": {
              "initial_state": 0.4,
              "action": 0.6,
              "physics": 0.3
          },
          "ground_truth": {
              "correct_action": "EXECUTE_CATCH",
              "impact_point_world": [0.12, 0.0, -0.34],
              "time_to_impact": 0.42,
              "reachable_mask": true,
              "safety_label": "safe",
              "object_value": 0.2
          },
          "camera": {
              "intrinsics": {"fx": ..., "fy": ..., "cx": ..., "cy": ...},
              "extrinsics": {"R": [...9...], "t": [...3...]}
          },
          "extras": { ... }
        }

    Unknown top-level keys are preserved verbatim under ``extras`` so
    that downstream code can read newly-added fields without forcing a
    schema bump.
    """
    spec_path = Path(spec_path)
    with open(spec_path, "r") as f:
        raw = json.load(f)

    diff = raw["difficulty"]
    difficulty = DifficultyVector(
        initial_state=float(diff["initial_state"]),
        action=float(diff["action"]),
        physics=float(diff["physics"]),
    )

    gt = raw["ground_truth"]
    ground_truth = GroundTruth(
        correct_action=ActionPrimitive(gt["correct_action"]),
        impact_point_world=tuple(gt["impact_point_world"]),  # type: ignore[arg-type]
        time_to_impact=float(gt["time_to_impact"]),
        reachable_mask=bool(gt["reachable_mask"]),
        safety_label=SafetyLabel(gt["safety_label"]),
        object_value=float(gt.get("object_value", 0.0)),
    )

    cam = raw.get("camera", {})
    intrinsics = cam.get("intrinsics", {})
    extrinsics = cam.get("extrinsics", {})

    extras = {
        k: v for k, v in raw.items()
        if k not in ("scene_id", "task_family", "difficulty", "ground_truth", "camera")
    }
    extras.update(raw.get("extras", {}))

    return SceneSpec(
        scene_id=raw["scene_id"],
        task_family=TaskFamily(raw["task_family"]),
        difficulty=difficulty,
        ground_truth=ground_truth,
        camera_intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
        extras=extras,
    )


def load_dataset(root: str | Path) -> List[SceneSpec]:
    """Walk a dataset root and load every `spec.json` it contains.

    Assumes the directory layout::

        <root>/<scene_id>/spec.json
        <root>/<scene_id>/metadata.json
        <root>/<scene_id>/video_observer.mp4
        <root>/<scene_id>/frame_start.png
        ...

    which is what `genesis_scene_generation` writes today.
    """
    root = Path(root)
    specs: List[SceneSpec] = []
    for spec_path in sorted(root.rglob("spec.json")):
        specs.append(load_scene_spec(spec_path))
    return specs


# --------------------------------------------------------------------------- #
# Prediction loader
# --------------------------------------------------------------------------- #
def load_predictions(jsonl_path: str | Path) -> List[ModelPrediction]:
    """Load model predictions from a JSON-Lines file.

    Each line is expected to be a JSON object with at least the fields
    of :class:`ModelPrediction`. Missing optional fields default to
    ``None`` / ``""``.
    """
    out: List[ModelPrediction] = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(
                ModelPrediction(
                    scene_id=d["scene_id"],
                    action=ActionPrimitive(d["action"]),
                    target_point_pixel=(
                        tuple(d["target_point_pixel"])  # type: ignore[arg-type]
                        if d.get("target_point_pixel") is not None else None
                    ),
                    target_point_world=(
                        tuple(d["target_point_world"])  # type: ignore[arg-type]
                        if d.get("target_point_world") is not None else None
                    ),
                    latency_seconds=float(d.get("latency_seconds", 0.0)),
                    rationale=d.get("rationale", ""),
                )
            )
    return out
