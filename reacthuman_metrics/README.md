# ReactHuman — Evaluation Metric Suite

[![tests](https://img.shields.io/badge/tests-14%20passed-brightgreen.svg)](#testing)
[![python](https://img.shields.io/badge/python-≥3.9-blue.svg)](#installation)

A reference implementation of the evaluation metric suite for the
**ReactHuman** benchmark. ReactHuman measures whether Multimodal Large
Language Models (MLLMs) can produce reactive, embodied decisions in
the face of sudden physical events. This package provides the
metrics, schemas, and command-line tooling that take a model's
predictions and the simulator's ground truth and return a
publication-ready benchmark report.

The package is deliberately decoupled from the scene generator
(`genesis_scene_generation/`) and from any specific model API. It
consumes the on-disk artefacts produced by the generator and the
JSON-Lines predictions emitted by a model harness; this separation
mirrors common practice in established benchmarks (e.g. PhysBench,
LeVERB-Bench) and makes the metric suite reusable across model
families and across future versions of the dataset.

## Contents

- [Motivation and scope](#motivation-and-scope)
- [Metric definitions](#metric-definitions)
- [Data interfaces](#data-interfaces)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line usage](#command-line-usage)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Reproducibility notes](#reproducibility-notes)
- [Citation](#citation)

---

## Motivation and scope

Existing physical-reasoning benchmarks for MLLMs are predominantly
*passive*: the model selects an answer from a multiple-choice menu
(PhysBench, PhysReason) or rates the plausibility of a generated
video (WorldBench, VideoPhy). LeVERB-Bench and HumanoidBench take a
step toward embodiment, but their tasks are long-horizon and
quasi-static, not reactive. ReactHuman closes this gap by evaluating
*sub-second decision making under sudden physical events*, where the
correct response is a discrete action primitive paired with a
metric-space interception target.

The metric suite is organised around the dual-track design of the
benchmark:

- **Track 1 — Semantic & Instinct.** Whether the model chose the
  correct action primitive (`EXECUTE_CATCH`, `TRIGGER_DODGE`,
  `BRACE_FOR_IMPACT`).
- **Track 2 — Physical & Kinematic.** Whether the model's predicted
  interception point and time-to-collision are close enough to the
  simulator ground truth to be physically executable.

Both tracks are reported jointly: a model that succeeds on Track 1
but fails on Track 2 is "text-smart but spatially blind", while one
that succeeds on Track 2 without Track 1 is "geometrically precise
but semantically reckless". Discriminating between these failure
modes is the primary diagnostic value of the benchmark.

Every scene is additionally annotated with a three-axis
*difficulty vector* (`initial_state`, `action`, `physics`) so that
all headline numbers can be sliced by difficulty bin. This permits
the analyses that PhysBench's category-only taxonomy cannot support,
notably "On which axis does each model fail first?".

## Metric definitions

| Metric | Track | Range | Direction | Description |
|---|---|---|---|---|
| `semantic_action_accuracy` | 1 | [0, 1] | ↑ | Fraction of scenes whose predicted primitive equals the ground-truth primitive. |
| `fatal_execution_rate` | 1 | [0, 1] | ↓ | Fraction of scenes whose predicted primitive would cause catastrophic harm if executed (dangerous object + `EXECUTE_CATCH`). |
| `adversarial_fool_rate` | 1 | [0, 1] ∪ {None} | ↓ | Among adversarial scenes, fraction in which the model picked the action implied by surface appearance rather than by physics. `None` when the corpus contains no adversarial scenes. |
| `trajectory_error_cm` | 2 | [0, ∞) cm | ↓ | Euclidean distance between predicted and ground-truth interception points. Computed only when the ground-truth action is `EXECUTE_CATCH`. |
| `time_to_collision_error` | 2 | (-∞, ∞) s | →0 | Signed gap between predicted and true time-to-impact (negative = late). |
| `reachable_prediction_rate` | 2 | [0, 1] ∪ {None} | ↑ | Fraction of catch predictions that lie inside the humanoid's kinematic reachability hull. Novel to ReactHuman. |

For each metric, the rationale, exclusion conditions, and aggregation
behaviour are documented in the module-level docstring of
`reacthuman.metrics.track1` and `reacthuman.metrics.track2`.

### Difficulty-conditioned breakdown

`aggregate_results()` additionally returns the action accuracy
broken down into four bins per axis:

| Bin | Boundary |
|---|---|
| `easy` | [0.00, 0.25) |
| `medium` | [0.25, 0.50) |
| `hard` | [0.50, 0.75) |
| `adversarial` | [0.75, 1.00] |

These bins power the four-panel radar figure that we recommend as
the paper's main result.

## Data interfaces

The metric suite operates on two typed objects, both serialisable
as JSON:

### `SceneSpec` — produced by the scene generator

```json
{
  "scene_id": "bc51dcdc",
  "task_family": "object_drop",
  "difficulty": {
    "initial_state": 0.40,
    "action":        0.60,
    "physics":       0.30
  },
  "ground_truth": {
    "correct_action":     "EXECUTE_CATCH",
    "impact_point_world": [0.12, 0.00, -0.34],
    "time_to_impact":     0.42,
    "reachable_mask":     true,
    "safety_label":       "safe",
    "object_value":       0.20
  },
  "camera": {
    "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0},
    "extrinsics": {"R": [1,0,0, 0,1,0, 0,0,1], "t": [0,0,0]}
  },
  "extras": {
    "appearance_implied_action": "TRIGGER_DODGE",
    "asset_glb": "objaverse/knife_steel_0023.glb"
  }
}
```

### `ModelPrediction` — produced by the model harness

One JSON object per line, in a `.jsonl` file:

```jsonl
{"scene_id": "bc51dcdc", "action": "EXECUTE_CATCH", "target_point_world": [0.11, 0.0, -0.33], "latency_seconds": 1.81, "rationale": "The plate is reachable and not dangerous."}
{"scene_id": "ec91e8c0", "action": "TRIGGER_DODGE", "target_point_pixel": [342.0, 198.0], "latency_seconds": 1.42, "rationale": "Falling knife, side-step."}
```

The harness may emit either `target_point_world` (3-D) or
`target_point_pixel` (2-D). In the latter case the metric stack
back-projects through the depth buffer if supplied a
`scene_id → depth` lookup at the predicted pixel.

## Installation

```bash
git clone https://github.com/BangLab-UdeM-Mila/WorldGen.git
cd WorldGen
git checkout mengyang/physics-benchmark
pip install -e .[dev]
```

Python 3.9 or newer; the only mandatory dependency is `numpy ≥ 1.24`.
`pytest` is required for the test suite.

## Quick start

```python
from reacthuman.metrics import aggregate_results, evaluate_scene
from reacthuman.io import load_dataset, load_predictions

specs = load_dataset("path/to/dataset_furniture_test")
preds = load_predictions("path/to/claude_predictions.jsonl")
pred_by_id = {p.scene_id: p for p in preds}

results = [
    evaluate_scene(spec, pred_by_id[spec.scene_id])
    for spec in specs
    if spec.scene_id in pred_by_id
]
report = aggregate_results("claude-opus-4.7", results)
report.to_json("report_claude.json")
```

The runnable demonstration in `examples/run_demo.py` constructs three
synthetic scenes in memory and prints a formatted report. Use it to
verify the installation before plugging in real predictions.

## Command-line usage

```bash
reacthuman \
    --dataset      path/to/dataset_furniture_test \
    --predictions  path/to/claude_predictions.jsonl \
    --model-name   claude-opus-4.7 \
    --out          report_claude.json
```

For predictions that contain pixel coordinates only, supply
`--depth-lookup path/to/depths.json`.

## Project layout

```
reacthuman/
├── reacthuman/
│   ├── __init__.py
│   ├── schemas/__init__.py        # typed dataclasses & enums
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── track1.py              # semantic metrics
│   │   ├── track2.py              # kinematic metrics
│   │   ├── scene.py               # per-scene evaluation
│   │   └── aggregate.py           # per-corpus aggregation
│   ├── io.py                      # JSON ⇄ schema converters
│   └── cli.py                     # `reacthuman` command-line tool
├── tests/test_metrics.py          # 14 unit tests
├── examples/run_demo.py           # in-memory end-to-end demo
├── configs/                       # placeholder for model-specific configs
├── docs/                          # extended design notes
├── setup.cfg
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Testing

```bash
pytest -q
```

All 14 unit tests must pass before any merge to
`mengyang/physics-benchmark`. Failures during development are usually
traceable to either (i) a schema field that was added to the scene
generator without being added to `schemas/__init__.py`, or (ii) a
prediction file emitted before the harness was updated to the
current `ActionPrimitive` enum values.

## Reproducibility notes

- Random seeds are not consumed by this package; determinism is
  established upstream in the scene generator. Every `spec.json`
  carries a stable `scene_id`, and aggregation is deterministic
  given a fixed multiset of `SceneResult`s.
- All metric definitions are pure functions of the schema objects;
  there is no global state.
- The aggregator's bin boundaries are exposed via
  `reacthuman.metrics.aggregate.DIFFICULTY_BINS` so that downstream
  plotting code uses identical edges.

## Citation

If this metric suite is used in published work, please cite the
ReactHuman paper:

```bibtex
@inproceedings{reacthuman2026,
  title = {ReactHuman: A Dual-Track Benchmark for Evaluating
           Reactive Decision-Making of Multimodal Large Language
           Models under Sudden Physical Events},
  author = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026}
}
```

## License

Released under the MIT License. See `LICENSE`.
