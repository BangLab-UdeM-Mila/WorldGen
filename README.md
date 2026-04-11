# ReactHuman — Scene Generation Pipeline

Automated dataset generator for the **ReactHuman** benchmark: a dual-track evaluation of socio-physical reasoning and kinematic prediction in Multimodal Large Language Models (MLLMs).

---

## Overview

This pipeline generates large-scale, photo-physically simulated videos of household objects dropping off tables inside realistic indoor rooms. Every generated scene is:

- **Reproducible** — given the same seed, the simulation is bit-for-bit identical.
- **Labeled** — each video ships with ground-truth action labels, physical trajectories, and interception coordinates.
- **Scalable** — one command generates 10 000+ scenes in parallel.
- **LLM-compatible** — natural-language descriptions are automatically converted into structured scene specs via the Claude API.

The pipeline supports **two generation modes**:

| Mode | Description | LLM needed? |
|------|-------------|-------------|
| `procedural` | Fully random sampling from asset library | No |
| `llm` | Natural-language descriptions → physics variants | Yes (Claude API) |

---

## Architecture

```
Natural Language Descriptions (optional)
              │
              ▼
  ┌───────────────────────┐
  │   LLM Planner         │  pipeline/llm_planner.py
  │   (Claude API)        │  • Parses description
  │   object + room type  │  • Validates against catalogue
  └───────────┬───────────┘
              │  (object_name, room_type)
              ▼
  ┌───────────────────────┐
  │   Randomizer          │  pipeline/randomizer.py
  │   (pure Python)       │  • Samples all physics parameters
  │                       │  • Deterministic from seed
  └───────────┬───────────┘
              │  SceneSpec (Pydantic)
              ▼
  ┌───────────────────────┐
  │   Batch Runner        │  pipeline/batch_runner.py
  │                       │  • Spawns one subprocess per scene
  │                       │  • Parallel execution
  └───────────┬───────────┘
              │  subprocess per scene
              ▼
  ┌───────────────────────┐
  │   Scene Builder       │  scene_builder.py
  │   (Genesis physics)   │  • Builds room + table + object
  │                       │  • Runs 240 Hz simulation
  │                       │  • Records 3 camera streams
  └───────────┬───────────┘
              │
              ▼
     dataset/<scene_id>/
       ├── spec.json          ← full SceneSpec (reproducible)
       ├── metadata.json      ← ground-truth labels + physics measurements
       ├── video_observer.mp4
       ├── video_closeup.mp4
       └── video_overhead.mp4
```

---

## Project Structure

```
genesis_scene_generation/
│
├── generate_dataset.py        ← Main CLI entry point
├── scene_builder.py           ← Generic Genesis scene from any SceneSpec
├── scene_runner.py            ← Subprocess entry point (one scene per call)
│
├── pipeline/
│   ├── scene_spec.py          ← Pydantic schema (the "language" of a scene)
│   ├── randomizer.py          ← Procedural parameter sampling
│   ├── llm_planner.py         ← Claude API integration
│   └── batch_runner.py        ← Parallel subprocess management
│
├── asset_library/
│   ├── objects.yaml           ← Object catalogue (10 objects, ground-truth labels)
│   └── rooms.yaml             ← Room templates (4 types) + lighting presets (4)
│
├── assets/
│   ├── download_textures.py   ← Downloads CC0 textures from PolyHaven
│   └── textures/
│       ├── floor/             ← Herringbone parquet (diff/rough/nor)
│       └── wall/              ← Plastered wall (diff/rough/nor)
│
├── scenes/
│   └── plate_drop.py          ← Legacy single-scene script (plate only)
├── config/
│   └── plate_drop.yaml        ← Legacy config for single-scene testing
└── outputs/                   ← Output from single-scene runs
```

---

## Environment Setup

The virtual environment is already configured at `/home/ubuntu/Yizhan3d/.venv`.

```bash
source /home/ubuntu/Yizhan3d/.venv/bin/activate
cd /home/ubuntu/Yizhan3d/genesis_scene_generation
```

**Required environment variable for headless rendering:**
```bash
export PYOPENGL_PLATFORM=egl
```

**For LLM mode only:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Download textures (one-time):**
```bash
python assets/download_textures.py
```

---

## Quick Start

### 1. Dry-run (no simulation, just preview specs)

```bash
python generate_dataset.py --n 10 --dry-run
```

Output:
```
     0  id=5e49a38d  seed=0  obj=foam_anvil         room=living   gt=EXECUTE_CATCH  adv=True
     1  id=114cd2f6  seed=1  obj=plastic_bottle     room=living   gt=EXECUTE_CATCH  adv=False
     2  id=e07cdeef  seed=2  obj=foam_anvil         room=kitchen  gt=EXECUTE_CATCH  adv=True
     ...
```

### 2. Generate scenes (procedural mode)

```bash
# 10 scenes, all objects and rooms
python generate_dataset.py --n 10 --output dataset/

# 500 scenes, only dangerous objects in kitchen/dining rooms
python generate_dataset.py --n 500 \
    --objects chef_knife,hot_iron,glass_shard \
    --rooms kitchen,dining \
    --output dataset/dangerous_kitchen/

# Extend an existing dataset (start seed at 1000 to avoid overlap)
python generate_dataset.py --n 500 --seed-start 1000 --output dataset/
```

### 3. Generate scenes (LLM mode)

Create a text file with one scenario description per line:

```
# scenarios.txt
A ceramic plate slides off the kitchen counter while someone is cooking.
A sharp chef's knife tips off the edge of the dining table.
A child's red apple rolls off the coffee table in the living room.
A hot clothes iron falls from the ironing board in the home office.
```

Then run:

```bash
python generate_dataset.py \
    --mode llm \
    --descriptions scenarios.txt \
    --variants 20 \
    --output dataset/llm_generated/
```

This calls the Claude API **once per line** to select the appropriate object and room, then generates 20 physics variants per description (different seeds, velocities, camera angles). 80 scenes total from 4 descriptions.

---

## CLI Reference

```
generate_dataset.py [OPTIONS]

Mode
  --mode {procedural,llm}     Generation mode. Default: procedural

Procedural mode
  --n INT                     Number of scenes to generate. Default: 10
  --seed-start INT            First random seed. Default: 0
  --objects STR               Comma-separated object names (see asset library).
                              Default: all objects.
  --rooms STR                 Comma-separated room types: dining,kitchen,living,office.
                              Default: all rooms.
  --adversarial-prob FLOAT    Fraction of adversarial scenes. Default: 0.15

LLM mode
  --descriptions FILE         Text file: one natural-language description per line.
  --variants INT              Physics variants per description. Default: 10

Execution
  --output DIR                Root output directory. Default: dataset/
  --workers INT               Parallel worker processes. Default: 1
                              Set to 1 on a single GPU to avoid OOM.
  --dry-run                   Preview SceneSpecs without running simulation.
```

---

## Asset Library

### Objects (`asset_library/objects.yaml`)

| Name | Category | Track-1 GT Action | Mass Hint | Description |
|------|----------|-------------------|-----------|-------------|
| `ceramic_plate` | safe | `EXECUTE_CATCH` | light | White ceramic dinner plate |
| `red_apple` | safe | `EXECUTE_CATCH` | light | Ripe red apple |
| `coffee_mug` | safe | `EXECUTE_CATCH` | light | White ceramic mug |
| `paperback_book` | safe | `EXECUTE_CATCH` | light | Lightweight paperback |
| `plastic_bottle` | safe | `EXECUTE_CATCH` | light | Half-empty water bottle |
| `chef_knife` | dangerous | `TRIGGER_DODGE` | medium | Sharp chef's knife (mesh) |
| `hot_iron` | dangerous | `BRACE_FOR_IMPACT` | heavy | Heavy clothes iron |
| `glass_shard` | dangerous | `TRIGGER_DODGE` | light | Thin, transparent glass shard |
| `foam_anvil` | **adversarial** | `EXECUTE_CATCH` | light | *Looks* like cast-iron anvil, actually foam |
| `steel_apple` | **adversarial** | `BRACE_FOR_IMPACT` | heavy | *Looks* like red apple, actually steel |

**Adversarial objects** deliberately mislead visual priors — the correct action can only be inferred from the object's acceleration in the first frames (Track 2 physics reasoning).

**Adding a new object:** append an entry to `asset_library/objects.yaml`. No code changes needed.

### Rooms (`asset_library/rooms.yaml`)

| Type | Lighting Preset | Floor | Wall | Window |
|------|----------------|-------|------|--------|
| `dining` | Warm pendant lamp | Herringbone parquet | Off-white plaster | North wall |
| `kitchen` | Bright overhead | Tile | Cream plaster | North wall |
| `living` | Soft dual-point | Laminate wood | Warm beige | North wall |
| `office` | Cool fluorescent | Laminate | Cool grey | North wall |

Each room has slight randomisation of wall colour tint, table position offset, and light intensity across seeds.

**Adding a new room:** append an entry to `asset_library/rooms.yaml`. No code changes needed.

---

## Dataset Output Format

Each scene produces a directory `<output>/<scene_id>/`:

```
dataset/bf67e379/
├── spec.json          ← Full SceneSpec (all parameters, 100% reproducible)
├── metadata.json      ← Ground-truth labels and physics measurements
├── video_observer.mp4 ← Standing-observer angle (1280×720 @ 60fps)
├── video_closeup.mp4  ← Close-up beside table edge
└── video_overhead.mp4 ← Bird's-eye view
```

### `metadata.json` fields

```json
{
  "scene_id": "bf67e379",
  "seed": 11,
  "object": "red_apple",
  "room": "dining",
  "ground_truth_action": "EXECUTE_CATCH",
  "safety_label": "safe",
  "adversarial": false,
  "time_to_floor_s": 0.5083,
  "interception_point_3d": [0.6001, -0.0689, 0.812]
}
```

| Field | Description |
|-------|-------------|
| `ground_truth_action` | Track-1 label: `EXECUTE_CATCH` / `TRIGGER_DODGE` / `BRACE_FOR_IMPACT` |
| `safety_label` | `safe` / `caution` / `dangerous` |
| `adversarial` | `true` if visual appearance contradicts physics |
| `time_to_floor_s` | Track-2 GT: time from first frame to floor contact |
| `interception_point_3d` | Track-2 GT: 3D world position when object crosses table edge |

### `spec.json`

The complete `SceneSpec` in JSON format. Pass it to `scene_runner.py` to reproduce the exact simulation:

```bash
python scene_runner.py --spec dataset/bf67e379/spec.json --output /tmp/repro/
```

---

## SceneSpec Schema

The `SceneSpec` (defined in `pipeline/scene_spec.py`) is the central data structure. Every configurable aspect of a scene is expressed as a validated Pydantic model.

```
SceneSpec
├── scene_id, seed, description, adversarial
├── room: RoomSpec          (type, dimensions, colours, window)
├── table: TableSpec        (size, position, material)
├── object: ObjectSpec      (morph, physics, colour, ground-truth label)
├── drop: DropSpec          (start position, initial velocity, rotation)
├── cameras: [CameraSpec]   (3 cameras: observer, closeup, overhead)
├── lighting: LightingSpec  (ambient, directional/point lights)
└── simulation params       (dt, substeps, duration)
```

---

## Evaluation Tracks

### Track 1 — Semantic & Instinct

Input: video frames (no coordinates).
Task: classify the correct response action.
Metric: `Semantic Action Accuracy (%)`, `Fatal Execution Rate`.

Ground truth: `ground_truth_action` field in `metadata.json`.

### Track 2 — Physical & Kinematic

Input: video frames + initial state variables (position, velocity).
Task: predict the 3D interception point and time to floor contact.
Metrics: `Euclidean Error (cm)`, `Time-to-Collision Error (s)`.

Ground truth: `interception_point_3d` and `time_to_floor_s` in `metadata.json`.

---

## Reproducibility

Every scene is keyed by its integer `seed`. To reproduce scene `bf67e379` from scratch:

```python
from pipeline.randomizer import Randomizer

r    = Randomizer()
spec = r.sample(seed=11)          # bf67e379 was generated with seed=11
print(spec.to_json())             # identical to dataset/bf67e379/spec.json
```

Or re-run the simulation directly:

```bash
python scene_runner.py \
    --spec dataset/bf67e379/spec.json \
    --output /tmp/repro_bf67e379/
```

---

## Paper Citation Method

This pipeline was used to generate the ReactHuman benchmark dataset. When citing the generation methodology in a paper:

> *"Dataset Generation. We develop a two-stage procedural generation framework. In Stage 1, domain experts specify scenario categories in natural language. A frozen Claude-3.5-Sonnet model [citation] parses each description into a structured SceneSpec object, selecting the object and room from a validated catalogue. In Stage 2, a deterministic randomizer (NumPy default_rng) samples physical parameters — including object overhang fraction, initial velocity, table position offset, camera jitter, and lighting intensity — from pre-defined distributions conditioned on the seed. All scenes are simulated in Genesis [citation] at 240 Hz using a rigid-body solver with CoACD collision decomposition. Given a seed, any scene is bit-for-bit reproducible."*

The Claude API is a **documented component** of the methodology (analogous to using SMPL for human bodies or PhysX for physics), not an undisclosed tool.

---

## Extending the Pipeline

### Add a new object

1. (Optional) Place a `.obj` mesh in `../assets/meshes/`.
2. Append an entry to `asset_library/objects.yaml` following the existing schema.
3. No code changes required.

### Add a new room type

1. Append an entry under `rooms:` in `asset_library/rooms.yaml`.
2. Optionally add a new entry under `lighting_presets:`.
3. No code changes required.

### Add a new camera angle

Edit `pipeline/randomizer.py → _make_cameras()` to add a fourth camera spec, or add a fixed camera in `scene_builder.py → _add_cameras()`.

### Swap the renderer to RayTracer (photorealistic)

Install LuisaRender:
```bash
pip install genesis-world[raytrace]
```

The `scene_builder.py` will automatically use it when available.  
Change `renderer.type` to `RayTracer` in `config/plate_drop.yaml` for the single-scene runner.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` | `sudo apt-get install libegl-mesa0` then `export PYOPENGL_PLATFORM=egl` |
| `ModuleNotFoundError: No module named 'LuisaRenderPy'` | Renderer falls back to Rasterizer automatically. Install with `pip install genesis-world[raytrace]` for path-tracing. |
| `TypeError: Could not resolve authentication method` | `export ANTHROPIC_API_KEY=sk-ant-...` (LLM mode only) |
| OOM / CUDA out of memory with `--workers > 1` | Set `--workers 1` on a single-GPU machine. Genesis holds GPU memory for the full scene lifetime. |
| Scene times out (default 180s) | Reduce `--n` or increase timeout in `pipeline/batch_runner.py → BatchRunner(timeout=...)`. |
