# LLM Mode — How It Works

LLM mode converts **natural-language scenario descriptions** into fully reproducible
`SceneSpec` objects using the Claude API.  It is the counterpart to procedural mode:
instead of randomly sampling from the asset library, you describe what you want in
plain English and the system figures out the task type, selects or creates the right
object, and generates N physics variants of that scenario.

---

## Pipeline Overview

```
Description (text)
        │
        ▼
  Claude API call   ←──  System prompt (task catalogue + rules)
        │                User message  (object/room catalogues + description)
        ▼
  JSON response:
    task_type, object_name, room_type
    [+ hints block if speed/height cues detected]
    [+ new_object block if object is unknown]
        │
        ├── object already in catalogue? ──► pass to Randomizer
        │
        └── new object? ──► download GLB from Objaverse
                            normalise + simplify mesh
                            register in objects.yaml permanently
                            reload catalogues
                            pass to Randomizer
        │
        ▼
  Randomizer.sample_with_constraints(seed, force_object, force_room, hints)
        │   (one call per variant; Claude is invoked only ONCE per description)
        ▼
  SceneSpec  ×  N variants
```

The Claude API is called **once per description** regardless of how many variants
you request.  All N variants are produced by the Randomizer using different seeds,
holding object and room fixed.

---

## What the LLM Decides

| Field | How it is chosen |
|---|---|
| `task_type` | Inferred from the description — Claude picks one of 15 physics task types |
| `object_name` | Matched to the catalogue, or a new object is proposed and registered |
| `room_type` | Selected from the 5 available room types |
| `hints` | Optional speed / height bias extracted from description phrasing |

Everything else (spawn position, height, velocity, camera angles, lighting, etc.)
is handled deterministically by the `Randomizer` after the LLM call.

---

## The 15 Task Types

| Task type | What happens | Object pool |
|---|---|---|
| `object_drop` | Small/medium object falls off a table or counter | Object Drop (12) |
| `furniture_tip` | Tall furniture topples toward a person | Furniture Tip (7) |
| `hanging_fall` | Wall/ceiling-mounted fixture falls when mount breaks | Hanging Fall (7) |
| `stack_collapse` | Stack of identical objects collapses | Stack Collapse (4) |
| `sliding_object` | Object slides down a ramp and flies off | Object Drop (12) |
| `rolling_ball` | Ball rolls off a table edge | Rolling Ball (6) |
| `shelf_slide` | Object slides off a high wall shelf | Shelf Slide (5) |
| `door_swing` | Door swings uncontrolled toward a person | Door Swing (5) |
| `thrown_object` | Object deliberately thrown at the observer | Thrown Object (6) |
| `pendulum_swing` | Heavy object on cable/rope swings at observer | Pendulum Swing (5) |
| `bouncing_object` | Ball falls, bounces off floor, flies at observer | Bouncing Object (6) |
| `ladder_slip` | Leaning ladder slides and tips toward observer | Ladder Slip (7) |
| `chain_reaction` | One object falls and triggers a cascade | Object Drop (12) |
| `ceiling_drop` | Object falls straight down from ceiling height | Object Drop (12) |
| `stair_tumble` | Object tumbles down a staircase toward the observer | Stair Tumble (7) |

`sliding_object`, `chain_reaction`, and `ceiling_drop` share the Object Drop pool.
Total catalogue size: **75 objects** across 12 distinct pools.

---

## Room Catalogue

Five indoor environments are available:

| Room type | Description | Used by |
|---|---|---|
| `dining` | Dining room with table, chairs, sideboard | most tasks |
| `kitchen` | Kitchen with counters, appliances | most tasks |
| `living` | Living room with sofa, coffee table, TV unit | most tasks |
| `office` | Office with desk, shelving, floor lamp | most tasks |
| `stair_landing` | Stair landing with 7-step staircase | `stair_tumble` only |

---

## Physics Hints

When a description contains **explicit speed or height cues**, Claude emits a `hints`
block that biases the Randomizer's sampling ranges.  The planner passes this block
directly to `Randomizer.sample_with_constraints(hints=…)`.

### Speed hint

| Value | Velocity scale | Example phrases |
|---|---|---|
| `"slow"` | 40 % of normal | "slowly", "gently", "lightly", "soft toss" |
| `"normal"` | 100 % (default) | *(no hint emitted)* |
| `"fast"` | 180 % of normal | "quickly", "rapidly", "hard throw", "high speed" |
| `"very_fast"` | 300 % of normal | "hurled", "slammed", "extremely fast", "full force" |

Parameters that scale with the speed hint (by task type):

| Task type | Scaled parameter |
|---|---|
| `object_drop` | `vel_x` (push-off speed) |
| `furniture_tip` | `angular_vel` |
| `hanging_fall` | `angular_vel` |
| `sliding_object` | `angle_deg` (ramp steepness) |
| `stack_collapse` | `angular_vel` |
| `rolling_ball` | `vel_x` |
| `shelf_slide` | `vel_y` |
| `door_swing` | `angular_vel` |
| `thrown_object` | `vel_y` |
| `pendulum_swing` | `initial_angle_deg` (capped at 85°) |
| `bouncing_object` | `drop_h`, `vel_y` |
| `ladder_slip` | `angular_vel` |
| `chain_reaction` | `trigger_vel_y` |
| `stair_tumble` | `nudge_vel_y` |

### Height hint

| Value | Effect | Example phrases |
|---|---|---|
| `"low"` | Lower spawn / attach point | "low shelf", "bottom step", "near the floor" |
| `"normal"` | Default range (default) | *(no hint emitted)* |
| `"high"` | Higher spawn / attach point | "top shelf", "upper landing", "near the ceiling" |

Parameters shifted by the height hint:

| Task type | Shifted parameter |
|---|---|
| `hanging_fall` (wall) | `attach_z` (wall bracket height) |
| `shelf_slide` | `height` (shelf mounting height) |
| `stair_tumble` | `start_step` (which step the object spawns on) |

### Hints in practice

```python
# "A ball is hurled at high speed across the living room."
# → hints = {"speed": "very_fast"}
spec = planner.plan("A ball is hurled at high speed across the living room.")

# "A vase gently slides off the top shelf in a kitchen."
# → hints = {"speed": "slow", "height": "high"}
spec = planner.plan("A vase gently slides off the top shelf in a kitchen.")

# "A bookshelf tips over." (neutral)
# → hints = {}  (omitted entirely)
spec = planner.plan("A bookshelf tips over.")
```

Hints are applied **consistently across all N variants** of a description — the
Claude call happens once, so the same hints are reused for every seed.

---

## New Object Registration (Objaverse)

When the description names an object **not in any catalogue**, the LLM proposes full
physics parameters and the planner automatically:

1. Looks up the object's LVIS 1.0 label in the Objaverse LVIS annotations index
2. Downloads the best-matching GLB mesh
3. Normalises it to the target size (longest dimension = `target_size_m`)
4. Simplifies to ≤ 8,000 faces if needed
5. Exports as `.obj` to `assets/meshes/<name>.obj`
6. Appends a validated entry to `asset_library/objects.yaml` (permanent)
7. Reloads catalogues and invalidates the Randomizer cache

**The new object is permanently available** for all future procedural and LLM
runs — you never download or register it twice.

Physics parameters the LLM provides for new objects:

| Field | Meaning |
|---|---|
| `lvis_label` | LVIS 1.0 category used to find the mesh on Objaverse |
| `target_size_m` | Longest bounding-box dimension in metres |
| `density` | kg/m³ (foam ~50, wood ~600, ceramic ~2400, steel ~7800) |
| `friction` | 0.0 – 1.0 |
| `restitution` | 0.0 – 1.0 (bounciness) |
| `color_rgb` | [R, G, B] floats in 0.0 – 1.0 |
| `catch_safe` | Whether a human can safely catch/intercept this object |
| `safety_label` | `"safe"` / `"caution"` / `"dangerous"` |

---

## CLI Usage

```bash
# Activate environment first
source /home/ubuntu/Yizhan3d/venv/bin/activate
export ANTHROPIC_API_KEY=sk-ant-...

# Generate 20 variants from a single description
python generate_dataset.py \
  --mode llm \
  --descriptions my_scenarios.txt \
  --variants 20 \
  --output dataset/

# Dry-run to inspect specs without simulating
python generate_dataset.py \
  --mode llm \
  --descriptions my_scenarios.txt \
  --variants 5 \
  --dry-run
```

`my_scenarios.txt` — one description per line:
```
A ripe banana falls off a kitchen counter.
A heavy bookshelf tips over toward someone in a living room.
A crystal chandelier drops from the ceiling.
A bat is thrown across an office.
```

### Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--descriptions` | required | Text file, one scenario per line |
| `--variants` | 10 | Physics variants per description |
| `--output` | `dataset/` | Root output directory |
| `--workers` | 1 | Parallel simulation workers |
| `--seed-start` | 0 | First seed (increment to extend an existing dataset) |
| `--dry-run` | off | Print specs, skip simulation |

---

## Python API

```python
from pipeline.llm_planner import LLMPlanner

planner = LLMPlanner()   # reads ANTHROPIC_API_KEY from environment

# Single spec
spec = planner.plan("A glass shard slides off a kitchen counter.", seed=42)

# N variants — Claude called once, Randomizer called N times
specs = planner.plan_with_variants(
    "A heavy bookshelf tips over in a living room.",
    n=20,
    seed_start=0,
)
```

---

## What the LLM Cannot Change

The LLM sets `task_type`, `object_name`, `room_type`, and optionally `hints`.
The Randomizer controls everything else:

- Exact spawn position (only the range is biased by hints, not the absolute value)
- Table / fixture geometry
- Camera positions and FOV
- Lighting colour and intensity
- Whether the scene is adversarial
- Simulation timestep, substeps, and solver settings

This design keeps physics variation fast (no extra API calls per variant) and
reproducible (fully determined by the integer seed + hints).

---

## Model

Default model: `claude-sonnet-4-6`.  Override at construction:

```python
planner = LLMPlanner(model="claude-opus-4-7")
```

Each `plan()` call uses `max_tokens=512` and returns a single JSON object.
