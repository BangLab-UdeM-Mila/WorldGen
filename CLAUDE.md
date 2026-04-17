# ReactHuman Pipeline — Developer Guide

This file is a checklist and reference for contributors. Follow it when adding new task types or mesh objects to avoid the classes of bugs that required a full debugging session (see CHANGELOG.md).

---

## Adding a New Task Type

A task type defines a distinct physics scenario (e.g. `object_drop`, `furniture_tip`, `hanging_fall`). Each requires coordinated changes across **four files** in a specific order. Do not write camera code before Step 2 — this was the root cause of the blank `hanging_fall` video bug.

### Step 1 — Define the spec dataclass in `pipeline/scene_spec.py`

Add a new `XxxSpec(BaseModel)` with all scenario-specific fields (start position, attachment point, etc.).

Add the spec as an optional field on `SceneSpec`:

```python
class SceneSpec(BaseModel):
    ...
    xxx: Optional[XxxSpec] = None
```

### Step 2 — Implement `_xxx_world_pos()` in `pipeline/randomizer.py` FIRST

**Before writing any camera code**, add the static helper that computes the object's world-space centre at t=0:

```python
@staticmethod
def _xxx_world_pos(xxx_spec: XxxSpec, obj: ObjectSpec) -> tuple[float, float, float]:
    _, _, hz = Randomizer._obj_phys_half(obj)
    x = xxx_spec.start_x
    y = xxx_spec.start_y
    z = xxx_spec.surface_height + hz     # example — adapt to your geometry
    return x, y, z
```

Use `_obj_phys_half()` for the physical size — never use `obj.size[0]` directly for mesh objects (size[0] is a scale factor, not metres).

### Step 3 — Wire up `_xxx_world_pos()` in two places

**3a. Camera generation** — pass the computed position to `_make_cameras_with_retry`:

```python
obj_pos = self._xxx_world_pos(xxx_spec, obj_spec)
cameras = self._make_cameras_with_retry(
    rng,
    lambda r: self._make_xxx_cameras(r, xxx_spec, obj_spec),
    obj_pos,
)
```

**3b. `SceneSpec.object_world_pos()`** — add an elif branch so the geometry validator can compute the same position:

```python
def object_world_pos(self) -> tuple[float, float, float]:
    ...
    elif self.task_type == "xxx":
        return Randomizer._xxx_world_pos(self.xxx, self.object)
```

The formula in 3a and 3b **must be identical**. This is the single source of truth — if they diverge, `validate_geometry()` will catch it at dry-run time, not after a 180-second simulation timeout.

### Step 4 — Implement `scene_builder.py` placement

In `SceneBuilder._add_xxx_object()` (or equivalent), compute the spawn position using the **same formula** as `_xxx_world_pos()`. Add the task type to:
- `SceneBuilder.build()` dispatch
- `SceneSpec.validate_geometry()` if there are task-specific boundary constraints

### Step 5 — Smoke test before committing

```bash
# Geometry check only (fast — no simulation)
python generate_dataset.py --task-type xxx --n 50 --validate-only

# Dry-run to spot obvious coordinate errors
python generate_dataset.py --task-type xxx --n 5 --dry-run

# Single full simulation
python generate_dataset.py --task-type xxx --n 1 --output /tmp/xxx_test/
```

All three should pass with zero warnings before opening a PR.

---

## Adding a New Mesh Object

Mesh objects use `.obj` files in `assets/meshes/`. Unlike primitives, their `size` field is a **scale factor** (`[1.0]`), not physical dimensions. Physical dimensions must be specified separately.

### Checklist

- [ ] Download/create `assets/meshes/<name>.obj`
- [ ] Verify face count ≤ 15,000:
  ```python
  import trimesh
  m = trimesh.load("assets/meshes/<name>.obj", force="mesh")
  print(len(m.faces))  # must be ≤ 15,000
  ```
  If over limit, simplify:
  ```python
  m2 = trimesh.simplify_quadric_decimation(m, face_count=4000)
  m2.export("assets/meshes/<name>.obj")
  ```
- [ ] Add the entry to `asset_library/objects.yaml` with `morph: mesh`, `mesh_path: <name>.obj`, `size: [1.0]`
- [ ] Measure physical extents and fill `phys_half_x` and `phys_half_z`:
  ```bash
  python assets/validate_assets.py --fix-hint
  # prints: [hint] <name>: phys_half_x: X.XXX  phys_half_z: Z.ZZZ
  ```
  Copy those values into the YAML entry. A value of 0.0 will cause objects to spawn at wrong positions.
- [ ] Run the asset linter — must exit 0:
  ```bash
  python assets/validate_assets.py
  ```
- [ ] Run geometry validation — must exit 0:
  ```bash
  python generate_dataset.py --objects <name> --n 20 --validate-only
  ```

---

## Key Invariants

These invariants are enforced by `validate_geometry()` and `validate_assets.py`. Breaking them silently causes wrong-looking simulations.

| Invariant | Enforcement |
|-----------|-------------|
| All cameras must see the object (angle < FOV × 85%) | `validate_geometry()` → `batch_runner.py` pre-check |
| Object must be inside room bounds at t=0 | `validate_geometry()` |
| Object must be above floor at t=0 | `validate_geometry()` |
| Mesh objects must have `phys_half_x/z > 0` | `validate_assets.py` |
| Mesh face count ≤ 15,000 | `validate_assets.py` |
| `_xxx_world_pos()` formula must match `scene_builder` placement | Manual — test with `--validate-only` |

---

## Useful Commands

```bash
# Validate all mesh objects (run after adding/editing any object)
python assets/validate_assets.py

# Get suggested phys_half values from mesh bounds
python assets/validate_assets.py --fix-hint

# Geometry smoke test (no simulation, runs in <1s per spec)
python generate_dataset.py --task-type object_drop --n 200 --validate-only
python generate_dataset.py --task-type furniture_tip --n 200 --validate-only
python generate_dataset.py --task-type hanging_fall --n 200 --validate-only

# Dry-run to inspect scene summaries
python generate_dataset.py --n 10 --dry-run

# Generate a small batch with parallel workers
python generate_dataset.py --n 20 --workers 4 --output dataset/
```
