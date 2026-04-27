# Environment & Rendering Troubleshooting

Errors encountered during development and their fixes. Check this before debugging a new machine setup.

---

## 1. Genesis not found — use the project venv

**Error**
```
ModuleNotFoundError: No module named 'genesis'
```

**Cause**
Genesis is installed in the project virtualenv, not the system Python. The batch runner spawns subprocesses using bare `python`, which resolves to `/usr/bin/python`.

**Fix**
Set `GENESIS_PYTHON` to the venv interpreter before running the generator:
```bash
export GENESIS_PYTHON=/lambda/nfs/Yizhan3d/.venv/bin/python
python generate_dataset.py --task-type object_drop --n 10 --output dataset/
```

The `batch_runner.py` reads `GENESIS_PYTHON` from the environment and uses it for every subprocess. If the variable is unset it falls back to `python` (system Python, which will fail).

---

## 2. EGL not installed — headless GPU rendering fails

**Error**
```
AttributeError: 'NoneType' object has no attribute 'eglGetCurrentContext'
```
or
```
pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"
```

**Cause**
Genesis uses EGL for headless (off-screen) GPU rendering. The `libegl1` library is missing on a fresh Ubuntu install.

**Fix**
```bash
sudo apt-get install -y libegl1
```

Also ensure the environment variable is set for subprocesses (already done in `batch_runner.py`):
```bash
export PYOPENGL_PLATFORM=egl
```

---

## 3. Pydantic not in system Python

**Error**
```
ModuleNotFoundError: No module named 'pydantic'
```

**Cause**
`generate_dataset.py` itself (the CLI entry point) imports from `pipeline/`, which uses Pydantic. This runs under the system Python, not the venv.

**Fix**
```bash
pip install pydantic
```
Or run the CLI via the venv too:
```bash
/lambda/nfs/Yizhan3d/.venv/bin/python generate_dataset.py ...
```

---

## 4. MJCF `shininess` attribute rejected

**Error**
```
ValueError: XML Error: Schema violation: unrecognized attribute: 'shininess'
```

**Cause**
When generating MJCF XML for articulated bodies (e.g. door panel), the `shininess` attribute was included in the `<geom>` tag. MuJoCo's schema does not recognise it.

**Fix**
Remove `shininess` from the geom attributes. Visual material properties (roughness, reflectance) are not supported directly in MuJoCo geom tags — set them on the Genesis `Surface` object instead, after loading the MJCF.

---

## 5. `set_dofs_position()` rejects 2-D tensor

**Error**
```
genesis.GenesisException: Invalid input shape: torch.Size([1, 1]). Expecting at most 1D tensor.
```

**Cause**
Genesis DOF setters have inconsistent shape requirements:
- `set_dofs_velocity(vel)` — accepts `[1, n_dofs]` (2-D, batch × dofs)
- `set_dofs_position(pos)` — requires at most `[n_dofs]` (1-D)

**Fix**
Use a 1-D tensor for position:
```python
# WRONG
pos_t = torch.zeros(1, n_dofs, dtype=torch.float32)
obj.set_dofs_position(pos_t)

# CORRECT
pos_t = torch.tensor([angle_rad], dtype=torch.float32)
obj.set_dofs_position(pos_t)
```

---

## 6. CUDA tensor → numpy without `.cpu()` — crashes in simulation loop

**Error**
```
TypeError: can't convert cuda:0 device type tensor to numpy.
Use Tensor.cpu() to copy the tensor to host memory first.
```

**Cause**
`obj.get_dofs_position()` returns a CUDA tensor. Any call to `np.asarray()` or `.numpy()` on it without first moving to CPU raises this error.

**Fix**
Always move to CPU before converting:
```python
q = obj.get_dofs_position()
if hasattr(q, "cpu"):
    q = q.cpu()
angle = float(np.asarray(q).flatten()[0])
```

---

## 7. Blank / black video frames from overhead or observer cameras

**Symptom**
Video file is suspiciously small (≈ 21 KB) and `validate_scene.py` reports:
```
✗ video_overhead object visible  (pixel std=0.0 < 8.0 (black/blank frame))
✗ video_overhead motion present  (changed_px=0 < 200 (frozen/static render))
```

**Causes and fixes**

| Root cause | Check | Fix |
|---|---|---|
| Camera Y is outside south wall (`pos_y < -room_depth/2`) | Print `spec.cameras[*].pos` vs `room.depth/2` | Clamp camera Y: `max(-depth/2 + margin, desired_y)` |
| Camera Z is above ceiling (`pos_z > room_height`) | Print `spec.cameras[*].pos[2]` vs `room.height` | Clamp camera Z: `min(room_height - 0.25, desired_z)` |
| Camera inside wall geometry | Camera Y very close to wall surface | Add margin (≥ 0.15 m) from all wall surfaces |

The office room (depth=3.5 m, height=2.7 m) is the tightest room. Camera placement that works in the larger living room (depth=5.0 m, height=2.8 m) can clip through walls or ceiling in the office.

---

## 8. Motion detection false-negative — frozen video reported for valid simulation

**Symptom**
`validate_scene.py` flags a real simulation as having no motion even though the object clearly moves.

**Cause**
The original validator used mean absolute pixel difference across the whole frame. For a small object (e.g. a bottle falling) this mean is only ≈ 0.6–1.1 out of 255 — well below the old threshold of 4.0.

**Fix (already applied)**
Count individual changed pixels instead of taking the mean:
```python
n_changed = int(np.sum(np.abs(late - early) > 5))
motion_ok = n_changed >= MOTION_PIXELS_THRESHOLD   # 200
```
A small object changes 800–4500 pixels; a genuinely frozen/blank video changes 0.

---

## Quick environment check

Run this before starting a generation session on a new machine:

```bash
# 1. EGL present?
ldconfig -p | grep libEGL

# 2. Venv Python reachable?
/lambda/nfs/Yizhan3d/.venv/bin/python -c "import genesis; print('genesis OK')"

# 3. Pydantic in system Python (needed by CLI entry point)?
python -c "import pydantic; print('pydantic OK')"

# 4. Geometry smoke test (no GPU needed)
python generate_dataset.py --task-type object_drop --n 50 --validate-only

# 5. Single scene end-to-end
GENESIS_PYTHON=/lambda/nfs/Yizhan3d/.venv/bin/python \
  python generate_dataset.py --task-type object_drop --n 1 --output /tmp/smoke_test/
```
