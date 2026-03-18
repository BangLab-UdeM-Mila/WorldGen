# PhysicsBenchmark

**Physics-Consistent Generation Pipeline — Benchmark Module**  
*BangLab · Université de Montréal · Mila*

---

## Overview

This module establishes physics-consistency baselines for the WorldGen pipeline:
**natural language prompt → task classification → physical model → PyBullet code → simulation.**

Three canonical benchmark experiments are implemented to validate whether
LLM-generated simulation code correctly reproduces ground-truth physical dynamics.
Objects are procedurally generated in Blender and exported as URDF files with
mass and inertia tensors computed analytically from mesh volume.

---

## Pipeline

```
Blender (procedural mesh generation)
        ↓
  .obj  visual mesh  +  .obj  collision mesh
        ↓
  URDF  (mass, CoM, inertia tensor from trimesh)
        ↓
  PyBullet simulation  (240 Hz, PGS solver, 50 iterations)
        ↓
  Benchmark logs  (JSON)  +  Figures  (PNG)
```

---

## Benchmark Objects

| Object | Material | Density (kg/m³) | Physics Regime |
|--------|----------|-----------------|----------------|
| UV-Sphere (r = 5 cm) | Rubber | 1100 | Hertzian contact, rolling |
| Beveled Box (6×4×3 cm) | Wood | 700 | Planar contact, stacking |
| L-Bracket (8×2×2 cm arms) | Aluminum | 2700 | Asymmetric inertia tensor |

Objects are inspired by the YCB benchmark object set (Calli et al., RA-L 2017).

---

## Experiments

### E1 — Free-Fall Bounce (Sphere)
Drop from h = 0.5 m onto a flat plane.  
**Physics check:** bounce height follows $h_n = e^{2n} \cdot h_0$ where $e$ is the coefficient of restitution.  
Used to detect incorrect `restitution` parameters in LLM-generated URDF files.

### E2 — Inclined-Plane Sliding (Beveled Box)
Object released on a 20° incline.  
**Physics check:** acceleration $a = g(\sin\theta - \mu\cos\theta)$.  
Measured $a$ is fitted from simulation data and compared to the theoretical value.

### E3 — Torque-Free Rotation (L-Bracket)
Initial angular velocity $\boldsymbol{\omega} = [1.0,\ 0.1,\ 0.0]$ rad/s, no gravity.  
**Physics check:** angular momentum $\mathbf{L} = \mathbf{I}\boldsymbol{\omega}$ is conserved.  
Off-diagonal inertia terms ($I_{xz}$ in particular) cause axis-coupling, validating
the full 3×3 inertia tensor in the URDF.

---

## Results

| Experiment | Key Metric | Value |
|------------|-----------|-------|
| E1 Free-fall | Energy retained after first bounce | ~10.7% (e ≈ 0.33) |
| E2 Incline | Measured acceleration | 0.904 m/s² |
| E3 Rotation | Angular momentum drift over 2 s | < 5% (numerical damping) |


---

## Repository Structure

```
PhysicsBenchmark/
├── scripts/
│   ├── 00_check_setup.sh          # Environment verification
│   ├── 01_generate_objects.py     # Blender procedural mesh generation
│   ├── 02_generate_urdfs.py       # URDF generation with inertia computation
│   └── 03_run_simulation.py       # PyBullet benchmark experiments
├── objects/                       # Generated .obj meshes (visual + collision)
├── urdfs/                         # Generated .urdf files
│   └── objects_summary.json
├── logs/                          # Simulation outputs
│   ├── benchmark_results.json
│   ├── e1_freefall.png
│   ├── e2_incline.png
│   └── e3_rotation.png
└── README.md
```

---

## Requirements

```bash
pip install pybullet trimesh numpy scipy matplotlib tqdm
# Blender 4.x or 5.x: https://www.blender.org/download/
```

## Usage

```bash
# Step 1 — Generate meshes (run inside Blender)
blender --background --python scripts/01_generate_objects.py -- --output ./objects

# Step 2 — Generate URDFs
python scripts/02_generate_urdfs.py --objects_dir ./objects --urdfs_dir ./urdfs

# Step 3 — Run benchmark simulation
python scripts/03_run_simulation.py --urdfs_dir ./urdfs --logs_dir ./logs
```

---

## References

- Calli et al., *The YCB Object and Model Set*, RA-L 2017.
- Mahler et al., *Dex-Net 2.0*, RSS 2017.
- Coumans & Bai, *PyBullet Physics Simulation*, 2016–2021.

---

*Branch: `mengyang/physics-benchmark` — Mengyang Xiong, March 2026*
