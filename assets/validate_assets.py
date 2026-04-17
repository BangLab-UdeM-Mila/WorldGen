#!/usr/bin/env python3
"""
Asset Library Validator
=======================
Checks objects.yaml and mesh files for completeness before running simulations.

Usage
-----
    python assets/validate_assets.py              # check everything
    python assets/validate_assets.py --fix-hint   # also print suggested phys_half values
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml

HERE       = Path(__file__).parent
REPO_ROOT  = HERE.parent
YAML_PATH  = REPO_ROOT / "asset_library" / "objects.yaml"
MESHES_DIR = HERE / "meshes"

# Maximum acceptable face count for physics simulation
MAX_FACES = 15_000

# Fields every object must have regardless of morph
REQUIRED_FIELDS = (
    "name", "display_name", "category", "morph",
    "density", "friction", "restitution",
    "color_rgb", "roughness",
    "catch_safe", "safety_label", "mass_hint",
)


def check_objects(fix_hint: bool = False) -> list[str]:
    errors: list[str] = []
    objects = yaml.safe_load(YAML_PATH.read_text())

    for obj in objects:
        name  = obj.get("name", "<unnamed>")
        morph = obj.get("morph", "")

        # ── Required fields for ALL objects ───────────────────
        for field in REQUIRED_FIELDS:
            if field not in obj:
                errors.append(f"[{name}] missing required field: '{field}'")

        # ── Mesh-specific checks ───────────────────────────────
        if morph == "mesh":
            mesh_path_val = obj.get("mesh_path")
            if not mesh_path_val:
                errors.append(f"[{name}] morph=mesh but mesh_path is missing")
                continue

            mesh_file = MESHES_DIR / mesh_path_val
            if not mesh_file.exists():
                errors.append(
                    f"[{name}] mesh_path '{mesh_path_val}' "
                    f"not found in assets/meshes/"
                )
            else:
                # Face count check (requires trimesh)
                try:
                    import trimesh
                    m = trimesh.load(str(mesh_file), force="mesh")
                    n_faces = len(m.faces) if hasattr(m, "faces") else 0
                    if n_faces > MAX_FACES:
                        errors.append(
                            f"[{name}] mesh has {n_faces:,} faces "
                            f"(limit {MAX_FACES:,}) — "
                            f"simplify with trimesh.simplify_quadric_decimation()"
                        )
                    # Print suggested phys_half values if missing
                    if fix_hint:
                        bounds = m.bounds
                        ext    = bounds[1] - bounds[0]
                        hx = round(float(ext[0]) / 2, 3)
                        hz = round(float(ext[2]) / 2, 3)
                        if obj.get("phys_half_x", 0.0) == 0.0 or obj.get("phys_half_z", 0.0) == 0.0:
                            print(f"  [hint] {name}: "
                                  f"phys_half_x: {hx}  "
                                  f"phys_half_z: {hz}  "
                                  f"(mesh extents: {ext[0]:.3f} x {ext[1]:.3f} x {ext[2]:.3f} m)")
                except ImportError:
                    pass  # trimesh not available — skip face/bounds check

            # phys_half_x/z must be set and non-zero
            if obj.get("phys_half_x", 0.0) == 0.0:
                errors.append(
                    f"[{name}] morph=mesh but phys_half_x is 0 or missing  "
                    f"→ run with --fix-hint to get suggested values"
                )
            if obj.get("phys_half_z", 0.0) == 0.0:
                errors.append(
                    f"[{name}] morph=mesh but phys_half_z is 0 or missing  "
                    f"→ run with --fix-hint to get suggested values"
                )

        # ── size sanity ────────────────────────────────────────
        size = obj.get("size", [])
        if morph == "mesh" and len(size) != 1:
            errors.append(
                f"[{name}] morph=mesh: size should be [scale_factor], got {size}"
            )
        elif morph == "sphere" and len(size) < 1:
            errors.append(f"[{name}] morph=sphere: size needs at least [radius]")
        elif morph in ("box", "cylinder") and len(size) < 3:
            errors.append(
                f"[{name}] morph={morph}: size needs [x, y, z] half-extents, got {size}"
            )

        # ── colour sanity ──────────────────────────────────────
        rgb = obj.get("color_rgb", [])
        if len(rgb) != 3 or not all(0.0 <= c <= 1.0 for c in rgb):
            errors.append(f"[{name}] color_rgb must be [R, G, B] in 0–1, got {rgb}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ReactHuman asset library (objects.yaml + mesh files)"
    )
    parser.add_argument(
        "--fix-hint", action="store_true",
        help="Print suggested phys_half_x/z values for mesh objects missing them"
    )
    args = parser.parse_args()

    print(f"Checking {YAML_PATH} …")
    if args.fix_hint:
        print("(--fix-hint mode: printing suggested phys_half values)\n")

    errors = check_objects(fix_hint=args.fix_hint)

    if errors:
        print(f"\n{len(errors)} issue(s) found:\n")
        for e in errors:
            print(f"  ✗  {e}")
        sys.exit(1)
    else:
        n = len(yaml.safe_load(YAML_PATH.read_text()))
        print(f"\n  All {n} objects passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
