#!/usr/bin/env python3
"""
Objaverse 3D Model Downloader
==============================
Downloads realistic household-object meshes from Objaverse (CC-by-4.0),
converts them from .glb to .obj, normalises scale and origin, then saves
them to assets/meshes/ ready for use by Genesis.

Mesh normalisation (done by mesh_utils.normalize_mesh):
  - Longest bounding-box dimension scaled to target_longest_dim metres.
  - Bottom face translated to z = 0  (bottom_z_offset: 0.0 in objects.yaml).
  - Centroid centred at x = y = 0.

Usage
-----
    cd genesis_scene_generation
    python assets/download_models.py

After running, objects.yaml entries are updated automatically by the script
(or you can copy the printed YAML hints manually).

Dependencies
------------
    pip install objaverse trimesh

Licence
-------
All downloaded models are CC-by-4.0.
Cite: Deitke et al., "Objaverse: A Universe of Annotated 3D Objects", CVPR 2023.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Dependency check ──────────────────────────────────────────

def _require(pkg: str) -> None:
    import importlib
    if importlib.util.find_spec(pkg) is None:
        sys.exit(f"Missing dependency: pip install {pkg}")

_require("objaverse")
_require("trimesh")

import objaverse
from assets.mesh_utils import load_as_single_trimesh, normalize_mesh

# ── Paths ─────────────────────────────────────────────────────

HERE     = Path(__file__).parent                      # assets/
RAW_DIR  = HERE / "raw_glb"                           # temporary GLB cache
MESH_DIR = HERE / "meshes"                            # final .obj output
RAW_DIR.mkdir(parents=True, exist_ok=True)
MESH_DIR.mkdir(parents=True, exist_ok=True)

# ── Target objects ────────────────────────────────────────────
# lvis_label   : LVIS 1.0 category used to search Objaverse.
# longest_dim  : Desired longest bounding-box side in metres after normalisation.

TARGETS: dict[str, dict] = {
    "red_apple": {
        "lvis_label": "apple",
        "longest_dim": 0.085,
        "notes": "Also reused by steel_apple (same mesh, different density)",
    },
    "coffee_mug": {
        "lvis_label": "mug",
        "longest_dim": 0.110,
        "notes": "",
    },
    "plastic_bottle": {
        "lvis_label": "water_bottle",
        "longest_dim": 0.220,
        "notes": "",
    },
    "hot_iron": {
        "lvis_label": "iron_(for_clothing)",
        "longest_dim": 0.240,
        "notes": "",
    },
    "foam_anvil": {
        "lvis_label": "hammer",
        "longest_dim": 0.160,
        "notes": "Adversarial: looks heavy but is foam",
    },
}


# ── Helpers ───────────────────────────────────────────────────

def _best_uid(lvis_label: str, lvis_anns: dict) -> str | None:
    """Return the first (alphabetically stable) UID for a LVIS label."""
    candidates = lvis_anns.get(lvis_label, [])
    return sorted(candidates)[0] if candidates else None


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    import shutil

    print("Loading Objaverse LVIS annotations (first run downloads ~50 MB) …")
    lvis_anns = objaverse.load_lvis_annotations()
    print(f"  {len(lvis_anns)} categories loaded.\n")

    uids_needed: dict[str, str] = {}
    for name, cfg in TARGETS.items():
        label = cfg["lvis_label"]
        uid   = _best_uid(label, lvis_anns)
        if uid is None:
            print(f"[WARN] No UID for LVIS '{label}' (object '{name}'). Skipping.")
            continue
        uids_needed[name] = uid
        print(f"  {name:<20}  lvis='{label}'  uid={uid[:20]}…")

    if not uids_needed:
        sys.exit("No UIDs resolved — nothing to download.")

    print(f"\nDownloading {len(uids_needed)} GLB files …")
    uid_to_path = objaverse.load_objects(
        uids=list(uids_needed.values()),
        download_processes=1,
    )
    print()

    results = []
    for name, uid in uids_needed.items():
        glb_src = uid_to_path.get(uid)
        if not glb_src or not Path(glb_src).exists():
            print(f"[ERROR] Download failed for '{name}'. Skipping.")
            continue

        raw_dest = RAW_DIR / f"{name}.glb"
        shutil.copy2(glb_src, raw_dest)

        cfg = TARGETS[name]
        print(f"Processing {name} …")
        mesh = load_as_single_trimesh(raw_dest)
        if mesh is None:
            print(f"  [ERROR] Could not load mesh. Skipping.")
            continue

        mesh    = normalize_mesh(mesh, cfg["longest_dim"])
        extents = mesh.bounding_box.extents
        print(f"  extents (x,y,z): {[round(v, 4) for v in extents.tolist()]} m")

        out_path = MESH_DIR / f"{name}.obj"
        mesh.export(str(out_path), file_type="obj", include_normals=True)
        print(f"  Saved → {out_path.relative_to(Path.cwd())}")

        results.append({"name": name, "extents": extents.tolist()})
        print()

    # ── YAML hints ───────────────────────────────────────────
    print("=" * 62)
    print("objects.yaml update hints")
    print("=" * 62)
    for r in results:
        name = r["name"]
        ex   = r["extents"]
        print(f"- name: {name}")
        print(f"  morph: mesh")
        print(f"  mesh_path: {name}.obj")
        print(f"  size: [1.0]             # pre-normalised")
        print(f"  bottom_z_offset: 0.0")
        print(f"  # half-extents: [{ex[0]/2:.4f}, {ex[1]/2:.4f}, {ex[2]/2:.4f}] m")
        if name == "red_apple":
            print(f"  # NOTE: steel_apple should reuse mesh_path: red_apple.obj")
        print()

    print(f"Done. {len(results)}/{len(uids_needed)} objects processed.")


if __name__ == "__main__":
    main()
