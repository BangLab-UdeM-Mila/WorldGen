"""
Shared mesh utilities for Objaverse download and normalisation.
Used by both assets/download_models.py and pipeline/llm_planner.py.
"""

from __future__ import annotations

from pathlib import Path


def load_as_single_trimesh(path: str | Path):
    """Load a GLB/OBJ file and merge all sub-meshes into one Trimesh."""
    import trimesh
    obj = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(obj, trimesh.Scene):
        meshes = [g for g in obj.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        return trimesh.util.concatenate(meshes) if meshes else None
    return obj if isinstance(obj, trimesh.Trimesh) else None


def normalize_mesh(mesh, target_size_m: float):
    """
    Scale mesh uniformly so its longest bounding-box dimension equals
    target_size_m, then translate so the bottom face sits at z = 0
    and the centroid is at x = y = 0.

    Returns the modified mesh (in-place).
    """
    scale = target_size_m / max(mesh.bounding_box.extents)
    mesh.apply_scale(scale)
    b = mesh.bounds
    mesh.apply_translation([
        -(b[0][0] + b[1][0]) / 2,
        -(b[0][1] + b[1][1]) / 2,
        -b[0][2],
    ])
    return mesh


def download_from_objaverse(
    lvis_label: str,
    target_size_m: float,
    out_obj_path: str | Path,
    raw_glb_dir: str | Path | None = None,
) -> dict:
    """
    Find the first (alphabetically stable) Objaverse UID for a LVIS label,
    download its GLB, normalise it, and export as .obj.

    Returns
    -------
    dict with keys: uid, vertices, faces, extents
    """
    import objaverse
    import shutil

    out_obj_path = Path(out_obj_path)
    out_obj_path.parent.mkdir(parents=True, exist_ok=True)

    lvis_anns  = objaverse.load_lvis_annotations()
    candidates = lvis_anns.get(lvis_label, [])
    if not candidates:
        raise ValueError(
            f"No Objaverse objects found for LVIS label '{lvis_label}'."
        )
    uid = sorted(candidates)[0]

    uid_to_path = objaverse.load_objects(uids=[uid], download_processes=1)
    glb_src = uid_to_path.get(uid)
    if not glb_src or not Path(glb_src).exists():
        raise RuntimeError(f"Objaverse download failed for uid={uid}")

    if raw_glb_dir is not None:
        raw_glb_dir = Path(raw_glb_dir)
        raw_glb_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(glb_src, raw_glb_dir / out_obj_path.stem + ".glb")

    mesh = load_as_single_trimesh(glb_src)
    if mesh is None:
        raise RuntimeError(f"Could not load mesh from {glb_src}")

    mesh = normalize_mesh(mesh, target_size_m)
    mesh.export(str(out_obj_path), file_type="obj", include_normals=True)

    return {
        "uid":      uid,
        "vertices": len(mesh.vertices),
        "faces":    len(mesh.faces),
        "extents":  mesh.bounding_box.extents.tolist(),
    }
