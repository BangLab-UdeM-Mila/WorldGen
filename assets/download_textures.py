"""
Download CC0 textures from PolyHaven for the plate-drop scene.

Textures used:
  - wood_table_001   : walnut-style tabletop
  - concrete_floor   : floor surface

Run once before generating scenes:
    python assets/download_textures.py
"""

import os
import sys
import urllib.request
from pathlib import Path

TEXTURES_ROOT = Path(__file__).parent / "textures"

# PolyHaven 1k JPG base URL
PH_BASE = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k"

DOWNLOADS = [
    # (texture_slug, map_type, local_subdir, local_filename)
    # ---- Parquet floor ----
    ("herringbone_parquet", "diff",   "floor", "floor_diff.jpg"),
    ("herringbone_parquet", "rough",  "floor", "floor_rough.jpg"),
    ("herringbone_parquet", "nor_gl", "floor", "floor_nor.jpg"),
    # ---- Wall plaster ----
    ("plastered_wall", "diff",   "wall", "wall_diff.jpg"),
    ("plastered_wall", "rough",  "wall", "wall_rough.jpg"),
    ("plastered_wall", "nor_gl", "wall", "wall_nor.jpg"),
]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  Downloading {dest.name} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"  [ok]   {dest.name}")
    except Exception as e:
        print(f"  [fail] {url}: {e}", file=sys.stderr)


def download_all() -> None:
    print("Downloading PolyHaven CC0 textures ...")
    for slug, map_type, subdir, filename in DOWNLOADS:
        url = f"{PH_BASE}/{slug}/{slug}_{map_type}_1k.jpg"
        dest = TEXTURES_ROOT / subdir / filename
        _download(url, dest)
    print("Done.")


def check_textures() -> dict[str, Path | None]:
    """Return a dict of texture key -> Path (or None if missing)."""
    mapping = {
        "floor_diff":  TEXTURES_ROOT / "floor" / "floor_diff.jpg",
        "floor_rough": TEXTURES_ROOT / "floor" / "floor_rough.jpg",
        "floor_nor":   TEXTURES_ROOT / "floor" / "floor_nor.jpg",
        "wall_diff":   TEXTURES_ROOT / "wall"  / "wall_diff.jpg",
        "wall_rough":  TEXTURES_ROOT / "wall"  / "wall_rough.jpg",
        "wall_nor":    TEXTURES_ROOT / "wall"  / "wall_nor.jpg",
    }
    return {k: (v if v.exists() else None) for k, v in mapping.items()}


if __name__ == "__main__":
    download_all()
    missing = [k for k, v in check_textures().items() if v is None]
    if missing:
        print(f"\nWarning: {len(missing)} textures failed: {missing}")
        print("Scene will fall back to solid colours.")
    else:
        print("\nAll textures ready.")
