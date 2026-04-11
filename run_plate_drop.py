#!/usr/bin/env python3
"""
Entry point for the plate-drop scene.

Usage
-----
    # from the genesis_scene_generation/ directory:
    python run_plate_drop.py

    # with custom config / output:
    python run_plate_drop.py --config config/plate_drop.yaml --output outputs/

    # download CC0 textures first (optional, improves realism):
    python assets/download_textures.py
"""

import argparse
import sys
from pathlib import Path

# Ensure project root on path when executed directly
sys.path.insert(0, str(Path(__file__).parent))

from scenes.plate_drop import PlateDrop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a plate-drop simulation video with Genesis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="config/plate_drop.yaml",
        help="Path to scene YAML config (relative to this script or absolute).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output directory for MP4 files. Defaults to config 'output.dir'.",
    )
    p.add_argument(
        "--download-textures",
        action="store_true",
        help="Download CC0 textures from PolyHaven before building the scene.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve config path relative to this script
    script_dir  = Path(__file__).parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (script_dir / config_path).resolve()

    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")

    # Optional: download textures
    if args.download_textures:
        from assets.download_textures import download_all
        download_all()

    print("=" * 60)
    print("  ReactHuman — Plate Drop Scene Generator")
    print("=" * 60)
    print(f"  Config  : {config_path}")
    print(f"  Backend : Genesis GPU (RayTracer)")
    print()

    # Build and run
    scene = PlateDrop.from_config(config_path)
    scene.build()
    scene.run(output_dir=args.output)


if __name__ == "__main__":
    main()
