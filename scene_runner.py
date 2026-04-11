"""
Scene Runner — subprocess entry point.

Each worker process calls this script with a path to a SceneSpec JSON.
Keeps Genesis fully isolated per process (no shared GPU state).

Usage (internal, called by batch_runner):
    python scene_runner.py --spec /path/to/spec.json --output /path/to/scene_dir/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.scene_spec import SceneSpec
from scene_builder import SceneBuilder


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one ReactHuman scene.")
    parser.add_argument("--spec",   required=True, help="Path to SceneSpec JSON file.")
    parser.add_argument("--output", required=True, help="Output directory for this scene.")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    out_dir   = Path(args.output)

    if not spec_path.exists():
        print(f"[scene_runner] ERROR: spec not found: {spec_path}", file=sys.stderr)
        return 1

    spec = SceneSpec.from_json(spec_path.read_text())
    print(f"[scene_runner] scene_id={spec.scene_id}  "
          f"object={spec.object.name}  room={spec.room.type}  seed={spec.seed}")

    try:
        builder = SceneBuilder(spec)
        builder.build()
        metadata = builder.run(output_dir=out_dir)
        print(f"[scene_runner] Done. ttf={metadata.get('time_to_floor_s')}s  "
              f"videos={list(metadata.get('videos', {}).keys())}")
        return 0
    except Exception as e:
        import traceback
        print(f"[scene_runner] FAILED: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
