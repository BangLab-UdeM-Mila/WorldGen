"""
WorldGen Video Assembler
========================
Combines Blender PNG frame sequences into MP4 videos.

Usage:
  python scripts\04b_assemble_video.py --renders_dir .\renders
"""

import os, sys, argparse, glob
import numpy as np

try:
    import imageio
except ImportError:
    print("ERROR: pip install imageio[ffmpeg]")
    sys.exit(1)

FPS = 30

def assemble(frames_dir, out_path):
    pattern = os.path.join(frames_dir, "frame_*.png")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"  [SKIP] No frames found in {frames_dir}")
        return

    print(f"  {len(files)} frames → {out_path}")
    writer = imageio.get_writer(out_path, fps=FPS, quality=8, macro_block_size=1)
    for f in files:
        writer.append_data(imageio.imread(f))
    writer.close()
    print(f"  Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders_dir", default="./renders")
    args = parser.parse_args()

    for exp in ["e1", "e2", "e3"]:
        frames_dir = os.path.join(args.renders_dir, f"{exp}_frames")
        names = {"e1":"e1_freefall","e2":"e2_incline","e3":"e3_rotation"}
        out   = os.path.join(args.renders_dir, f"{names[exp]}.mp4")
        if os.path.isdir(frames_dir):
            assemble(frames_dir, out)

    print("\n[Done] Videos in:", os.path.abspath(args.renders_dir))

if __name__ == "__main__":
    main()
