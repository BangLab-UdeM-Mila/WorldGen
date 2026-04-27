#!/usr/bin/env python3
"""
Scene Output Validator — Layer 1 + Layer 2
==========================================
Validates generated scene directories without running any simulation.

Layer 1 — metadata completeness & physics sanity  (fast, no video read)
  • spec.json and metadata.json exist and parse correctly
  • all 3 video files exist and have reasonable file size
  • time_to_floor_s is present and within expected range per task type
  • ground_truth_action is a valid label

Layer 2 — video frame heuristics  (reads a few frames per video, ~0.1s each)
  • each video has enough frames
  • middle frame has sufficient pixel variance (object visible, not black)
  • scene contains motion (early vs late frames differ)

Usage
-----
  # Validate a single scene
  python validate_scene.py dataset/bf67e379/

  # Validate an entire dataset root
  python validate_scene.py dataset/

  # Show only failures (quiet mode)
  python validate_scene.py dataset/ --failures-only

  # Skip Layer 2 video checks (fastest)
  python validate_scene.py dataset/ --layer1-only

  # JSON output for downstream processing
  python validate_scene.py dataset/ --json > results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_NAMES = ["video_observer", "video_closeup", "video_overhead"]

# Minimum video file size in bytes — anything smaller is likely a corrupt/empty file
MIN_VIDEO_BYTES = 50_000   # 50 KB

# Minimum number of frames in a valid video
MIN_FRAMES = 30

# Middle-frame pixel std dev (grayscale 0-255) — catches black / uniform frames
VARIANCE_THRESHOLD = 8.0

# Number of pixels (out of total) that changed by >5 intensity units between
# early (10%) and late (90%) frames — catches frozen/static renders.
# Small objects change ~500-5000 pixels; furniture ~10k-70k; frozen = 0.
MOTION_PIXELS_THRESHOLD = 200

# Expected time_to_floor_s ranges per task type (inclusive).
# None → task type not recognised — uses _DEFAULT_TTF_RANGE.
# Upper limit is slightly below simulation duration (3.0s) to catch "stuck" objects.
_TTF_RANGES: dict[str, tuple[float, float]] = {
    "object_drop":    (0.08, 2.88),
    "sliding_object": (0.15, 2.88),
    "stack_collapse": (0.15, 2.88),
    "hanging_fall":   (0.15, 2.88),
    "furniture_tip":  (0.25, 2.88),
    "rolling_ball":   (0.10, 2.88),
    "shelf_slide":    (0.25, 2.88),
    "door_swing":     (0.30, 2.60),   # time for door angle to drop below 20° (nearly closed)
    "thrown_object":  (0.08, 0.80),   # time for thrown object to cross y=0 toward observer
    "pendulum_swing":  (0.25, 1.20),   # time for bob to reach vertical (angle=0, max speed)
    "bouncing_object": (0.20, 2.88),   # time for ball to complete post-bounce arc
    "ladder_slip":     (0.30, 2.88),   # time for ladder COM to fall nearly horizontal
    "chain_reaction":  (0.10, 2.88),   # time for target B to hit the floor
}
_DEFAULT_TTF_RANGE = (0.05, 2.90)

_VALID_GT_ACTIONS = {"EXECUTE_CATCH", "TRIGGER_DODGE", "BRACE_FOR_IMPACT"}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    passed: bool
    level: str = "error"   # "error" | "warning"
    message: str = ""

    def __str__(self) -> str:
        icon = "✓" if self.passed else ("✗" if self.level == "error" else "⚠")
        suffix = f"  ({self.message})" if self.message and not self.passed else ""
        return f"  {icon} {self.name}{suffix}"


@dataclass
class SceneReport:
    scene_id: str
    scene_dir: Path
    checks: list[Check] = field(default_factory=list)

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.level == "error"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.level == "warning"]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "passed": self.passed,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "checks": [
                {"name": c.name, "passed": c.passed,
                 "level": c.level, "message": c.message}
                for c in self.checks
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — metadata checks
# ─────────────────────────────────────────────────────────────────────────────

def _layer1(scene_dir: Path) -> tuple[list[Check], dict, dict]:
    """Run all Layer 1 checks.  Returns (checks, spec_data, meta_data)."""
    checks: list[Check] = []
    spec_data: dict = {}
    meta_data: dict = {}

    # ── 1a. spec.json exists and parses ──────────────────────────────────────
    spec_path = scene_dir / "spec.json"
    if not spec_path.exists():
        checks.append(Check("spec.json exists", False, "error", "file missing"))
        return checks, spec_data, meta_data
    try:
        spec_data = json.loads(spec_path.read_text())
        checks.append(Check("spec.json parses", True))
    except json.JSONDecodeError as e:
        checks.append(Check("spec.json parses", False, "error", str(e)))
        return checks, spec_data, meta_data

    # ── 1b. metadata.json exists and parses ──────────────────────────────────
    meta_path = scene_dir / "metadata.json"
    if not meta_path.exists():
        checks.append(Check("metadata.json exists", False, "error", "file missing"))
    else:
        try:
            meta_data = json.loads(meta_path.read_text())
            checks.append(Check("metadata.json parses", True))
        except json.JSONDecodeError as e:
            checks.append(Check("metadata.json parses", False, "error", str(e)))

    # ── 1c. all video files exist ─────────────────────────────────────────────
    for vname in VIDEO_NAMES:
        vpath = scene_dir / f"{vname}.mp4"
        checks.append(Check(
            f"{vname}.mp4 exists",
            vpath.exists(),
            "error",
            "" if vpath.exists() else "file missing",
        ))

    # ── 1d. video file sizes ──────────────────────────────────────────────────
    for vname in VIDEO_NAMES:
        vpath = scene_dir / f"{vname}.mp4"
        if vpath.exists():
            size = vpath.stat().st_size
            ok   = size >= MIN_VIDEO_BYTES
            checks.append(Check(
                f"{vname}.mp4 size",
                ok,
                "error",
                "" if ok else f"only {size:,} B (< {MIN_VIDEO_BYTES:,} B)",
            ))

    # ── 1e. ground_truth_action ────────────────────────────────────────────────
    # Use spec.json (set before simulation) for semantic labels
    gt = spec_data.get("ground_truth_action")
    gt_ok = gt in _VALID_GT_ACTIONS
    checks.append(Check(
        "ground_truth_action valid",
        gt_ok,
        "error",
        "" if gt_ok else f"got {gt!r}",
    ))

    # ── 1f. time_to_floor_s ────────────────────────────────────────────────────
    # metadata.json contains the post-simulation SceneSpec with ttf filled in
    source = meta_data if meta_data else spec_data
    ttf = source.get("time_to_floor_s")
    task_type = (spec_data.get("task_type") or "object_drop")
    lo, hi = _TTF_RANGES.get(task_type, _DEFAULT_TTF_RANGE)

    if ttf is None:
        checks.append(Check(
            "time_to_floor_s present",
            False,
            "warning",
            "None — object may not have landed (physics failure or old dataset)",
        ))
    else:
        ttf_ok = lo <= ttf <= hi
        checks.append(Check(
            "time_to_floor_s in range",
            ttf_ok,
            "warning" if not ttf_ok else "error",
            "" if ttf_ok else f"{ttf:.3f}s outside [{lo}, {hi}]s for {task_type}",
        ))

    # ── 1g. interception_point_3d ─────────────────────────────────────────────
    ipt3d = source.get("interception_point_3d")
    if ipt3d is None:
        checks.append(Check(
            "interception_point_3d present",
            False,
            "warning",
            "None — object may not have crossed the event boundary",
        ))
    else:
        checks.append(Check("interception_point_3d present", True))

    return checks, spec_data, meta_data


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — video frame heuristics
# ─────────────────────────────────────────────────────────────────────────────

def _sample_frames(video_path: Path, positions: list[float]) -> list[np.ndarray]:
    """Return frames from the video at relative positions (0.0–1.0)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total == 0:
        cap.release()
        return frames
    for p in positions:
        idx = min(int(total * p), total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def _layer2_video(vpath: Path) -> list[Check]:
    """Run Layer 2 checks on one video file."""
    checks: list[Check] = []
    vname = vpath.stem

    if not vpath.exists():
        return checks   # already caught by Layer 1

    # Frame count
    cap  = cv2.VideoCapture(str(vpath))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frames_ok = n_frames >= MIN_FRAMES
    checks.append(Check(
        f"{vname} frame count",
        frames_ok,
        "error",
        "" if frames_ok else f"only {n_frames} frames (min {MIN_FRAMES})",
    ))
    if not frames_ok:
        return checks

    # Sample 5 frames spread across the video
    sampled = _sample_frames(vpath, [0.10, 0.25, 0.50, 0.75, 0.90])
    if len(sampled) < 2:
        checks.append(Check(f"{vname} frame read", False, "error", "could not read frames"))
        return checks

    # Pixel variance on the middle frame (index 2 → 50%)
    mid_frame = sampled[len(sampled) // 2]
    gray_mid  = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY).astype(float)
    variance  = float(np.std(gray_mid))
    var_ok    = variance >= VARIANCE_THRESHOLD
    checks.append(Check(
        f"{vname} object visible",
        var_ok,
        "error",
        "" if var_ok else f"pixel std={variance:.1f} < {VARIANCE_THRESHOLD} (black/blank frame)",
    ))

    # Motion: count pixels that changed by >5 intensity units between 10% and 90% frames.
    # Mean-diff fails for small objects (only ~1k pixels change in a 1280×720 frame).
    early = cv2.cvtColor(sampled[0], cv2.COLOR_BGR2GRAY).astype(float)
    late  = cv2.cvtColor(sampled[-1], cv2.COLOR_BGR2GRAY).astype(float)
    n_changed = int(np.sum(np.abs(late - early) > 5))
    motion_ok = n_changed >= MOTION_PIXELS_THRESHOLD
    checks.append(Check(
        f"{vname} motion present",
        motion_ok,
        "error",
        "" if motion_ok else f"changed_px={n_changed} < {MOTION_PIXELS_THRESHOLD} (frozen/static render)",
    ))

    return checks


def _layer2(scene_dir: Path) -> list[Check]:
    """Run Layer 2 checks on all videos in the scene directory."""
    checks: list[Check] = []
    for vname in VIDEO_NAMES:
        vpath = scene_dir / f"{vname}.mp4"
        checks.extend(_layer2_video(vpath))
    return checks


# ─────────────────────────────────────────────────────────────────────────────
# Scene validation entry point
# ─────────────────────────────────────────────────────────────────────────────

def validate_scene(scene_dir: Path, skip_layer2: bool = False) -> SceneReport:
    """Validate one scene directory.  Returns a SceneReport."""
    scene_id = scene_dir.name
    report   = SceneReport(scene_id=scene_id, scene_dir=scene_dir)

    l1_checks, spec_data, meta_data = _layer1(scene_dir)
    report.checks.extend(l1_checks)

    if not skip_layer2 and _HAS_CV2:
        # Only run Layer 2 if Layer 1 video-existence checks passed
        videos_exist = all(
            (scene_dir / f"{v}.mp4").exists() for v in VIDEO_NAMES
        )
        if videos_exist:
            report.checks.extend(_layer2(scene_dir))
    elif not skip_layer2 and not _HAS_CV2:
        report.checks.append(Check(
            "Layer 2 available",
            False,
            "warning",
            "cv2 not installed — install opencv-python to enable frame checks",
        ))

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_scene_dirs(root: Path) -> list[Path]:
    """Return all scene directories under root (each has a spec.json)."""
    if (root / "spec.json").exists():
        return [root]
    dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p / "spec.json").exists()
    )
    return dirs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(report: SceneReport, verbose: bool, failures_only: bool) -> None:
    if failures_only and report.passed:
        return
    status = "PASS" if report.passed else "FAIL"
    n_err  = len(report.errors)
    n_warn = len(report.warnings)
    line   = f"[{status}] {report.scene_id}"
    if not report.passed:
        line += f"  {n_err} error(s)"
    if n_warn:
        line += f"  {n_warn} warning(s)"
    print(line)
    if verbose or not report.passed:
        for c in report.checks:
            if not c.passed or verbose:
                print(str(c))
        print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="ReactHuman Scene Validator — Layer 1 + Layer 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("path", help="Scene directory or dataset root directory.")
    p.add_argument("--layer1-only",    action="store_true",
                   help="Skip Layer 2 video frame checks.")
    p.add_argument("--failures-only",  action="store_true",
                   help="Only print failed scenes.")
    p.add_argument("--verbose", "-v",  action="store_true",
                   help="Print all check results, including passes.")
    p.add_argument("--json",           action="store_true",
                   help="Output results as a JSON array (for downstream tools).")
    args = p.parse_args()

    root      = Path(args.path)
    scene_dirs = find_scene_dirs(root)

    if not scene_dirs:
        print(f"[validate] No scene directories found under: {root}", file=sys.stderr)
        sys.exit(1)

    reports: list[SceneReport] = []
    for sd in scene_dirs:
        report = validate_scene(sd, skip_layer2=args.layer1_only)
        reports.append(report)
        if not args.json:
            _print_report(report, verbose=args.verbose, failures_only=args.failures_only)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total   = len(reports)
    n_pass    = sum(1 for r in reports if r.passed)
    n_fail    = n_total - n_pass
    n_warn    = sum(len(r.warnings) for r in reports)

    if args.json:
        out = {
            "total": n_total,
            "passed": n_pass,
            "failed": n_fail,
            "total_warnings": n_warn,
            "scenes": [r.to_dict() for r in reports],
        }
        print(json.dumps(out, indent=2))
    else:
        print("─" * 60)
        print(f"  Total   : {n_total}")
        print(f"  Passed  : {n_pass}")
        print(f"  Failed  : {n_fail}  (errors blocking quality)")
        print(f"  Warnings: {n_warn}  (advisory, scene usable)")

        if not _HAS_CV2 and not args.layer1_only:
            print("\n  [note] Install opencv-python to enable Layer 2 frame checks.")

        if n_fail > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
