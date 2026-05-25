#!/usr/bin/env python3
"""
Dataset Generator — Main CLI Entry Point
=========================================
Generates the ReactHuman benchmark dataset at any scale.

Two modes
---------
procedural  (default)
    Pure random sampling.  No LLM needed.  Fast, scalable to 10,000+.

    python generate_dataset.py --n 100 --output dataset/

llm
    Natural-language descriptions → physics variants via Claude API.
    Requires ANTHROPIC_API_KEY in the environment.

    python generate_dataset.py --mode llm \\
        --descriptions scenarios.txt --variants 20 --output dataset/

Key flags
---------
--n             Number of scenes (procedural mode).
--seed-start    First seed value (default 0). Increment to extend dataset.
--objects       Comma-separated object names to include (default: all).
--rooms         Comma-separated room types (default: all).
--adversarial-prob  Fraction of adversarial scenes (default 0.15).
--workers       Parallel subprocesses (default 1 for single GPU).
--dry-run       Print specs without running simulation.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.randomizer import Randomizer
from pipeline.batch_runner import BatchRunner
from pipeline.scene_spec import SceneSpec


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ReactHuman Dataset Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["procedural", "llm"], default="procedural",
                   help="Generation mode.")
    p.add_argument("--task-type",
                   choices=["object_drop", "furniture_tip", "hanging_fall",
                            "stack_collapse", "sliding_object",
                            "rolling_ball", "shelf_slide", "door_swing",
                            "thrown_object", "pendulum_swing",
                            "bouncing_object", "ladder_slip", "chain_reaction",
                            "ceiling_drop"],
                   default="object_drop",
                   help="Physics task type to generate.")

    # Procedural mode
    p.add_argument("--n",           type=int,   default=10,
                   help="Number of scenes to generate (procedural).")
    p.add_argument("--seed-start",  type=int,   default=0,
                   help="Starting random seed.")
    p.add_argument("--objects",     type=str,   default=None,
                   help="Comma-separated object names (e.g. ceramic_plate,chef_knife).")
    p.add_argument("--rooms",       type=str,   default=None,
                   help="Comma-separated room types (e.g. dining,kitchen).")
    p.add_argument("--adversarial-prob", type=float, default=0.15,
                   help="Fraction of adversarial scenes.")

    # LLM mode
    p.add_argument("--descriptions", type=str, default=None,
                   help="Text file with one natural-language scenario per line (llm mode).")
    p.add_argument("--variants",    type=int, default=10,
                   help="Physics variants per description (llm mode).")

    # Output & execution
    p.add_argument("--output",   type=str, default="dataset/",
                   help="Root output directory.")
    p.add_argument("--workers",  type=int, default=1,
                   help="Parallel worker processes.")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print SceneSpecs without running simulation.")
    p.add_argument("--validate-only", action="store_true",
                   help="Run geometry validation on all specs and exit. "
                        "No simulation is started. Use after adding a new task type or object.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

def build_specs_procedural(args: argparse.Namespace) -> list[SceneSpec]:
    obj_filter  = [o.strip() for o in args.objects.split(",")] if args.objects else None
    room_filter = [r.strip() for r in args.rooms.split(",")]   if args.rooms   else None

    # Map object names → categories for the Randomizer filter
    if obj_filter:
        # Load library to check categories
        import yaml
        lib = yaml.safe_load((Path(__file__).parent / "asset_library" / "objects.yaml").read_text())
        name_to_cat = {o["name"]: o["category"] for o in lib}
        categories  = list({name_to_cat[n] for n in obj_filter if n in name_to_cat})
    else:
        categories = None

    rand = Randomizer(
        task_type=args.task_type,
        object_categories=categories,
        room_types=room_filter,
        adversarial_prob=args.adversarial_prob,
    )

    if obj_filter:
        # Sample with forced object selection
        specs = []
        for i in range(args.n):
            # Cycle through requested objects
            force = obj_filter[i % len(obj_filter)]
            spec  = rand.sample_with_constraints(
                seed=args.seed_start + i,
                force_object=force,
                force_room=room_filter[i % len(room_filter)] if room_filter else None,
            )
            specs.append(spec)
    else:
        specs = rand.sample_batch(n=args.n, seed_start=args.seed_start)

    return specs


def build_specs_llm(args: argparse.Namespace) -> list[SceneSpec]:
    if not args.descriptions:
        sys.exit("--descriptions required in llm mode.")

    from pipeline.llm_planner import LLMPlanner

    desc_file = Path(args.descriptions)
    if not desc_file.exists():
        sys.exit(f"Descriptions file not found: {desc_file}")

    descriptions = [l.strip() for l in desc_file.read_text().splitlines() if l.strip()]
    print(f"[generate] {len(descriptions)} descriptions × {args.variants} variants "
          f"= up to {len(descriptions) * args.variants} scenes")

    planner = LLMPlanner()
    specs   = []
    seed    = args.seed_start

    for i, desc in enumerate(descriptions):
        print(f"  [{i+1}/{len(descriptions)}] Planning: {desc[:70]}…")
        try:
            batch = planner.plan_with_variants(desc, n=args.variants, seed_start=seed)
            specs.extend(batch)
            seed += args.variants
        except Exception as e:
            print(f"  [WARN] LLM planning failed for description {i+1}: {e}")

    return specs


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 62)
    print("  ReactHuman — Dataset Generator")
    print("=" * 62)
    print(f"  Mode       : {args.mode}")
    print(f"  Task type  : {args.task_type}")
    print(f"  Output     : {args.output}")
    print(f"  Workers    : {args.workers}")
    print()

    # ── Build scene specifications ────────────────────────────
    if args.mode == "procedural":
        specs = build_specs_procedural(args)
    else:
        specs = build_specs_llm(args)

    print(f"[generate] {len(specs)} SceneSpecs ready.")

    # ── Validate-only: geometry checks, no simulation ─────────
    if args.validate_only:
        print("\n[validate] Running geometry checks …\n")
        n_warn = 0
        for s in specs:
            warnings = s.validate_geometry()
            if warnings:
                n_warn += 1
                print(f"  WARN  seed={s.seed:<6}  obj={s.object.name:<20}")
                for w in warnings:
                    print(f"          {w}")
        if n_warn:
            print(f"\n[validate] {n_warn}/{len(specs)} specs have geometry issues.")
            sys.exit(1)
        else:
            print(f"[validate] All {len(specs)} specs passed geometry checks.")
        return

    # ── Dry-run: just print summaries ─────────────────────────
    if args.dry_run:
        print("\n[dry-run] Scene summaries:\n")
        for i, s in enumerate(specs):
            print(f"  {i:>4}  id={s.scene_id}  seed={s.seed:>6}  "
                  f"obj={s.object.name:<20}  room={s.room.type:<8}  "
                  f"gt={s.ground_truth_action}  adv={s.adversarial}")
        print(f"\n[dry-run] Total: {len(specs)} scenes. "
              "Re-run without --dry-run to simulate.")
        return

    # ── Run simulations ───────────────────────────────────────
    runner = BatchRunner(
        output_dir=args.output,
        workers=args.workers,
    )
    results = runner.run(specs)

    n_ok = sum(1 for r in results if r["success"])
    print(f"\n[generate] Dataset complete: {n_ok}/{len(specs)} scenes OK → {args.output}")


if __name__ == "__main__":
    main()
