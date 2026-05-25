#!/usr/bin/env python3
"""
Quick smoke-test for LLMPlanner covering all 13 task types.
Usage:
    export ANTHROPIC_API_KEY=sk-...
    python test_llm_planner.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.llm_planner import LLMPlanner, _TASK_TYPES

# One representative description per task type
TEST_CASES = [
    ("object_drop",    "A ceramic coffee mug tips off the kitchen counter."),
    ("furniture_tip",  "A tall wooden bookshelf starts to topple forward in the living room."),
    ("hanging_fall",   "A framed picture falls from the wall when its nail pulls out."),
    ("stack_collapse", "A stack of tin cans collapses off a dining room shelf."),
    ("sliding_object", "A plastic bottle slides down a ramp and flies toward the camera."),
    ("rolling_ball",   "A billiard ball rolls off the edge of the dining table."),
    ("shelf_slide",    "A glass spice bottle slides off a high kitchen shelf."),
    ("door_swing",     "A heavy wooden door swings open out of control."),
    ("thrown_object",  "Someone throws a baseball directly at the observer."),
    ("pendulum_swing", "A large sandbag on a rope swings toward the camera like a wrecking ball."),
    ("bouncing_object","A rubber ball is dropped from above and bounces toward the observer."),
    ("ladder_slip",    "An aluminum ladder leaning against the wall slides and tips forward."),
    ("chain_reaction", "A ceramic plate falls, then knocks over a knife and apple in a chain."),
]


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    print("Initialising LLMPlanner …")
    planner = LLMPlanner(api_key=api_key)

    print(f"\nAll registered task types: {_TASK_TYPES}\n")
    print(f"{'Task type':<18} {'Expected':<18} {'Got task':<18} {'Object':<25} {'Room':<10} {'OK?'}")
    print("-" * 100)

    n_pass = 0
    for expected_task, desc in TEST_CASES:
        try:
            spec = planner.plan(desc, seed=42)
            match = spec.task_type == expected_task
            n_pass += match
            status = "PASS" if match else "FAIL (task mismatch)"
            print(f"{expected_task:<18} {expected_task:<18} {spec.task_type:<18} "
                  f"{spec.object.name:<25} {spec.room.type:<10} {status}")
        except Exception as exc:
            print(f"{expected_task:<18} {expected_task:<18} {'ERROR':<18} "
                  f"{'':<25} {'':<10} FAIL: {exc}")

    print(f"\n{'='*100}")
    print(f"Results: {n_pass}/{len(TEST_CASES)} passed")
    sys.exit(0 if n_pass == len(TEST_CASES) else 1)


if __name__ == "__main__":
    main()
