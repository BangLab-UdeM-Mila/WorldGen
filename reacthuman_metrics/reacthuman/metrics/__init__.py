"""
Evaluation metrics for the ReactHuman benchmark.

The module is organised so that each metric is independently importable
and unit-testable. Higher-level orchestration (aggregation across many
scenes, slicing by difficulty axis) lives in :mod:`reacthuman.metrics.aggregate`.
"""
from .track1 import (
    semantic_action_accuracy,
    fatal_execution_rate,
    adversarial_fool_rate,
)
from .track2 import (
    trajectory_error_cm,
    time_to_collision_error,
    reachable_prediction_rate,
)
from .scene import evaluate_scene
from .aggregate import aggregate_results

__all__ = [
    "semantic_action_accuracy",
    "fatal_execution_rate",
    "adversarial_fool_rate",
    "trajectory_error_cm",
    "time_to_collision_error",
    "reachable_prediction_rate",
    "evaluate_scene",
    "aggregate_results",
]
