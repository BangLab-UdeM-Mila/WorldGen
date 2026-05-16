"""
Unit tests for the metric stack.

The tests use synthetic SceneSpec / ModelPrediction pairs so that they
neither depend on Genesis nor on any actual model API. Run with::

    pytest -q
"""
from __future__ import annotations

import math

import pytest

from reacthuman.metrics import (
    adversarial_fool_rate,
    aggregate_results,
    evaluate_scene,
    fatal_execution_rate,
    reachable_prediction_rate,
    semantic_action_accuracy,
    trajectory_error_cm,
)
from reacthuman.schemas import (
    ActionPrimitive,
    DifficultyVector,
    GroundTruth,
    ModelPrediction,
    SafetyLabel,
    SceneSpec,
    TaskFamily,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _spec(
    sid: str,
    *,
    correct=ActionPrimitive.EXECUTE_CATCH,
    impact=(0.1, 0.0, -0.2),
    safety=SafetyLabel.SAFE,
    reachable=True,
    initial_state=0.3,
    action=0.3,
    physics=0.3,
    extras=None,
) -> SceneSpec:
    return SceneSpec(
        scene_id=sid,
        task_family=TaskFamily.OBJECT_DROP,
        difficulty=DifficultyVector(initial_state, action, physics),
        ground_truth=GroundTruth(
            correct_action=correct,
            impact_point_world=impact,
            time_to_impact=0.5,
            reachable_mask=reachable,
            safety_label=safety,
        ),
        extras=extras or {},
    )


def _pred(sid: str, action: ActionPrimitive, world=None) -> ModelPrediction:
    return ModelPrediction(scene_id=sid, action=action, target_point_world=world)


# --------------------------------------------------------------------------- #
# Track 1
# --------------------------------------------------------------------------- #
def test_semantic_action_accuracy_simple():
    specs = [
        _spec("s1", correct=ActionPrimitive.EXECUTE_CATCH),
        _spec("s2", correct=ActionPrimitive.TRIGGER_DODGE),
        _spec("s3", correct=ActionPrimitive.BRACE_FOR_IMPACT),
    ]
    preds = [
        _pred("s1", ActionPrimitive.EXECUTE_CATCH),
        _pred("s2", ActionPrimitive.EXECUTE_CATCH),
        _pred("s3", ActionPrimitive.BRACE_FOR_IMPACT),
    ]
    assert semantic_action_accuracy(preds, specs) == pytest.approx(2 / 3)


def test_semantic_action_accuracy_empty_intersection():
    specs = [_spec("s1")]
    preds = [_pred("s2", ActionPrimitive.EXECUTE_CATCH)]
    assert semantic_action_accuracy(preds, specs) == 0.0


def test_fatal_execution_rate_counts_dangerous_catches():
    specs = [
        _spec("s1", correct=ActionPrimitive.TRIGGER_DODGE, safety=SafetyLabel.DANGEROUS),
        _spec("s2", correct=ActionPrimitive.EXECUTE_CATCH, safety=SafetyLabel.SAFE),
    ]
    preds = [
        _pred("s1", ActionPrimitive.EXECUTE_CATCH),  # fatal
        _pred("s2", ActionPrimitive.EXECUTE_CATCH),  # fine
    ]
    assert fatal_execution_rate(preds, specs) == pytest.approx(0.5)


def test_adversarial_fool_rate_only_counts_adversarial():
    specs = [
        _spec("s1", safety=SafetyLabel.ADVERSARIAL,
              extras={"appearance_implied_action": "TRIGGER_DODGE"},
              correct=ActionPrimitive.EXECUTE_CATCH),
        _spec("s2", safety=SafetyLabel.ADVERSARIAL,
              extras={"appearance_implied_action": "EXECUTE_CATCH"},
              correct=ActionPrimitive.TRIGGER_DODGE),
        _spec("s3", safety=SafetyLabel.SAFE),
    ]
    preds = [
        _pred("s1", ActionPrimitive.TRIGGER_DODGE),  # fooled
        _pred("s2", ActionPrimitive.TRIGGER_DODGE),  # not fooled
        _pred("s3", ActionPrimitive.EXECUTE_CATCH),  # ignored
    ]
    assert adversarial_fool_rate(preds, specs) == pytest.approx(0.5)


def test_adversarial_fool_rate_returns_none_when_no_adversarial():
    specs = [_spec("s1", safety=SafetyLabel.SAFE)]
    preds = [_pred("s1", ActionPrimitive.EXECUTE_CATCH)]
    assert adversarial_fool_rate(preds, specs) is None


# --------------------------------------------------------------------------- #
# Track 2
# --------------------------------------------------------------------------- #
def test_trajectory_error_cm_zero_when_exact():
    specs = [_spec("s1", impact=(0.1, 0.2, 0.3))]
    preds = [_pred("s1", ActionPrimitive.EXECUTE_CATCH, world=(0.1, 0.2, 0.3))]
    errs = trajectory_error_cm(preds, specs)
    assert errs == [pytest.approx(0.0)]


def test_trajectory_error_cm_correct_distance():
    # GT at origin, pred 1 cm = 0.01 m along x
    specs = [_spec("s1", impact=(0.0, 0.0, 0.0))]
    preds = [_pred("s1", ActionPrimitive.EXECUTE_CATCH, world=(0.01, 0.0, 0.0))]
    errs = trajectory_error_cm(preds, specs)
    assert errs == [pytest.approx(1.0, abs=1e-9)]


def test_trajectory_error_only_catch():
    specs = [_spec("s1", correct=ActionPrimitive.TRIGGER_DODGE)]
    preds = [_pred("s1", ActionPrimitive.TRIGGER_DODGE, world=(0.5, 0.5, 0.5))]
    assert trajectory_error_cm(preds, specs) == []


def test_reachable_prediction_rate():
    specs = [
        _spec("s1", reachable=True, impact=(0.0, 0.0, 0.0)),
        _spec("s2", reachable=True, impact=(0.0, 0.0, 0.0)),
        _spec("s3", reachable=False),  # excluded
    ]
    preds = [
        _pred("s1", ActionPrimitive.EXECUTE_CATCH, world=(0.02, 0.0, 0.0)),  # 2 cm
        _pred("s2", ActionPrimitive.EXECUTE_CATCH, world=(0.20, 0.0, 0.0)),  # 20 cm
        _pred("s3", ActionPrimitive.EXECUTE_CATCH, world=(0.0, 0.0, 0.0)),
    ]
    rate = reachable_prediction_rate(preds, specs, reach_margin_cm=5.0)
    assert rate == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Scene evaluation
# --------------------------------------------------------------------------- #
def test_evaluate_scene_full():
    spec = _spec("s1", impact=(0.0, 0.0, 0.0))
    pred = _pred("s1", ActionPrimitive.EXECUTE_CATCH, world=(0.03, 0.0, 0.0))
    r = evaluate_scene(spec, pred)
    assert r.action_correct
    assert not r.fatal_outcome
    assert r.trajectory_error_cm == pytest.approx(3.0)


def test_evaluate_scene_scene_id_mismatch():
    spec = _spec("s1")
    pred = _pred("s2", ActionPrimitive.EXECUTE_CATCH)
    with pytest.raises(ValueError):
        evaluate_scene(spec, pred)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_results_headline_numbers():
    specs = [
        _spec("s1", correct=ActionPrimitive.EXECUTE_CATCH),
        _spec("s2", correct=ActionPrimitive.TRIGGER_DODGE, safety=SafetyLabel.DANGEROUS),
    ]
    preds = [
        _pred("s1", ActionPrimitive.EXECUTE_CATCH, world=(0.0, 0.0, 0.0)),
        _pred("s2", ActionPrimitive.EXECUTE_CATCH),  # fatal
    ]
    results = [evaluate_scene(s, p) for s, p in zip(specs, preds)]
    report = aggregate_results("dummy", results)
    assert report.semantic_action_accuracy == pytest.approx(0.5)
    assert report.fatal_execution_rate == pytest.approx(0.5)


def test_difficulty_vector_validation():
    with pytest.raises(ValueError):
        DifficultyVector(initial_state=1.2, action=0.5, physics=0.5)


def test_difficulty_overall_geometric_mean():
    v = DifficultyVector(0.1, 0.5, 0.9)
    assert v.overall == pytest.approx((0.1 * 0.5 * 0.9) ** (1 / 3))
