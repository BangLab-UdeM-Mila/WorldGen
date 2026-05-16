"""
Aggregate per-scene results into a publishable benchmark report.

The aggregator produces the headline numbers (Track 1 / Track 2) and
also the per-axis difficulty breakdown that powers the main figures
of the paper.

Difficulty binning
------------------
Each continuous axis (initial_state, action, physics) is discretised
into four levels — ``easy`` [0, 0.25), ``medium`` [0.25, 0.5),
``hard`` [0.5, 0.75), ``adversarial`` [0.75, 1.0]. The boundaries are
exposed via :data:`DIFFICULTY_BINS` so other modules can recompute
identical bins.
"""
from __future__ import annotations

from typing import Iterable, Dict, List, Optional, Sequence

import numpy as np

from ..schemas import BenchmarkReport, SceneResult


DIFFICULTY_BINS: Sequence[tuple] = (
    ("easy", 0.0, 0.25),
    ("medium", 0.25, 0.5),
    ("hard", 0.5, 0.75),
    ("adversarial", 0.75, 1.0 + 1e-9),  # right edge inclusive
)


def _bin_label(value: float) -> str:
    for label, lo, hi in DIFFICULTY_BINS:
        if lo <= value < hi:
            return label
    return "adversarial"  # numerical safety net


def _safe_mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    arr = np.array([x for x in xs if x is not None and np.isfinite(x)],
                   dtype=np.float64)
    return float(arr.mean()) if arr.size else None


def _safe_median(xs: Iterable[Optional[float]]) -> Optional[float]:
    arr = np.array([x for x in xs if x is not None and np.isfinite(x)],
                   dtype=np.float64)
    return float(np.median(arr)) if arr.size else None


def aggregate_results(
    model_name: str,
    results: Iterable[SceneResult],
) -> BenchmarkReport:
    """Aggregate a list of per-scene results into a :class:`BenchmarkReport`.

    Parameters
    ----------
    model_name :
        Free-form identifier for the model under evaluation. Echoed
        into the returned report.
    results :
        Iterable of :class:`SceneResult` objects.
    """
    results = list(results)
    if not results:
        raise ValueError("Cannot aggregate over an empty iterable.")

    n = len(results)

    # ----- Track 1 -----
    sem_acc = sum(r.action_correct for r in results) / n
    fatal_rate = sum(r.fatal_outcome for r in results) / n

    adv_evaluable = [r for r in results if r.adversarial_fooled is not None]
    adv_rate = (
        sum(r.adversarial_fooled for r in adv_evaluable) / len(adv_evaluable)
        if adv_evaluable else None
    )

    # ----- Track 2 -----
    traj_errs = [r.trajectory_error_cm for r in results]
    ttc_errs = [
        abs(r.ttc_error_seconds) for r in results
        if r.ttc_error_seconds is not None
    ]
    reach_evaluable = [r for r in results if r.reachable_prediction is not None]
    reach_rate = (
        sum(r.reachable_prediction for r in reach_evaluable) / len(reach_evaluable)
        if reach_evaluable else None
    )

    # ----- Difficulty-conditioned accuracy -----
    by_initial_state: Dict[str, List[bool]] = {b: [] for b, _, _ in DIFFICULTY_BINS}
    by_action: Dict[str, List[bool]] = {b: [] for b, _, _ in DIFFICULTY_BINS}
    by_physics: Dict[str, List[bool]] = {b: [] for b, _, _ in DIFFICULTY_BINS}
    by_task: Dict[str, List[bool]] = {}

    for r in results:
        by_initial_state[_bin_label(r.difficulty.initial_state)].append(r.action_correct)
        by_action[_bin_label(r.difficulty.action)].append(r.action_correct)
        by_physics[_bin_label(r.difficulty.physics)].append(r.action_correct)
        by_task.setdefault(r.task_family.value, []).append(r.action_correct)

    def _bin_to_acc(d: Dict[str, List[bool]]) -> Dict[str, float]:
        return {k: (sum(v) / len(v) if v else float("nan")) for k, v in d.items()}

    return BenchmarkReport(
        model_name=model_name,
        n_scenes=n,
        semantic_action_accuracy=sem_acc,
        fatal_execution_rate=fatal_rate,
        adversarial_fool_rate=adv_rate,
        trajectory_error_cm_mean=_safe_mean(traj_errs),
        trajectory_error_cm_median=_safe_median(traj_errs),
        ttc_error_abs_mean=_safe_mean(ttc_errs) if ttc_errs else None,
        reachable_prediction_rate=reach_rate,
        by_initial_state=_bin_to_acc(by_initial_state),
        by_action=_bin_to_acc(by_action),
        by_physics=_bin_to_acc(by_physics),
        by_task_family=_bin_to_acc(by_task),
    )
