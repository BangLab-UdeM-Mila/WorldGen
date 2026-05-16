"""
Track 2 — Physical & Kinematic metrics.

These metrics evaluate whether the MLLM not only chose the correct
discrete action, but also localised the interception point with
sufficient spatial and temporal precision to be physically executable.

References
----------
ReactHuman paper, Section "Evaluation Methodology — Dual-Track Design".
Trajectory and TTC errors follow the conventions of WorldBench (2025)
and PhysCtrl (2025); the *reachability-aware* variant is novel to
ReactHuman.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple, List

import numpy as np

from ..schemas import ModelPrediction, SceneSpec, ActionPrimitive


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _unproject_pixel(
    pixel: Tuple[float, float],
    intrinsics: dict,
    extrinsics: dict,
    depth: float,
) -> np.ndarray:
    """Back-project a pixel `(u, v)` at a known metric depth `z` to a
    3-D point in the world frame.

    Uses the standard pinhole model. The harness is expected to supply
    `depth` from the Z-buffer of the same frame the model observed.

    Parameters
    ----------
    intrinsics :
        Dict with keys `fx`, `fy`, `cx`, `cy` (pixels).
    extrinsics :
        Dict with keys `R` (3x3 row-major) and `t` (length-3) defining
        the camera-to-world transform.
    """
    u, v = pixel
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    # camera-frame ray
    x_c = (u - cx) * depth / fx
    y_c = (v - cy) * depth / fy
    z_c = depth
    p_c = np.array([x_c, y_c, z_c], dtype=np.float64)

    R = np.array(extrinsics["R"], dtype=np.float64).reshape(3, 3)
    t = np.array(extrinsics["t"], dtype=np.float64)
    return R @ p_c + t


# --------------------------------------------------------------------------- #
# Trajectory error
# --------------------------------------------------------------------------- #
def trajectory_error_cm(
    predictions: Iterable[ModelPrediction],
    specs: Iterable[SceneSpec],
    *,
    depth_lookup: Optional[dict] = None,
    only_catch: bool = True,
) -> List[float]:
    """Per-scene Euclidean distance, in centimetres, between the model's
    predicted interception point and the simulator's ground-truth impact
    point.

    Parameters
    ----------
    depth_lookup :
        Optional mapping `scene_id -> depth_metres` providing the depth
        at the predicted pixel. If `target_point_world` is already
        populated on the prediction, depth is not needed.
    only_catch :
        If True (default), restrict the metric to scenes whose
        ground-truth action is `EXECUTE_CATCH`. For DODGE / BRACE
        scenes the notion of "interception error" is ill-defined and
        the prediction is allowed to be `None`.

    Returns
    -------
    list[float]
        One float per evaluable scene. Use NumPy reductions on the
        returned list for summary statistics.
    """
    pred_map = {p.scene_id: p for p in predictions}
    spec_map = {s.scene_id: s for s in specs}

    errors: List[float] = []
    for sid in pred_map.keys() & spec_map.keys():
        spec = spec_map[sid]
        pred = pred_map[sid]

        if only_catch and spec.ground_truth.correct_action != ActionPrimitive.EXECUTE_CATCH:
            continue

        # Obtain a world-frame point from the prediction.
        if pred.target_point_world is not None:
            p_world = np.array(pred.target_point_world, dtype=np.float64)
        elif pred.target_point_pixel is not None and depth_lookup is not None:
            depth = depth_lookup.get(sid)
            if depth is None:
                continue
            p_world = _unproject_pixel(
                pred.target_point_pixel,
                spec.camera_intrinsics,
                spec.camera_extrinsics,
                depth,
            )
        else:
            # model refused to localise; treat as the worst case.
            errors.append(float("inf"))
            continue

        gt_world = np.array(spec.ground_truth.impact_point_world, dtype=np.float64)
        err_m = float(np.linalg.norm(p_world - gt_world))
        errors.append(err_m * 100.0)  # → centimetres
    return errors


# --------------------------------------------------------------------------- #
# Time-to-collision error
# --------------------------------------------------------------------------- #
def time_to_collision_error(
    predictions: Iterable[ModelPrediction],
    specs: Iterable[SceneSpec],
    predicted_ttcs: dict,
) -> List[float]:
    """Signed TTC errors, in seconds, between each model's predicted
    time-to-collision and the simulator's ground-truth value.

    Parameters
    ----------
    predicted_ttcs :
        Mapping `scene_id -> ttc_seconds`. This is kept separate from
        :class:`ModelPrediction` because not every prompt template
        elicits a TTC estimate; storing it here keeps the schema lean.

    Returns
    -------
    list[float]
        Negative values mean the model predicted *late* (i.e. would
        arrive after the actual impact); positive values mean *early*.
    """
    spec_map = {s.scene_id: s for s in specs}
    pred_ids = {p.scene_id for p in predictions}

    errors: List[float] = []
    for sid, ttc_hat in predicted_ttcs.items():
        if sid not in spec_map or sid not in pred_ids:
            continue
        ttc_true = spec_map[sid].ground_truth.time_to_impact
        errors.append(ttc_hat - ttc_true)
    return errors


# --------------------------------------------------------------------------- #
# Reachability-aware metric (novel to ReactHuman)
# --------------------------------------------------------------------------- #
def reachable_prediction_rate(
    predictions: Iterable[ModelPrediction],
    specs: Iterable[SceneSpec],
    *,
    reach_margin_cm: float = 5.0,
    depth_lookup: Optional[dict] = None,
) -> Optional[float]:
    """Fraction of catch-type predictions that land inside the
    humanoid's kinematic reachability hull.

    Rationale
    ---------
    Standard Euclidean trajectory error can be small while the
    predicted point is physically unreachable (e.g. one centimetre
    behind the humanoid's shoulder). For embodied evaluation it is
    more honest to ask whether the prediction is *executable*. The
    reachability mask is precomputed by the scene generator at the
    moment of freeze, conditioned on the humanoid's frozen pose, and
    encoded as `GroundTruth.reachable_mask`.

    A predicted point is considered reachable iff

    1.  The ground-truth impact point is itself reachable
        (`reachable_mask = True`); otherwise the scene is excluded
        because the prompted correct action could not have been
        CATCH.
    2.  The predicted point lies within `reach_margin_cm` of the
        ground-truth point. We use proximity to the ground-truth
        reachable point as a proxy for being inside the hull, which
        avoids loading the full SMPL-X mesh at evaluation time.

    Returns
    -------
    float or None
        Rate in [0, 1], or `None` if no scene is evaluable.
    """
    pred_map = {p.scene_id: p for p in predictions}
    spec_map = {s.scene_id: s for s in specs}

    evaluable, reachable = 0, 0
    for sid in pred_map.keys() & spec_map.keys():
        spec = spec_map[sid]
        if not spec.ground_truth.reachable_mask:
            continue
        if spec.ground_truth.correct_action != ActionPrimitive.EXECUTE_CATCH:
            continue

        pred = pred_map[sid]
        if pred.target_point_world is not None:
            p_world = np.array(pred.target_point_world, dtype=np.float64)
        elif pred.target_point_pixel is not None and depth_lookup is not None:
            depth = depth_lookup.get(sid)
            if depth is None:
                continue
            p_world = _unproject_pixel(
                pred.target_point_pixel,
                spec.camera_intrinsics,
                spec.camera_extrinsics,
                depth,
            )
        else:
            continue

        gt = np.array(spec.ground_truth.impact_point_world, dtype=np.float64)
        evaluable += 1
        if float(np.linalg.norm(p_world - gt)) * 100.0 <= reach_margin_cm:
            reachable += 1

    if evaluable == 0:
        return None
    return reachable / evaluable
