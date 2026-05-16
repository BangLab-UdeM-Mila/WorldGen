"""
Schemas for ReactHuman benchmark data structures.

Defines strict typed structures for:
- Scene specification (ground truth)
- Model prediction (MLLM output)
- Per-scene evaluation result
- Aggregated benchmark report

All schemas follow the dual-track design described in the benchmark paper
(Track 1: Semantic & Instinct; Track 2: Physical & Kinematic).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import json


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class ActionPrimitive(str, Enum):
    """High-level action primitive selectable by the MLLM (Track 1).

    The names mirror the discrete decision space used by the digital-human
    controller. They are kept stable to enable cross-paper comparison.
    """
    EXECUTE_CATCH = "EXECUTE_CATCH"
    TRIGGER_DODGE = "TRIGGER_DODGE"
    BRACE_FOR_IMPACT = "BRACE_FOR_IMPACT"
    NO_ACTION = "NO_ACTION"  # included for parity with the baselines


class TaskFamily(str, Enum):
    """Top-level task family in ReactHuman's taxonomy."""
    OBJECT_DROP = "object_drop"            # A1
    SLIDING_OBJECT = "sliding_object"      # A2
    ROLLING_BALL = "rolling_ball"          # A3
    STACK_COLLAPSE = "stack_collapse"      # A4
    SHELF_SLIDE = "shelf_slide"            # A5
    HANGING_FALL_WALL = "hanging_fall_wall"        # B1
    HANGING_FALL_CEILING = "hanging_fall_ceiling"  # B2
    CURTAIN_ROD_FALL = "curtain_rod_fall"  # B3
    CEILING_TILE_FALL = "ceiling_tile_fall"        # B4
    FURNITURE_TIP = "furniture_tip"        # C1
    DOOR_SWING = "door_swing"              # C2
    LADDER_SLIP = "ladder_slip"            # C3
    THROWN_OBJECT = "thrown_object"        # D1
    BOUNCING_OBJECT = "bouncing_object"    # D2
    PENDULUM_SWING = "pendulum_swing"      # E1
    CHAIN_REACTION = "chain_reaction"      # F1


class SafetyLabel(str, Enum):
    """Semantic safety label of the threatening object."""
    SAFE = "safe"
    DANGEROUS = "dangerous"
    ADVERSARIAL = "adversarial"  # appearance and physics disagree


# --------------------------------------------------------------------------- #
# Difficulty
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DifficultyVector:
    """Three orthogonal difficulty axes assigned to every scene.

    Each value is normalised to [0, 1] where 0 is trivial and 1 is the
    hardest variant encountered in the corpus. See Section "Difficulty
    Axes" in the paper for the calibration protocol.

    Attributes
    ----------
    initial_state :
        Observation sparsity. Lower observation window, stronger
        occlusion, more oblique camera angles all push this toward 1.
    action :
        Embodied feasibility. Reflects reachability margin, available
        reaction time, and pose-induced kinematic constraints.
    physics :
        Physical-reasoning load. Increases with non-trivial initial
        velocities, angular momentum, multi-body interaction, and
        appearance/physics mismatch.
    """
    initial_state: float
    action: float
    physics: float

    def __post_init__(self) -> None:
        for name in ("initial_state", "action", "physics"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"DifficultyVector.{name} must be in [0, 1], got {v}"
                )

    @property
    def overall(self) -> float:
        """Geometric mean of the three axes.

        We use geometric (not arithmetic) mean because difficulty is
        multiplicatively bottlenecked: a scene that is trivial on two
        axes but adversarial on the third remains hard overall.
        """
        return (self.initial_state * self.action * self.physics) ** (1 / 3)


# --------------------------------------------------------------------------- #
# Ground truth (per scene)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GroundTruth:
    """Ground-truth state produced by the physics simulator.

    Attributes
    ----------
    correct_action :
        The single primitive that the benchmark accepts as correct.
        May be defined as the action that simultaneously (i) avoids
        catastrophic harm to the humanoid and (ii) minimises damage
        to objects with positive value.
    impact_point_world :
        3-D world coordinate (metres) at which the threatening object
        will first contact a ground-or-body surface, computed from the
        full simulator rollout.
    time_to_impact :
        Seconds between the freeze instant and the impact event.
    reachable_mask :
        Whether `impact_point_world` lies inside the SMPL-X kinematic
        reachability hull of the humanoid given its frozen pose.
    safety_label :
        Semantic safety label of the threatening object.
    object_value :
        Monetary or symbolic value of the object on a 0..1 scale.
        Used in dilemma scenes where two objects compete for the
        humanoid's single available end-effector.
    """
    correct_action: ActionPrimitive
    impact_point_world: Tuple[float, float, float]
    time_to_impact: float
    reachable_mask: bool
    safety_label: SafetyLabel
    object_value: float = 0.0


# --------------------------------------------------------------------------- #
# Scene specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SceneSpec:
    """Full specification of one benchmark scene.

    This mirrors the on-disk `spec.json` produced by
    `genesis_scene_generation`. The fields not consumed by the metric
    module are loaded into the `extras` dict to keep forward
    compatibility as the scene generator evolves.
    """
    scene_id: str
    task_family: TaskFamily
    difficulty: DifficultyVector
    ground_truth: GroundTruth
    camera_intrinsics: Dict[str, float] = field(default_factory=dict)
    camera_extrinsics: Dict[str, List[float]] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Model output
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelPrediction:
    """Structured output produced by the model being evaluated.

    Attributes
    ----------
    scene_id :
        Identifier matching the corresponding `SceneSpec`.
    action :
        Chosen primitive (Track 1).
    target_point_pixel :
        Pixel coordinate `(u, v)` of the intended interception point
        in the observer-camera frame (Track 2). May be `None` if the
        model declines to localise (e.g. for BRACE_FOR_IMPACT).
    target_point_world :
        Optional 3-D world coordinate. Only used when the harness
        un-projects the pixel via depth, otherwise left empty.
    latency_seconds :
        Wall-clock time spent on the model call. Reported but not
        scored, because the freeze-and-predict paradigm decouples
        scoring from latency.
    rationale :
        Free-form chain-of-thought string returned by the model.
        Stored for qualitative analysis only, never scored.
    """
    scene_id: str
    action: ActionPrimitive
    target_point_pixel: Optional[Tuple[float, float]] = None
    target_point_world: Optional[Tuple[float, float, float]] = None
    latency_seconds: float = 0.0
    rationale: str = ""


# --------------------------------------------------------------------------- #
# Per-scene result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SceneResult:
    """Atomic per-scene evaluation result.

    Contains both Track 1 (semantic) and Track 2 (kinematic) outcomes
    so that downstream aggregation can slice along any axis without
    re-running the simulator.
    """
    scene_id: str
    task_family: TaskFamily
    difficulty: DifficultyVector

    # Track 1 — semantic
    action_correct: bool
    fatal_outcome: bool
    adversarial_fooled: Optional[bool]  # None if scene is not adversarial

    # Track 2 — kinematic
    trajectory_error_cm: Optional[float]     # Euclidean distance, None if N/A
    ttc_error_seconds: Optional[float]       # signed; negative = late
    reachable_prediction: Optional[bool]     # prediction inside reach hull?

    # Diagnostics
    latency_seconds: float


# --------------------------------------------------------------------------- #
# Aggregated report
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkReport:
    """Aggregated benchmark report for one (model, evaluation-run) pair.

    The class is mutable so that the aggregator can populate slices
    progressively. Conversion to a flat dict for JSON / TSV export is
    provided through :meth:`to_dict`.
    """
    model_name: str
    n_scenes: int

    # Track 1 headline numbers
    semantic_action_accuracy: float
    fatal_execution_rate: float
    adversarial_fool_rate: Optional[float]

    # Track 2 headline numbers
    trajectory_error_cm_mean: Optional[float]
    trajectory_error_cm_median: Optional[float]
    ttc_error_abs_mean: Optional[float]
    reachable_prediction_rate: Optional[float]

    # Difficulty-conditioned breakdown
    by_initial_state: Dict[str, float] = field(default_factory=dict)
    by_action: Dict[str, float] = field(default_factory=dict)
    by_physics: Dict[str, float] = field(default_factory=dict)
    by_task_family: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str, indent: int = 2) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)
