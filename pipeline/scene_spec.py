"""
SceneSpec — the universal schema for a single simulation scenario.

Every scene in the ReactHuman benchmark is fully described by a SceneSpec.
Given the same SceneSpec, the simulation is 100% reproducible.

Design principles
-----------------
- Self-contained: no external state needed to run a scene.
- JSON-serialisable: specs can be stored, versioned, and shared.
- Validated: Pydantic catches bad values before Genesis sees them.
- LLM-friendly: the schema is small enough to fit in an API tool definition.
"""

from __future__ import annotations

import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────────────────────

class ObjectSpec(BaseModel):
    """Describes the object that will be dropped."""
    name: str
    display_name: str
    category: Literal["safe", "dangerous", "adversarial"]
    morph: Literal["mesh", "sphere", "box", "cylinder"]
    # mesh_path: relative to repo assets/meshes/ (only for morph=mesh)
    mesh_path: Optional[str] = None
    # bottom_z_offset: local Z of the lowest point of a mesh (signed, metres)
    bottom_z_offset: float = 0.0
    # size: [x,y,z] half-extents for primitives, or [scale] for mesh
    size: list[float] = Field(default_factory=lambda: [0.05, 0.05, 0.05])
    density: float                  # kg/m³
    friction: float
    restitution: float
    color_rgb: list[float]          # [R,G,B] in 0–1
    roughness: float
    ior: float = 1.0
    # Ground-truth semantics
    catch_safe: bool
    safety_label: Literal["safe", "caution", "dangerous"]
    mass_hint: Literal["light", "medium", "heavy"]
    description: str = ""

    @model_validator(mode="after")
    def _check_mesh_path(self) -> "ObjectSpec":
        if self.morph == "mesh" and self.mesh_path is None:
            raise ValueError("mesh_path required when morph='mesh'")
        return self


class TableSpec(BaseModel):
    """Dining/kitchen table."""
    width: float = 1.20        # x
    depth: float = 0.80        # y
    height: float = 0.77       # z of top surface
    thickness: float = 0.04
    leg_size: float = 0.06
    pos_x: float = 0.0         # table centre in world x
    pos_y: float = 0.0         # table centre in world y
    color_rgb: list[float] = Field(default_factory=lambda: [0.35, 0.22, 0.10])
    roughness: float = 0.55
    friction: float = 0.6


class RoomSpec(BaseModel):
    """Room geometry and materials."""
    type: Literal["dining", "kitchen", "living", "office"]
    display_name: str
    width: float
    depth: float
    height: float
    wall_thickness: float = 0.12
    wall_color_rgb: list[float]
    ceiling_color_rgb: list[float]
    floor_color_rgb: list[float]
    floor_roughness: float = 0.55
    # Window emissive panel
    window_wall: Literal["north", "east", "south", "west"] = "north"
    window_pos_x: float = 0.0
    window_pos_z: float = 1.45
    window_width: float = 1.20
    window_height: float = 1.00
    window_emissive: list[float] = Field(default_factory=lambda: [6.0, 5.8, 5.2])


class DropSpec(BaseModel):
    """How and where the object starts its drop."""
    # Position on the table (world coords, table top surface is z=table.height)
    start_x: float          # > table_width/2 − object_radius → overhangs → gravity tip
    start_y: float = 0.0
    # Initial rotation in degrees (Euler x-y-z)
    euler_z: float = 0.0    # spin around vertical axis for variety
    # Optional initial velocity (m/s)
    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_z: float = 0.0


class CameraSpec(BaseModel):
    name: str
    pos: list[float]       # [x, y, z]
    lookat: list[float]    # [x, y, z]
    fov: float = 55.0


class LightSpec(BaseModel):
    type: Literal["point", "directional", "ambient"]
    pos: Optional[list[float]] = None   # point only
    dir: Optional[list[float]] = None   # directional only
    color: list[float]
    intensity: float


class LightingSpec(BaseModel):
    ambient_light: list[float]
    background_color: list[float]
    shadow: bool = True
    lights: list[LightSpec]


# ─────────────────────────────────────────────────────────────────────────────
# Top-level SceneSpec
# ─────────────────────────────────────────────────────────────────────────────

class SceneSpec(BaseModel):
    """
    Complete, self-contained specification for one simulation scenario.
    JSON-serialisable. Given the same spec, results are reproducible.
    """
    scene_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    seed: int
    description: str = ""       # natural-language origin (if LLM-generated)
    adversarial: bool = False   # True → visual appearance contradicts physics

    # Scene components
    room: RoomSpec
    table: TableSpec
    object: ObjectSpec
    drop: DropSpec
    cameras: list[CameraSpec]
    lighting: LightingSpec

    # Simulation parameters
    dt: float = 1 / 240
    substeps: int = 4
    duration_s: float = 3.0

    # Ground-truth labels (for benchmark evaluation)
    ground_truth_action: Literal["EXECUTE_CATCH", "TRIGGER_DODGE", "BRACE_FOR_IMPACT"]
    safety_label: Literal["safe", "caution", "dangerous"]
    # Track-2: filled in post-simulation
    interception_point_2d: Optional[list[float]] = None  # [px_x, px_y]
    interception_point_3d: Optional[list[float]] = None  # [x, y, z] metres
    time_to_floor_s: Optional[float] = None

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "SceneSpec":
        return cls.model_validate_json(text)
