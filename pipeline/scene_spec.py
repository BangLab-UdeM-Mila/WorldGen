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
    # phys_half_x/z: actual physical half-extents for placement & floor-hit (metres).
    # Required for mesh morphs; 0.0 means fall back to size[0]/size[2].
    phys_half_x: float = 0.0
    phys_half_z: float = 0.0
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


class RampSpec(BaseModel):
    """Inclined surface for a sliding-object event.

    The ramp sits on the floor with its LOWER end at (pos_x, pos_y) and
    rises toward +Y. Objects start at 80% up the ramp and slide down
    toward the observer (-Y direction).
    """
    pos_x: float = 0.0         # X of ramp centre (world)
    pos_y: float = 0.0         # Y of ramp LOWER end (where objects fly off)
    length: float = 0.90       # ramp length along slope (metres)
    width: float = 0.35        # ramp width (metres)
    angle_deg: float = 25.0    # inclination angle from horizontal (degrees)
    euler_z: float = 0.0       # horizontal rotation for visual variety (degrees)


class StackSpec(BaseModel):
    """Starting condition for a stack-collapse event.

    N identical objects are stacked on a table and given an initial lean
    past vertical toward -Y (the observer side). Gravity does the rest.
    """
    n_items: int = 3               # number of items in the stack (2–5)
    start_x: float = 0.0          # world X of stack centre (on table top)
    start_y: float = 0.0          # world Y of stack centre
    tilt_angle_deg: float = 8.0   # initial lean past vertical toward -Y (°)
    angular_vel: float = 0.5      # initial angular velocity on top item (rad/s)
    euler_z: float = 0.0          # in-plane Z rotation of stack for variety (°)


class HangingSpec(BaseModel):
    """Starting condition for a wall- or ceiling-mounted object that falls when its mount breaks.

    The object is placed at its hanging position with a small outward tilt
    and angular velocity to simulate the mount suddenly failing.
    """
    attachment: Literal["wall_north", "ceiling"]
    pos_x: float = 0.0        # world X of object centre
    pos_y: float = 0.0        # world Y — used for ceiling mounts; wall mounts auto-compute Y
    attach_z: float = 1.80    # height of object centre when hanging (metres above floor)
    tilt_deg: float = 4.0     # initial tilt away from mounting surface (°)
    angular_vel: float = 0.25 # initial angular velocity to simulate sudden mount failure (rad/s)
    euler_z: float = 0.0      # random in-plane rotation for visual variety (°)


class TipSpec(BaseModel):
    """Starting condition for a furniture-tipping event.

    The furniture is placed upright on the floor, tilted slightly past vertical
    in the -Y direction (toward the observer camera), then released.
    """
    start_x: float = 0.0          # world X centre of furniture
    start_y: float = 0.8          # world Y centre (positive = away from observer)
    tilt_angle_deg: float = 8.0   # initial tilt past vertical toward -Y (observer side)
    angular_vel: float = 0.4      # extra angular velocity (rad/s) about +X axis → tips toward -Y
    euler_z: float = 0.0          # furniture's own Z-rotation (facing variety)


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

    # Task type — determines which scene builder path is used
    task_type: str = "object_drop"   # "object_drop" | "furniture_tip" | "hanging_fall"

    # Scene components
    # object_drop    : table + drop required
    # furniture_tip  : tip required
    # hanging_fall   : hanging required
    # stack_collapse : table + stack required
    room: RoomSpec
    table: Optional[TableSpec] = None
    object: ObjectSpec
    drop: Optional[DropSpec] = None
    tip: Optional[TipSpec] = None
    hanging: Optional[HangingSpec] = None
    stack: Optional[StackSpec] = None
    ramp: Optional[RampSpec] = None
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

    # ── Geometry validation ───────────────────────────────────

    def object_world_pos(self) -> tuple[float, float, float]:
        """Return the world-space centre of the object at t=0."""
        o = self.object
        hz = o.phys_half_z if o.phys_half_z > 0 else (o.size[2] if len(o.size) >= 3 else o.size[0])

        if self.task_type == "object_drop":
            assert self.drop and self.table
            if o.morph == "mesh":
                obj_z = self.table.height - o.bottom_z_offset + hz
            else:
                obj_z = self.table.height + hz
            return self.drop.start_x, self.drop.start_y, obj_z

        elif self.task_type == "furniture_tip":
            assert self.tip
            return self.tip.start_x, self.tip.start_y, hz

        elif self.task_type == "hanging_fall":
            assert self.hanging
            if self.hanging.attachment == "wall_north":
                hy = o.phys_half_x if o.phys_half_x > 0 else (o.size[1] if len(o.size) >= 2 else 0.03)
                wall_y = self.room.depth / 2
                return self.hanging.pos_x, wall_y - hy, self.hanging.attach_z
            else:
                return self.hanging.pos_x, self.hanging.pos_y, self.hanging.attach_z

        elif self.task_type == "stack_collapse":
            assert self.stack and self.table
            # Track the midpoint of the stack (camera targeting reference)
            n = self.stack.n_items
            stack_mid_z = self.table.height + hz * n
            return self.stack.start_x, self.stack.start_y, stack_mid_z

        elif self.task_type == "sliding_object":
            import math
            assert self.ramp
            r = self.ramp
            a = math.radians(r.angle_deg)
            t = 0.025   # ramp board half-thickness
            d = r.length * 0.80   # distance from low end (80% up ramp)
            obj_x = r.pos_x
            obj_y = r.pos_y + d * math.cos(a) - hz * math.sin(a)
            obj_z = t + d * math.sin(a) + hz * math.cos(a)
            return obj_x, obj_y, obj_z

        raise ValueError(f"Unknown task_type: {self.task_type}")

    def validate_geometry(self) -> list[str]:
        """Return a list of geometry warnings. Empty list = all checks passed.

        Checks:
        - Object is inside room bounds
        - Object is above the floor
        - Each camera's lookat angle to the object is within the FOV
        """
        import math
        warnings: list[str] = []
        r = self.room

        ox, oy, oz = self.object_world_pos()

        # 1. Object inside room (horizontal)
        # wall_north hanging objects legitimately sit at y ≈ depth/2 — skip that edge
        is_wall_mounted = (
            self.task_type == "hanging_fall"
            and self.hanging is not None
            and self.hanging.attachment == "wall_north"
        )
        margin = 0.05
        if abs(ox) > r.width / 2 - margin:
            warnings.append(f"Object X={ox:.3f} outside room width {r.width}")
        if not is_wall_mounted and abs(oy) > r.depth / 2 - margin:
            warnings.append(f"Object Y={oy:.3f} outside room depth {r.depth}")
        if oz < 0:
            warnings.append(f"Object Z={oz:.3f} is below the floor")

        # 2. Each camera can see the object (lookat ≈ object direction)
        for cam in self.cameras:
            px, py, pz = cam.pos
            lx, ly, lz = cam.lookat

            # Vector from camera to lookat
            dl = (lx - px, ly - py, lz - pz)
            dl_len = math.sqrt(sum(v*v for v in dl))
            if dl_len < 1e-6:
                warnings.append(f"Camera '{cam.name}' has zero-length lookat vector")
                continue
            dl_unit = tuple(v / dl_len for v in dl)

            # Vector from camera to object
            do = (ox - px, oy - py, oz - pz)
            do_len = math.sqrt(sum(v*v for v in do))
            if do_len < 1e-6:
                continue  # camera is at the object, skip
            do_unit = tuple(v / do_len for v in do)

            cos_angle = sum(a*b for a, b in zip(dl_unit, do_unit))
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle_deg = math.degrees(math.acos(cos_angle))

            # Object must be within 85% of half-FOV (visible, not necessarily centred)
            max_angle = cam.fov * 0.85
            if angle_deg > max_angle:
                warnings.append(
                    f"Camera '{cam.name}': object is {angle_deg:.1f}° off-centre "
                    f"(FOV={cam.fov:.1f}°, limit={max_angle:.1f}°)"
                )

        return warnings
