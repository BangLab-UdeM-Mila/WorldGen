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


class RollSpec(BaseModel):
    """Starting condition for a rolling-ball event.

    The ball starts on the table surface with an initial velocity directed
    toward the table edge (+X direction). angular_vel_y is set for
    rolling-without-slipping so the ball rolls rather than slides.
    """
    start_x: float          # ball centre world x (inside table, will roll to edge)
    start_y: float = 0.0    # ball centre world y
    vel_x: float = 0.8      # initial linear velocity toward table edge (m/s)
    vel_y: float = 0.0      # lateral velocity component (m/s)
    angular_vel_y: float = 0.0  # wy (rad/s) = vel_x / radius for rolling without slipping


class ShelfSpec(BaseModel):
    """Wall-mounted shelf from which an object slides off toward the observer.

    The shelf is a fixed box attached to the north-wall interior face.
    The object starts near the shelf front edge and is given a small
    initial velocity toward the observer (-Y direction) to trigger the slide.
    """
    height: float = 1.60    # z of the shelf top surface (metres above floor)
    pos_x: float = 0.0      # x centre of shelf and object start position
    depth: float = 0.18     # shelf depth in y (from wall to front edge, metres)
    width: float = 0.60     # shelf width in x (metres)
    thickness: float = 0.03 # shelf board full thickness (metres)
    vel_y: float = -0.35    # initial velocity toward observer (-Y) to trigger slide-off


class ThrownSpec(BaseModel):
    """Starting condition for a thrown-object event.

    The object is spawned at (launch_x, launch_y, launch_z) with initial
    linear velocity (vel_x, vel_y, vel_z).  Positive Y is north (away from
    observer); vel_y must be negative so the object flies toward the observer.

    The event detected is when the object COM crosses y = 0 (room midpoint)
    on its way toward the observer.
    """
    launch_x: float = 0.0   # world X of spawn point
    launch_y: float = 1.0   # world Y of spawn point (positive = north of centre)
    launch_z: float = 1.50  # world Z height of spawn point (metres)
    vel_x: float = 0.0      # initial X velocity (lateral drift)
    vel_y: float = -3.0     # initial Y velocity (negative = toward observer)
    vel_z: float = 0.0      # initial Z velocity (upward arc or flat)


class PendulumSpec(BaseModel):
    """Starting condition for a pendulum-swing event.

    The bob is a rigid body attached to a fixed ceiling pivot via a MuJoCo
    hinge joint (axis = world X).  Positive Y is north (away from observer).

    initial_angle_deg is the joint angle at t=0 in degrees:
      0   → bob hangs straight down (vertical, maximum speed position)
      +θ  → bob pulled toward north (+Y), released toward observer

    angular_vel ≤ 0 adds an initial push southward on top of gravity.
    The event detected is when the joint angle crosses 0 (bob at vertical,
    maximum speed, heading toward the observer).
    """
    pivot_x: float = 0.0          # world X of ceiling attachment
    pivot_y: float = 0.0          # world Y of ceiling attachment
    pivot_z: float = 2.50         # world Z of ceiling attachment (below ceiling)
    length: float = 1.20          # rope / rod length (metres)
    bob_half_z: float = 0.12      # half-height of bob (floor clearance check)
    initial_angle_deg: float = 45.0   # initial angle from vertical (positive = north)
    angular_vel: float = 0.0      # initial angular velocity (rad/s, 0 = pure release)


class BounceSpec(BaseModel):
    """Starting condition for a bouncing-object event.

    Rather than relying on floor-contact restitution (unreliable in Genesis),
    the ball spawns just above the floor at the first-bounce location with
    the post-bounce upward and southward velocity it would have after the
    initial impact, producing a parabolic arc toward the observer.

    start_z ≈ ball radius (just off the floor at the bounce point)
    vel_z > 0 : upward component (proportional to drop height × restitution)
    vel_y < 0 : southward component toward the observer

    Time-to-floor event = when the ball completes its arc and lands again.
    """
    start_x: float = 0.0    # world X of first-bounce location
    start_y: float = 0.5    # world Y of first-bounce location (positive = north)
    start_z: float = 0.05   # world Z at t=0 (set equal to ball radius in _make_bounce)
    vel_x: float = 0.0      # lateral drift (m/s)
    vel_y: float = -1.2     # velocity toward observer (m/s, negative)
    vel_z: float = 2.5      # upward post-bounce velocity (m/s)


class LadderSpec(BaseModel):
    """Starting condition for a ladder-slip event.

    The ladder (a tall box) is placed in mid-slip: already tilted past vertical
    toward the observer (-Y direction), as if its base has just slipped out on
    the smooth floor.  This avoids wall-contact issues and produces clean physics.

    Convention matches furniture_tip: lean_deg is the tilt angle from vertical
    toward -Y (observer side).  COM is placed at (base_x, start_y, hz) on the
    floor, with euler_x = +lean_deg.

    Geometry:
      pos = (base_x, start_y, hz)          # hz = object.size[2], half-height
      euler = (lean_deg, 0, euler_z)        # tilt toward -Y (observer)
    """
    lean_deg: float = 30.0         # tilt from vertical toward observer (degrees, 20–45 typical)
    base_x: float = 0.0            # world X of the ladder COM
    start_y: float = 0.60          # world Y of the ladder COM (north half of room)
    angular_vel: float = 0.6       # initial wx (rad/s) to accelerate the fall toward -Y
    euler_z: float = 0.0           # in-plane Z rotation for visual variety (degrees)


class ChainSpec(BaseModel):
    """Starting condition for a chain-reaction event.

    A trigger box (A) slides southward on a table and collides with the main
    object (B), which is perched at the south edge of the table.  B then falls
    off the table toward the observer.

    Positive Y is north; the observer is at -Y.

    A is always a book-like box (parameterised by trigger_* fields).
    B is the SceneSpec.object at (target_x, target_y) on the table.
    """
    # Target object B (SceneSpec.object) position on table (world coords)
    target_x: float = 0.0           # world X of target B centre
    target_y: float = -0.25         # world Y of target B (near south table edge)
    euler_z: float = 0.0            # target's Z-rotation for variety
    # Trigger object A (book-like box)
    trigger_x: float = 0.0          # world X of trigger A start (≈ target_x)
    trigger_y: float = 0.10         # world Y of trigger A start (north of target)
    trigger_vel_y: float = -1.2     # initial velocity toward target (m/s, negative=south)
    trigger_size: list[float] = Field(default_factory=lambda: [0.075, 0.055, 0.015])
    trigger_density: float = 650.0  # kg/m³ (book-like)
    trigger_color: list[float] = Field(default_factory=lambda: [0.20, 0.40, 0.72])


class DoorSpec(BaseModel):
    """Starting condition for a door-swing event.

    The door panel is a rigid box hinged at one vertical edge via a revolute
    joint (Z axis = vertical).  The hinge sits at (hinge_x, door_y, 0.01)
    where door_y is the SOUTH WALL inner surface (≈ -room_depth/2 + wall_thickness).

    initial_angle_deg is the revolute joint angle at t=0 in degrees:
      0  → door panel points east  (+X from hinge) — flush with south wall, CLOSED
      90 → door panel points north (+Y from hinge) — perpendicular to wall, fully OPEN

    angular_vel < 0 rotates clockwise from above → door swings from open toward
    closed (back into the south wall).  The "event" detected during simulation is
    the door angle dropping below 20° (nearly closed, door threatening the doorway).
    """
    width: float = 0.90              # door panel width (metres)
    height: float = 2.10             # door panel height (metres)
    thickness: float = 0.045         # door panel thickness (metres)
    hinge_x: float = 0.0             # world X of the vertical hinge axis
    door_y: float = -1.65            # world Y of hinge = south wall inner surface
    initial_angle_deg: float = 85.0  # revolute joint angle at t=0 (degrees)
    angular_vel: float = -1.5        # initial angular velocity (rad/s), -ve → closing


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
    # "object_drop" | "sliding_object" | "stack_collapse" |
    # "hanging_fall" | "furniture_tip" | "rolling_ball" | "shelf_slide" |
    # "door_swing"  | "thrown_object" | "pendulum_swing" | "bouncing_object" |
    # "ladder_slip" | "chain_reaction"
    task_type: str = "object_drop"

    # Scene components (required fields vary by task_type)
    # object_drop      : table + drop
    # sliding_object   : ramp
    # stack_collapse   : table + stack
    # hanging_fall     : hanging
    # furniture_tip    : tip
    # rolling_ball     : table + roll
    # shelf_slide      : shelf
    # door_swing       : door
    # thrown_object    : thrown
    # pendulum_swing   : pendulum
    # bouncing_object  : bounce
    # ladder_slip      : ladder
    # chain_reaction   : table + chain
    room: RoomSpec
    table: Optional[TableSpec] = None
    object: ObjectSpec
    drop: Optional[DropSpec] = None
    tip: Optional[TipSpec] = None
    hanging: Optional[HangingSpec] = None
    stack: Optional[StackSpec] = None
    ramp: Optional[RampSpec] = None
    roll: Optional[RollSpec] = None
    shelf: Optional[ShelfSpec] = None
    door: Optional[DoorSpec] = None
    thrown: Optional[ThrownSpec] = None
    pendulum: Optional[PendulumSpec] = None
    bounce: Optional[BounceSpec] = None
    ladder: Optional[LadderSpec] = None
    chain: Optional[ChainSpec] = None
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

        elif self.task_type == "rolling_ball":
            assert self.roll and self.table
            radius = o.size[0]   # sphere: size[0] = radius
            obj_z = self.table.height + radius
            return self.roll.start_x, self.roll.start_y, obj_z

        elif self.task_type == "shelf_slide":
            assert self.shelf
            wall_y = self.room.depth / 2
            obj_y = wall_y - self.shelf.depth * 0.70
            obj_z = self.shelf.height + hz
            return self.shelf.pos_x, obj_y, obj_z

        elif self.task_type == "door_swing":
            import math
            assert self.door
            angle = math.radians(self.door.initial_angle_deg)
            cx = self.door.hinge_x + (self.door.width / 2) * math.cos(angle)
            cy = self.door.door_y + (self.door.width / 2) * math.sin(angle)
            cz = 0.01 + self.door.height / 2   # 0.01 m floor gap
            return cx, cy, cz

        elif self.task_type == "thrown_object":
            assert self.thrown
            return self.thrown.launch_x, self.thrown.launch_y, self.thrown.launch_z

        elif self.task_type == "pendulum_swing":
            import math
            assert self.pendulum
            p = self.pendulum
            angle = math.radians(p.initial_angle_deg)
            bx = p.pivot_x
            by = p.pivot_y + p.length * math.sin(angle)
            bz = p.pivot_z - p.length * math.cos(angle)
            return bx, by, bz

        elif self.task_type == "bouncing_object":
            assert self.bounce
            return self.bounce.start_x, self.bounce.start_y, self.bounce.start_z

        elif self.task_type == "ladder_slip":
            assert self.ladder
            o = self.object
            hz = o.size[2] if len(o.size) >= 3 else o.size[0]
            # Same convention as furniture_tip: COM on floor at hz height
            return self.ladder.base_x, self.ladder.start_y, hz

        elif self.task_type == "chain_reaction":
            assert self.chain and self.table
            o = self.object
            hz = o.phys_half_z if o.phys_half_z > 0 else (o.size[2] if len(o.size) >= 3 else o.size[0])
            if o.morph == "mesh":
                obj_z = self.table.height - o.bottom_z_offset + hz
            else:
                obj_z = self.table.height + hz
            return self.chain.target_x, self.chain.target_y, obj_z

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
        # Objects near the north wall (hanging or shelf) legitimately sit at y ≈ depth/2
        is_wall_mounted = (
            self.task_type == "hanging_fall"
            and self.hanging is not None
            and self.hanging.attachment == "wall_north"
        ) or (
            self.task_type == "shelf_slide"
        ) or (
            self.task_type == "pendulum_swing"  # bob position handled by custom check below
        )
        margin = 0.05
        if abs(ox) > r.width / 2 - margin:
            warnings.append(f"Object X={ox:.3f} outside room width {r.width}")
        if not is_wall_mounted and abs(oy) > r.depth / 2 - margin:
            warnings.append(f"Object Y={oy:.3f} outside room depth {r.depth}")
        if oz < 0:
            warnings.append(f"Object Z={oz:.3f} is below the floor")

        # pendulum_swing: bob must clear the floor at vertical and stay inside room
        if self.task_type == "pendulum_swing" and self.pendulum is not None:
            import math as _math
            p = self.pendulum
            bob_bottom_z = p.pivot_z - p.length - p.bob_half_z
            if bob_bottom_z < 0.05:
                warnings.append(
                    f"Pendulum bob bottom z={bob_bottom_z:.3f} below floor clearance"
                )
            max_bob_y = p.pivot_y + p.length * _math.sin(_math.radians(p.initial_angle_deg))
            if max_bob_y > r.depth / 2 - margin:
                warnings.append(
                    f"Pendulum bob max Y={max_bob_y:.3f} outside room depth {r.depth}"
                )
            if abs(p.pivot_x) > r.width / 2 - margin:
                warnings.append(
                    f"Pendulum pivot X={p.pivot_x:.3f} outside room width {r.width}"
                )

        # ladder_slip: ladder COM must be in north half and base_x inside room
        if self.task_type == "ladder_slip" and self.ladder is not None:
            lad = self.ladder
            if lad.start_y < 0:
                warnings.append(
                    f"Ladder start_y={lad.start_y:.3f} should be positive (north half)"
                )
            if lad.start_y > r.depth / 2 - margin:
                warnings.append(
                    f"Ladder start_y={lad.start_y:.3f} too close to north wall"
                )
            if abs(lad.base_x) > r.width / 2 - margin:
                warnings.append(
                    f"Ladder base x={lad.base_x:.3f} outside room width {r.width}"
                )

        # chain_reaction: trigger and target must fit on the table
        if self.task_type == "chain_reaction" and self.chain is not None and self.table is not None:
            t = self.table
            ch = self.chain
            half_w = t.width / 2
            half_d = t.depth / 2
            if abs(ch.target_x - t.pos_x) > half_w - margin:
                warnings.append(
                    f"Chain target x={ch.target_x:.3f} outside table width"
                )
            if ch.target_y < t.pos_y - half_d + margin:
                warnings.append(
                    f"Chain target y={ch.target_y:.3f} outside table south edge"
                )
            if ch.trigger_y > t.pos_y + half_d - margin:
                warnings.append(
                    f"Chain trigger y={ch.trigger_y:.3f} outside table north edge"
                )

        # door_swing: door panel must not protrude through east/west walls when open
        if self.task_type == "door_swing" and self.door is not None:
            far_x = abs(self.door.hinge_x) + self.door.width
            if far_x > r.width / 2 - margin:
                warnings.append(
                    f"Door far end x={far_x:.3f} too close to east/west wall x={r.width/2:.3f}"
                )

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
