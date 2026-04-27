"""
Procedural Scene Randomizer
============================
Generates SceneSpec objects from a seed without any LLM calls.
All randomness is driven by numpy's default_rng(seed), making
every generated scene 100% reproducible.

Usage
-----
    from pipeline.randomizer import Randomizer
    r = Randomizer()
    spec = r.sample(seed=42)
    specs = r.sample_batch(n=1000, seed_start=0)
"""

from __future__ import annotations

import math
from pathlib import Path

from typing import Optional

import numpy as np
import yaml

from .scene_spec import (
    BounceSpec, CameraSpec, ChainSpec, DoorSpec, DropSpec, HangingSpec,
    LadderSpec, LightingSpec, LightSpec,
    ObjectSpec, PendulumSpec, RampSpec, RollSpec, RoomSpec, SceneSpec,
    ShelfSpec, StackSpec, TableSpec, TipSpec, ThrownSpec,
)

# ── Asset library paths ───────────────────────────────────────
_HERE = Path(__file__).parent.parent
_OBJECTS_YAML = _HERE / "asset_library" / "objects.yaml"
_ROOMS_YAML   = _HERE / "asset_library" / "rooms.yaml"


def _load_objects(task_type: str = "object_drop") -> list[dict]:
    with open(_OBJECTS_YAML) as f:
        all_objs = yaml.safe_load(f)
    # sliding_object and chain_reaction reuse the object_drop pool
    if task_type in ("sliding_object", "chain_reaction"):
        effective = "object_drop"
    else:
        effective = task_type
    return [o for o in all_objs if o.get("task_type", "object_drop") == effective]


def _load_rooms() -> tuple[list[dict], dict]:
    with open(_ROOMS_YAML) as f:
        data = yaml.safe_load(f)
    rooms   = data.get("rooms", [])
    presets = data.get("lighting_presets", {})
    return rooms, presets


# ─────────────────────────────────────────────────────────────────────────────

class Randomizer:
    """
    Samples SceneSpec objects procedurally.

    Parameters
    ----------
    task_type : str
        Which task to generate. "object_drop" (default) or "furniture_tip".
    object_categories : list of str, optional
        Filter object pool to these categories.
        Choices: "safe", "dangerous", "adversarial".
        Default: all categories.
    room_types : list of str, optional
        Filter room pool to these types.
        Choices: "dining", "kitchen", "living", "office".
        Default: all types.
    adversarial_prob : float
        Probability that an adversarial object is selected
        (only when "adversarial" in object_categories). Default 0.15.
    """

    def __init__(
        self,
        task_type: str = "object_drop",
        object_categories: Optional[list[str]] = None,
        room_types: Optional[list[str]] = None,
        adversarial_prob: float = 0.15,
    ):
        self._task_type = task_type
        all_objects = _load_objects(task_type)
        all_rooms, self._lighting_presets = _load_rooms()

        cats = set(object_categories) if object_categories else {"safe", "dangerous", "adversarial"}
        self._objects = [o for o in all_objects if o["category"] in cats]
        if not self._objects:
            raise ValueError(f"No objects match categories: {object_categories}")

        rtypes = set(room_types) if room_types else {"dining", "kitchen", "living", "office"}
        self._rooms = [r for r in all_rooms if r["type"] in rtypes]
        if not self._rooms:
            raise ValueError(f"No rooms match types: {room_types}")

        self._adversarial_prob = adversarial_prob

    # ── Public API ────────────────────────────────────────────

    def sample(self, seed: int) -> SceneSpec:
        """Draw one SceneSpec from the given seed."""
        rng = np.random.default_rng(seed)
        return self._build_spec(rng, seed)

    def sample_batch(
        self,
        n: int,
        seed_start: int = 0,
        object_name: Optional[str] = None,
    ) -> list[SceneSpec]:
        """
        Draw `n` specs starting at `seed_start`.
        If `object_name` is given, all specs use that object.
        """
        specs = []
        for i in range(n):
            rng = np.random.default_rng(seed_start + i)
            spec = self._build_spec(rng, seed_start + i, force_object=object_name)
            specs.append(spec)
        return specs

    def sample_with_constraints(
        self,
        seed: int,
        force_object: Optional[str] = None,
        force_room: Optional[str] = None,
    ) -> SceneSpec:
        """
        Draw one SceneSpec from `seed`, optionally pinning object and/or room.
        Used by both LLM mode (via llm_planner) and procedural mode (--objects flag).
        """
        rng = np.random.default_rng(seed)

        orig_objects = self._objects
        orig_rooms   = self._rooms

        if force_object:
            matched = [o for o in orig_objects if o["name"] == force_object]
            if matched:
                self._objects = matched
        if force_room:
            matched = [r for r in orig_rooms if r["type"] == force_room]
            if matched:
                self._rooms = matched

        spec = self._build_spec(rng, seed)

        self._objects = orig_objects
        self._rooms   = orig_rooms
        return spec

    # ── Internal builders ─────────────────────────────────────

    def _build_spec(
        self,
        rng: np.random.Generator,
        seed: int,
        force_object: Optional[str] = None,
    ) -> SceneSpec:
        if self._task_type == "furniture_tip":
            return self._build_furniture_tip_spec(rng, seed, force_object)
        if self._task_type == "hanging_fall":
            return self._build_hanging_fall_spec(rng, seed, force_object)
        if self._task_type == "stack_collapse":
            return self._build_stack_collapse_spec(rng, seed, force_object)
        if self._task_type == "sliding_object":
            return self._build_sliding_object_spec(rng, seed, force_object)
        if self._task_type == "rolling_ball":
            return self._build_rolling_ball_spec(rng, seed, force_object)
        if self._task_type == "shelf_slide":
            return self._build_shelf_slide_spec(rng, seed, force_object)
        if self._task_type == "door_swing":
            return self._build_door_swing_spec(rng, seed, force_object)
        if self._task_type == "thrown_object":
            return self._build_thrown_object_spec(rng, seed, force_object)
        if self._task_type == "pendulum_swing":
            return self._build_pendulum_swing_spec(rng, seed, force_object)
        if self._task_type == "bouncing_object":
            return self._build_bouncing_object_spec(rng, seed, force_object)
        if self._task_type == "ladder_slip":
            return self._build_ladder_slip_spec(rng, seed, force_object)
        if self._task_type == "chain_reaction":
            return self._build_chain_reaction_spec(rng, seed, force_object)
        return self._build_object_drop_spec(rng, seed, force_object)

    def _build_object_drop_spec(
        self,
        rng: np.random.Generator,
        seed: int,
        force_object: Optional[str] = None,
    ) -> SceneSpec:
        # ── 1. Sample object ──────────────────────────────────
        if force_object:
            candidates = [o for o in self._objects if o["name"] == force_object]
            obj_dict = candidates[0] if candidates else rng.choice(self._objects)
        else:
            obj_dict = self._objects[rng.integers(len(self._objects))]

        # ── 2. Sample room ────────────────────────────────────
        room_dict = self._rooms[rng.integers(len(self._rooms))]

        # ── 3. Table ──────────────────────────────────────────
        table = self._make_table(rng, room_dict)

        # ── 4. Drop spec ──────────────────────────────────────
        obj_spec  = self._make_object_spec(obj_dict)
        drop_spec = self._make_drop(rng, table, obj_spec)

        # ── 5. Room spec ──────────────────────────────────────
        room_spec = self._make_room(rng, room_dict)

        # ── 6. Cameras (with geometry-validated retry) ───────────
        obj_pos = self._drop_world_pos(table, drop_spec, obj_spec)
        cameras = self._make_cameras_with_retry(
            rng,
            lambda r: self._make_cameras(r, table, drop_spec, obj_spec),
            obj_pos,
        )

        # ── 7. Lighting ───────────────────────────────────────
        lighting = self._make_lighting(rng, room_dict)

        # ── 8. Ground truth ───────────────────────────────────
        if obj_spec.catch_safe:
            gt_action = "EXECUTE_CATCH"
        elif obj_spec.mass_hint == "heavy":
            gt_action = "BRACE_FOR_IMPACT"
        else:
            gt_action = "TRIGGER_DODGE"

        return SceneSpec(
            seed=seed,
            task_type="object_drop",
            adversarial=(obj_dict["category"] == "adversarial"),
            room=room_spec,
            table=table,
            object=obj_spec,
            drop=drop_spec,
            cameras=cameras,
            lighting=lighting,
            ground_truth_action=gt_action,
            safety_label=obj_spec.safety_label,
        )

    def _build_furniture_tip_spec(
        self,
        rng: np.random.Generator,
        seed: int,
        force_object: Optional[str] = None,
    ) -> SceneSpec:
        # ── 1. Sample furniture object ────────────────────────
        if force_object:
            candidates = [o for o in self._objects if o["name"] == force_object]
            obj_dict = candidates[0] if candidates else self._objects[rng.integers(len(self._objects))]
        else:
            obj_dict = self._objects[rng.integers(len(self._objects))]

        # ── 2. Sample room ────────────────────────────────────
        room_dict = self._rooms[rng.integers(len(self._rooms))]

        # ── 3. Object + tip spec ──────────────────────────────
        obj_spec = self._make_object_spec(obj_dict)
        tip_spec = self._make_tip(rng, room_dict, obj_spec)

        # ── 4. Room spec ──────────────────────────────────────
        room_spec = self._make_room(rng, room_dict)

        # ── 5. Cameras (with geometry-validated retry) ───────────
        tip_pos = self._tip_world_pos(tip_spec, obj_spec)
        cameras = self._make_cameras_with_retry(
            rng,
            lambda r: self._make_furniture_cameras(r, tip_spec, obj_spec),
            tip_pos,
        )

        # ── 6. Lighting ───────────────────────────────────────
        lighting = self._make_lighting(rng, room_dict)

        # ── 7. Ground truth ───────────────────────────────────
        # Furniture always tips toward the observer — no catching
        if obj_spec.mass_hint == "heavy":
            gt_action = "BRACE_FOR_IMPACT"
        else:
            gt_action = "TRIGGER_DODGE"

        return SceneSpec(
            seed=seed,
            task_type="furniture_tip",
            adversarial=(obj_dict["category"] == "adversarial"),
            room=room_spec,
            table=None,
            object=obj_spec,
            drop=None,
            tip=tip_spec,
            cameras=cameras,
            lighting=lighting,
            ground_truth_action=gt_action,
            safety_label=obj_spec.safety_label,
        )

    def _build_hanging_fall_spec(
        self,
        rng: np.random.Generator,
        seed: int,
        force_object: Optional[str] = None,
    ) -> SceneSpec:
        # ── 1. Sample object ──────────────────────────────────
        if force_object:
            candidates = [o for o in self._objects if o["name"] == force_object]
            obj_dict = candidates[0] if candidates else self._objects[rng.integers(len(self._objects))]
        else:
            obj_dict = self._objects[rng.integers(len(self._objects))]

        # ── 2. Sample room ────────────────────────────────────
        room_dict = self._rooms[rng.integers(len(self._rooms))]

        # ── 3. Object + hanging spec ──────────────────────────
        obj_spec     = self._make_object_spec(obj_dict)
        hanging_spec = self._make_hanging(rng, room_dict, obj_dict, obj_spec)

        # ── 4. Room spec ──────────────────────────────────────
        room_spec = self._make_room(rng, room_dict)

        # ── 5. Cameras (with geometry-validated retry) ───────────
        hang_pos = self._hanging_world_pos(room_dict, hanging_spec, obj_spec)
        cameras = self._make_cameras_with_retry(
            rng,
            lambda r: self._make_hanging_cameras(r, hanging_spec, obj_spec, room_dict),
            hang_pos,
        )

        # ── 6. Lighting ───────────────────────────────────────
        lighting = self._make_lighting(rng, room_dict)

        # ── 7. Ground truth ───────────────────────────────────
        if obj_spec.catch_safe:
            gt_action = "EXECUTE_CATCH"
        elif obj_spec.mass_hint == "heavy":
            gt_action = "BRACE_FOR_IMPACT"
        else:
            gt_action = "TRIGGER_DODGE"

        return SceneSpec(
            seed=seed,
            task_type="hanging_fall",
            adversarial=(obj_dict["category"] == "adversarial"),
            room=room_spec,
            table=None,
            object=obj_spec,
            drop=None,
            tip=None,
            hanging=hanging_spec,
            cameras=cameras,
            lighting=lighting,
            ground_truth_action=gt_action,
            safety_label=obj_spec.safety_label,
        )

    # ── Sub-builders ──────────────────────────────────────────

    def _make_object_spec(self, d: dict) -> ObjectSpec:
        return ObjectSpec(
            name=d["name"],
            display_name=d["display_name"],
            category=d["category"],
            morph=d["morph"],
            mesh_path=d.get("mesh_path"),
            bottom_z_offset=d.get("bottom_z_offset", 0.0),
            size=d.get("size", [0.05, 0.05, 0.05]),
            phys_half_x=d.get("phys_half_x", 0.0),
            phys_half_z=d.get("phys_half_z", 0.0),
            density=d["density"],
            friction=d["friction"],
            restitution=d["restitution"],
            color_rgb=d["color_rgb"],
            roughness=d["roughness"],
            ior=d.get("ior", 1.0),
            catch_safe=d["catch_safe"],
            safety_label=d["safety_label"],
            mass_hint=d["mass_hint"],
            description=d.get("description", ""),
        )

    def _make_table(self, rng: np.random.Generator, room: dict) -> TableSpec:
        # Slight position jitter within the room
        tx, ty = room.get("table_pos", [0.0, 0.0])
        tx += float(rng.uniform(-0.15, 0.15))
        ty += float(rng.uniform(-0.10, 0.10))
        # Colour variation: darken/lighten the base walnut
        base = [0.35, 0.22, 0.10]
        tint = float(rng.uniform(0.85, 1.15))
        color = [min(1.0, c * tint) for c in base]
        return TableSpec(
            pos_x=round(tx, 3),
            pos_y=round(ty, 3),
            color_rgb=color,
        )

    # ── Camera validation + retry ─────────────────────────────

    @staticmethod
    def _cameras_pass(cameras: list, obj_pos: tuple[float,float,float]) -> bool:
        """Quick check: every camera has the object within 85% of half-FOV."""
        import math
        ox, oy, oz = obj_pos
        for cam in cameras:
            px, py, pz = cam.pos
            lx, ly, lz = cam.lookat
            dl = (lx-px, ly-py, lz-pz); dll = math.sqrt(sum(v*v for v in dl))
            if dll < 1e-6:
                return False
            do_ = (ox-px, oy-py, oz-pz); dol = math.sqrt(sum(v*v for v in do_))
            if dol < 1e-6:
                continue
            cos_a = max(-1.0, min(1.0, sum(a*b/dll/dol for a, b in zip(dl, do_))))
            if math.degrees(math.acos(cos_a)) > cam.fov * 0.85:
                return False
        return True

    def _make_cameras_with_retry(
        self,
        rng: np.random.Generator,
        make_fn,                         # callable(rng) -> list[CameraSpec]
        obj_pos: tuple[float,float,float],
        max_tries: int = 6,
    ) -> list:
        """Call make_fn up to max_tries times; return first result that passes geometry."""
        for _ in range(max_tries):
            cameras = make_fn(rng)
            if self._cameras_pass(cameras, obj_pos):
                return cameras
        # Fallback: aim all cameras directly at object (guaranteed to pass)
        ox, oy, oz = obj_pos
        return cameras   # return last attempt; validate_geometry() will still flag it

    # ── Geometry helpers ──────────────────────────────────────

    @staticmethod
    def _obj_phys_half(obj: ObjectSpec) -> tuple[float, float, float]:
        """Return (half_x, half_y, half_z) physical half-extents for any morph.

        For mesh objects, phys_half_x/z must be set in the YAML;
        for primitives, derived from size[].
        """
        if obj.morph == "mesh":
            hx = obj.phys_half_x if obj.phys_half_x > 0 else 0.05
            hz = obj.phys_half_z if obj.phys_half_z > 0 else 0.05
            return hx, hx, hz          # assume hy ≈ hx for placement purposes
        elif obj.morph == "sphere":
            r = obj.size[0]
            return r, r, r
        elif obj.morph == "cylinder":
            return obj.size[0], obj.size[0], obj.size[2]
        else:  # box
            return obj.size[0], obj.size[1], obj.size[2]

    @staticmethod
    def _drop_world_pos(table: TableSpec, drop: "DropSpec", obj: ObjectSpec) -> tuple[float, float, float]:
        """World-space position of the object centre at t=0 for object_drop."""
        _, _, hz = Randomizer._obj_phys_half(obj)
        # For mesh: bottom_z_offset shifts the mesh so bottom=0, so centre is at table.height + hz
        if obj.morph == "mesh":
            obj_z = table.height - obj.bottom_z_offset + hz
        else:
            obj_z = table.height + hz
        return drop.start_x, drop.start_y, obj_z

    @staticmethod
    def _hanging_world_pos(room: dict, hanging: "HangingSpec", obj: ObjectSpec) -> tuple[float, float, float]:
        """World-space position of the object centre at t=0 for hanging_fall."""
        if hanging.attachment == "wall_north":
            _, hy, _ = Randomizer._obj_phys_half(obj)
            wall_y = room["depth"] / 2
            return hanging.pos_x, wall_y - hy, hanging.attach_z
        else:  # ceiling
            return hanging.pos_x, hanging.pos_y, hanging.attach_z

    @staticmethod
    def _tip_world_pos(tip: "TipSpec", obj: ObjectSpec) -> tuple[float, float, float]:
        """World-space position of the furniture centre at t=0 for furniture_tip."""
        _, _, hz = Randomizer._obj_phys_half(obj)
        return tip.start_x, tip.start_y, hz

    @staticmethod
    def _sliding_world_pos(ramp: "RampSpec", obj: ObjectSpec) -> tuple[float, float, float]:
        """World-space position of object at t=0 for sliding_object (80% up from low end).

        Ramp euler=(+angle_deg): +Y local is the HIGH end, -Y local is the LOW end.
        Surface normal = (0, -sin(a), cos(a)).
        Object centre = surface_contact + hz * normal.
        """
        _, _, hz = Randomizer._obj_phys_half(obj)
        a = math.radians(ramp.angle_deg)
        t = 0.025   # ramp board half-thickness
        d = ramp.length * 0.80  # distance from low end
        obj_x = ramp.pos_x
        obj_y = ramp.pos_y + d * math.cos(a) - hz * math.sin(a)
        obj_z = t + d * math.sin(a) + hz * math.cos(a)
        return obj_x, obj_y, obj_z

    @staticmethod
    def _stack_world_pos(table: "TableSpec", stack: "StackSpec", obj: ObjectSpec) -> tuple[float, float, float]:
        """World-space midpoint of the stack at t=0 for stack_collapse.

        Item i centre z = table.height + hz * (2*i + 1).
        Midpoint of n items = table.height + hz * n_items.
        """
        _, _, hz = Randomizer._obj_phys_half(obj)
        stack_mid_z = table.height + hz * stack.n_items
        return stack.start_x, stack.start_y, stack_mid_z

    @staticmethod
    def _roll_world_pos(table: "TableSpec", roll: "RollSpec", obj: ObjectSpec) -> tuple[float, float, float]:
        """World-space position of ball centre at t=0 for rolling_ball.

        Ball rests on the table top: z = table.height + radius.
        """
        radius = obj.size[0]   # sphere morph: size[0] = radius
        obj_z = table.height + radius
        return roll.start_x, roll.start_y, obj_z

    @staticmethod
    def _shelf_world_pos(room: dict, shelf: "ShelfSpec", obj: ObjectSpec) -> tuple[float, float, float]:
        """World-space position of object centre at t=0 for shelf_slide.

        Object starts at 70% of shelf depth from the north wall (near front edge).
        z = shelf.height + obj half-height.
        """
        _, _, hz = Randomizer._obj_phys_half(obj)
        wall_y = room["depth"] / 2
        obj_y = wall_y - shelf.depth * 0.70
        obj_z = shelf.height + hz
        return shelf.pos_x, obj_y, obj_z

    # ─────────────────────────────────────────────────────────

    def _make_drop(
        self,
        rng: np.random.Generator,
        table: TableSpec,
        obj: ObjectSpec,
    ) -> DropSpec:
        # Object radius in x (for overhang calculation)
        # Mesh morphs have size=[scale], not physical half-extents — use phys_half_x
        if obj.morph == "mesh" and obj.phys_half_x > 0:
            obj_rx = obj.phys_half_x
        else:
            obj_rx = obj.size[0]

        # Table edge in world x
        edge_x = table.pos_x + table.width / 2

        # Choose fall mode
        use_velocity = bool(rng.random() < 0.35)   # 35% velocity-driven, 65% gravity-tip

        if use_velocity:
            # Object starts fully on table but gets a push
            overhang = float(rng.uniform(0.05, 0.30))  # small overhang
            start_x  = edge_x - obj_rx + overhang * 2 * obj_rx
            vel_x    = float(rng.uniform(0.35, 0.90))
        else:
            # Centre of mass past edge → gravity tips it off
            overhang = float(rng.uniform(0.50, 0.80))
            start_x  = edge_x - obj_rx + overhang * 2 * obj_rx
            vel_x    = 0.0

        start_y  = table.pos_y + float(rng.uniform(-0.12, 0.12))
        euler_z  = float(rng.uniform(-30, 30))   # random in-plane rotation

        return DropSpec(
            start_x=round(start_x, 4),
            start_y=round(start_y, 4),
            euler_z=round(euler_z, 1),
            vel_x=round(vel_x, 3),
        )

    def _make_room(self, rng: np.random.Generator, d: dict) -> RoomSpec:
        # Slight colour variation on walls
        def _jitter(rgb):
            t = float(rng.uniform(0.95, 1.05))
            return [min(1.0, max(0.0, c * t)) for c in rgb]

        win = d.get("window", {})
        return RoomSpec(
            type=d["type"],
            display_name=d["display_name"],
            width=d["width"],
            depth=d["depth"],
            height=d["height"],
            wall_thickness=d.get("wall_thickness", 0.12),
            wall_color_rgb=_jitter(d["wall_color_rgb"]),
            ceiling_color_rgb=_jitter(d["ceiling_color_rgb"]),
            floor_color_rgb=d["floor_color_rgb"],
            floor_roughness=d.get("floor_roughness", 0.55),
            window_wall=win.get("wall", "north"),
            window_pos_x=win.get("pos_x", 0.0),
            window_pos_z=win.get("pos_z", 1.45),
            window_width=win.get("width", 1.20),
            window_height=win.get("height", 1.00),
            window_emissive=win.get("emissive", [6.0, 5.8, 5.2]),
        )

    def _make_cameras(
        self,
        rng: np.random.Generator,
        table: TableSpec,
        drop: DropSpec,
        obj: ObjectSpec,
    ) -> list[CameraSpec]:
        """Three cameras: observer, closeup, overhead — all slightly randomised."""
        ox, oy, oz = self._drop_world_pos(table, drop, obj)  # actual object start pos
        tx, ty = table.pos_x, table.pos_y
        # Midpoint between start position and landing: good lookat target
        mid_x = (ox + tx) * 0.5
        mid_z = oz * 0.4   # roughly halfway down

        def _j(v, s=0.08): return round(v + float(rng.uniform(-s, s)), 3)

        return [
            CameraSpec(
                name="observer",
                pos=[_j(tx - 0.6), _j(ty - 1.80, 0.15), _j(1.50, 0.10)],
                lookat=[_j(mid_x), _j(oy), _j(mid_z, 0.05)],
                fov=float(rng.uniform(50, 58)),
            ),
            CameraSpec(
                name="closeup",
                pos=[_j(ox + 0.65), _j(oy - 1.10, 0.12), _j(oz * 0.8, 0.10)],
                lookat=[_j(ox), _j(oy), _j(oz, 0.05)],
                fov=float(rng.uniform(48, 58)),
            ),
            CameraSpec(
                name="overhead",
                pos=[_j(ox - 0.05, 0.05), _j(oy + 0.20, 0.10), _j(2.05, 0.08)],
                lookat=[_j(ox, 0.03), _j(oy + 0.10, 0.05), _j(table.height, 0.02)],
                fov=float(rng.uniform(50, 60)),
            ),
        ]

    def _make_tip(
        self,
        rng: np.random.Generator,
        room: dict,
        obj: ObjectSpec,
    ) -> TipSpec:
        """Sample TipSpec for a furniture-tipping event."""
        # Furniture placed slightly off-centre for variety
        start_x = float(rng.uniform(-0.30, 0.30))
        # Distance from observer: furniture centre between 0.5 and 1.2 m on +Y side
        start_y = float(rng.uniform(0.50, 1.20))
        # Tilt past vertical: enough to guarantee tipping (depends on aspect ratio)
        # For a box half-extents [x, y, z]: critical tilt = arctan(x/z) degrees
        half_w = obj.size[0]
        half_h = obj.size[2]
        critical_deg = math.degrees(math.atan2(half_w, half_h))
        tilt = float(rng.uniform(critical_deg + 2.0, critical_deg + 12.0))
        # Small extra angular velocity to ensure reliable tipping
        ang_vel = float(rng.uniform(0.2, 0.6))
        # Slight random facing rotation (furniture not always perfectly square to camera)
        euler_z = float(rng.uniform(-15.0, 15.0))
        return TipSpec(
            start_x=round(start_x, 3),
            start_y=round(start_y, 3),
            tilt_angle_deg=round(tilt, 2),
            angular_vel=round(ang_vel, 3),
            euler_z=round(euler_z, 1),
        )

    def _make_furniture_cameras(
        self,
        rng: np.random.Generator,
        tip: TipSpec,
        obj: ObjectSpec,
    ) -> list[CameraSpec]:
        """Three cameras positioned for a tall furniture tipping event."""
        fx, fy, _ = self._tip_world_pos(tip, obj)
        _, _, hz = self._obj_phys_half(obj)
        full_h = hz * 2   # full height of furniture

        def _j(v, s=0.10): return round(v + float(rng.uniform(-s, s)), 3)

        # Observer: eye-level, 2–2.5 m in front of furniture (-Y side)
        obs_dist = float(rng.uniform(2.0, 2.5))
        obs_y    = fy - obs_dist
        return [
            CameraSpec(
                name="observer",
                pos=[_j(fx, 0.12), _j(obs_y, 0.15), _j(1.45, 0.10)],
                lookat=[_j(fx, 0.06), _j(fy, 0.06), _j(full_h * 0.5, 0.08)],
                fov=float(rng.uniform(52, 62)),
            ),
            CameraSpec(
                name="closeup",
                # Side angle: captures the top of the furniture and fall arc
                pos=[_j(fx + 1.60, 0.15), _j(fy - 0.60, 0.12), _j(full_h * 0.65, 0.10)],
                lookat=[_j(fx, 0.06), _j(fy - full_h * 0.4, 0.08), _j(full_h * 0.3, 0.08)],
                fov=float(rng.uniform(48, 58)),
            ),
            CameraSpec(
                name="overhead",
                pos=[_j(fx, 0.06), _j(fy - full_h * 0.3, 0.10), _j(full_h * 1.1, 0.08)],
                lookat=[_j(fx, 0.04), _j(fy - full_h * 0.5, 0.06), 0.0],
                fov=float(rng.uniform(55, 65)),
            ),
        ]

    def _make_hanging(
        self,
        rng: np.random.Generator,
        room: dict,
        obj_dict: dict,
        obj: ObjectSpec,
    ) -> HangingSpec:
        """Sample HangingSpec for a wall- or ceiling-mounted object."""
        attachment = obj_dict.get("attachment", "wall_north")

        # Lateral position: random within middle half of the room
        pos_x = float(rng.uniform(-room["width"] / 4, room["width"] / 4))

        if attachment == "ceiling":
            pos_y      = float(rng.uniform(-room["depth"] / 4, room["depth"] / 4))
            # Hang object centre below ceiling; allow some cable variation
            half_h     = obj.size[2] if len(obj.size) >= 3 else obj.size[0]
            cable_slack = float(rng.uniform(0.25, 0.70))
            attach_z   = room["height"] - half_h - cable_slack
            tilt_deg   = float(rng.uniform(0.0, 4.0))
            ang_vel    = float(rng.uniform(0.0, 0.15))
        else:  # wall_north
            pos_y    = 0.0
            attach_z = float(rng.uniform(1.40, 2.00))
            tilt_deg = float(rng.uniform(3.0, 8.0))   # enough to guarantee tipping
            ang_vel  = float(rng.uniform(0.20, 0.50))

        return HangingSpec(
            attachment=attachment,
            pos_x=round(pos_x, 3),
            pos_y=round(pos_y, 3),
            attach_z=round(attach_z, 3),
            tilt_deg=round(tilt_deg, 2),
            angular_vel=round(ang_vel, 3),
            euler_z=round(float(rng.uniform(-20.0, 20.0)), 1),
        )

    def _make_hanging_cameras(
        self,
        rng: np.random.Generator,
        hanging: HangingSpec,
        obj: ObjectSpec,
        room: dict,
    ) -> list[CameraSpec]:
        """Three cameras for a hanging-fall event.

        Uses _hanging_world_pos() so lookat always tracks the actual object start
        position, regardless of attachment type or room size.
        """
        ox, oy, oz = self._hanging_world_pos(room, hanging, obj)  # actual start pos

        def _j(v, s=0.10): return round(v + float(rng.uniform(-s, s)), 3)

        if hanging.attachment == "wall_north":
            # Observer: south side of room, looking toward north (object) at mid-fall height
            mid_y  = oy * 0.55           # between start y and room centre
            obs_y  = float(rng.uniform(-room["depth"] / 2 + 0.30, -room["depth"] / 4))
            return [
                CameraSpec(
                    name="observer",
                    pos=[_j(ox, 0.12), _j(obs_y, 0.15), _j(1.20, 0.15)],
                    lookat=[_j(ox), _j(mid_y, 0.10), _j(oz * 0.55, 0.08)],
                    fov=float(rng.uniform(55, 65)),
                ),
                CameraSpec(
                    name="closeup",
                    pos=[_j(ox + 0.80, 0.15), _j(oy - 1.20, 0.12), _j(oz * 0.85, 0.10)],
                    lookat=[_j(ox), _j(oy, 0.08), _j(oz * 0.50, 0.08)],
                    fov=float(rng.uniform(45, 55)),
                ),
                CameraSpec(
                    name="overhead",
                    pos=[_j(ox, 0.06), _j(oy * 0.3, 0.10), _j(room["height"] * 0.88, 0.08)],
                    lookat=[_j(ox), _j(oy, 0.08), _j(oz * 0.40, 0.08)],
                    fov=float(rng.uniform(60, 70)),
                ),
            ]
        else:  # ceiling
            obs_dist = float(rng.uniform(2.0, 2.8))
            mid_z = oz * 0.55   # midpoint between hang height and floor
            return [
                CameraSpec(
                    name="observer",
                    pos=[_j(ox + 0.25, 0.12), _j(oy - obs_dist, 0.15), _j(1.30, 0.12)],
                    lookat=[_j(ox), _j(oy), _j(mid_z, 0.10)],
                    fov=float(rng.uniform(52, 60)),
                ),
                CameraSpec(
                    name="closeup",
                    pos=[_j(ox + 1.20, 0.15), _j(oy - 0.80, 0.12), _j(oz * 0.75, 0.10)],
                    lookat=[_j(ox), _j(oy), _j(oz * 0.60, 0.08)],
                    fov=float(rng.uniform(45, 55)),
                ),
                CameraSpec(
                    name="overhead",
                    pos=[_j(ox, 0.05), _j(oy + 0.20, 0.10), _j(room["height"] * 0.92, 0.08)],
                    lookat=[_j(ox), _j(oy), _j(oz * 0.40, 0.08)],
                    fov=float(rng.uniform(60, 70)),
                ),
            ]

    def _build_sliding_object_spec(
        self,
        rng: np.random.Generator,
        seed: int,
        force_object: Optional[str] = None,
    ) -> SceneSpec:
        # ── 1. Sample object ──────────────────────────────────
        if force_object:
            candidates = [o for o in self._objects if o["name"] == force_object]
            obj_dict = candidates[0] if candidates else self._objects[rng.integers(len(self._objects))]
        else:
            obj_dict = self._objects[rng.integers(len(self._objects))]

        # ── 2. Sample room ────────────────────────────────────
        room_dict = self._rooms[rng.integers(len(self._rooms))]

        # ── 3. Object + ramp spec ─────────────────────────────
        obj_spec  = self._make_object_spec(obj_dict)
        ramp_spec = self._make_ramp(rng, room_dict)

        # ── 4. Room spec ──────────────────────────────────────
        room_spec = self._make_room(rng, room_dict)

        # ── 5. Cameras (with geometry-validated retry) ────────
        slide_pos = self._sliding_world_pos(ramp_spec, obj_spec)
        cameras = self._make_cameras_with_retry(
            rng,
            lambda r: self._make_sliding_cameras(r, ramp_spec, obj_spec),
            slide_pos,
        )

        # ── 6. Lighting ───────────────────────────────────────
        lighting = self._make_lighting(rng, room_dict)

        # ── 7. Ground truth ───────────────────────────────────
        if obj_spec.catch_safe:
            gt_action = "EXECUTE_CATCH"
        elif obj_spec.mass_hint == "heavy":
            gt_action = "BRACE_FOR_IMPACT"
        else:
            gt_action = "TRIGGER_DODGE"

        return SceneSpec(
            seed=seed,
            task_type="sliding_object",
            adversarial=(obj_dict["category"] == "adversarial"),
            room=room_spec,
            table=None,
            object=obj_spec,
            drop=None,
            tip=None,
            hanging=None,
            stack=None,
            ramp=ramp_spec,
            cameras=cameras,
            lighting=lighting,
            ground_truth_action=gt_action,
            safety_label=obj_spec.safety_label,
        )

    def _make_ramp(self, rng: np.random.Generator, room: dict) -> RampSpec:
        """Sample a RampSpec: random position in room, angle, and size."""
        pos_x = float(rng.uniform(-room["width"] * 0.20, room["width"] * 0.20))
        # Lower end of ramp: near centre of room (objects slide toward observer at -Y)
        pos_y = float(rng.uniform(-0.20, 0.30))
        length = float(rng.uniform(0.70, 1.10))
        width  = float(rng.uniform(0.28, 0.42))
        # Steep enough to ensure reliable sliding
        angle_deg = float(rng.uniform(20.0, 35.0))
        euler_z   = float(rng.uniform(-12.0, 12.0))
        return RampSpec(
            pos_x=round(pos_x, 3),
            pos_y=round(pos_y, 3),
            length=round(length, 3),
            width=round(width, 3),
            angle_deg=round(angle_deg, 2),
            euler_z=round(euler_z, 1),
        )

    def _make_sliding_cameras(
        self,
        rng: np.random.Generator,
        ramp: RampSpec,
        obj: ObjectSpec,
    ) -> list[CameraSpec]:
        """Three cameras for a sliding-object event."""
        ox, oy, oz = self._sliding_world_pos(ramp, obj)
        a = math.radians(ramp.angle_deg)
        # Lower-end position of the ramp (where objects launch off)
        launch_y = ramp.pos_y
        launch_z = 0.0
        # Midpoint between object start and launch point (good lookat for wide shots)
        mid_y = (oy + launch_y) * 0.5
        mid_z = oz * 0.5

        def _j(v, s=0.08): return round(v + float(rng.uniform(-s, s)), 3)

        return [
            CameraSpec(
                # Observer: front-on, eye level, watching the slide approach
                name="observer",
                pos=[_j(ramp.pos_x - 0.3, 0.15), _j(launch_y - 1.60, 0.15), _j(1.10, 0.12)],
                lookat=[_j(ramp.pos_x), _j(mid_y, 0.08), _j(mid_z, 0.06)],
                fov=float(rng.uniform(52, 62)),
            ),
            CameraSpec(
                # Closeup: side angle, captures object sliding and becoming airborne
                name="closeup",
                pos=[_j(ramp.pos_x + 1.00, 0.12), _j(mid_y - 0.60, 0.12), _j(oz * 0.85, 0.10)],
                lookat=[_j(ramp.pos_x), _j(mid_y, 0.06), _j(mid_z, 0.06)],
                fov=float(rng.uniform(46, 56)),
            ),
            CameraSpec(
                # Overhead: top-down view along the ramp axis
                name="overhead",
                pos=[_j(ramp.pos_x, 0.05), _j(mid_y + 0.10, 0.10), _j(oz + 1.40, 0.10)],
                lookat=[_j(ramp.pos_x, 0.03), _j(mid_y, 0.05), _j(0.15, 0.03)],
                fov=float(rng.uniform(55, 65)),
            ),
        ]

    def _build_stack_collapse_spec(
        self,
        rng: np.random.Generator,
        seed: int,
        force_object: Optional[str] = None,
    ) -> SceneSpec:
        # ── 1. Sample object ──────────────────────────────────
        if force_object:
            candidates = [o for o in self._objects if o["name"] == force_object]
            obj_dict = candidates[0] if candidates else self._objects[rng.integers(len(self._objects))]
        else:
            obj_dict = self._objects[rng.integers(len(self._objects))]

        # ── 2. Sample room ────────────────────────────────────
        room_dict = self._rooms[rng.integers(len(self._rooms))]

        # ── 3. Table + stack spec ─────────────────────────────
        table     = self._make_table(rng, room_dict)
        obj_spec  = self._make_object_spec(obj_dict)
        stack_spec = self._make_stack(rng, table, obj_spec)

        # ── 4. Room spec ──────────────────────────────────────
        room_spec = self._make_room(rng, room_dict)

        # ── 5. Cameras (with geometry-validated retry) ────────
        stack_pos = self._stack_world_pos(table, stack_spec, obj_spec)
        cameras = self._make_cameras_with_retry(
            rng,
            lambda r: self._make_stack_cameras(r, table, stack_spec, obj_spec),
            stack_pos,
        )

        # ── 6. Lighting ───────────────────────────────────────
        lighting = self._make_lighting(rng, room_dict)

        # ── 7. Ground truth ───────────────────────────────────
        # Stack always collapses toward observer — no catching reflex for a cascade
        if obj_spec.mass_hint == "heavy":
            gt_action = "BRACE_FOR_IMPACT"
        elif obj_spec.catch_safe:
            gt_action = "EXECUTE_CATCH"
        else:
            gt_action = "TRIGGER_DODGE"

        return SceneSpec(
            seed=seed,
            task_type="stack_collapse",
            adversarial=(obj_dict["category"] == "adversarial"),
            room=room_spec,
            table=table,
            object=obj_spec,
            drop=None,
            tip=None,
            hanging=None,
            stack=stack_spec,
            cameras=cameras,
            lighting=lighting,
            ground_truth_action=gt_action,
            safety_label=obj_spec.safety_label,
        )

    def _make_stack(
        self,
        rng: np.random.Generator,
        table: TableSpec,
        obj: ObjectSpec,
    ) -> StackSpec:
        """Sample a StackSpec: random position on table, lean, and item count."""
        # Stack position: within the middle half of the table
        start_x = table.pos_x + float(rng.uniform(-table.width * 0.25, table.width * 0.25))
        start_y = table.pos_y + float(rng.uniform(-table.depth * 0.20, table.depth * 0.20))

        n_items = int(rng.integers(2, 5))  # 2–4 items

        # Tilt past the critical angle to guarantee collapse
        _, _, hz = self._obj_phys_half(obj)
        hx, _, _ = self._obj_phys_half(obj)
        critical_deg = math.degrees(math.atan2(hx, hz * n_items)) if hz > 0 else 8.0
        tilt = float(rng.uniform(max(5.0, critical_deg + 2.0), max(12.0, critical_deg + 10.0)))

        return StackSpec(
            n_items=n_items,
            start_x=round(start_x, 3),
            start_y=round(start_y, 3),
            tilt_angle_deg=round(tilt, 2),
            angular_vel=round(float(rng.uniform(0.3, 0.7)), 3),
            euler_z=round(float(rng.uniform(-20.0, 20.0)), 1),
        )

    def _make_stack_cameras(
        self,
        rng: np.random.Generator,
        table: TableSpec,
        stack: StackSpec,
        obj: ObjectSpec,
    ) -> list[CameraSpec]:
        """Three cameras for a stack-collapse event."""
        ox, oy, oz = self._stack_world_pos(table, stack, obj)
        _, _, hz = self._obj_phys_half(obj)
        stack_top_z = table.height + hz * (2 * stack.n_items - 1)

        def _j(v, s=0.08): return round(v + float(rng.uniform(-s, s)), 3)

        return [
            CameraSpec(
                # Observer: front-on, slightly above table to capture full collapse arc
                name="observer",
                pos=[_j(ox - 0.3, 0.15), _j(oy - 1.80, 0.15), _j(1.20, 0.12)],
                lookat=[_j(ox), _j(oy), _j(oz, 0.05)],
                fov=float(rng.uniform(52, 62)),
            ),
            CameraSpec(
                # Closeup: 45° side angle, shows cascade and items scattering
                name="closeup",
                pos=[_j(ox + 0.90, 0.12), _j(oy - 1.00, 0.12), _j(stack_top_z * 0.90, 0.10)],
                lookat=[_j(ox), _j(oy), _j(oz, 0.05)],
                fov=float(rng.uniform(46, 56)),
            ),
            CameraSpec(
                # Overhead: top-down to show items scattering across table
                name="overhead",
                pos=[_j(ox, 0.05), _j(oy + 0.10, 0.10), _j(stack_top_z + 1.20, 0.10)],
                lookat=[_j(ox, 0.03), _j(oy, 0.05), _j(table.height, 0.02)],
                fov=float(rng.uniform(55, 65)),
            ),
        ]

    # ── Rolling ball ──────────────────────────────────────────

    def _build_rolling_ball_spec(
        self,
        rng: np.random.Generator,
        seed: int,
        force_object: Optional[str] = None,
    ) -> SceneSpec:
        # 1. Sample ball object
        if force_object:
            candidates = [o for o in self._objects if o["name"] == force_object]
            obj_dict = candidates[0] if candidates else self._objects[rng.integers(len(self._objects))]
        else:
            obj_dict = self._objects[rng.integers(len(self._objects))]

        # 2. Sample room
        room_dict = self._rooms[rng.integers(len(self._rooms))]

        # 3. Table + roll spec
        table    = self._make_table(rng, room_dict)
        obj_spec = self._make_object_spec(obj_dict)
        roll_spec = self._make_roll(rng, table, obj_spec)

        # 4. Room spec
        room_spec = self._make_room(rng, room_dict)

        # 5. Cameras with geometry-validated retry
        roll_pos = self._roll_world_pos(table, roll_spec, obj_spec)
        cameras = self._make_cameras_with_retry(
            rng,
            lambda r: self._make_rolling_cameras(r, table, roll_spec, obj_spec),
            roll_pos,
        )

        # 6. Lighting
        lighting = self._make_lighting(rng, room_dict)

        # 7. Ground truth
        if obj_spec.catch_safe:
            gt_action = "EXECUTE_CATCH"
        elif obj_spec.mass_hint == "heavy":
            gt_action = "BRACE_FOR_IMPACT"
        else:
            gt_action = "TRIGGER_DODGE"

        return SceneSpec(
            seed=seed,
            task_type="rolling_ball",
            adversarial=(obj_dict["category"] == "adversarial"),
            room=room_spec,
            table=table,
            object=obj_spec,
            roll=roll_spec,
            cameras=cameras,
            lighting=lighting,
            ground_truth_action=gt_action,
            safety_label=obj_spec.safety_label,
        )

    def _make_roll(
        self,
        rng: np.random.Generator,
        table: TableSpec,
        obj: ObjectSpec,
    ) -> RollSpec:
        """Sample a RollSpec: ball placed on table, given velocity to roll off the edge."""
        radius = obj.size[0]
        edge_x = table.pos_x + table.width / 2

        # Ball starts 25–55 cm from the edge (enough roll to build realism)
        dist_from_edge = float(rng.uniform(0.25, 0.55))
        start_x = edge_x - dist_from_edge
        start_y = table.pos_y + float(rng.uniform(-0.12, 0.12))

        # Initial velocity toward table edge
        vel_x = float(rng.uniform(0.60, 1.20))
        vel_y = float(rng.uniform(-0.10, 0.10))

        # Rolling without slipping: wy = vel_x / radius
        angular_vel_y = vel_x / radius

        return RollSpec(
            start_x=round(start_x, 4),
            start_y=round(start_y, 4),
            vel_x=round(vel_x, 3),
            vel_y=round(vel_y, 3),
            angular_vel_y=round(angular_vel_y, 3),
        )

    def _make_rolling_cameras(
        self,
        rng: np.random.Generator,
        table: TableSpec,
        roll: RollSpec,
        obj: ObjectSpec,
    ) -> list[CameraSpec]:
        """Three cameras for a rolling-ball event.

        Observer watches from in front of the table to see the ball approach and fall.
        Closeup is near the table edge. Overhead shows the roll path across the table.
        """
        ox, oy, oz = self._roll_world_pos(table, roll, obj)
        edge_x = table.pos_x + table.width / 2
        mid_x  = (ox + edge_x) * 0.5
        tx, ty = table.pos_x, table.pos_y

        def _j(v, s=0.08): return round(v + float(rng.uniform(-s, s)), 3)

        return [
            CameraSpec(
                name="observer",
                pos=[_j(tx - 0.5, 0.12), _j(ty - 1.80, 0.15), _j(1.40, 0.10)],
                lookat=[_j(edge_x, 0.05), _j(oy, 0.05), _j(table.height, 0.04)],
                fov=float(rng.uniform(50, 60)),
            ),
            CameraSpec(
                name="closeup",
                pos=[_j(edge_x + 0.55, 0.10), _j(oy - 1.00, 0.12), _j(oz + 0.35, 0.10)],
                lookat=[_j(edge_x, 0.05), _j(oy, 0.05), _j(table.height, 0.03)],
                fov=float(rng.uniform(46, 56)),
            ),
            CameraSpec(
                name="overhead",
                pos=[_j(mid_x, 0.06), _j(oy + 0.10, 0.08), _j(1.80, 0.10)],
                lookat=[_j(mid_x, 0.04), _j(oy, 0.04), _j(table.height, 0.02)],
                fov=float(rng.uniform(55, 65)),
            ),
        ]

    # ── Shelf slide ───────────────────────────────────────────

    def _build_shelf_slide_spec(
        self,
        rng: np.random.Generator,
        seed: int,
        force_object: Optional[str] = None,
    ) -> SceneSpec:
        # 1. Sample object
        if force_object:
            candidates = [o for o in self._objects if o["name"] == force_object]
            obj_dict = candidates[0] if candidates else self._objects[rng.integers(len(self._objects))]
        else:
            obj_dict = self._objects[rng.integers(len(self._objects))]

        # 2. Sample room
        room_dict = self._rooms[rng.integers(len(self._rooms))]

        # 3. Object + shelf spec
        obj_spec   = self._make_object_spec(obj_dict)
        shelf_spec = self._make_shelf(rng, room_dict)

        # 4. Room spec
        room_spec = self._make_room(rng, room_dict)

        # 5. Cameras with geometry-validated retry
        shelf_pos = self._shelf_world_pos(room_dict, shelf_spec, obj_spec)
        cameras = self._make_cameras_with_retry(
            rng,
            lambda r: self._make_shelf_cameras(r, room_dict, shelf_spec, obj_spec),
            shelf_pos,
        )

        # 6. Lighting
        lighting = self._make_lighting(rng, room_dict)

        # 7. Ground truth
        if obj_spec.catch_safe:
            gt_action = "EXECUTE_CATCH"
        elif obj_spec.mass_hint == "heavy":
            gt_action = "BRACE_FOR_IMPACT"
        else:
            gt_action = "TRIGGER_DODGE"

        return SceneSpec(
            seed=seed,
            task_type="shelf_slide",
            adversarial=(obj_dict["category"] == "adversarial"),
            room=room_spec,
            object=obj_spec,
            shelf=shelf_spec,
            cameras=cameras,
            lighting=lighting,
            ground_truth_action=gt_action,
            safety_label=obj_spec.safety_label,
        )

    def _make_shelf(self, rng: np.random.Generator, room: dict) -> ShelfSpec:
        """Sample a ShelfSpec: random position on north wall at varied height."""
        pos_x     = float(rng.uniform(-room["width"] * 0.25, room["width"] * 0.25))
        height    = float(rng.uniform(1.40, 1.90))
        depth     = float(rng.uniform(0.14, 0.22))
        width     = float(rng.uniform(0.45, 0.80))
        vel_y     = float(rng.uniform(-0.55, -0.25))
        return ShelfSpec(
            height=round(height, 3),
            pos_x=round(pos_x, 3),
            depth=round(depth, 3),
            width=round(width, 3),
            vel_y=round(vel_y, 3),
        )

    def _make_shelf_cameras(
        self,
        rng: np.random.Generator,
        room: dict,
        shelf: ShelfSpec,
        obj: ObjectSpec,
    ) -> list[CameraSpec]:
        """Three cameras for a shelf-slide event.

        Observer is ~2m in front of the shelf watching the fall arc.
        Closeup is at a side angle, aimed at the object's starting height
        (not the floor mid-point) so the angle check passes reliably.
        Overhead looks down at the fall path.
        """
        ox, oy, oz = self._shelf_world_pos(room, shelf, obj)
        fall_mid_z = oz * 0.50   # midpoint of fall arc (for observer lookat)

        def _j(v, s=0.10): return round(v + float(rng.uniform(-s, s)), 3)

        obs_dist = float(rng.uniform(1.80, 2.40))
        obs_y    = oy - obs_dist

        return [
            CameraSpec(
                name="observer",
                pos=[_j(ox, 0.12), _j(obs_y, 0.15), _j(1.20, 0.12)],
                lookat=[_j(ox), _j(oy, 0.10), _j(fall_mid_z, 0.08)],
                fov=float(rng.uniform(55, 65)),
            ),
            CameraSpec(
                # Side angle: camera is 0.9m to +X and 1m in front of shelf object.
                # Lookat Z = oz * 0.80 keeps the lookat close to the object's actual
                # start height, ensuring the angle check passes.
                name="closeup",
                pos=[_j(ox + 0.90, 0.15), _j(oy - 1.00, 0.12), _j(oz * 0.85, 0.10)],
                lookat=[_j(ox), _j(oy, 0.08), _j(oz * 0.80, 0.08)],
                fov=float(rng.uniform(50, 62)),
            ),
            CameraSpec(
                name="overhead",
                pos=[_j(ox, 0.05), _j(oy + 0.20, 0.10), _j(oz + 0.80, 0.08)],
                lookat=[_j(ox), _j(oy - 0.30, 0.08), _j(0.30, 0.05)],
                fov=float(rng.uniform(60, 70)),
            ),
        ]

    # ── Door swing ────────────────────────────────────────────

    def _build_door_swing_spec(
        self,
        rng: np.random.Generator,
        seed: int,
        force_object: Optional[str] = None,
    ) -> SceneSpec:
        # 1. Sample door object (defines material/visual)
        if force_object:
            candidates = [o for o in self._objects if o["name"] == force_object]
            obj_dict = candidates[0] if candidates else self._objects[rng.integers(len(self._objects))]
        else:
            obj_dict = self._objects[rng.integers(len(self._objects))]

        # 2. Sample room
        room_dict = self._rooms[rng.integers(len(self._rooms))]

        # 3. Build sub-specs
        obj_spec  = self._make_object_spec(obj_dict)
        door_spec = self._make_door(rng, room_dict, obj_spec)
        room_spec = self._make_room(rng, room_dict)

        # 4. Cameras with geometry-validated retry
        door_pos = self._door_world_pos(door_spec)
        cameras  = self._make_cameras_with_retry(
            rng,
            lambda r: self._make_door_cameras(r, door_spec, room_dict),
            door_pos,
        )

        # 5. Lighting
        lighting = self._make_lighting(rng, room_dict)

        # 6. Ground truth
        if obj_spec.category == "adversarial":
            gt_action = "TRIGGER_DODGE"   # foam_door: looks heavy, actually weightless
        elif obj_spec.mass_hint == "heavy":
            gt_action = "BRACE_FOR_IMPACT"
        else:
            gt_action = "TRIGGER_DODGE"

        return SceneSpec(
            seed=seed,
            task_type="door_swing",
            adversarial=(obj_dict["category"] == "adversarial"),
            room=room_spec,
            object=obj_spec,
            door=door_spec,
            cameras=cameras,
            lighting=lighting,
            ground_truth_action=gt_action,
            safety_label=obj_spec.safety_label,
        )

    def _make_door(
        self,
        rng: np.random.Generator,
        room: dict,
        obj: ObjectSpec,
    ) -> DoorSpec:
        """Sample a DoorSpec with the hinge on the south wall inner surface.

        The door starts open (initial_angle ≈ 85°, panel pointing into room)
        and swings closed (angular_vel < 0 → angle → 0°, panel flush with wall).
        The event detected is angle < 20° (door nearly closed, threatening doorway).

        Constraint: hinge_x placed so the door panel stays clear of east/west walls.
        """
        thickness = float(obj.size[1]) if len(obj.size) >= 2 else 0.045
        width  = float(rng.uniform(0.80, 1.05))
        height = float(rng.uniform(1.90, 2.20))

        # Hinge on south wall inner surface
        wall_thickness = room.get("wall_thickness", 0.1)
        door_y = -(room["depth"] / 2) + wall_thickness + thickness / 2

        # Place hinge so door panel clears east/west walls; allow slight off-centre
        half_room_x = room["width"] / 2
        clearance = 0.25
        hinge_x_max = half_room_x - width - clearance
        hinge_x = float(rng.uniform(-hinge_x_max, hinge_x_max))

        initial_angle_deg = float(rng.uniform(80.0, 88.0))
        angular_vel = float(rng.uniform(-2.0, -1.0))

        return DoorSpec(
            width=round(width, 3),
            height=round(height, 3),
            thickness=round(thickness, 4),
            hinge_x=round(hinge_x, 3),
            door_y=round(door_y, 4),
            initial_angle_deg=round(initial_angle_deg, 1),
            angular_vel=round(angular_vel, 3),
        )

    @staticmethod
    def _door_world_pos(door: DoorSpec) -> tuple[float, float, float]:
        """World-space COM of door panel at t=0 (before velocity is applied)."""
        angle = math.radians(door.initial_angle_deg)
        cx = door.hinge_x + (door.width / 2) * math.cos(angle)
        cy = door.door_y + (door.width / 2) * math.sin(angle)
        cz = 0.01 + door.height / 2   # 0.01 m floor gap
        return cx, cy, cz

    def _make_door_cameras(
        self,
        rng: np.random.Generator,
        door: DoorSpec,
        room_dict: Optional[dict] = None,
    ) -> list[CameraSpec]:
        """Three cameras for a door-swing event.

        Observer: south of the door watching it swing toward them.
        Closeup:  east side, angled toward the door panel.
        Overhead: looking straight down at the sweep arc.
        """
        ox, oy, oz = self._door_world_pos(door)
        hx, hy = door.hinge_x, door.door_y

        # Door is on south wall at hy ≈ -depth/2. Cameras sit INSIDE the room
        # (positive y side) looking toward the south wall.
        room_depth  = room_dict.get("depth",  5.0) if room_dict else 5.0
        room_height = room_dict.get("height", 2.8) if room_dict else 2.8
        max_cam_z   = room_height - 0.25   # stay 25 cm below ceiling

        def _j(v, s=0.08): return round(v + float(rng.uniform(-s, s)), 3)
        def _clamp_z(z): return min(max_cam_z, z)

        # Observer: inside room, 1.5–2.0 m north of the hinge, at eye height
        obs_dist = float(rng.uniform(1.50, 2.00))
        # Closeup: east of door panel (panel extends +x from hinge), angled back
        closeup_x = hx + door.width * 0.70
        # Overhead: above the doorway, looking straight down at the sweep arc
        return [
            CameraSpec(
                name="observer",
                pos=[_j(hx, 0.12), _j(hy + obs_dist, 0.12), _j(1.20, 0.10)],
                lookat=[_j(ox, 0.08), _j(hy + 0.20, 0.08), _j(oz * 0.55, 0.06)],
                fov=float(rng.uniform(52, 64)),
            ),
            CameraSpec(
                name="closeup",
                pos=[_j(closeup_x, 0.12), _j(hy + 0.80, 0.12), _j(oz * 0.85, 0.10)],
                lookat=[_j(ox, 0.06), _j(hy + door.width * 0.40, 0.06), _j(oz, 0.06)],
                fov=float(rng.uniform(48, 58)),
            ),
            CameraSpec(
                name="overhead",
                pos=[_j(hx + door.width * 0.30, 0.08),
                     _j(hy + door.width * 0.50, 0.08),
                     _clamp_z(_j(2.30, 0.10))],
                lookat=[_j(ox, 0.06), _j(hy + 0.20, 0.06), _j(0.50, 0.05)],
                fov=float(rng.uniform(60, 72)),
            ),
        ]

    # ── Thrown object ─────────────────────────────────────────────────────────

    def _build_thrown_object_spec(
        self,
        rng: np.random.Generator,
        seed: int,
        force_object: Optional[str] = None,
    ) -> SceneSpec:
        if force_object:
            candidates = [o for o in self._objects if o["name"] == force_object]
            obj_dict = candidates[0] if candidates else self._objects[rng.integers(len(self._objects))]
        else:
            obj_dict = self._objects[rng.integers(len(self._objects))]

        room_dict = self._rooms[rng.integers(len(self._rooms))]
        obj_spec  = self._make_object_spec(obj_dict)
        room_spec = self._make_room(rng, room_dict)
        thrown    = self._make_thrown(rng, room_dict)
        lighting  = self._make_lighting(rng, room_dict)

        obj_pos  = self._thrown_world_pos(thrown)
        cameras  = self._make_cameras_with_retry(
            rng,
            lambda r: self._make_thrown_cameras(r, thrown, room_dict),
            obj_pos,
        )

        gt_action = "EXECUTE_CATCH" if obj_spec.catch_safe else "TRIGGER_DODGE"

        return SceneSpec(
            seed=seed,
            task_type="thrown_object",
            adversarial=(obj_dict["category"] == "adversarial"),
            room=room_spec,
            object=obj_spec,
            thrown=thrown,
            cameras=cameras,
            lighting=lighting,
            ground_truth_action=gt_action,
            safety_label=obj_spec.safety_label,
        )

    def _make_thrown(
        self,
        rng: np.random.Generator,
        room: dict,
    ) -> ThrownSpec:
        """Sample a ThrownSpec.

        Launch from the north half of the room at throwing height.
        vel_y is negative (toward south/observer).  Guaranteed that
        the object crosses y=0 within 3 s.
        """
        room_depth = room["depth"]
        room_width = room["width"]

        # Spawn in north half, clear of north wall
        launch_y = float(rng.uniform(0.40, min(1.60, room_depth / 2 - 0.40)))
        launch_x = float(rng.uniform(-room_width * 0.25, room_width * 0.25))
        launch_z = float(rng.uniform(1.20, 1.80))

        # Velocity toward observer; |vel_y| ∈ [2.5, 5.0] guarantees fast crossing
        vel_y = float(rng.uniform(-5.0, -2.5))
        vel_x = float(rng.uniform(-0.60, 0.60))
        vel_z = float(rng.uniform(-0.30, 0.50))  # slight upward arc or flat

        return ThrownSpec(
            launch_x=round(launch_x, 3),
            launch_y=round(launch_y, 3),
            launch_z=round(launch_z, 3),
            vel_x=round(vel_x, 3),
            vel_y=round(vel_y, 3),
            vel_z=round(vel_z, 3),
        )

    @staticmethod
    def _thrown_world_pos(thrown: ThrownSpec) -> tuple[float, float, float]:
        """World-space position of the thrown object at t=0."""
        return thrown.launch_x, thrown.launch_y, thrown.launch_z

    def _make_thrown_cameras(
        self,
        rng: np.random.Generator,
        thrown: ThrownSpec,
        room_dict: Optional[dict] = None,
    ) -> list[CameraSpec]:
        """Three cameras for a thrown-object event.

        Observer: south of the trajectory, watching the object fly toward them.
        Closeup:  east side at mid-trajectory height.
        Overhead: top-down view of the full arc.
        """
        lx, ly, lz = thrown.launch_x, thrown.launch_y, thrown.launch_z

        room_depth  = room_dict.get("depth",  5.0) if room_dict else 5.0
        room_height = room_dict.get("height", 2.8) if room_dict else 2.8
        south_wall_y = -room_depth / 2
        max_cam_z    = room_height - 0.25

        def _j(v, s=0.08): return round(v + float(rng.uniform(-s, s)), 3)
        def _clamp_y(y): return max(south_wall_y + 0.20, y)
        def _clamp_z(z): return min(max_cam_z, z)

        # Observer is south of centre, looking north toward launch point
        obs_y = _clamp_y(-room_depth * 0.30)
        # Closeup is east of the mid-trajectory point, looking at the arc
        mid_y = ly / 2   # approximate mid-flight y
        mid_z = lz - 0.30  # object will have dropped slightly by midpoint
        return [
            CameraSpec(
                name="observer",
                pos=[_j(lx, 0.15), _j(obs_y, 0.12), _j(lz * 0.75, 0.10)],
                lookat=[_j(lx, 0.08), _j(ly * 0.60, 0.08), _j(lz * 0.55, 0.06)],
                fov=float(rng.uniform(52, 62)),
            ),
            CameraSpec(
                name="closeup",
                pos=[_j(lx + 1.40, 0.12), _j(mid_y, 0.12), _j(mid_z, 0.10)],
                lookat=[_j(lx, 0.06), _j(mid_y, 0.06), _j(mid_z, 0.06)],
                fov=float(rng.uniform(46, 56)),
            ),
            CameraSpec(
                name="overhead",
                pos=[_j(lx, 0.10), _j(ly * 0.40, 0.10), _clamp_z(_j(2.50, 0.10))],
                lookat=[_j(lx, 0.06), _j(ly * 0.80, 0.08), _j(lz * 0.70, 0.06)],
                fov=float(rng.uniform(62, 74)),
            ),
        ]

    # ── Pendulum swing ────────────────────────────────────────────────────────

    def _build_pendulum_swing_spec(
        self,
        rng: np.random.Generator,
        seed: int,
        force_object: Optional[str] = None,
    ) -> SceneSpec:
        if force_object:
            candidates = [o for o in self._objects if o["name"] == force_object]
            obj_dict = candidates[0] if candidates else self._objects[rng.integers(len(self._objects))]
        else:
            obj_dict = self._objects[rng.integers(len(self._objects))]

        room_dict = self._rooms[rng.integers(len(self._rooms))]
        obj_spec  = self._make_object_spec(obj_dict)
        room_spec = self._make_room(rng, room_dict)
        pendulum  = self._make_pendulum(rng, room_dict, obj_spec)
        lighting  = self._make_lighting(rng, room_dict)

        obj_pos  = self._pendulum_world_pos(pendulum)
        cameras  = self._make_cameras_with_retry(
            rng,
            lambda r: self._make_pendulum_cameras(r, pendulum, room_dict),
            obj_pos,
        )

        gt_action = "BRACE_FOR_IMPACT" if obj_spec.mass_hint == "heavy" else "TRIGGER_DODGE"

        return SceneSpec(
            seed=seed,
            task_type="pendulum_swing",
            adversarial=(obj_dict["category"] == "adversarial"),
            room=room_spec,
            object=obj_spec,
            pendulum=pendulum,
            cameras=cameras,
            lighting=lighting,
            ground_truth_action=gt_action,
            safety_label=obj_spec.safety_label,
        )

    def _make_pendulum(
        self,
        rng: np.random.Generator,
        room: dict,
        obj_spec: ObjectSpec,
    ) -> PendulumSpec:
        """Sample a PendulumSpec.

        Pivot is placed just below the ceiling, slightly north of centre.
        Length is chosen so the bob clears the floor at vertical.
        Initial angle 30–55° gives natural swing TTF of 0.35–0.80 s.
        """
        room_height = room["height"]
        room_depth  = room["depth"]
        room_width  = room["width"]

        pivot_x = float(rng.uniform(-room_width * 0.20, room_width * 0.20))
        pivot_y = float(rng.uniform(0.0, min(0.50, room_depth / 2 - 0.50)))
        pivot_z = room_height - 0.06

        # Bob half-height for floor clearance
        o = obj_spec
        if o.morph == "sphere":
            bob_half_z = o.size[0]
        elif o.morph == "box" and len(o.size) >= 3:
            bob_half_z = o.size[2]
        else:
            bob_half_z = 0.15

        # Length: bob bottom (pivot_z - length - bob_half_z) must clear floor by 0.10 m
        max_length = pivot_z - bob_half_z - 0.10
        length = float(rng.uniform(0.80, min(1.60, max_length)))

        initial_angle_deg = float(rng.uniform(30.0, 55.0))

        return PendulumSpec(
            pivot_x=round(pivot_x, 3),
            pivot_y=round(pivot_y, 3),
            pivot_z=round(pivot_z, 3),
            length=round(length, 3),
            bob_half_z=round(bob_half_z, 3),
            initial_angle_deg=round(initial_angle_deg, 1),
            angular_vel=0.0,
        )

    @staticmethod
    def _pendulum_world_pos(p: PendulumSpec) -> tuple[float, float, float]:
        """World-space COM of pendulum bob at t=0."""
        angle = math.radians(p.initial_angle_deg)
        bx = p.pivot_x
        by = p.pivot_y + p.length * math.sin(angle)
        bz = p.pivot_z - p.length * math.cos(angle)
        return bx, by, bz

    def _make_pendulum_cameras(
        self,
        rng: np.random.Generator,
        p: PendulumSpec,
        room_dict: Optional[dict] = None,
    ) -> list[CameraSpec]:
        """Three cameras for a pendulum swing.

        Observer: south of pivot at eye height, watching the bob swing toward them.
        Closeup:  east side at bob height showing the arc.
        Overhead: above pivot looking down at the swing path.
        """
        bx, by, bz = self._pendulum_world_pos(p)
        room_depth  = room_dict.get("depth",  5.0) if room_dict else 5.0
        room_height = room_dict.get("height", 2.8) if room_dict else 2.8
        max_cam_z   = room_height - 0.25

        # Bob at vertical (event position): (pivot_x, pivot_y, pivot_z - length)
        bob_vert_y = p.pivot_y
        bob_vert_z = p.pivot_z - p.length

        def _j(v, s=0.08): return round(v + float(rng.uniform(-s, s)), 3)
        def _clamp_z(z): return min(max_cam_z, z)

        obs_y = max(-room_depth / 2 + 0.20, -room_depth * 0.30)
        return [
            CameraSpec(
                name="observer",
                pos=[_j(p.pivot_x, 0.12), _j(obs_y, 0.12), _j(bob_vert_z * 0.75 + 0.30, 0.10)],
                lookat=[_j(p.pivot_x, 0.08), _j(bob_vert_y, 0.08), _j(bob_vert_z, 0.06)],
                fov=float(rng.uniform(52, 62)),
            ),
            CameraSpec(
                name="closeup",
                pos=[_j(p.pivot_x + 1.60, 0.12), _j(by * 0.40, 0.12), _j(bz * 0.85, 0.10)],
                lookat=[_j(p.pivot_x, 0.06), _j(by * 0.80, 0.06), _j(bz, 0.06)],
                fov=float(rng.uniform(52, 62)),
            ),
            CameraSpec(
                name="overhead",
                pos=[_j(p.pivot_x, 0.08), _j(p.pivot_y - p.length * 0.40, 0.10),
                     _clamp_z(_j(p.pivot_z + 0.15, 0.08))],
                lookat=[_j(p.pivot_x, 0.06), _j(by * 0.60, 0.08), _j(bob_vert_z, 0.06)],
                fov=float(rng.uniform(60, 72)),
            ),
        ]

    # ── Bouncing object ───────────────────────────────────────────────────────

    def _build_bouncing_object_spec(
        self,
        rng: np.random.Generator,
        seed: int,
        force_object: Optional[str] = None,
    ) -> SceneSpec:
        if force_object:
            candidates = [o for o in self._objects if o["name"] == force_object]
            obj_dict = candidates[0] if candidates else self._objects[rng.integers(len(self._objects))]
        else:
            obj_dict = self._objects[rng.integers(len(self._objects))]

        room_dict = self._rooms[rng.integers(len(self._rooms))]
        obj_spec  = self._make_object_spec(obj_dict)
        room_spec = self._make_room(rng, room_dict)
        bounce    = self._make_bounce(rng, room_dict, obj_spec)
        lighting  = self._make_lighting(rng, room_dict)

        obj_pos  = self._bounce_world_pos(bounce)
        cameras  = self._make_cameras_with_retry(
            rng,
            lambda r: self._make_bounce_cameras(r, bounce, room_dict),
            obj_pos,
        )

        gt_action = "EXECUTE_CATCH" if obj_spec.catch_safe else "TRIGGER_DODGE"

        return SceneSpec(
            seed=seed,
            task_type="bouncing_object",
            adversarial=(obj_dict["category"] == "adversarial"),
            room=room_spec,
            object=obj_spec,
            bounce=bounce,
            cameras=cameras,
            lighting=lighting,
            ground_truth_action=gt_action,
            safety_label=obj_spec.safety_label,
        )

    def _make_bounce(
        self,
        rng: np.random.Generator,
        room: dict,
        obj_spec: ObjectSpec,
    ) -> BounceSpec:
        """Sample a BounceSpec.

        Ball spawns just above the floor at the first-bounce location with
        post-bounce upward + southward velocity.  This avoids relying on
        Genesis floor restitution, which is unreliable.

        Physics: simulating a ball dropped from drop_height and bouncing with
        obj_spec.restitution:
          vel_z = restitution × sqrt(2 × g × drop_height)
          vel_y = lateral speed before/after bounce (unchanged)
        """
        room_depth = room["depth"]
        room_width = room["width"]
        g = 9.81

        # Bounce point in northern half, clear of walls
        start_y = float(rng.uniform(0.30, min(1.20, room_depth / 2 - 0.40)))
        start_x = float(rng.uniform(-room_width * 0.22, room_width * 0.22))
        # Ball spawns at exactly its radius above the floor
        radius   = obj_spec.size[0]   # sphere: size[0] = radius
        start_z  = round(radius, 4)

        # Simulate drop from 0.60–1.50 m with the object's actual restitution.
        # Clamp to 2.5 m/s: ensures the ball is still airborne when the early-exit
        # fires (floor_hit_step+120 ≈ step 123, and arc lasts ~120 steps at 2.5 m/s),
        # avoiding the expensive floor-rolling phase for low-restitution balls.
        drop_h   = float(rng.uniform(0.60, 1.50))
        vel_z    = max(2.5, float(obj_spec.restitution * math.sqrt(2 * g * drop_h)))

        # Lateral speed: enough to cross y=0 within 3 s arc time
        vel_y    = float(rng.uniform(-2.5, -0.8))
        vel_x    = float(rng.uniform(-0.30, 0.30))

        return BounceSpec(
            start_x=round(start_x, 3),
            start_y=round(start_y, 3),
            start_z=start_z,
            vel_x=round(vel_x, 3),
            vel_y=round(vel_y, 3),
            vel_z=round(vel_z, 3),
        )

    @staticmethod
    def _bounce_world_pos(bounce: BounceSpec) -> tuple[float, float, float]:
        """World-space position of the bouncing ball at t=0."""
        return bounce.start_x, bounce.start_y, bounce.start_z

    def _make_bounce_cameras(
        self,
        rng: np.random.Generator,
        bounce: BounceSpec,
        room_dict: Optional[dict] = None,
    ) -> list[CameraSpec]:
        """Three cameras for a bouncing-object event.

        Observer: south of room at eye height, watching ball fall, bounce, and
                  fly toward them.
        Closeup:  east-side view showing the floor impact.
        Overhead: top-down view of the full fall + bounce path.
        """
        bx, by, bz = bounce.start_x, bounce.start_y, bounce.start_z

        room_depth  = room_dict.get("depth",  5.0) if room_dict else 5.0
        room_height = room_dict.get("height", 2.8) if room_dict else 2.8
        south_wall_y = -room_depth / 2
        max_cam_z    = room_height - 0.25

        def _j(v, s=0.08): return round(v + float(rng.uniform(-s, s)), 3)
        def _clamp_y(y): return max(south_wall_y + 0.20, y)
        def _clamp_z(z): return min(max_cam_z, z)

        # Ball spawns at floor level (bz ≈ radius) and flies upward then south.
        # Peak height: vel_z² / (2g), arc landing y: start_y + vel_y * (2*vel_z/g)
        import math as _math
        g          = 9.81
        peak_z     = bounce.vel_z ** 2 / (2 * g)           # max height of arc
        arc_time   = 2 * bounce.vel_z / g                  # time for full arc
        land_y     = by + bounce.vel_y * arc_time           # approx landing y
        mid_y      = (by + land_y) / 2                     # arc midpoint y
        mid_z      = peak_z * 0.75                         # ~¾ peak for lookat height

        obs_y      = _clamp_y(-room_depth * 0.30)

        return [
            CameraSpec(
                name="observer",
                # South of room, watching the ball's parabola arrive
                pos=[_j(bx, 0.15), _j(obs_y, 0.12), _j(min(peak_z * 0.60 + 0.20, 1.50), 0.12)],
                lookat=[_j(bx, 0.08), _j(mid_y * 0.35, 0.10), _j(mid_z * 0.60, 0.08)],
                fov=float(rng.uniform(54, 64)),
            ),
            CameraSpec(
                name="closeup",
                # East side, watching the ball's arc cross the room
                pos=[_j(bx + 1.50, 0.12), _j(mid_y, 0.12), _j(mid_z * 0.90, 0.10)],
                lookat=[_j(bx, 0.06), _j(by, 0.06), _j(bz, 0.06)],
                fov=float(rng.uniform(52, 62)),
            ),
            CameraSpec(
                name="overhead",
                pos=[_j(bx, 0.10), _j(by * 0.20, 0.10), _clamp_z(_j(2.50, 0.10))],
                lookat=[_j(bx, 0.06), _j(mid_y, 0.08), _j(0.10, 0.06)],
                fov=float(rng.uniform(62, 74)),
            ),
        ]

    # ── Ladder slip ───────────────────────────────────────────────────────────

    def _build_ladder_slip_spec(
        self,
        rng: np.random.Generator,
        seed: int,
        force_object: Optional[str] = None,
    ) -> SceneSpec:
        if force_object:
            candidates = [o for o in self._objects if o["name"] == force_object]
            obj_dict = candidates[0] if candidates else self._objects[rng.integers(len(self._objects))]
        else:
            obj_dict = self._objects[rng.integers(len(self._objects))]

        room_dict = self._rooms[rng.integers(len(self._rooms))]
        obj_spec  = self._make_object_spec(obj_dict)
        room_spec = self._make_room(rng, room_dict)
        ladder    = self._make_ladder(rng, room_dict, obj_spec)
        lighting  = self._make_lighting(rng, room_dict)

        obj_pos  = self._ladder_world_pos(room_dict, ladder, obj_spec)
        cameras  = self._make_cameras_with_retry(
            rng,
            lambda r: self._make_ladder_cameras(r, room_dict, ladder, obj_spec),
            obj_pos,
        )

        # GT: ladder always falls toward observer — no catching.
        # Light ladder → dodge sideways; heavy → brace
        if obj_spec.mass_hint == "heavy":
            gt_action = "BRACE_FOR_IMPACT"
        else:
            gt_action = "TRIGGER_DODGE"

        return SceneSpec(
            seed=seed,
            task_type="ladder_slip",
            adversarial=(obj_dict["category"] == "adversarial"),
            room=room_spec,
            object=obj_spec,
            ladder=ladder,
            cameras=cameras,
            lighting=lighting,
            ground_truth_action=gt_action,
            safety_label=obj_spec.safety_label,
        )

    def _make_ladder(
        self,
        rng: np.random.Generator,
        room: dict,
        obj_spec: ObjectSpec,
    ) -> LadderSpec:
        """Sample a LadderSpec.

        Mid-slip convention (matches furniture_tip):
          lean_deg = already-tilted angle from vertical toward observer (20–40°)
          start_y  = world Y of COM, north half of room (clear of walls)
          angular_vel = small positive push (wx > 0) → continues falling toward -Y
        """
        room_width = room["width"]
        room_depth = room.get("depth", 5.0)

        lean_deg    = float(rng.uniform(20.0, 40.0))
        base_x      = float(rng.uniform(-room_width * 0.25, room_width * 0.25))
        angular_vel = float(rng.uniform(0.30, 0.70))
        euler_z     = float(rng.uniform(-8.0, 8.0))
        start_y     = float(rng.uniform(0.20, min(0.80, room_depth * 0.25)))

        return LadderSpec(
            lean_deg=round(lean_deg, 1),
            base_x=round(base_x, 3),
            start_y=round(start_y, 3),
            angular_vel=round(angular_vel, 3),
            euler_z=round(euler_z, 1),
        )

    @staticmethod
    def _ladder_world_pos(
        room: dict,
        ladder: LadderSpec,
        obj: ObjectSpec,
    ) -> tuple[float, float, float]:
        """World-space COM of the ladder at t=0 (mid-slip convention)."""
        hz = obj.size[2] if len(obj.size) >= 3 else obj.size[0]
        return ladder.base_x, ladder.start_y, hz

    def _make_ladder_cameras(
        self,
        rng: np.random.Generator,
        room_dict: dict,
        ladder: LadderSpec,
        obj: ObjectSpec,
    ) -> list[CameraSpec]:
        """Three cameras for a ladder-slip event.

        Observer: south of room at eye height, watching the ladder fall toward them.
        Closeup:  east-side view showing the fall arc.
        Overhead: top-down view of the fall path.
        """
        ox, oy, oz = self._ladder_world_pos(room_dict, ladder, obj)
        room_depth  = room_dict.get("depth",  5.0)
        room_height = room_dict.get("height", 2.8)
        south_wall_y = -room_depth / 2
        max_cam_z    = room_height - 0.25

        def _j(v, s=0.10): return round(v + float(rng.uniform(-s, s)), 3)
        def _clamp_y(y): return max(south_wall_y + 0.20, y)
        def _clamp_z(z): return min(max_cam_z, z)

        obs_y = _clamp_y(-room_depth * 0.35)

        return [
            CameraSpec(
                name="observer",
                pos=[_j(ox, 0.12), _j(obs_y, 0.12), _j(1.20, 0.10)],
                lookat=[_j(ox, 0.08), _j(oy * 0.40, 0.10), _j(oz * 0.60, 0.08)],
                fov=float(rng.uniform(52, 62)),
            ),
            CameraSpec(
                name="closeup",
                pos=[_j(ox + 1.40, 0.12), _j(oy * 0.50, 0.12), _j(oz * 0.80, 0.10)],
                lookat=[_j(ox, 0.06), _j(oy * 0.60, 0.08), _j(oz * 0.70, 0.06)],
                fov=float(rng.uniform(50, 60)),
            ),
            CameraSpec(
                name="overhead",
                pos=[_j(ox, 0.08), _j(oy * 0.20, 0.10), _clamp_z(_j(2.30, 0.10))],
                lookat=[_j(ox, 0.06), _j(oy * 0.50, 0.08), _j(0.50, 0.05)],
                fov=float(rng.uniform(60, 72)),
            ),
        ]

    # ── Chain reaction ────────────────────────────────────────────────────────

    def _build_chain_reaction_spec(
        self,
        rng: np.random.Generator,
        seed: int,
        force_object: Optional[str] = None,
    ) -> SceneSpec:
        # Target B — any object_drop object (the threatening one that hits the observer)
        if force_object:
            candidates = [o for o in self._objects if o["name"] == force_object]
            obj_dict = candidates[0] if candidates else self._objects[rng.integers(len(self._objects))]
        else:
            obj_dict = self._objects[rng.integers(len(self._objects))]

        room_dict = self._rooms[rng.integers(len(self._rooms))]
        obj_spec  = self._make_object_spec(obj_dict)
        table     = self._make_table(rng, room_dict)
        room_spec = self._make_room(rng, room_dict)
        chain     = self._make_chain(rng, room_dict, table, obj_spec)
        lighting  = self._make_lighting(rng, room_dict)

        obj_pos  = self._chain_world_pos(table, chain, obj_spec)
        cameras  = self._make_cameras_with_retry(
            rng,
            lambda r: self._make_chain_cameras(r, room_dict, table, chain, obj_spec),
            obj_pos,
        )

        # GT based on target B's properties (same logic as object_drop)
        if obj_spec.catch_safe:
            gt_action = "EXECUTE_CATCH"
        elif obj_spec.mass_hint == "heavy":
            gt_action = "BRACE_FOR_IMPACT"
        else:
            gt_action = "TRIGGER_DODGE"

        return SceneSpec(
            seed=seed,
            task_type="chain_reaction",
            adversarial=(obj_dict["category"] == "adversarial"),
            room=room_spec,
            table=table,
            object=obj_spec,
            chain=chain,
            cameras=cameras,
            lighting=lighting,
            ground_truth_action=gt_action,
            safety_label=obj_spec.safety_label,
        )

    def _make_chain(
        self,
        rng: np.random.Generator,
        room: dict,
        table: TableSpec,
        obj_spec: ObjectSpec,
    ) -> ChainSpec:
        """Sample a ChainSpec.

        Target B is placed near the south edge of the table.
        Trigger A starts 20–35 cm north of B, given southward velocity.
        """
        _, _, hz_b = self._obj_phys_half(obj_spec)

        # Target B: near south edge of the table, leaving enough room to slide
        table_south_edge = table.pos_y - table.depth / 2
        target_y = float(table_south_edge + hz_b * 2 + rng.uniform(0.02, 0.08))
        target_x = float(table.pos_x + rng.uniform(-table.width * 0.20, table.width * 0.20))

        # Trigger A: north of B by gap_y, same X
        gap_y     = float(rng.uniform(0.18, 0.32))
        trigger_y = target_y + gap_y
        trigger_x = target_x   # aligned in X for a direct head-on collision

        # Trigger velocity — enough to push B off the table after the collision
        trigger_vel_y = float(rng.uniform(-2.0, -1.2))

        # Random trigger look (book-like flat box)
        trigger_size    = [
            round(float(rng.uniform(0.06, 0.09)), 3),
            round(float(rng.uniform(0.04, 0.07)), 3),
            round(float(rng.uniform(0.010, 0.020)), 3),
        ]
        trigger_density = float(rng.uniform(500.0, 850.0))
        hue = float(rng.uniform(0.0, 1.0))
        trigger_color   = list(self._hsv_to_rgb(hue, 0.55, 0.60))
        euler_z         = float(rng.uniform(-15.0, 15.0))

        return ChainSpec(
            target_x=round(target_x, 3),
            target_y=round(target_y, 3),
            euler_z=round(euler_z, 1),
            trigger_x=round(trigger_x, 3),
            trigger_y=round(trigger_y, 3),
            trigger_vel_y=round(trigger_vel_y, 3),
            trigger_size=trigger_size,
            trigger_density=round(trigger_density, 1),
            trigger_color=trigger_color,
        )

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> list[float]:
        """Convert HSV (all in 0–1) to [R, G, B] in 0–1."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return [round(r, 3), round(g, 3), round(b, 3)]

    @staticmethod
    def _chain_world_pos(
        table: TableSpec,
        chain: ChainSpec,
        obj: ObjectSpec,
    ) -> tuple[float, float, float]:
        """World-space COM of target B at t=0."""
        _, _, hz = Randomizer._obj_phys_half(obj)
        if obj.morph == "mesh":
            obj_z = table.height - obj.bottom_z_offset + hz
        else:
            obj_z = table.height + hz
        return chain.target_x, chain.target_y, obj_z

    def _make_chain_cameras(
        self,
        rng: np.random.Generator,
        room_dict: dict,
        table: TableSpec,
        chain: ChainSpec,
        obj: ObjectSpec,
    ) -> list[CameraSpec]:
        """Three cameras for a chain-reaction event.

        Observer: south of table at eye height, watching B fly toward them.
        Closeup:  east-side view showing A→B collision and B's fall.
        Overhead: top-down view of the full chain path.
        """
        ox, oy, oz = self._chain_world_pos(table, chain, obj)
        room_depth  = room_dict.get("depth",  5.0)
        room_height = room_dict.get("height", 2.8)
        south_wall_y = -room_depth / 2
        max_cam_z    = room_height - 0.25

        def _j(v, s=0.08): return round(v + float(rng.uniform(-s, s)), 3)
        def _clamp_y(y): return max(south_wall_y + 0.20, y)
        def _clamp_z(z): return min(max_cam_z, z)

        obs_y = _clamp_y(-room_depth * 0.30)
        # Midpoint between trigger start and table south edge
        trigger_y = chain.trigger_y
        chain_mid_y = (trigger_y + oy) / 2

        return [
            CameraSpec(
                name="observer",
                pos=[_j(ox, 0.15), _j(obs_y, 0.12), _j(1.15, 0.10)],
                lookat=[_j(ox, 0.08), _j(chain_mid_y * 0.20, 0.10), _j(oz * 0.70, 0.08)],
                fov=float(rng.uniform(52, 62)),
            ),
            CameraSpec(
                name="closeup",
                pos=[_j(ox + 1.20, 0.12), _j(chain_mid_y, 0.12), _j(table.height + 0.45, 0.10)],
                lookat=[_j(ox, 0.06), _j(oy, 0.06), _j(oz, 0.06)],
                fov=float(rng.uniform(48, 58)),
            ),
            CameraSpec(
                name="overhead",
                pos=[_j(ox, 0.10), _j(chain_mid_y * 0.30, 0.10), _clamp_z(_j(2.20, 0.10))],
                lookat=[_j(ox, 0.06), _j(oy, 0.08), _j(oz * 0.30, 0.05)],
                fov=float(rng.uniform(60, 72)),
            ),
        ]

    # ── Lighting ──────────────────────────────────────────────────────────────

    def _make_lighting(self, rng: np.random.Generator, room: dict) -> LightingSpec:
        preset_name = room.get("lighting_preset", "warm_pendant")
        preset = self._lighting_presets.get(preset_name, {})

        # Scale intensities slightly
        scale = float(rng.uniform(0.85, 1.15))

        lights = []
        for ls in preset.get("lights", []):
            lights.append(LightSpec(
                type=ls["type"],
                pos=ls.get("pos"),
                dir=ls.get("dir"),
                color=ls["color"],
                intensity=round(ls["intensity"] * scale, 2),
            ))

        # Ambient jitter
        base_amb = preset.get("ambient_light", [0.28, 0.25, 0.22])
        jitter    = float(rng.uniform(0.90, 1.10))
        ambient   = [min(1.0, c * jitter) for c in base_amb]

        return LightingSpec(
            ambient_light=ambient,
            background_color=preset.get("background_color", [0.15, 0.14, 0.13]),
            shadow=bool(preset.get("shadow", True)),
            lights=lights,
        )
