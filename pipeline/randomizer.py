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
    CameraSpec, DropSpec, HangingSpec, LightingSpec, LightSpec,
    ObjectSpec, RampSpec, RoomSpec, SceneSpec, StackSpec, TableSpec, TipSpec,
)

# ── Asset library paths ───────────────────────────────────────
_HERE = Path(__file__).parent.parent
_OBJECTS_YAML = _HERE / "asset_library" / "objects.yaml"
_ROOMS_YAML   = _HERE / "asset_library" / "rooms.yaml"


def _load_objects(task_type: str = "object_drop") -> list[dict]:
    with open(_OBJECTS_YAML) as f:
        all_objs = yaml.safe_load(f)
    # sliding_object reuses the object_drop pool (same table-sized objects)
    effective = "object_drop" if task_type == "sliding_object" else task_type
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
