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
    CameraSpec, DropSpec, LightingSpec, LightSpec,
    ObjectSpec, RoomSpec, SceneSpec, TableSpec,
)

# ── Asset library paths ───────────────────────────────────────
_HERE = Path(__file__).parent.parent
_OBJECTS_YAML = _HERE / "asset_library" / "objects.yaml"
_ROOMS_YAML   = _HERE / "asset_library" / "rooms.yaml"


def _load_objects() -> list[dict]:
    with open(_OBJECTS_YAML) as f:
        return yaml.safe_load(f)


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
        object_categories: Optional[list[str]] = None,
        room_types: Optional[list[str]] = None,
        adversarial_prob: float = 0.15,
    ):
        all_objects = _load_objects()
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

    # ── Internal builders ─────────────────────────────────────

    def _build_spec(
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

        # ── 6. Cameras ────────────────────────────────────────
        cameras = self._make_cameras(rng, table, drop_spec)

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

    def _make_drop(
        self,
        rng: np.random.Generator,
        table: TableSpec,
        obj: ObjectSpec,
    ) -> DropSpec:
        # Object radius in x (for overhang calculation)
        obj_rx = obj.size[0]   # half-extent in x

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
    ) -> list[CameraSpec]:
        """Three cameras: observer, closeup, overhead — all slightly randomised."""
        tx, ty = table.pos_x, table.pos_y
        fall_x  = drop.start_x       # approx landing x
        lookat  = [fall_x * 0.6, ty, 0.55]   # midpoint between edge and floor

        def _j(v, s=0.08): return round(v + float(rng.uniform(-s, s)), 3)

        return [
            CameraSpec(
                name="observer",
                pos=[_j(tx - 0.6), _j(ty - 1.80, 0.15), _j(1.50, 0.10)],
                lookat=[_j(lookat[0]), _j(lookat[1]), _j(lookat[2], 0.05)],
                fov=float(rng.uniform(50, 58)),
            ),
            CameraSpec(
                name="closeup",
                pos=[_j(fall_x + 0.65), _j(ty - 1.10, 0.12), _j(1.10, 0.08)],
                lookat=[_j(fall_x), _j(ty), _j(0.50, 0.05)],
                fov=float(rng.uniform(45, 55)),
            ),
            CameraSpec(
                name="overhead",
                pos=[_j(fall_x - 0.05, 0.05), _j(ty + 0.20, 0.10), _j(2.05, 0.08)],
                lookat=[_j(fall_x, 0.03), _j(ty + 0.10, 0.05), _j(table.height, 0.02)],
                fov=float(rng.uniform(50, 60)),
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
