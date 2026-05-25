"""
LLM Scene Planner
==================
Converts natural-language scenario descriptions into SceneSpec objects
via the Claude API.

What changed from v1
---------------------
- Automatically determines task_type (object_drop / furniture_tip) from the
  description text, so callers no longer need to pass --task-type.
- When the description names an object not in the catalogue, the LLM proposes
  full physics parameters; the planner downloads a matching mesh from Objaverse,
  normalises it, and permanently registers it in objects.yaml + assets/meshes/.
- One Randomizer per task_type is kept in a cache; the cache is invalidated
  whenever a new object is registered, forcing a fresh YAML read.

Paper citation pattern
-----------------------
"We use Claude as a scene planner to map natural-language descriptions to
structured SceneSpec objects. The planner selects task type, object, and room
from a validated catalogue, or proposes new objects whose meshes are
automatically sourced from Objaverse (Deitke et al., 2023) and added to the
asset library permanently."

Usage
-----
    from pipeline.llm_planner import LLMPlanner
    planner = LLMPlanner()
    spec  = planner.plan("A chef's knife slides off a kitchen counter.")
    specs = planner.plan_with_variants(description, n=20)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import yaml

from .scene_spec import SceneSpec
from .randomizer import Randomizer

# ── Paths ─────────────────────────────────────────────────────
_HERE      = Path(__file__).parent.parent            # genesis_scene_generation/
_OBJ_YAML  = _HERE / "asset_library" / "objects.yaml"
_ROOM_YAML = _HERE / "asset_library" / "rooms.yaml"
_MESHES_DIR = _HERE / "assets" / "meshes"            # genesis_scene_generation/assets/meshes/
_RAW_DIR    = _HERE / "assets" / "raw_glb"          # temp GLB cache

_TASK_TYPES = [
    "object_drop",       # small object falls off a table / counter
    "furniture_tip",     # tall furniture topples toward a person
    "hanging_fall",      # wall/ceiling-mounted object falls when mount breaks
    "stack_collapse",    # stacked items collapse toward the observer
    "sliding_object",    # object slides down a ramp and flies off toward the observer
    "rolling_ball",      # ball rolls off a table toward the observer
    "shelf_slide",       # object slides off a high wall shelf toward the observer
    "door_swing",        # door swings uncontrolled toward the observer
    "thrown_object",     # object deliberately thrown toward the observer
    "pendulum_swing",    # heavy object swings on a cable/rope toward the observer
    "bouncing_object",   # ball falls, bounces off floor, flies toward the observer
    "ladder_slip",       # ladder slides/tips on slippery floor toward the observer
    "chain_reaction",    # one object falls and triggers a cascade toward the observer
    "ceiling_drop",      # object spawned at ceiling height falls straight down to floor
    "stair_tumble",      # object tumbles down a staircase toward the observer
]


# ── System prompt ─────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a scene-specification planner for the ReactHuman physics benchmark.

## Task Types
- "object_drop"     : A small or medium object falls off a table or counter surface.
  Use for: objects sliding, rolling, tipping, or being knocked off surfaces.
- "furniture_tip"   : A tall piece of furniture topples toward a nearby person.
  Use for: bookshelves, cabinets, floor lamps, large vases tipping over.
- "hanging_fall"    : A wall- or ceiling-mounted object falls when its mount suddenly breaks.
  Use for: framed paintings, clocks, shelves, hanging lamps, potted plants on brackets.
- "stack_collapse"  : A stack of identical objects collapses toward the observer.
  Use for: stacked cans, books, crates, boxes — any repeated item piled vertically.
- "sliding_object"  : An object slides down a ramp and flies off toward the observer.
  Use for: any table-top-sized object given an initial push down an inclined surface.
  Objects from the Object Drop Catalogue are used for this task type.
- "rolling_ball"    : A ball rolls off a table edge toward the observer.
  Use for: balls of any type rolling off surfaces.
- "shelf_slide"     : An object slides off a high wall shelf and falls toward the observer.
  Use for: items on high shelves that slide off — plants, bottles, vases, toolboxes.
- "door_swing"      : A door swings rapidly/uncontrolled toward a person.
  Use for: any type of door swinging open violently toward the observer.
- "thrown_object"   : An object is deliberately thrown toward the observer.
  Use for: balls, knives, or any object hurled at the camera.
- "pendulum_swing"  : A heavy object suspended on a cable or rope swings toward the observer.
  Use for: chandeliers, wrecking-ball-style weights, sandbags on ropes swinging.
- "bouncing_object" : A ball falls from above, bounces off the floor, and flies toward the observer.
  Use for: any bouncing ball scenario where the ball rebounds toward the camera.
- "ladder_slip"     : A leaning ladder slides on a slippery floor and tips toward the observer.
  Use for: ladders of any type leaning against a wall that slip and fall.
- "chain_reaction"  : One object falls and triggers a cascade of objects toward the observer.
  Use for: domino-style scenarios where the initial object triggers additional hazards.
  Objects from the Object Drop Catalogue are used for this task type.
- "ceiling_drop"    : An object falls straight down from ceiling height to the floor.
  Use for: objects knocked off a very high shelf or overhead surface, ceiling debris.
  Objects from the Object Drop Catalogue are used for this task type.
- "stair_tumble"    : An object tumbles down a staircase toward the observer.
  Use for: any object rolling, sliding, or bouncing down a set of stairs.

## Hints

In addition to selecting the task type, object, and room, you may emit a `hints` block
to bias the physics sampler when the description contains clear speed or height cues.
Include `hints` only when the description explicitly implies a non-default value;
omit it entirely when the description is neutral.

  speed values: "slow" | "normal" | "fast" | "very_fast"
    slow      → 40 % of normal velocity ranges
    normal    → default ranges (omit hints block)
    fast      → 180 % of normal velocity ranges
    very_fast → 300 % of normal velocity ranges

  height values: "low" | "normal" | "high"
    low    → object spawns at / attaches at the lower end of its range
    normal → default height ranges (omit hints block)
    high   → object spawns at / attaches at the upper end of its range

Hint triggers:
  speed=slow       : "slowly", "gently", "lightly", "soft toss"
  speed=fast       : "quickly", "rapidly", "hard throw", "high speed"
  speed=very_fast  : "hurled", "slammed", "extremely fast", "full force"
  height=low       : "low shelf", "bottom step", "near the floor"
  height=high      : "top shelf", "upper landing", "near the ceiling"

## Output format

Case A — the description's object exactly matches a catalogue entry (same object, same name):
{
  "task_type":   "<one of the fifteen task types above>",
  "object_name": "<exact name from the matching catalogue>",
  "room_type":   "<exact type from Rooms Catalogue>",
  "hints":       {"speed": "fast"}   // omit entirely if neutral
}

Case B — the description names a specific object NOT in any catalogue:
{
  "task_type":   "<one of the fifteen task types above>",
  "object_name": "<new_snake_case_identifier>",
  "room_type":   "<exact type from Rooms Catalogue>",
  "hints":       {"speed": "slow", "height": "high"},   // omit entirely if neutral
  "new_object": {
    "display_name":  "Human Readable Name",
    "lvis_label":    "<real LVIS 1.0 category, snake_case, e.g. banana, scissors, ceramic_bowl>",
    "target_size_m": 0.XX,
    "category":      "safe" | "dangerous" | "adversarial",
    "density":       <integer kg/m³>,
    "friction":      <float 0.0-1.0>,
    "restitution":   <float 0.0-1.0>,
    "color_rgb":     [R, G, B],   (floats in 0.0–1.0 range, NOT 0–255)
    "roughness":     <float 0.0-1.0>,
    "ior":           1.0,
    "catch_safe":    true | false,
    "safety_label":  "safe" | "caution" | "dangerous",
    "mass_hint":     "light" | "medium" | "heavy",
    "description":   "one-line English description"
  }
}

## Rules
1. OBJECT MATCHING: Always prefer an existing catalogue entry.
   - Match by physical type, size, and behaviour — not by exact name.
   - Examples: "football" / "soccer ball" → nearest sphere in the matching catalogue;
     "dining chair" → nearest chair-like object; "wine bottle" → nearest bottle object.
   - Use Case A whenever a catalogue object is a reasonable physical stand-in for the
     described object (same morph, similar size, similar mass category).
   - Use Case B (new_object + Objaverse download) ONLY when no existing catalogue object
     is remotely similar in shape or physical behaviour.
2. Match task_type to the matching catalogue:
   object_drop/sliding_object/chain_reaction/ceiling_drop → Object Drop Catalogue,
   furniture_tip → Furniture Tip Catalogue,
   hanging_fall → Hanging Objects Catalogue,
   stack_collapse → Stack Collapse Catalogue,
   rolling_ball → Rolling Ball Catalogue,
   shelf_slide → Shelf Slide Catalogue,
   door_swing → Door Swing Catalogue,
   thrown_object → Thrown Object Catalogue,
   pendulum_swing → Pendulum Swing Catalogue,
   bouncing_object → Bouncing Object Catalogue,
   ladder_slip → Ladder Slip Catalogue,
   stair_tumble → Stair Tumble Catalogue.
3. "object_drop", "sliding_object", "chain_reaction", and "ceiling_drop" objects must be table-top sized — never furniture.
4. "furniture_tip" objects must be tall, stand-alone furniture pieces.
5. "hanging_fall" objects must be wall- or ceiling-mounted items.
6. "stack_collapse" objects must be stackable (roughly uniform shape).
7. "rolling_ball" and "bouncing_object" objects must be spherical.
8. "ladder_slip" objects must be ladders.
9. "door_swing" objects must be doors.
10. lvis_label must be a real LVIS 1.0 snake_case category name.
11. Output ONLY the JSON object — no markdown fences, no explanation.
12. Omit the "hints" key entirely when the description gives no speed or height cue.

## Reference densities (kg/m³)
foam/sponge 30-100 | wood 400-700 | plastic 200-1200
ceramic 2000-2800 | glass 2500 | aluminium 2700 | steel/iron 7000-8000

## Common LVIS label corrections (use these exact strings)
"football" → "soccer_ball" or "football_(American)"
"knife" → "kitchen_knife"  |  "cup" → "mug"  |  "pot" → "cooking_pot"
"ball" → "ball" (generic)  |  "bottle" → "bottle"  |  "box" → "cardboard_box"
"lamp" → "table_lamp" or "floor_lamp"  |  "vase" → "vase"
"""


def _build_user_message(
    description: str,
    drop_obj_names: list[str],
    tip_obj_names: list[str],
    hanging_obj_names: list[str],
    stack_obj_names: list[str],
    roll_obj_names: list[str],
    shelf_obj_names: list[str],
    door_obj_names: list[str],
    thrown_obj_names: list[str],
    pendulum_obj_names: list[str],
    bounce_obj_names: list[str],
    ladder_obj_names: list[str],
    stair_obj_names: list[str],
    room_types: list[str],
) -> str:
    return (
        f"Object Drop Catalogue     (task_type=object_drop OR sliding_object OR chain_reaction OR ceiling_drop): {drop_obj_names}\n"
        f"Furniture Tip Catalogue   (task_type=furniture_tip):    {tip_obj_names}\n"
        f"Hanging Objects Catalogue (task_type=hanging_fall):     {hanging_obj_names}\n"
        f"Stack Collapse Catalogue  (task_type=stack_collapse):   {stack_obj_names}\n"
        f"Rolling Ball Catalogue    (task_type=rolling_ball):     {roll_obj_names}\n"
        f"Shelf Slide Catalogue     (task_type=shelf_slide):      {shelf_obj_names}\n"
        f"Door Swing Catalogue      (task_type=door_swing):       {door_obj_names}\n"
        f"Thrown Object Catalogue   (task_type=thrown_object):    {thrown_obj_names}\n"
        f"Pendulum Swing Catalogue  (task_type=pendulum_swing):   {pendulum_obj_names}\n"
        f"Bouncing Object Catalogue (task_type=bouncing_object):  {bounce_obj_names}\n"
        f"Ladder Slip Catalogue     (task_type=ladder_slip):      {ladder_obj_names}\n"
        f"Stair Tumble Catalogue    (task_type=stair_tumble):     {stair_obj_names}\n"
        f"Rooms Catalogue: {room_types}\n\n"
        f"Description: {description}\n\n"
        "Return the JSON selection:"
    )


# ── Mesh helpers ──────────────────────────────────────────────

def _load_as_single_trimesh(path: Path):
    """Load a GLB/OBJ and merge all sub-meshes into one Trimesh."""
    import trimesh
    obj = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(obj, trimesh.Scene):
        meshes = [g for g in obj.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        return trimesh.util.concatenate(meshes) if meshes else None
    return obj if isinstance(obj, trimesh.Trimesh) else None


def _normalize_mesh(mesh, target_size_m: float):
    """Scale to target longest dim and translate bottom to z=0."""
    import numpy as np
    scale = target_size_m / max(mesh.bounding_box.extents)
    mesh.apply_scale(scale)
    b = mesh.bounds
    mesh.apply_translation([
        -(b[0][0] + b[1][0]) / 2,
        -(b[0][1] + b[1][1]) / 2,
        -b[0][2],
    ])
    return mesh


# ─────────────────────────────────────────────────────────────────────────────

class LLMPlanner:
    """
    Maps natural-language descriptions → SceneSpec objects.

    - task_type is inferred automatically from the description.
    - Unknown objects trigger an Objaverse download and are permanently
      registered in objects.yaml and assets/meshes/.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
    ):
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
        except ImportError:
            raise ImportError("pip install anthropic  # to use LLMPlanner")

        self._model = model
        self._load_catalogues()
        # Lazy Randomizer cache — one entry per task_type string.
        # Invalidated when a new object is registered so the next call
        # re-reads objects.yaml and picks up the new entry.
        self._randomizer_cache: dict[str, Randomizer] = {}

    # ── Public ────────────────────────────────────────────────

    def plan(self, description: str, seed: int = 0) -> SceneSpec:
        """Convert one natural-language description to a single SceneSpec."""
        task_type, obj_name, room_type, hints = self._llm_select(description)
        spec = self._get_randomizer(task_type).sample_with_constraints(
            seed=seed,
            force_object=obj_name,
            force_room=room_type,
            hints=hints,
        )
        spec.description = description
        return spec

    def plan_with_variants(
        self,
        description: str,
        n: int = 10,
        seed_start: int = 0,
    ) -> list[SceneSpec]:
        """
        Produce N physics variants of one description.
        The Claude API is called exactly once per description.
        Hints from the LLM response are applied to every variant.
        """
        task_type, obj_name, room_type, hints = self._llm_select(description)
        rand = self._get_randomizer(task_type)
        specs = []
        for i in range(n):
            spec = rand.sample_with_constraints(
                seed=seed_start + i,
                force_object=obj_name,
                force_room=room_type,
                hints=hints,
            )
            spec.description = description
            specs.append(spec)
        return specs

    # ── Catalogue helpers ─────────────────────────────────────

    def _load_catalogues(self) -> None:
        """Read objects.yaml and rooms.yaml into memory."""
        with open(_OBJ_YAML) as f:
            obj_list = yaml.safe_load(f)

        self._obj_names           = [o["name"] for o in obj_list]
        self._drop_obj_names      = [
            o["name"] for o in obj_list
            if o.get("task_type", "object_drop") == "object_drop"
        ]
        self._tip_obj_names       = [
            o["name"] for o in obj_list
            if o.get("task_type") == "furniture_tip"
        ]
        self._hanging_obj_names   = [
            o["name"] for o in obj_list
            if o.get("task_type") == "hanging_fall"
        ]
        self._stack_obj_names     = [
            o["name"] for o in obj_list
            if o.get("task_type") == "stack_collapse"
        ]
        self._roll_obj_names      = [
            o["name"] for o in obj_list
            if o.get("task_type") == "rolling_ball"
        ]
        self._shelf_obj_names     = [
            o["name"] for o in obj_list
            if o.get("task_type") == "shelf_slide"
        ]
        self._door_obj_names      = [
            o["name"] for o in obj_list
            if o.get("task_type") == "door_swing"
        ]
        self._thrown_obj_names    = [
            o["name"] for o in obj_list
            if o.get("task_type") == "thrown_object"
        ]
        self._pendulum_obj_names  = [
            o["name"] for o in obj_list
            if o.get("task_type") == "pendulum_swing"
        ]
        self._bounce_obj_names    = [
            o["name"] for o in obj_list
            if o.get("task_type") == "bouncing_object"
        ]
        self._ladder_obj_names    = [
            o["name"] for o in obj_list
            if o.get("task_type") == "ladder_slip"
        ]
        self._stair_obj_names     = [
            o["name"] for o in obj_list
            if o.get("task_type") == "stair_tumble"
        ]
        # sliding_object, chain_reaction, and ceiling_drop reuse the object_drop pool
        self._sliding_obj_names   = self._drop_obj_names
        self._chain_obj_names     = self._drop_obj_names
        self._ceiling_drop_obj_names = self._drop_obj_names

        # Map task_type → catalogue (for validation in _llm_select)
        self._task_catalogue: dict[str, list[str]] = {
            "object_drop":    self._drop_obj_names,
            "furniture_tip":  self._tip_obj_names,
            "hanging_fall":   self._hanging_obj_names,
            "stack_collapse": self._stack_obj_names,
            "sliding_object": self._sliding_obj_names,
            "rolling_ball":   self._roll_obj_names,
            "shelf_slide":    self._shelf_obj_names,
            "door_swing":     self._door_obj_names,
            "thrown_object":  self._thrown_obj_names,
            "pendulum_swing": self._pendulum_obj_names,
            "bouncing_object": self._bounce_obj_names,
            "ladder_slip":    self._ladder_obj_names,
            "chain_reaction": self._chain_obj_names,
            "ceiling_drop":   self._ceiling_drop_obj_names,
            "stair_tumble":   self._stair_obj_names,
        }

        with open(_ROOM_YAML) as f:
            rooms_raw = yaml.safe_load(f)
        self._room_types = [r["type"] for r in rooms_raw.get("rooms", [])]

    def _get_randomizer(self, task_type: str) -> Randomizer:
        if task_type not in self._randomizer_cache:
            self._randomizer_cache[task_type] = Randomizer(task_type=task_type)
        return self._randomizer_cache[task_type]

    def _invalidate_randomizer(self, task_type: str) -> None:
        """Force a fresh Randomizer after a new object is registered."""
        self._randomizer_cache.pop(task_type, None)

    # ── LLM interaction ───────────────────────────────────────

    def _llm_select(self, description: str) -> tuple[str, str, str, dict]:
        """
        Call Claude and return (task_type, object_name, room_type, hints).
        If the LLM proposes a new object, download and register it first.
        """
        user_msg = _build_user_message(
            description,
            self._drop_obj_names,
            self._tip_obj_names,
            self._hanging_obj_names,
            self._stack_obj_names,
            self._roll_obj_names,
            self._shelf_obj_names,
            self._door_obj_names,
            self._thrown_obj_names,
            self._pendulum_obj_names,
            self._bounce_obj_names,
            self._ladder_obj_names,
            self._stair_obj_names,
            self._room_types,
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()

        # Strip optional markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data      = json.loads(raw)
        task_type = data["task_type"]
        obj_name  = data["object_name"]
        room_type = data["room_type"]
        hints     = data.get("hints") or {}
        new_obj   = data.get("new_object")

        # Validate task_type
        if task_type not in _TASK_TYPES:
            raise ValueError(
                f"LLM returned unknown task_type '{task_type}'. "
                f"Valid: {_TASK_TYPES}"
            )
        # Validate room_type
        if room_type not in self._room_types:
            raise ValueError(
                f"LLM returned unknown room '{room_type}'. "
                f"Valid: {self._room_types}"
            )

        # Handle new object: download, normalise, register, reload
        if new_obj is not None and obj_name not in self._obj_names:
            print(
                f"  [LLMPlanner] '{obj_name}' not in catalogue — "
                "downloading from Objaverse …"
            )
            self._download_and_register(obj_name, task_type, new_obj)
            self._load_catalogues()            # reload YAML (now includes new entry)
            self._invalidate_randomizer(task_type)

        # Validate object_name against the correct per-task catalogue
        valid_for_task = self._task_catalogue.get(task_type, self._obj_names)
        if obj_name not in valid_for_task:
            raise ValueError(
                f"LLM returned object '{obj_name}' which is not valid for "
                f"task_type='{task_type}'. Valid: {valid_for_task}"
            )

        return task_type, obj_name, room_type, hints

    # ── Objaverse integration ─────────────────────────────────

    def _download_and_register(
        self,
        obj_name: str,
        task_type: str,
        spec: dict,
    ) -> None:
        """
        Download a mesh from Objaverse, normalise it, write to assets/meshes/,
        and permanently append a validated entry to objects.yaml.

        Parameters
        ----------
        obj_name  : snake_case identifier for the new object
        task_type : one of the thirteen task types (sliding_object/chain_reaction
                    are treated as object_drop for registration purposes)
        spec      : the "new_object" dict from the LLM response
        """
        try:
            import objaverse
        except ImportError:
            raise ImportError(
                "pip install objaverse trimesh  # for dynamic object download"
            )

        lvis_label  = spec["lvis_label"]
        target_size = float(spec["target_size_m"])

        # ── 1. Find a stable UID via LVIS annotations ──────────
        print(f"    Searching Objaverse for LVIS label '{lvis_label}' …")
        lvis_anns  = objaverse.load_lvis_annotations()
        candidates = lvis_anns.get(lvis_label, [])
        if not candidates:
            raise ValueError(
                f"No Objaverse objects found for LVIS label '{lvis_label}'. "
                "Try a different lvis_label (must be a real LVIS 1.0 category)."
            )
        uid = sorted(candidates)[0]   # alphabetic sort → stable, reproducible

        # ── 2. Download GLB ─────────────────────────────────────
        print(f"    Downloading uid={uid[:20]}… ")
        _RAW_DIR.mkdir(parents=True, exist_ok=True)
        uid_to_path = objaverse.load_objects(uids=[uid], download_processes=1)
        glb_src = uid_to_path.get(uid)
        if not glb_src or not Path(glb_src).exists():
            raise RuntimeError(f"Objaverse download failed for uid={uid}")

        raw_dest = _RAW_DIR / f"{obj_name}.glb"
        shutil.copy2(glb_src, raw_dest)

        # ── 3. Load mesh + normalise ────────────────────────────
        mesh = _load_as_single_trimesh(raw_dest)
        if mesh is None:
            raise RuntimeError(
                f"Could not load a valid mesh from the downloaded GLB for '{obj_name}'."
            )
        mesh = _normalize_mesh(mesh, target_size)

        # ── 3b. Simplify if over face-count limit ───────────────
        _MAX_FACES = 8_000
        if len(mesh.faces) > _MAX_FACES:
            print(
                f"    Simplifying {len(mesh.faces):,} → {_MAX_FACES:,} faces …"
            )
            mesh = mesh.simplify_quadric_decimation(face_count=_MAX_FACES)
            # Re-normalise after decimation (bounds may shift slightly)
            mesh = _normalize_mesh(mesh, target_size)

        ex = mesh.bounding_box.extents

        # ── 4. Export .obj ──────────────────────────────────────
        _MESHES_DIR.mkdir(parents=True, exist_ok=True)
        obj_path = _MESHES_DIR / f"{obj_name}.obj"
        mesh.export(str(obj_path), file_type="obj", include_normals=True)
        print(
            f"    Saved {obj_path.name}  "
            f"verts={len(mesh.vertices)}  faces={len(mesh.faces)}  "
            f"extents=[{ex[0]:.3f}, {ex[1]:.3f}, {ex[2]:.3f}] m"
        )

        # ── 5. Build validated YAML entry ───────────────────────
        # phys_half_x/z: actual physical half-extents in metres.
        # For a normalised mesh sitting at z=0, half-extents are extents/2.
        phys_half_x = round(float(ex[0]) / 2, 4)
        phys_half_z = round(float(ex[2]) / 2, 4)

        entry: dict = {
            "name":            obj_name,
            "display_name":    spec["display_name"],
            "category":        spec["category"],
            "morph":           "mesh",
            "mesh_path":       f"{obj_name}.obj",
            "size":            [1.0],
            "bottom_z_offset": 0.0,
            "phys_half_x":     phys_half_x,
            "phys_half_z":     phys_half_z,
            "density":         int(spec["density"]),
            "friction":        round(float(spec["friction"]), 3),
            "restitution":     round(float(spec["restitution"]), 3),
            "color_rgb":       [round(float(c) / 255.0, 3) if float(c) > 1.0 else round(float(c), 3)
                               for c in spec["color_rgb"]],
            "roughness":       round(float(spec["roughness"]), 3),
            "ior":             round(float(spec.get("ior", 1.0)), 3),
            "catch_safe":      bool(spec["catch_safe"]),
            "safety_label":    spec["safety_label"],
            "mass_hint":       spec["mass_hint"],
            "description":     spec["description"],
        }
        # sliding_object, chain_reaction, and ceiling_drop share the object_drop pool
        # in objects.yaml, so new objects for those task types must be registered as object_drop.
        _POOL_ALIASES = {"sliding_object", "chain_reaction", "ceiling_drop"}
        effective_task_type = "object_drop" if task_type in _POOL_ALIASES else task_type
        # object_drop is the default (no field needed); all others must be explicit
        if effective_task_type != "object_drop":
            entry["task_type"] = effective_task_type

        # ── 6. Append to objects.yaml ───────────────────────────
        # Read → append → write (safe: preserves existing content)
        with open(_OBJ_YAML) as f:
            existing_text = f.read()

        new_block = (
            "\n# Auto-registered by LLMPlanner\n"
            + yaml.dump([entry], allow_unicode=True, default_flow_style=False)
        )
        with open(_OBJ_YAML, "a") as f:
            f.write(new_block)

        print(f"    Registered '{obj_name}' in objects.yaml.")
