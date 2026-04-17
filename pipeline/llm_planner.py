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
    "object_drop",      # small object falls off a table / counter
    "furniture_tip",    # tall furniture topples toward a person
    "hanging_fall",     # wall/ceiling-mounted object falls when mount breaks
    "stack_collapse",   # stacked items collapse toward the observer
    "sliding_object",   # object slides down a ramp and flies off toward the observer
]


# ── System prompt ─────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a scene-specification planner for the ReactHuman physics benchmark.

## Task Types
- "object_drop"    : A small or medium object falls off a table or counter surface.
  Use for: objects sliding, rolling, tipping, or being knocked off surfaces.
- "furniture_tip"  : A tall piece of furniture topples toward a nearby person.
  Use for: bookshelves, cabinets, ladders, floor lamps, large vases tipping over.
- "hanging_fall"   : A wall- or ceiling-mounted object falls when its mount suddenly breaks.
  Use for: framed paintings, clocks, shelves, hanging lamps, potted plants on brackets.
- "stack_collapse" : A stack of identical objects collapses toward the observer.
  Use for: stacked cans, books, crates, boxes — any repeated item piled vertically.
- "sliding_object" : An object slides down a ramp and flies off toward the observer.
  Use for: any table-top-sized object given an initial push down an inclined surface.
  Objects from the Object Drop Catalogue are used for this task type.

## Output format

Case A — object already in catalogue (preferred, faster):
{
  "task_type":   "<one of the five task types above>",
  "object_name": "<exact name from the matching catalogue>",
  "room_type":   "<exact type from Rooms Catalogue>"
}

Case B — object NOT in any catalogue (use sparingly):
{
  "task_type":   "<one of the five task types above>",
  "object_name": "<new_snake_case_identifier>",
  "room_type":   "<exact type from Rooms Catalogue>",
  "new_object": {
    "display_name":  "Human Readable Name",
    "lvis_label":    "<real LVIS 1.0 category, snake_case, e.g. scissors, ceramic_bowl>",
    "target_size_m": 0.XX,
    "category":      "safe" | "dangerous" | "adversarial",
    "density":       <integer kg/m³>,
    "friction":      <float 0.0-1.0>,
    "restitution":   <float 0.0-1.0>,
    "color_rgb":     [R, G, B],
    "roughness":     <float 0.0-1.0>,
    "ior":           1.0,
    "catch_safe":    true | false,
    "safety_label":  "safe" | "caution" | "dangerous",
    "mass_hint":     "light" | "medium" | "heavy",
    "description":   "one-line English description"
  }
}

## Rules
1. Prefer catalogue objects whenever they fit the description.
2. Match task_type to the matching catalogue: object_drop/sliding_object → Object Drop,
   furniture_tip → Furniture Tip, hanging_fall → Hanging Objects,
   stack_collapse → Stack Collapse.
3. "object_drop" and "sliding_object" objects must be table-top sized — never furniture.
4. "furniture_tip" objects must be tall, stand-alone furniture pieces.
5. "hanging_fall" objects must be wall- or ceiling-mounted items.
6. "stack_collapse" objects must be stackable (roughly uniform shape).
7. lvis_label must be a real LVIS 1.0 snake_case category name.
8. Output ONLY the JSON object — no markdown fences, no explanation.

## Reference densities (kg/m³)
foam/sponge 30-100 | wood 400-700 | plastic 200-1200
ceramic 2000-2800 | glass 2500 | aluminium 2700 | steel/iron 7000-8000
"""


def _build_user_message(
    description: str,
    drop_obj_names: list[str],
    tip_obj_names: list[str],
    hanging_obj_names: list[str],
    stack_obj_names: list[str],
    room_types: list[str],
) -> str:
    return (
        f"Object Drop Catalogue   (task_type=object_drop OR sliding_object): {drop_obj_names}\n"
        f"Furniture Tip Catalogue (task_type=furniture_tip):                  {tip_obj_names}\n"
        f"Hanging Objects Catalogue (task_type=hanging_fall):                 {hanging_obj_names}\n"
        f"Stack Collapse Catalogue  (task_type=stack_collapse):               {stack_obj_names}\n"
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
        task_type, obj_name, room_type = self._llm_select(description)
        spec = self._get_randomizer(task_type).sample_with_constraints(
            seed=seed,
            force_object=obj_name,
            force_room=room_type,
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
        """
        task_type, obj_name, room_type = self._llm_select(description)
        rand = self._get_randomizer(task_type)
        specs = []
        for i in range(n):
            spec = rand.sample_with_constraints(
                seed=seed_start + i,
                force_object=obj_name,
                force_room=room_type,
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
        # sliding_object reuses the object_drop pool
        self._sliding_obj_names   = self._drop_obj_names

        # Map task_type → catalogue (for validation in _llm_select)
        self._task_catalogue: dict[str, list[str]] = {
            "object_drop":    self._drop_obj_names,
            "furniture_tip":  self._tip_obj_names,
            "hanging_fall":   self._hanging_obj_names,
            "stack_collapse": self._stack_obj_names,
            "sliding_object": self._sliding_obj_names,
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

    def _llm_select(self, description: str) -> tuple[str, str, str]:
        """
        Call Claude and return (task_type, object_name, room_type).
        If the LLM proposes a new object, download and register it first.
        """
        user_msg = _build_user_message(
            description,
            self._drop_obj_names,
            self._tip_obj_names,
            self._hanging_obj_names,
            self._stack_obj_names,
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

        return task_type, obj_name, room_type

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
        task_type : "object_drop" | "furniture_tip"
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
            import trimesh
            print(
                f"    Simplifying {len(mesh.faces):,} → {_MAX_FACES:,} faces …"
            )
            mesh = trimesh.simplify_quadric_decimation(mesh, face_count=_MAX_FACES)
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
            "color_rgb":       [round(float(c), 3) for c in spec["color_rgb"]],
            "roughness":       round(float(spec["roughness"]), 3),
            "ior":             round(float(spec.get("ior", 1.0)), 3),
            "catch_safe":      bool(spec["catch_safe"]),
            "safety_label":    spec["safety_label"],
            "mass_hint":       spec["mass_hint"],
            "description":     spec["description"],
        }
        # object_drop is the default (no field needed); all others must be explicit
        if task_type != "object_drop":
            entry["task_type"] = task_type

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
