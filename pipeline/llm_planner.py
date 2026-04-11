"""
LLM Scene Planner
==================
Converts natural-language scenario descriptions into SceneSpec objects
by calling the Claude API.  The LLM only handles semantic parsing;
all physics parameters come from the validated asset library.

Paper citation pattern
-----------------------
"We use Claude-3.5-Sonnet as a scene planner to map natural-language
scenario descriptions to structured SceneSpec objects. The planner
output is validated against a Pydantic schema and subsequently
executed by our deterministic Genesis physics pipeline."

Usage
-----
    from pipeline.llm_planner import LLMPlanner
    planner = LLMPlanner()
    spec = planner.plan("A chef's knife slides off a kitchen counter
                          toward a cook's hand.")
    variants = planner.plan_with_variants(description, n=20)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import yaml

from .scene_spec import SceneSpec
from .randomizer import Randomizer

_HERE       = Path(__file__).parent.parent
_OBJ_YAML   = _HERE / "asset_library" / "objects.yaml"
_ROOM_YAML  = _HERE / "asset_library" / "rooms.yaml"


# ── System prompt given to the LLM ───────────────────────────

_SYSTEM_PROMPT = """\
You are a scene-specification planner for the ReactHuman physics benchmark.
Your job: given a plain-English description of a physical drop event in a home,
select the most appropriate object and room from the provided catalogues and
return a JSON object with ONLY these two fields:

  {
    "object_name": "<name from objects catalogue>",
    "room_type":   "<type from rooms catalogue>"
  }

Rules:
- object_name MUST exactly match a name in the Objects Catalogue.
- room_type MUST exactly match a type in the Rooms Catalogue.
- If the description mentions a dangerous object (knife, iron, scissors),
  prefer the matching dangerous object.
- If the description is ambiguous, prefer "safe" objects.
- Output ONLY the JSON object, no markdown, no explanation.
"""


def _build_user_message(description: str, obj_names: list[str], room_types: list[str]) -> str:
    return (
        f"Objects Catalogue: {obj_names}\n"
        f"Rooms Catalogue:   {room_types}\n\n"
        f"Description: {description}\n\n"
        "Return the JSON selection:"
    )


# ─────────────────────────────────────────────────────────────────────────────

class LLMPlanner:
    """
    Uses the Claude API to map natural language → (object_name, room_type),
    then delegates all physics parameter sampling to the Randomizer.

    Parameters
    ----------
    model : str
        Claude model ID. Defaults to claude-sonnet-4-6.
    api_key : str, optional
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
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

        # Load catalogues for the prompt
        with open(_OBJ_YAML) as f:
            objs = yaml.safe_load(f)
        with open(_ROOM_YAML) as f:
            rooms_raw = yaml.safe_load(f)

        self._obj_names   = [o["name"] for o in objs]
        self._room_types  = [r["type"] for r in rooms_raw if isinstance(r, dict) and "type" in r]
        self._randomizer  = Randomizer()

    # ── Public ────────────────────────────────────────────────

    def plan(self, description: str, seed: int = 0) -> SceneSpec:
        """
        Convert one natural-language description to a SceneSpec.

        The LLM selects object + room; the Randomizer fills all
        physics parameters deterministically from `seed`.
        """
        obj_name, room_type = self._llm_select(description)
        spec = self._randomizer.sample_with_constraints(
            seed=seed,
            force_object=obj_name,
            force_room=room_type,
        )
        # Annotate with the original description
        spec.description = description
        return spec

    def plan_with_variants(
        self,
        description: str,
        n: int = 10,
        seed_start: int = 0,
    ) -> list[SceneSpec]:
        """
        Produce `n` physics variants of a single natural-language description.
        The LLM is called only once; the Randomizer produces N distinct seeds.
        """
        obj_name, room_type = self._llm_select(description)
        specs = []
        for i in range(n):
            spec = self._randomizer.sample_with_constraints(
                seed=seed_start + i,
                force_object=obj_name,
                force_room=room_type,
            )
            spec.description = description
            specs.append(spec)
        return specs

    # ── Private ───────────────────────────────────────────────

    def _llm_select(self, description: str) -> tuple[str, str]:
        """Call the Claude API and return (object_name, room_type)."""
        user_msg = _build_user_message(description, self._obj_names, self._room_types)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=128,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()

        # Strip optional markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)
        obj_name  = data["object_name"]
        room_type = data["room_type"]

        # Validate selections against catalogue
        if obj_name not in self._obj_names:
            raise ValueError(
                f"LLM returned unknown object '{obj_name}'. "
                f"Valid: {self._obj_names}"
            )
        if room_type not in self._room_types:
            raise ValueError(
                f"LLM returned unknown room '{room_type}'. "
                f"Valid: {self._room_types}"
            )

        return obj_name, room_type


# ─────────────────────────────────────────────────────────────────────────────
# Extend Randomizer with constraint sampling
# (monkey-patched here to keep randomizer.py clean)
# ─────────────────────────────────────────────────────────────────────────────

def _sample_with_constraints(
    self: Randomizer,
    seed: int,
    force_object: Optional[str] = None,
    force_room: Optional[str] = None,
) -> SceneSpec:
    import numpy as np
    rng = np.random.default_rng(seed)

    # Temporarily restrict pools
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


Randomizer.sample_with_constraints = _sample_with_constraints
