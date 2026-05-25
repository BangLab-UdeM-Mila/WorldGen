"""
Generic Scene Builder
======================
Builds and runs a Genesis simulation from any SceneSpec.
Replaces the old plate-specific plate_drop.py.

Called by scene_runner.py (subprocess entry point) and can also
be used directly for single-scene testing.
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import genesis as gs
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="genesis")

from pipeline.scene_spec import SceneSpec, ObjectSpec, CameraSpec, LightingSpec

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
REPO_ROOT    = PROJECT_ROOT.parent
MESHES_DIR   = PROJECT_ROOT / "assets" / "meshes"
TEX_ROOT     = PROJECT_ROOT / "assets" / "textures"
ENV_MAP      = Path(gs.__file__).parent / "assets" / "textures" / "indoor_bright.png"


# ─────────────────────────────────────────────────────────────────────────────
# Material helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rgb(c: list[float]) -> tuple:
    r, g, b = c
    if max(r, g, b) > 1.0:
        return r / 255, g / 255, b / 255
    return float(r), float(g), float(b)


def _tex(path: Path, encoding: str = "srgb"):
    if path.exists():
        return gs.textures.ImageTexture(image_path=str(path), encoding=encoding)
    return None


def _solid(color, roughness=0.5, metallic=0.0, ior=1.0) -> gs.surfaces.BSDF:
    return gs.surfaces.BSDF(color=_rgb(color), roughness=roughness,
                             metallic=metallic, ior=ior, smooth=True)


def _textured(color, roughness, diff_tex=None, rough_tex=None, nor_tex=None) -> gs.surfaces.BSDF:
    if diff_tex is None:
        return _solid(color, roughness)
    return gs.surfaces.BSDF(
        diffuse_texture=diff_tex,
        roughness=roughness if rough_tex is None else None,
        roughness_texture=rough_tex,
        normal_texture=nor_tex,
        smooth=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SceneBuilder
# ─────────────────────────────────────────────────────────────────────────────

class SceneBuilder:
    """Builds and runs a Genesis scene from a SceneSpec."""

    def __init__(self, spec: SceneSpec):
        self.spec   = spec
        self.scene  = None
        self.obj    = None          # the primary tracked entity (top item for stack_collapse)
        self.stack_objs: list = []  # all stack items (stack_collapse only)
        self.cameras: dict = {}
        self._built = False

        # Ground-truth trajectory recording
        self._gt_positions: list[list[float]] = []   # [(x,y,z), …]
        self._floor_hit_step: int | None = None
        self._bounce_risen: bool = False  # for bouncing_object: track upward phase
        self._chain_trigger = None        # for chain_reaction: the trigger entity (A)

    # ── Public ────────────────────────────────────────────────

    def build(self) -> None:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        self._build_scene()
        self._built = True

    def run(self, output_dir: str | Path) -> dict:
        """
        Simulate and record videos.
        Returns a metadata dict suitable for writing to metadata.json.
        """
        if not self._built:
            raise RuntimeError("Call build() first.")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        spec    = self.spec
        n_steps = int(spec.duration_s / spec.dt)
        dt      = spec.dt

        self._apply_velocity()
        for cam in self.cameras.values():
            cam.start_recording()

        t0 = time.perf_counter()
        for step in range(n_steps):
            self.scene.step()
            for cam in self.cameras.values():
                cam.render()

            pos = self._obj_pos()
            self._gt_positions.append(pos.tolist())
            # Floor-hit threshold varies by task type
            if self._floor_hit_step is None:
                if self.spec.task_type == "furniture_tip":
                    half_w = self.spec.object.size[0]
                    if pos[2] < half_w + 0.05:
                        self._floor_hit_step = step
                elif self.spec.task_type == "hanging_fall":
                    o = self.spec.object
                    if o.phys_half_z > 0:
                        half_h = o.phys_half_z
                    elif len(o.size) >= 3:
                        half_h = o.size[2]
                    else:
                        half_h = o.size[0]
                    if pos[2] < half_h + 0.05:
                        self._floor_hit_step = step
                elif self.spec.task_type == "stack_collapse":
                    o = self.spec.object
                    half_h = o.phys_half_z if o.phys_half_z > 0 else (o.size[2] if len(o.size) >= 3 else o.size[0])
                    if pos[2] < half_h + 0.05:
                        self._floor_hit_step = step
                elif self.spec.task_type == "sliding_object":
                    if pos[2] < 0.05:
                        self._floor_hit_step = step
                elif self.spec.task_type == "rolling_ball":
                    # Ball hits floor when centre drops below radius + small margin
                    radius = self.spec.object.size[0]
                    if pos[2] < radius + 0.03:
                        self._floor_hit_step = step
                elif self.spec.task_type == "shelf_slide":
                    o = self.spec.object
                    half_h = o.phys_half_z if o.phys_half_z > 0 else (o.size[2] if len(o.size) >= 3 else o.size[0])
                    if pos[2] < half_h + 0.04:
                        self._floor_hit_step = step
                elif self.spec.task_type == "door_swing":
                    # Event: door nearly closed (angle < 20° = panel nearly flush with wall)
                    q = self.obj.get_dofs_position()
                    if hasattr(q, "cpu"):
                        q = q.cpu()
                    angle = float(np.asarray(q).flatten()[0])
                    if angle < math.radians(20.0):
                        self._floor_hit_step = step
                elif self.spec.task_type == "thrown_object":
                    # Event: object crosses y=0 (room midpoint, flying toward observer)
                    if pos[1] < 0:
                        self._floor_hit_step = step
                elif self.spec.task_type == "pendulum_swing":
                    # Event: joint angle crosses 0 (bob at vertical, max speed toward observer)
                    q = self.obj.get_dofs_position()
                    if hasattr(q, "cpu"):
                        q = q.cpu()
                    if float(np.asarray(q).flatten()[0]) < 0.0:
                        self._floor_hit_step = step
                elif self.spec.task_type == "bouncing_object":
                    # Event: ball lands after its post-bounce arc (second floor contact).
                    # Ball starts at z≈radius with vel_z>0. Use 1.5× radius as the
                    # "risen" threshold to handle low-restitution balls (e.g. restitution=0.2)
                    # which only reach ~2.7× radius at peak.
                    radius = self.spec.object.size[0]
                    if not self._bounce_risen and pos[2] > radius * 1.5:
                        self._bounce_risen = True
                    if self._bounce_risen and pos[2] < radius + 0.04:
                        self._floor_hit_step = step
                elif self.spec.task_type == "ladder_slip":
                    # Event: ladder has fallen significantly.
                    # Box/cylinder: COM z drops below narrow half-extent (nearly horizontal).
                    # Mesh: origin is at base (z≈0); detect south-ward displacement of base.
                    o = self.spec.object
                    if o.morph == "mesh":
                        # pos is mesh origin. Detect southward slide > 0.25 m.
                        if pos[1] < self.spec.ladder.start_y - 0.25:
                            self._floor_hit_step = step
                    else:
                        half_x = o.phys_half_x if o.phys_half_x > 0 else o.size[0]
                        if pos[2] < half_x + 0.12:
                            self._floor_hit_step = step
                elif self.spec.task_type == "chain_reaction":
                    # Event: target object B falls below the table surface (off the table)
                    o = self.spec.object
                    half_h = o.phys_half_z if o.phys_half_z > 0 else (o.size[2] if len(o.size) >= 3 else o.size[0])
                    if pos[2] < half_h + 0.05:
                        self._floor_hit_step = step
                elif self.spec.task_type == "ceiling_drop":
                    o = self.spec.object
                    if o.phys_half_z > 0:
                        half_h = o.phys_half_z
                    elif len(o.size) >= 3:
                        half_h = o.size[2]
                    else:
                        half_h = o.size[0]
                    if pos[2] < half_h + 0.05:
                        self._floor_hit_step = step
                elif self.spec.task_type == "stair_tumble":
                    # Event: object has cleared all steps and reached the floor.
                    # The bottom step tread is at z = step_rise; floor clearance
                    # is when z drops below one step_rise, meaning object has left
                    # the staircase entirely.
                    o  = self.spec.object
                    st = self.spec.stair
                    half_h = o.phys_half_z if o.phys_half_z > 0 else (o.size[2] if len(o.size) >= 3 else o.size[0])
                    if pos[2] < half_h + 0.08:
                        self._floor_hit_step = step
                else:
                    if pos[2] < 0.05:
                        self._floor_hit_step = step

            # bouncing_object: stop 0.5 s after the ball lands to avoid simulating
            # the expensive floor-rolling phase for high-density balls.
            if (self.spec.task_type == "bouncing_object"
                    and self._floor_hit_step is not None
                    and step > self._floor_hit_step + int(0.5 / dt)):
                break

            if (step + 1) % int(1.0 / dt) == 0:
                t_sim = (step + 1) * dt
                print(f"  t={t_sim:.2f}s  obj=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"
                      f"  [wall {time.perf_counter()-t0:.1f}s]")
                if pos[2] < -0.5:
                    break

        videos = {}
        fps = 60
        for name, cam in self.cameras.items():
            vpath = out / f"video_{name}.mp4"
            cam.stop_recording(save_to_filename=str(vpath), fps=fps)
            videos[name] = str(vpath)

        metadata = self._build_metadata(videos)
        (out / "metadata.json").write_text(
            self.spec.model_copy(
                update={
                    "interception_point_3d": metadata.get("interception_point_3d"),
                    "time_to_floor_s": metadata.get("time_to_floor_s"),
                }
            ).to_json(),
            encoding="utf-8",
        )
        return metadata

    # ── Scene assembly ────────────────────────────────────────

    def _build_scene(self) -> None:
        s   = self.spec
        renderer, vis_opt = self._make_renderer()
        kw = dict(
            sim_options=gs.options.SimOptions(
                dt=s.dt, substeps=s.substeps,
                gravity=(0, 0, -9.81),
            ),
            rigid_options=gs.options.RigidOptions(enable_collision=True),
            renderer=renderer,
            show_viewer=False,
        )
        if vis_opt:
            kw["vis_options"] = vis_opt
        self.scene = gs.Scene(**kw)

        self._add_floor()
        self._add_room()

        if s.task_type == "furniture_tip":
            self._add_furniture_standing()
        elif s.task_type == "hanging_fall":
            self._add_hanging_object()
        elif s.task_type == "stack_collapse":
            self._add_table()
            self._add_stack()
        elif s.task_type == "sliding_object":
            self._add_ramp()
            self._add_sliding_object()
        elif s.task_type == "rolling_ball":
            self._add_table()
            self._add_rolling_ball()
        elif s.task_type == "shelf_slide":
            self._add_shelf()
            self._add_shelf_object()
        elif s.task_type == "door_swing":
            self._add_door()
        elif s.task_type == "thrown_object":
            self._add_thrown_object()
        elif s.task_type == "pendulum_swing":
            self._add_pendulum()
        elif s.task_type == "bouncing_object":
            self._add_bouncing_object()
        elif s.task_type == "ladder_slip":
            self._add_ladder_leaning()
        elif s.task_type == "chain_reaction":
            self._add_table()
            self._add_chain_reaction_objects()
        elif s.task_type == "ceiling_drop":
            self._add_ceiling_drop_object()
        elif s.task_type == "stair_tumble":
            self._add_stair_geometry()
            self._add_stair_tumble_object()
        else:
            self._add_table()
            self._add_object()

        self._add_cameras()
        self.scene.build()

    # ── Renderer ──────────────────────────────────────────────

    def _make_renderer(self):
        li = self.spec.lighting
        gs_lights = []
        for l in li.lights:
            if l.type == "directional":
                gs_lights.append(gs.options.vis.DirectionalLight(
                    dir=tuple(l.dir), color=tuple(l.color), intensity=l.intensity))
            elif l.type == "point":
                gs_lights.append(gs.options.vis.PointLight(
                    pos=tuple(l.pos), color=tuple(l.color), intensity=l.intensity))
            elif l.type == "ambient":
                gs_lights.append(gs.options.vis.AmbientLight(
                    color=tuple(l.color), intensity=l.intensity))

        vis_kw = dict(
            shadow=li.shadow,
            ambient_light=tuple(li.ambient_light),
            background_color=tuple(li.background_color),
        )
        if gs_lights:
            vis_kw["lights"] = gs_lights

        return gs.renderers.Rasterizer(), gs.options.VisOptions(**vis_kw)

    # ── Floor ─────────────────────────────────────────────────

    def _add_floor(self) -> None:
        r = self.spec.room
        diff  = _tex(TEX_ROOT / "floor" / "floor_diff.jpg")
        rough = _tex(TEX_ROOT / "floor" / "floor_rough.jpg", "linear")
        nor   = _tex(TEX_ROOT / "floor" / "floor_nor.jpg",   "linear")
        surf  = _textured(r.floor_color_rgb, r.floor_roughness, diff, rough, nor)
        self.scene.add_entity(
            gs.morphs.Plane(),
            surface=surf,
            material=gs.materials.Rigid(friction=0.5),
        )

    # ── Room walls + ceiling ──────────────────────────────────

    def _add_room(self) -> None:
        r   = self.spec.room
        W, D, H = r.width, r.depth, r.height
        TH  = r.wall_thickness
        ws  = _solid(r.wall_color_rgb, roughness=0.75)
        cs  = _solid(r.ceiling_color_rgb, roughness=0.90)
        mat = gs.materials.Rigid(friction=0.8)

        # North / East / West walls + ceiling (always solid)
        other_walls = [
            ((0,   D/2 + TH/2, H/2), (W+2*TH, TH,       H      )),  # north
            ((-W/2 - TH/2, 0, H/2), (TH,     D,        H      )),  # west
            (( W/2 + TH/2, 0, H/2), (TH,     D,        H      )),  # east
            ((0,   0,  H + TH/2),   (W+2*TH, D+2*TH,  TH     )),  # ceiling
        ]
        surfs = [ws, ws, ws, cs]
        for (p, sz), surf in zip(other_walls, surfs):
            self.scene.add_entity(
                gs.morphs.Box(pos=p, size=sz, fixed=True),
                surface=surf, material=mat,
            )

        # South wall — split to leave a doorway opening for door_swing scenes
        sy = -D/2 - TH/2   # wall centre Y
        if self.spec.task_type == "door_swing" and self.spec.door is not None:
            dw  = self.spec.door.width
            dh  = self.spec.door.height
            hx  = self.spec.door.hinge_x
            x0  = -W/2 - TH   # west end of south wall
            x1  = W/2  + TH   # east end of south wall

            # Left segment: west end → hinge
            lw = hx - x0
            if lw > 0.01:
                self.scene.add_entity(
                    gs.morphs.Box(pos=(x0 + lw/2, sy, H/2), size=(lw, TH, H), fixed=True),
                    surface=ws, material=mat,
                )
            # Right segment: hinge+door_width → east end
            rw = x1 - (hx + dw)
            if rw > 0.01:
                self.scene.add_entity(
                    gs.morphs.Box(pos=(hx + dw + rw/2, sy, H/2), size=(rw, TH, H), fixed=True),
                    surface=ws, material=mat,
                )
            # Lintel: above the doorway up to ceiling
            lintel_h = H - dh
            if lintel_h > 0.01:
                self.scene.add_entity(
                    gs.morphs.Box(pos=(hx + dw/2, sy, dh + lintel_h/2),
                                  size=(dw, TH, lintel_h), fixed=True),
                    surface=ws, material=mat,
                )
        else:
            self.scene.add_entity(
                gs.morphs.Box(pos=(0, sy, H/2), size=(W+2*TH, TH, H), fixed=True),
                surface=ws, material=mat,
            )

        # Window emissive panel
        wy = D/2 - 0.01 if r.window_wall == "north" else -D/2 + 0.01
        wx = r.window_pos_x
        wz = r.window_pos_z
        self.scene.add_entity(
            gs.morphs.Box(
                pos=(wx, wy, wz),
                size=(r.window_width, 0.01, r.window_height),
                fixed=True, collision=False,
            ),
            surface=gs.surfaces.Emission(emissive=tuple(r.window_emissive)),
        )

    # ── Table ─────────────────────────────────────────────────

    def _add_table(self) -> None:
        t   = self.spec.table
        tx, ty = t.pos_x, t.pos_y
        H, TH   = t.height, t.thickness
        W, D    = t.width, t.depth
        LEG, LH = t.leg_size, H - TH
        surf = _solid(t.color_rgb, roughness=t.roughness)
        mat  = gs.materials.Rigid(friction=t.friction)

        # Tabletop
        self.scene.add_entity(
            gs.morphs.Box(pos=(tx, ty, H - TH/2), size=(W, D, TH), fixed=True),
            surface=surf, material=mat,
        )
        # Legs
        inset = 0.05
        hx = W/2 - inset - LEG/2
        hy = D/2 - inset - LEG/2
        for lx in (tx+hx, tx-hx):
            for ly in (ty+hy, ty-hy):
                self.scene.add_entity(
                    gs.morphs.Box(pos=(lx, ly, LH/2), size=(LEG, LEG, LH), fixed=True),
                    surface=surf, material=mat,
                )

    # ── Dropped object ────────────────────────────────────────

    def _add_object(self) -> None:
        o    = self.spec.object
        d    = self.spec.drop
        t    = self.spec.table
        surf = _solid(o.color_rgb, roughness=o.roughness, ior=o.ior)
        mat  = gs.materials.Rigid(
            rho=o.density,
            friction=o.friction,
            coup_restitution=o.restitution,
        )

        # Compute z so that the object's bottom rests on the table surface
        if o.morph == "mesh":
            obj_z = t.height - o.bottom_z_offset   # bottom_z_offset is negative for below-origin
        elif o.morph == "sphere":
            obj_z = t.height + o.size[0]            # radius
        elif o.morph == "cylinder":
            obj_z = t.height + o.size[2]            # half-height
        elif o.morph == "box":
            obj_z = t.height + o.size[2]            # half-height z
        else:
            obj_z = t.height + 0.05

        pos    = (d.start_x, d.start_y, obj_z)
        euler  = (0.0, 0.0, d.euler_z) if d.euler_z != 0 else None

        if o.morph == "mesh":
            mesh_path = (MESHES_DIR / o.mesh_path).resolve()
            # size=[scale] → single float = uniform mesh scale; size=[x,y,z] → use 1.0
            mesh_scale = o.size[0] if len(o.size) == 1 else 1.0
            morph = gs.morphs.Mesh(file=str(mesh_path), pos=pos,
                                   euler=euler, scale=mesh_scale)
        elif o.morph == "sphere":
            morph = gs.morphs.Sphere(pos=pos, radius=o.size[0])
        elif o.morph == "cylinder":
            morph = gs.morphs.Cylinder(pos=pos, radius=o.size[0],
                                       height=o.size[2]*2, euler=euler)
        elif o.morph == "box":
            morph = gs.morphs.Box(pos=pos, size=tuple(o.size), euler=euler)
        else:
            raise ValueError(f"Unknown morph: {o.morph}")

        self.obj = self.scene.add_entity(morph, surface=surf, material=mat)

    # ── Standing furniture (furniture_tip task) ───────────────

    def _add_furniture_standing(self) -> None:
        """Place the furniture upright on the floor, tilted past vertical toward -Y."""
        o    = self.spec.object
        tip  = self.spec.tip
        surf = _solid(o.color_rgb, roughness=o.roughness, ior=o.ior)
        mat  = gs.materials.Rigid(
            rho=o.density,
            friction=o.friction,
            coup_restitution=o.restitution,
        )

        # Centre of furniture is at half-height above the floor
        if o.morph in ("box", "cylinder"):
            obj_z = o.size[2]   # half-height = distance from floor to COM
        else:
            obj_z = max(o.size)

        pos   = (tip.start_x, tip.start_y, obj_z)
        # Tilt toward -Y (observer side) = positive rotation about +X axis
        euler = (tip.tilt_angle_deg, 0.0, tip.euler_z)

        if o.morph == "box":
            morph = gs.morphs.Box(pos=pos, size=tuple(o.size), euler=euler)
        elif o.morph == "cylinder":
            morph = gs.morphs.Cylinder(
                pos=pos,
                radius=o.size[0],
                height=o.size[2] * 2,
                euler=euler,
            )
        else:
            raise ValueError(f"Unsupported morph for furniture_tip: {o.morph}")

        self.obj = self.scene.add_entity(morph, surface=surf, material=mat)

    # ── Hanging object (hanging_fall task) ───────────────────

    def _add_hanging_object(self) -> None:
        """
        Place a wall- or ceiling-mounted object at its hanging position.

        Wall-mount (wall_north):
          The object hangs flat against the north wall (y = +depth/2).
          Its centre is offset inward by its own half-depth so the back
          face sits flush against the wall surface.
          A small positive tilt_deg leans it away from the wall.

        Ceiling-mount:
          The object hangs below the ceiling at attach_z (precomputed to
          account for cable slack). No wall proximity needed.

        In both cases the initial angular velocity (applied after build())
        simulates the mount suddenly snapping.
        """
        o   = self.spec.object
        h   = self.spec.hanging
        r   = self.spec.room
        surf = _solid(o.color_rgb, roughness=o.roughness, ior=o.ior)
        mat  = gs.materials.Rigid(
            rho=o.density,
            friction=o.friction,
            coup_restitution=o.restitution,
        )

        if h.attachment == "wall_north":
            # Half-depth of the object facing the wall
            if o.morph == "box":
                half_depth = o.size[1]
            elif o.morph in ("cylinder", "sphere"):
                half_depth = o.size[0]   # radius
            elif o.morph == "mesh":
                half_depth = 0.03        # thin default; normalised meshes start near 0
            else:
                half_depth = 0.03

            wall_y = r.depth / 2
            pos    = (h.pos_x, wall_y - half_depth, h.attach_z)
            # Tilt forward (away from wall = toward -Y) around the X axis
            euler  = (h.tilt_deg, 0.0, h.euler_z)
        else:  # ceiling
            pos   = (h.pos_x, h.pos_y, h.attach_z)
            euler = (h.tilt_deg, 0.0, h.euler_z)

        if o.morph == "mesh":
            mesh_path  = (MESHES_DIR / o.mesh_path).resolve()
            mesh_scale = o.size[0] if len(o.size) == 1 else 1.0
            morph = gs.morphs.Mesh(file=str(mesh_path), pos=pos, euler=euler,
                                   scale=mesh_scale)
        elif o.morph == "sphere":
            morph = gs.morphs.Sphere(pos=pos, radius=o.size[0])
        elif o.morph == "cylinder":
            morph = gs.morphs.Cylinder(pos=pos, radius=o.size[0],
                                       height=o.size[2] * 2, euler=euler)
        elif o.morph == "box":
            morph = gs.morphs.Box(pos=pos, size=tuple(o.size), euler=euler)
        else:
            raise ValueError(f"Unknown morph for hanging_fall: {o.morph}")

        self.obj = self.scene.add_entity(morph, surface=surf, material=mat)

    # ── Ramp (sliding_object task) ───────────────────────────

    def _add_ramp(self) -> None:
        """Place a fixed inclined box (ramp) on the floor.

        Ramp geometry:
          lower end at (pos_x, pos_y, ≈0), rises toward +Y at angle_deg.
          Box centre:  (pos_x, pos_y + L*cos(a), L*sin(a) + t*cos(a))
          Euler:       (-angle_deg, 0, euler_z)  — positive Y side is higher
        """
        r  = self.spec.ramp
        a  = math.radians(r.angle_deg)
        t  = 0.025          # board half-thickness
        hL = r.length / 2
        hW = r.width  / 2

        cx = r.pos_x
        cy = r.pos_y + hL * math.cos(a)
        cz = hL * math.sin(a) + t * math.cos(a)

        self.scene.add_entity(
            gs.morphs.Box(
                pos=(cx, cy, cz),
                size=(hW, hL, t),
                euler=(r.angle_deg, 0.0, r.euler_z),   # +angle: +Y end is HIGH
                fixed=True,
            ),
            surface=_solid([0.55, 0.42, 0.24], roughness=0.78),  # wood plank
            material=gs.materials.Rigid(friction=0.45),
        )

    def _add_sliding_object(self) -> None:
        """Place the object on the ramp at 80% of the way up, resting on the surface."""
        o    = self.spec.object
        r    = self.spec.ramp
        surf = _solid(o.color_rgb, roughness=o.roughness, ior=o.ior)
        mat  = gs.materials.Rigid(
            rho=o.density,
            friction=o.friction,
            coup_restitution=o.restitution,
        )

        a  = math.radians(r.angle_deg)
        t  = 0.025
        d  = r.length * 0.80

        # Physical half-height perpendicular to ramp surface
        if o.morph == "mesh" and o.phys_half_z > 0:
            hz = o.phys_half_z
        elif o.morph in ("box", "cylinder"):
            hz = o.size[2]
        elif o.morph == "sphere":
            hz = o.size[0]
        else:
            hz = 0.05

        # Surface normal with euler=(+angle_deg): local +Z → world (0, -sin(a), cos(a))
        # Object centre = surface_contact + hz * normal
        obj_x = r.pos_x
        obj_y = r.pos_y + d * math.cos(a) - hz * math.sin(a)
        obj_z = t + d * math.sin(a) + hz * math.cos(a)
        pos   = (obj_x, obj_y, obj_z)
        # Tilt object to lie flush on ramp (+angle_deg matches ramp orientation)
        euler = (r.angle_deg, 0.0, r.euler_z)

        if o.morph == "mesh":
            mesh_path  = (MESHES_DIR / o.mesh_path).resolve()
            mesh_scale = o.size[0] if len(o.size) == 1 else 1.0
            morph = gs.morphs.Mesh(file=str(mesh_path), pos=pos, euler=euler,
                                   scale=mesh_scale)
        elif o.morph == "sphere":
            morph = gs.morphs.Sphere(pos=pos, radius=o.size[0])
        elif o.morph == "cylinder":
            morph = gs.morphs.Cylinder(pos=pos, radius=o.size[0],
                                       height=o.size[2] * 2, euler=euler)
        elif o.morph == "box":
            morph = gs.morphs.Box(pos=pos, size=tuple(o.size), euler=euler)
        else:
            raise ValueError(f"Unknown morph for sliding_object: {o.morph}")

        self.obj = self.scene.add_entity(morph, surface=surf, material=mat)

    # ── Stack (stack_collapse task) ───────────────────────────

    def _add_stack(self) -> None:
        """Place N identical objects stacked on the table, all given the same
        initial tilt toward -Y. The top item is stored in self.obj so that
        _obj_pos() tracks it for floor-hit detection and metadata.

        Items are indexed 0 (bottom) to n-1 (top). Each item's centre Z:
            z_i = table.height + hz + 2 * hz * i
        """
        o     = self.spec.object
        t     = self.spec.table
        stack = self.spec.stack
        surf  = _solid(o.color_rgb, roughness=o.roughness, ior=o.ior)
        mat   = gs.materials.Rigid(
            rho=o.density,
            friction=o.friction,
            coup_restitution=o.restitution,
        )

        # Physical half-height of one item
        if o.morph == "box":
            hz = o.size[2]
        elif o.morph == "cylinder":
            hz = o.size[2]
        elif o.morph == "sphere":
            hz = o.size[0]
        else:
            hz = o.phys_half_z if o.phys_half_z > 0 else 0.05

        euler = (stack.tilt_angle_deg, 0.0, stack.euler_z)

        self.stack_objs = []
        for i in range(stack.n_items):
            item_z = t.height + hz + 2 * hz * i
            pos    = (stack.start_x, stack.start_y, item_z)

            if o.morph == "box":
                morph = gs.morphs.Box(pos=pos, size=tuple(o.size), euler=euler)
            elif o.morph == "cylinder":
                morph = gs.morphs.Cylinder(
                    pos=pos, radius=o.size[0], height=o.size[2] * 2, euler=euler,
                )
            elif o.morph == "sphere":
                morph = gs.morphs.Sphere(pos=pos, radius=o.size[0])
            elif o.morph == "mesh":
                mesh_path  = (MESHES_DIR / o.mesh_path).resolve()
                mesh_scale = o.size[0] if len(o.size) == 1 else 1.0
                morph = gs.morphs.Mesh(file=str(mesh_path), pos=pos,
                                       euler=euler, scale=mesh_scale)
            else:
                raise ValueError(f"Unsupported morph for stack_collapse: {o.morph}")

            entity = self.scene.add_entity(morph, surface=surf, material=mat)
            self.stack_objs.append(entity)

        # self.obj tracks the TOP item for _obj_pos() and floor-hit detection
        self.obj = self.stack_objs[-1]

    def _apply_stack_velocity(self) -> None:
        """Apply initial angular velocity to every stack item to trigger collapse.
        Top item gets full angular_vel; each lower item gets proportionally less
        (the lean at the bottom is steadied by the table friction).
        """
        stack = self.spec.stack
        if stack.angular_vel == 0:
            return
        try:
            import torch
            n_items = len(self.stack_objs)
            for i, entity in enumerate(self.stack_objs):
                # Scale angular velocity from 0.3× (bottom) to 1.0× (top)
                scale = 0.3 + 0.7 * (i / max(n_items - 1, 1))
                n   = entity.n_dofs
                vel = torch.zeros(1, n, dtype=torch.float32)
                vel[0, 3] = stack.angular_vel * scale   # wx > 0 → tip toward -Y
                entity.set_dofs_velocity(vel)
        except Exception as e:
            print(f"[scene_builder] stack velocity not applied: {e}", file=sys.stderr)

    # ── Cameras ───────────────────────────────────────────────

    def _add_cameras(self) -> None:
        for c in self.spec.cameras:
            self.cameras[c.name] = self.scene.add_camera(
                res=(1280, 720),
                pos=tuple(c.pos),
                lookat=tuple(c.lookat),
                fov=c.fov,
                GUI=False,
            )

    # ── Velocity / initial conditions ─────────────────────────

    def _apply_velocity(self) -> None:
        if self.spec.task_type == "furniture_tip":
            self._apply_tip_velocity()
        elif self.spec.task_type == "hanging_fall":
            self._apply_hanging_velocity()
        elif self.spec.task_type == "stack_collapse":
            self._apply_stack_velocity()
        elif self.spec.task_type == "rolling_ball":
            self._apply_roll_velocity()
        elif self.spec.task_type == "shelf_slide":
            self._apply_shelf_velocity()
        elif self.spec.task_type == "door_swing":
            self._apply_door_velocity()
        elif self.spec.task_type == "thrown_object":
            self._apply_thrown_velocity()
        elif self.spec.task_type == "pendulum_swing":
            self._apply_pendulum_velocity()
        elif self.spec.task_type == "bouncing_object":
            self._apply_bounce_velocity()
        elif self.spec.task_type == "ladder_slip":
            self._apply_ladder_velocity()
        elif self.spec.task_type == "chain_reaction":
            self._apply_chain_velocity()
        elif self.spec.task_type in ("sliding_object", "ceiling_drop"):
            pass   # gravity alone is sufficient; no extra velocity needed
        elif self.spec.task_type == "stair_tumble":
            self._apply_stair_velocity()
        else:
            self._apply_drop_velocity()

    def _apply_drop_velocity(self) -> None:
        d = self.spec.drop
        if d.vel_x == 0 and d.vel_y == 0 and d.vel_z == 0:
            return
        try:
            import torch
            n   = self.obj.n_dofs
            vel = torch.zeros(1, n, dtype=torch.float32)
            vel[0, 0] = d.vel_x
            vel[0, 1] = d.vel_y
            vel[0, 2] = d.vel_z
            self.obj.set_dofs_velocity(vel)
        except Exception as e:
            print(f"[scene_builder] drop velocity not applied: {e}", file=sys.stderr)

    def _apply_hanging_velocity(self) -> None:
        """
        Apply initial angular velocity to simulate the mount suddenly snapping.

        wall_north objects tip away from the north wall (toward -Y):
          rotation about +X axis → wx > 0  (same convention as furniture_tip)
        ceiling objects get a small random angular kick for realism.
        """
        h = self.spec.hanging
        if h.angular_vel == 0:
            return
        try:
            import torch
            n   = self.obj.n_dofs
            vel = torch.zeros(1, n, dtype=torch.float32)
            # DOF layout: [vx, vy, vz, wx, wy, wz]
            if h.attachment == "wall_north":
                vel[0, 3] = h.angular_vel   # wx > 0 → tip toward -Y
            else:                            # ceiling: small random tilt
                vel[0, 3] = h.angular_vel * 0.5
                vel[0, 4] = h.angular_vel * 0.5
            self.obj.set_dofs_velocity(vel)
        except Exception as e:
            print(f"[scene_builder] hanging velocity not applied: {e}", file=sys.stderr)

    def _apply_tip_velocity(self) -> None:
        """Apply initial angular velocity about +X to start the furniture tipping toward -Y."""
        tip = self.spec.tip
        if tip.angular_vel == 0:
            return
        try:
            import torch
            n   = self.obj.n_dofs
            vel = torch.zeros(1, n, dtype=torch.float32)
            # DOF layout for free rigid body: [vx, vy, vz, wx, wy, wz]
            # wx (DOF[3]) > 0 → tip toward -Y (toward observer)
            vel[0, 3] = tip.angular_vel
            self.obj.set_dofs_velocity(vel)
        except Exception as e:
            print(f"[scene_builder] tip velocity not applied: {e}", file=sys.stderr)

    # ── Rolling ball (rolling_ball task) ─────────────────────

    def _add_rolling_ball(self) -> None:
        """Place the ball on the table surface at its start position.

        Velocity is applied after scene.build() via _apply_roll_velocity().
        The ball has no initial euler rotation — spheres are orientation-agnostic.
        """
        o    = self.spec.object
        t    = self.spec.table
        roll = self.spec.roll
        surf = _solid(o.color_rgb, roughness=o.roughness, ior=o.ior)
        mat  = gs.materials.Rigid(
            rho=o.density,
            friction=o.friction,
            coup_restitution=o.restitution,
        )
        radius = o.size[0]
        pos    = (roll.start_x, roll.start_y, t.height + radius)
        self.obj = self.scene.add_entity(
            gs.morphs.Sphere(pos=pos, radius=radius),
            surface=surf, material=mat,
        )

    def _apply_roll_velocity(self) -> None:
        """Apply initial linear velocity and rolling angular velocity to the ball.

        For rolling without slipping in the +X direction:
          linear:  v_x = roll.vel_x
          angular: w_y = roll.angular_vel_y  (positive wy → rolls in +X)
        """
        roll = self.spec.roll
        if roll.vel_x == 0 and roll.vel_y == 0:
            return
        try:
            import torch
            n   = self.obj.n_dofs
            vel = torch.zeros(1, n, dtype=torch.float32)
            # DOF layout: [vx, vy, vz, wx, wy, wz]
            vel[0, 0] = roll.vel_x
            vel[0, 1] = roll.vel_y
            vel[0, 4] = roll.angular_vel_y   # wy > 0 → rolls in +X
            self.obj.set_dofs_velocity(vel)
        except Exception as e:
            print(f"[scene_builder] roll velocity not applied: {e}", file=sys.stderr)

    # ── Shelf (shelf_slide task) ──────────────────────────────

    def _add_shelf(self) -> None:
        """Add a fixed wall-mounted shelf to the interior face of the north wall.

        Shelf geometry:
          - Main board: full (width × depth × thickness) box centered at
            (pos_x, wall_y − depth/2, height − thickness/2).
          - Two support brackets (thin boxes) underneath the board ends.
        """
        s     = self.spec.shelf
        r     = self.spec.room
        wall_y = r.depth / 2

        surf = _solid([0.55, 0.42, 0.24], roughness=0.78)   # natural wood
        mat  = gs.materials.Rigid(friction=0.50)

        # Main shelf board
        self.scene.add_entity(
            gs.morphs.Box(
                pos=(s.pos_x, wall_y - s.depth / 2, s.height - s.thickness / 2),
                size=(s.width, s.depth, s.thickness),
                fixed=True,
            ),
            surface=surf, material=mat,
        )

        # Support brackets — one near each end of the shelf
        br_w = 0.025   # bracket width (full)
        br_d = s.depth
        br_h = 0.12    # bracket height (full)
        for bx_offset in (s.width / 2 - br_w / 2, -(s.width / 2 - br_w / 2)):
            self.scene.add_entity(
                gs.morphs.Box(
                    pos=(s.pos_x + bx_offset,
                         wall_y - s.depth / 2,
                         s.height - s.thickness - br_h / 2),
                    size=(br_w, br_d, br_h),
                    fixed=True,
                ),
                surface=surf, material=mat,
            )

    def _add_shelf_object(self) -> None:
        """Place the object on the shelf near its front edge.

        Object y = wall_y − depth × 0.70 (30% shelf depth from front edge).
        Object z = shelf.height + physical half-height.
        """
        o     = self.spec.object
        s     = self.spec.shelf
        r     = self.spec.room
        surf  = _solid(o.color_rgb, roughness=o.roughness, ior=o.ior)
        mat   = gs.materials.Rigid(
            rho=o.density,
            friction=o.friction,
            coup_restitution=o.restitution,
        )

        wall_y = r.depth / 2

        if o.morph == "cylinder":
            hz = o.size[2]
        elif o.morph == "box":
            hz = o.size[2]
        elif o.morph == "sphere":
            hz = o.size[0]
        elif o.morph == "mesh":
            hz = o.phys_half_z if o.phys_half_z > 0 else 0.05
        else:
            hz = 0.05

        obj_y = wall_y - s.depth * 0.70
        obj_z = s.height + hz
        pos   = (s.pos_x, obj_y, obj_z)

        if o.morph == "cylinder":
            morph = gs.morphs.Cylinder(pos=pos, radius=o.size[0], height=o.size[2] * 2)
        elif o.morph == "box":
            morph = gs.morphs.Box(pos=pos, size=tuple(o.size))
        elif o.morph == "sphere":
            morph = gs.morphs.Sphere(pos=pos, radius=o.size[0])
        elif o.morph == "mesh":
            mesh_path  = (MESHES_DIR / o.mesh_path).resolve()
            mesh_scale = o.size[0] if len(o.size) == 1 else 1.0
            morph = gs.morphs.Mesh(file=str(mesh_path), pos=pos, scale=mesh_scale)
        else:
            raise ValueError(f"Unknown morph for shelf_slide: {o.morph}")

        self.obj = self.scene.add_entity(morph, surface=surf, material=mat)

    def _apply_shelf_velocity(self) -> None:
        """Apply initial velocity toward the observer to trigger the shelf slide-off."""
        shelf = self.spec.shelf
        if shelf.vel_y == 0:
            return
        try:
            import torch
            n   = self.obj.n_dofs
            vel = torch.zeros(1, n, dtype=torch.float32)
            vel[0, 1] = shelf.vel_y   # vy < 0 → toward observer (-Y)
            self.obj.set_dofs_velocity(vel)
        except Exception as e:
            print(f"[scene_builder] shelf velocity not applied: {e}", file=sys.stderr)

    # ── Door swing (door_swing task) ──────────────────────────

    def _make_door_mjcf(self) -> str:
        """Generate a minimal MJCF XML for a door with revolute Z hinge.

        hinge_base is a massless body fixed to the world at (hinge_x, door_y, 0.01).
        door_panel is a box child with a hinge joint — 1 DOF total.
        Genesis merges the massless base into the world, so the entity ends up
        as a single-link revolute body at the hinge position.
        """
        s = self.spec.door
        o = self.spec.object
        hw = s.width     / 2   # half-width
        ht = s.thickness / 2   # half-thickness
        hh = s.height    / 2   # half-height
        r, g, b = o.color_rgb
        return (
            f'<mujoco>\n'
            f'  <worldbody>\n'
            f'    <body name="hinge_base" pos="{s.hinge_x:.4f} {s.door_y:.4f} 0.01">\n'
            f'      <body name="door_panel">\n'
            f'        <joint name="door_hinge" type="hinge" axis="0 0 1"/>\n'
            f'        <geom type="box"\n'
            f'              size="{hw:.4f} {ht:.4f} {hh:.4f}"\n'
            f'              pos="{hw:.4f} 0.0 {hh:.4f}"\n'
            f'              density="{o.density:.1f}"\n'
            f'              friction="{o.friction:.3f} 0.005 0.0001"\n'
            f'              rgba="{r:.3f} {g:.3f} {b:.3f} 1.0"/>\n'
            f'      </body>\n'
            f'    </body>\n'
            f'  </worldbody>\n'
            f'</mujoco>'
        )

    def _add_door(self) -> None:
        """Load a revolute-joint door via MJCF.

        The MJCF is written to a temp file and loaded with gs.morphs.MJCF.
        Initial joint angle and angular velocity are set in _apply_door_velocity()
        after scene.build().
        """
        mjcf_xml  = self._make_door_mjcf()
        tmp_dir   = Path(tempfile.mkdtemp(prefix="gs_door_"))
        mjcf_path = tmp_dir / "door.xml"
        mjcf_path.write_text(mjcf_xml, encoding="utf-8")
        self._door_tmp_dir = tmp_dir   # keep alive until scene is built

        self.obj = self.scene.add_entity(
            gs.morphs.MJCF(file=str(mjcf_path)),
        )

    def _apply_door_velocity(self) -> None:
        """Set initial revolute joint angle and angular velocity for the door."""
        import torch
        door = self.spec.door

        # Clean up the temp MJCF file now that the scene is built
        if hasattr(self, "_door_tmp_dir"):
            import shutil
            try:
                shutil.rmtree(self._door_tmp_dir, ignore_errors=True)
            except Exception:
                pass

        n = self.obj.n_dofs   # should be 1 (one revolute joint)
        angle_rad = math.radians(door.initial_angle_deg)

        pos_t = torch.tensor([angle_rad], dtype=torch.float32)
        self.obj.set_dofs_position(pos_t)

        if door.angular_vel != 0:
            vel_t = torch.zeros(1, n, dtype=torch.float32)
            vel_t[0, 0] = door.angular_vel
            self.obj.set_dofs_velocity(vel_t)

    # ── Thrown object (thrown_object task) ───────────────────────

    def _add_thrown_object(self) -> None:
        """Spawn the thrown object at its launch position (mid-air, no table)."""
        o    = self.spec.object
        t    = self.spec.thrown
        surf = _solid(o.color_rgb, roughness=o.roughness, ior=o.ior)
        mat  = gs.materials.Rigid(
            rho=o.density,
            friction=o.friction,
            coup_restitution=o.restitution,
        )
        pos = (t.launch_x, t.launch_y, t.launch_z)
        if o.morph == "sphere":
            morph = gs.morphs.Sphere(pos=pos, radius=o.size[0])
        elif o.morph == "box":
            morph = gs.morphs.Box(pos=pos, size=tuple(o.size))
        elif o.morph == "cylinder":
            morph = gs.morphs.Cylinder(pos=pos, radius=o.size[0],
                                       height=o.size[2] * 2)
        elif o.morph == "mesh":
            mesh_path  = (MESHES_DIR / o.mesh_path).resolve()
            mesh_scale = o.size[0] if len(o.size) == 1 else 1.0
            morph = gs.morphs.Mesh(file=str(mesh_path), pos=pos, scale=mesh_scale)
        else:
            raise ValueError(f"Unknown morph for thrown_object: {o.morph}")
        self.obj = self.scene.add_entity(morph, surface=surf, material=mat)

    def _apply_thrown_velocity(self) -> None:
        """Apply initial linear velocity to the thrown object."""
        import torch
        t   = self.spec.thrown
        n   = self.obj.n_dofs
        vel = torch.zeros(1, n, dtype=torch.float32)
        vel[0, 0] = t.vel_x
        vel[0, 1] = t.vel_y
        vel[0, 2] = t.vel_z
        self.obj.set_dofs_velocity(vel)

    # ── Ladder slip (ladder_slip task) ────────────────────────

    def _add_ladder_leaning(self) -> None:
        """Place a ladder leaning against the north wall, ready to slip.

        Wall-contact geometry:
          euler = (-lean_deg, 0, euler_z)   negative euler_x → top tilts toward +Y (north wall)
          start_y = y-base of ladder on floor (Randomizer computes this from lean_deg + room)

        Box morph: pos = (base_x, start_y + hz*sin_d, hz*cos_d)  — COM placed correctly
        Mesh morph: pos = (base_x, start_y, 0)                   — mesh origin at base on floor

        The north wall is a fixed rigid body in the scene, so the top of the ladder
        will be in contact with it at t=0 and slides down as the base slides south.
        """
        import math
        o      = self.spec.object
        ladder = self.spec.ladder
        surf   = _solid(o.color_rgb, roughness=o.roughness, ior=o.ior)
        mat    = gs.materials.Rigid(
            rho=o.density,
            friction=o.friction,
            coup_restitution=o.restitution,
        )

        sin_d = math.sin(math.radians(ladder.lean_deg))
        cos_d = math.cos(math.radians(ladder.lean_deg))
        # Negative euler_x: local +Z (ladder top) tilts toward +Y (north wall)
        euler = (-ladder.lean_deg, 0.0, ladder.euler_z)

        if o.morph == "box":
            hz  = o.size[2]   # half-height
            pos = (ladder.base_x, ladder.start_y + hz * sin_d, hz * cos_d)
            morph = gs.morphs.Box(pos=pos, size=tuple(o.size), euler=euler)
        elif o.morph == "mesh":
            # If the mesh has a built-in lean (mesh_base_y != 0), the mesh local
            # origin is NOT at the floor centroid.  Compute pos_z to shift the
            # origin so that the actual mesh base (at local y=mesh_base_y, z=0)
            # lands exactly on the floor after rotation Rx(-lean_deg):
            #   z_world(base) = pos_z - mesh_base_y * sin_d  → set = 0
            #   ⟹ pos_z = mesh_base_y * sin_d  (negative for pre-leaned meshes)
            pos_z = o.mesh_base_y * sin_d if o.mesh_base_y != 0.0 else 0.0
            pos = (ladder.base_x, ladder.start_y, pos_z)
            mesh_scale = o.size[0] if len(o.size) == 1 else 1.0
            mesh_path  = (MESHES_DIR / o.mesh_path).resolve()
            morph = gs.morphs.Mesh(file=str(mesh_path), pos=pos, euler=euler,
                                   scale=mesh_scale)
        elif o.morph == "cylinder":
            hz  = o.size[2]
            pos = (ladder.base_x, ladder.start_y + hz * sin_d, hz * cos_d)
            morph = gs.morphs.Cylinder(pos=pos, radius=o.size[0],
                                       height=hz * 2, euler=euler)
        else:
            raise ValueError(f"Unsupported morph for ladder_slip: {o.morph}")

        self.obj = self.scene.add_entity(morph, surface=surf, material=mat)

    def _apply_ladder_velocity(self) -> None:
        """Apply initial angular velocity to tip the ladder toward the observer (-Y)."""
        ladder = self.spec.ladder
        if ladder.angular_vel == 0:
            return
        try:
            import torch
            n   = self.obj.n_dofs
            vel = torch.zeros(1, n, dtype=torch.float32)
            vel[0, 3] = ladder.angular_vel   # wx > 0 → tip toward -Y (observer)
            self.obj.set_dofs_velocity(vel)
        except Exception as e:
            print(f"[scene_builder] ladder velocity not applied: {e}", file=sys.stderr)

    # ── Chain reaction (chain_reaction task) ──────────────────

    def _add_chain_reaction_objects(self) -> None:
        """Add trigger A (book-like box) and target B (main object) to the table.

        A is north of B, given initial velocity southward.
        A hits B → B slides off the south table edge → B falls toward observer.
        """
        o      = self.spec.object
        t      = self.spec.table
        ch     = self.spec.chain
        surf_b = _solid(o.color_rgb, roughness=o.roughness, ior=o.ior)
        mat_b  = gs.materials.Rigid(
            rho=o.density,
            friction=o.friction,
            coup_restitution=o.restitution,
        )

        # ── Target B (main tracked object) ───────────────────
        if o.morph == "mesh":
            obj_z = t.height - o.bottom_z_offset
            hz    = o.phys_half_z if o.phys_half_z > 0 else 0.05
        elif o.morph == "sphere":
            obj_z = t.height + o.size[0]
            hz    = o.size[0]
        elif o.morph == "cylinder":
            obj_z = t.height + o.size[2]
            hz    = o.size[2]
        else:  # box
            obj_z = t.height + o.size[2]
            hz    = o.size[2]
        _ = hz  # unused locally, tracked by floor-hit detection

        pos_b  = (ch.target_x, ch.target_y, obj_z)
        euler_b = (0.0, 0.0, ch.euler_z) if ch.euler_z != 0 else None

        if o.morph == "mesh":
            mesh_path  = (MESHES_DIR / o.mesh_path).resolve()
            mesh_scale = o.size[0] if len(o.size) == 1 else 1.0
            morph_b = gs.morphs.Mesh(file=str(mesh_path), pos=pos_b,
                                     euler=euler_b, scale=mesh_scale)
        elif o.morph == "sphere":
            morph_b = gs.morphs.Sphere(pos=pos_b, radius=o.size[0])
        elif o.morph == "cylinder":
            morph_b = gs.morphs.Cylinder(pos=pos_b, radius=o.size[0],
                                         height=o.size[2] * 2, euler=euler_b)
        else:
            morph_b = gs.morphs.Box(pos=pos_b, size=tuple(o.size), euler=euler_b)

        self.obj = self.scene.add_entity(morph_b, surface=surf_b, material=mat_b)

        # ── Trigger A (book-like box) ─────────────────────────
        trigger_hz = ch.trigger_size[2]
        pos_a      = (ch.trigger_x, ch.trigger_y, t.height + trigger_hz)
        surf_a     = _solid(ch.trigger_color, roughness=0.80)
        mat_a      = gs.materials.Rigid(rho=ch.trigger_density, friction=0.50,
                                        coup_restitution=0.05)
        self._chain_trigger = self.scene.add_entity(
            gs.morphs.Box(pos=pos_a, size=tuple(ch.trigger_size)),
            surface=surf_a, material=mat_a,
        )

    def _apply_chain_velocity(self) -> None:
        """Apply initial southward velocity to the trigger A."""
        ch = self.spec.chain
        if ch.trigger_vel_y == 0:
            return
        try:
            import torch
            n   = self._chain_trigger.n_dofs
            vel = torch.zeros(1, n, dtype=torch.float32)
            vel[0, 1] = ch.trigger_vel_y   # vy < 0 → toward observer (-Y)
            self._chain_trigger.set_dofs_velocity(vel)
        except Exception as e:
            print(f"[scene_builder] chain trigger velocity not applied: {e}", file=sys.stderr)

    # ── Ceiling drop (ceiling_drop task) ─────────────────────

    def _add_ceiling_drop_object(self) -> None:
        """Spawn the object just below the ceiling and release it under gravity.

        spawn_z = room.height − obj_half_height − 0.05 m clearance.
        No initial velocity; pure free-fall to the floor.
        """
        o  = self.spec.object
        cd = self.spec.ceiling_drop
        r  = self.spec.room
        surf = _solid(o.color_rgb, roughness=o.roughness, ior=o.ior)
        mat  = gs.materials.Rigid(
            rho=o.density,
            friction=o.friction,
            coup_restitution=o.restitution,
        )

        if o.morph == "mesh":
            hz = o.phys_half_z if o.phys_half_z > 0 else 0.05
        elif o.morph in ("box", "cylinder"):
            hz = o.size[2]
        elif o.morph == "sphere":
            hz = o.size[0]
        else:
            hz = 0.05

        spawn_z = r.height - hz - 0.05
        pos     = (cd.spawn_x, cd.spawn_y, spawn_z)
        euler   = (0.0, 0.0, cd.euler_z) if cd.euler_z != 0 else None

        if o.morph == "mesh":
            mesh_path  = (MESHES_DIR / o.mesh_path).resolve()
            mesh_scale = o.size[0] if len(o.size) == 1 else 1.0
            morph = gs.morphs.Mesh(file=str(mesh_path), pos=pos, euler=euler,
                                   scale=mesh_scale)
        elif o.morph == "sphere":
            morph = gs.morphs.Sphere(pos=pos, radius=o.size[0])
        elif o.morph == "cylinder":
            morph = gs.morphs.Cylinder(pos=pos, radius=o.size[0],
                                       height=o.size[2] * 2, euler=euler)
        elif o.morph == "box":
            morph = gs.morphs.Box(pos=pos, size=tuple(o.size), euler=euler)
        else:
            raise ValueError(f"Unknown morph for ceiling_drop: {o.morph}")

        self.obj = self.scene.add_entity(morph, surface=surf, material=mat)

    # ── Stair tumble (stair_tumble task) ─────────────────────

    def _add_stair_geometry(self) -> None:
        """Add N fixed step-blocks forming a staircase against the north wall.

        Each step block is a solid box that extends from the floor (z=0) up to the
        top surface of that step — the classic "staircase cut from a solid" approach.
        This is physically correct for collision and visually clean.

        Step k (0-indexed from bottom):
            size  = (stair_width, step_run, (k+1)*step_rise)
            pos   = (stair_x,
                     stair_start_y + k*step_run + step_run/2,
                     (k+1)*step_rise / 2)
        """
        st   = self.spec.stair
        surf = _solid([0.72, 0.60, 0.42], roughness=0.65)   # warm wood tone
        mat  = gs.materials.Rigid(friction=0.55)

        for k in range(st.n_steps):
            block_h   = (k + 1) * st.step_rise
            centre_y  = st.stair_start_y + k * st.step_run + st.step_run / 2
            centre_z  = block_h / 2
            self.scene.add_entity(
                gs.morphs.Box(
                    pos=(st.stair_x, centre_y, centre_z),
                    size=(st.stair_width, st.step_run, block_h),
                    fixed=True,
                ),
                surface=surf,
                material=mat,
            )

    def _add_stair_tumble_object(self) -> None:
        """Spawn the object on its designated step tread.

        spawn position:
            x = stair.start_x
            y = stair_start_y + start_step * step_run + step_run / 2
            z = (start_step + 1) * step_rise + obj_half_z
        """
        o    = self.spec.object
        st   = self.spec.stair
        surf = _solid(o.color_rgb, roughness=o.roughness, ior=o.ior)
        mat  = gs.materials.Rigid(
            rho=o.density,
            friction=o.friction,
            coup_restitution=o.restitution,
        )

        if o.morph == "sphere":
            hz = o.size[0]
        elif o.morph == "cylinder":
            hz = o.size[2]
        elif o.morph in ("box", "mesh"):
            hz = o.phys_half_z if o.phys_half_z > 0 else (o.size[2] if len(o.size) >= 3 else o.size[0])
        else:
            hz = 0.05

        tread_z = (st.start_step + 1) * st.step_rise
        obj_y   = st.stair_start_y + st.start_step * st.step_run + st.step_run / 2
        pos     = (st.start_x, obj_y, tread_z + hz)
        euler   = (0.0, 0.0, st.euler_z) if st.euler_z != 0 else None

        if o.morph == "sphere":
            morph = gs.morphs.Sphere(pos=pos, radius=o.size[0])
        elif o.morph == "cylinder":
            morph = gs.morphs.Cylinder(pos=pos, radius=o.size[0],
                                       height=o.size[2] * 2, euler=euler)
        elif o.morph == "box":
            morph = gs.morphs.Box(pos=pos, size=tuple(o.size), euler=euler)
        elif o.morph == "mesh":
            mesh_path  = (MESHES_DIR / o.mesh_path).resolve()
            mesh_scale = o.size[0] if len(o.size) == 1 else 1.0
            morph = gs.morphs.Mesh(file=str(mesh_path), pos=pos, euler=euler,
                                   scale=mesh_scale)
        else:
            raise ValueError(f"Unknown morph for stair_tumble: {o.morph}")

        self.obj = self.scene.add_entity(morph, surface=surf, material=mat)

    def _apply_stair_velocity(self) -> None:
        """Apply initial southward nudge to start the tumble down the stairs."""
        st = self.spec.stair
        if st.nudge_vel_y == 0:
            return
        try:
            import torch
            n   = self.obj.n_dofs
            vel = torch.zeros(1, n, dtype=torch.float32)
            vel[0, 1] = st.nudge_vel_y   # vy < 0 → toward observer (-Y)
            self.obj.set_dofs_velocity(vel)
        except Exception as e:
            print(f"[scene_builder] stair nudge velocity not applied: {e}", file=sys.stderr)

    # ── Bouncing object (bouncing_object task) ────────────────

    def _add_bouncing_object(self) -> None:
        """Spawn the ball at its start position with pre-calculated post-bounce velocity."""
        o    = self.spec.object
        b    = self.spec.bounce
        surf = _solid(o.color_rgb, roughness=o.roughness, ior=o.ior)
        # coup_restitution=0 — restitution is baked into vel_z by the randomizer;
        # non-zero values cause expensive micro-contacts when ball rolls on floor
        mat  = gs.materials.Rigid(
            rho=o.density,
            friction=o.friction,
            coup_restitution=0.0,
        )
        pos = (b.start_x, b.start_y, b.start_z)
        # bouncing_object always uses spheres
        morph = gs.morphs.Sphere(pos=pos, radius=o.size[0])
        self.obj = self.scene.add_entity(morph, surface=surf, material=mat)

    def _apply_bounce_velocity(self) -> None:
        """Apply initial horizontal velocity to the bouncing ball."""
        import torch
        b   = self.spec.bounce
        if b.vel_x == 0 and b.vel_y == 0 and b.vel_z == 0:
            return
        try:
            n   = self.obj.n_dofs
            vel = torch.zeros(1, n, dtype=torch.float32)
            vel[0, 0] = b.vel_x
            vel[0, 1] = b.vel_y
            vel[0, 2] = b.vel_z
            self.obj.set_dofs_velocity(vel)
        except Exception as e:
            print(f"[scene_builder] bounce velocity not applied: {e}", file=sys.stderr)

    # ── Pendulum swing (pendulum_swing task) ──────────────────

    def _make_pendulum_mjcf(self) -> str:
        p = self.spec.pendulum
        o = self.spec.object
        r, g, b = o.color_rgb

        if o.morph == "sphere":
            bob_geom = (
                f'        <geom name="bob" type="sphere" size="{o.size[0]:.4f}"\n'
                f'              pos="0 0 -{p.length:.4f}"\n'
                f'              density="{o.density:.1f}"\n'
                f'              friction="{o.friction:.3f} 0.005 0.0001"\n'
                f'              rgba="{r:.3f} {g:.3f} {b:.3f} 1.0"/>\n'
            )
        elif o.morph == "box":
            hx, hy, hz = o.size[0], o.size[1], o.size[2]
            bob_geom = (
                f'        <geom name="bob" type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}"\n'
                f'              pos="0 0 -{p.length:.4f}"\n'
                f'              density="{o.density:.1f}"\n'
                f'              friction="{o.friction:.3f} 0.005 0.0001"\n'
                f'              rgba="{r:.3f} {g:.3f} {b:.3f} 1.0"/>\n'
            )
        else:
            bob_geom = (
                f'        <geom name="bob" type="sphere" size="0.15"\n'
                f'              pos="0 0 -{p.length:.4f}"\n'
                f'              density="{o.density:.1f}"\n'
                f'              rgba="{r:.3f} {g:.3f} {b:.3f} 1.0"/>\n'
            )

        return (
            f'<mujoco>\n'
            f'  <worldbody>\n'
            f'    <body name="pivot" pos="{p.pivot_x:.4f} {p.pivot_y:.4f} {p.pivot_z:.4f}">\n'
            f'      <body name="pendulum">\n'
            f'        <joint name="swing" type="hinge" axis="1 0 0"/>\n'
            f'        <geom name="rod" type="capsule"\n'
            f'              fromto="0 0 0 0 0 -{p.length:.4f}"\n'
            f'              size="0.012" density="0.5"\n'
            f'              rgba="0.50 0.42 0.32 1.0"/>\n'
            f'{bob_geom}'
            f'      </body>\n'
            f'    </body>\n'
            f'  </worldbody>\n'
            f'</mujoco>'
        )

    def _add_pendulum(self) -> None:
        """Add ceiling anchor hook + MJCF pendulum to the scene."""
        import tempfile
        p = self.spec.pendulum

        # Visual anchor hook at ceiling
        self.scene.add_entity(
            gs.morphs.Cylinder(
                pos=(p.pivot_x, p.pivot_y, p.pivot_z + 0.05),
                radius=0.030, height=0.10, fixed=True,
            ),
            surface=_solid([0.35, 0.30, 0.28], roughness=0.55),
            material=gs.materials.Rigid(friction=0.8),
        )

        mjcf_xml  = self._make_pendulum_mjcf()
        tmp_dir   = tempfile.mkdtemp()
        self._pendulum_tmp_dir = tmp_dir
        mjcf_path = f"{tmp_dir}/pendulum.xml"
        with open(mjcf_path, "w") as fh:
            fh.write(mjcf_xml)

        self.obj = self.scene.add_entity(gs.morphs.MJCF(file=mjcf_path))

    def _apply_pendulum_velocity(self) -> None:
        """Set initial joint angle and angular velocity for the pendulum."""
        import torch
        p = self.spec.pendulum

        if hasattr(self, "_pendulum_tmp_dir"):
            import shutil
            shutil.rmtree(self._pendulum_tmp_dir, ignore_errors=True)

        angle_rad = math.radians(p.initial_angle_deg)
        pos_t = torch.tensor([angle_rad], dtype=torch.float32)
        self.obj.set_dofs_position(pos_t)

        if p.angular_vel != 0.0:
            n     = self.obj.n_dofs
            vel_t = torch.zeros(1, n, dtype=torch.float32)
            vel_t[0, 0] = p.angular_vel
            self.obj.set_dofs_velocity(vel_t)

    # ── Position query ────────────────────────────────────────

    def _obj_pos(self) -> np.ndarray:
        # chain_reaction: self.obj is B (target), already set correctly in _add_chain_reaction_objects
        try:
            if self.spec.task_type == "pendulum_swing":
                q = self.obj.get_dofs_position()
                if hasattr(q, "cpu"):
                    q = q.cpu()
                angle = float(np.asarray(q).flatten()[0])
                p  = self.spec.pendulum
                bx = p.pivot_x
                by = p.pivot_y + p.length * math.sin(angle)
                bz = p.pivot_z - p.length * math.cos(angle)
                return np.array([bx, by, bz])
            if self.spec.task_type == "door_swing":
                # Compute door panel COM analytically from the revolute joint angle.
                # This is robust regardless of how Genesis represents the MJCF links.
                q = self.obj.get_dofs_position()
                if hasattr(q, "cpu"):
                    q = q.cpu().numpy()
                angle = float(np.asarray(q).flatten()[0])
                d  = self.spec.door
                cx = d.hinge_x + (d.width / 2) * math.cos(angle)
                cy = d.door_y  + (d.width / 2) * math.sin(angle)
                cz = 0.01 + d.height / 2
                return np.array([cx, cy, cz])
            p = self.obj.get_pos()
            if hasattr(p, "cpu"):
                p = p.cpu().numpy()
            return np.asarray(p).reshape(-1)[:3]
        except Exception:
            return np.zeros(3)

    # ── Metadata ──────────────────────────────────────────────

    def _build_metadata(self, videos: dict) -> dict:
        ttf = (self._floor_hit_step * self.spec.dt
               if self._floor_hit_step is not None else None)

        if self.spec.task_type in ("furniture_tip", "hanging_fall", "stack_collapse", "sliding_object",
                                   "ladder_slip", "ceiling_drop", "stair_tumble"):
            # interception = position when object reaches the floor
            ipt3d = self._gt_positions[self._floor_hit_step] if self._floor_hit_step is not None else None
        elif self.spec.task_type == "rolling_ball":
            # interception = when ball crosses the table edge in X
            edge_x = self.spec.table.pos_x + self.spec.table.width / 2
            ipt3d  = None
            for pos in self._gt_positions:
                if pos[0] >= edge_x:
                    ipt3d = pos
                    break
        elif self.spec.task_type == "shelf_slide":
            # interception = when object crosses the shelf front edge in Y (toward -Y)
            front_edge_y = self.spec.room.depth / 2 - self.spec.shelf.depth
            ipt3d = None
            for pos in self._gt_positions:
                if pos[1] <= front_edge_y:
                    ipt3d = pos
                    break
        elif self.spec.task_type == "door_swing":
            # interception = when door panel COM sweeps past room midpoint (y < 0)
            ipt3d = self._gt_positions[self._floor_hit_step] if self._floor_hit_step is not None else None
        elif self.spec.task_type == "thrown_object":
            # interception = first position where object crosses y=0 toward observer
            ipt3d = None
            for pos in self._gt_positions:
                if pos[1] < 0:
                    ipt3d = pos
                    break
        elif self.spec.task_type == "pendulum_swing":
            # interception = bob position when it crosses vertical (angle = 0)
            ipt3d = self._gt_positions[self._floor_hit_step] if self._floor_hit_step is not None else None
        elif self.spec.task_type == "bouncing_object":
            # interception = first position where ball crosses y=0 after first bounce
            ipt3d = None
            bounced = False
            radius  = self.spec.object.size[0]
            for pos in self._gt_positions:
                if not bounced and pos[2] <= radius + 0.05:
                    bounced = True   # ball has touched the floor
                if bounced and pos[1] < 0:
                    ipt3d = pos
                    break
        elif self.spec.task_type == "chain_reaction":
            # interception = when target B crosses y=0 (mid-room, heading toward observer)
            ipt3d = None
            for pos in self._gt_positions:
                if pos[1] < 0:
                    ipt3d = pos
                    break
        else:
            # object_drop: interception point = when object crosses the table edge in X
            if self.spec.table is not None:
                edge_x = self.spec.table.pos_x + self.spec.table.width / 2
                ipt3d  = None
                for pos in self._gt_positions:
                    if pos[0] >= edge_x:
                        ipt3d = pos
                        break
            else:
                ipt3d = None

        return {
            "scene_id":              self.spec.scene_id,
            "seed":                  self.spec.seed,
            "task_type":             self.spec.task_type,
            "object":                self.spec.object.name,
            "room":                  self.spec.room.type,
            "ground_truth_action":   self.spec.ground_truth_action,
            "safety_label":          self.spec.safety_label,
            "adversarial":           self.spec.adversarial,
            "time_to_floor_s":       round(ttf, 4) if ttf else None,
            "interception_point_3d": [round(v, 4) for v in ipt3d] if ipt3d else None,
            "videos":                videos,
            "n_trajectory_frames":   len(self._gt_positions),
        }
