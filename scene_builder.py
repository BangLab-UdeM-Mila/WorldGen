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
                else:
                    if pos[2] < 0.05:
                        self._floor_hit_step = step

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

        walls = [
            # (pos, size)
            ((0,  -D/2 - TH/2, H/2), (W+2*TH, TH,       H      )),  # south
            ((0,   D/2 + TH/2, H/2), (W+2*TH, TH,       H      )),  # north
            ((-W/2 - TH/2, 0, H/2), (TH,     D,        H      )),  # west
            (( W/2 + TH/2, 0, H/2), (TH,     D,        H      )),  # east
            ((0,   0,  H + TH/2),   (W+2*TH, D+2*TH,  TH     )),  # ceiling
        ]
        surfs = [ws, ws, ws, ws, cs]
        for (p, sz), surf in zip(walls, surfs):
            self.scene.add_entity(
                gs.morphs.Box(pos=p, size=sz, fixed=True),
                surface=surf, material=mat,
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
        elif self.spec.task_type == "sliding_object":
            pass   # gravity on the ramp is sufficient; no extra velocity needed
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

    # ── Position query ────────────────────────────────────────

    def _obj_pos(self) -> np.ndarray:
        try:
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

        if self.spec.task_type in ("furniture_tip", "hanging_fall", "stack_collapse", "sliding_object"):
            # interception = position when object reaches the floor
            ipt3d = self._gt_positions[self._floor_hit_step] if self._floor_hit_step is not None else None
        else:
            # object_drop: interception point = when object crosses the table edge in X
            edge_x = self.spec.table.pos_x + self.spec.table.width / 2
            ipt3d  = None
            for pos in self._gt_positions:
                if pos[0] >= edge_x:
                    ipt3d = pos
                    break

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
