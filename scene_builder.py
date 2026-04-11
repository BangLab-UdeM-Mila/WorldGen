"""
Generic Scene Builder
======================
Builds and runs a Genesis simulation from any SceneSpec.
Replaces the old plate-specific plate_drop.py.

Called by scene_runner.py (subprocess entry point) and can also
be used directly for single-scene testing.
"""

from __future__ import annotations

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
MESHES_DIR   = REPO_ROOT / "assets" / "meshes"
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
        self.obj    = None          # the dropped entity
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
            if self._floor_hit_step is None and pos[2] < 0.05:
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
            morph = gs.morphs.Mesh(file=str(mesh_path), pos=pos,
                                   euler=euler, scale=1.0)
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

    # ── Velocity ──────────────────────────────────────────────

    def _apply_velocity(self) -> None:
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
            print(f"[scene_builder] velocity not applied: {e}", file=sys.stderr)

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
        # Interception point: position when object crosses table edge in x
        edge_x = self.spec.table.pos_x + self.spec.table.width / 2
        ipt3d  = None
        for pos in self._gt_positions:
            if pos[0] >= edge_x:
                ipt3d = pos
                break

        return {
            "scene_id":              self.spec.scene_id,
            "seed":                  self.spec.seed,
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
