"""
Indoor Plate-Drop Scene Generator
==================================
Simulates a ceramic dinner plate tipping off a walnut dining table inside a
realistic home interior (walls, ceiling, herringbone parquet floor, window
light).

Physics model
-------------
  Table   : fixed rigid bodies (top + 4 legs), dark walnut BSDF
  Plate   : free rigid body from plate.obj (real-world scale),
            porcelain BSDF, density 2300 kg/m³
  Floor   : infinite Plane with herringbone parquet ImageTexture
  Room    : Box slabs for walls + ceiling (off-white BSDF)
  Window  : emissive panel on the north wall (daylight fill)
  Gravity : (0, 0, −9.81) m/s²

Coordinate frame
----------------
  Z = up,  X = table width (plate slides in +X),  Y = table depth
  Table top surface at Z = cfg['table']['height']  (default 0.77 m)
  Room origin at (0, 0, 0) — centre of the floor.

Usage
-----
    from scenes.plate_drop import PlateDrop
    scene = PlateDrop.from_config("config/plate_drop.yaml")
    scene.build()
    scene.run()
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

import genesis as gs

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="genesis")

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent.resolve()
ASSETS_ROOT   = PROJECT_ROOT / "assets"
TEXTURES_ROOT = ASSETS_ROOT / "textures"
REPO_ROOT     = PROJECT_ROOT.parent
PLATE_MESH    = REPO_ROOT / "assets" / "meshes" / "plate.obj"
GENESIS_PKG   = Path(gs.__file__).parent
ENV_MAP       = GENESIS_PKG / "assets" / "textures" / "indoor_bright.png"


# ─────────────────────────────────────────────────────────────────────────────
# Tiny helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _rgb01(rgb: list[float]) -> tuple[float, float, float]:
    r, g, b = rgb
    if max(r, g, b) > 1.0:
        return r / 255, g / 255, b / 255
    return float(r), float(g), float(b)


def _tex(path: Path | None, encoding: str = "srgb"):
    """Return an ImageTexture if the file exists, else None."""
    if path is not None and path.exists():
        return gs.textures.ImageTexture(image_path=str(path), encoding=encoding)
    return None


def _solid_bsdf(color_rgb, roughness=0.5, metallic=0.0, ior=1.0) -> gs.surfaces.BSDF:
    """Solid-colour BSDF — safe for Box/Cylinder primitives (no UV coords)."""
    return gs.surfaces.BSDF(
        color=_rgb01(color_rgb),
        roughness=roughness,
        metallic=metallic,
        ior=ior,
        smooth=True,
    )


def _textured_bsdf(
    color_rgb, roughness,
    diffuse_tex=None, rough_tex=None, normal_tex=None,
    metallic=0.0, ior=1.0,
) -> gs.surfaces.BSDF:
    """
    Textured BSDF for geometry that has UV coordinates (Plane, Mesh).
    Falls back to solid colour when no textures are available.
    Genesis forbids setting both `color` and `diffuse_texture`.
    """
    if diffuse_tex is None:
        return _solid_bsdf(color_rgb, roughness, metallic, ior)
    return gs.surfaces.BSDF(
        diffuse_texture=diffuse_tex,
        roughness=roughness if rough_tex is None else None,
        roughness_texture=rough_tex,
        normal_texture=normal_tex,
        metallic=metallic,
        ior=ior,
        smooth=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scene
# ─────────────────────────────────────────────────────────────────────────────

class PlateDrop:
    """Full indoor home scene: room + dining table + ceramic plate."""

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.scene  = None
        self.plate  = None
        self.cameras: dict = {}
        self._built = False

    @classmethod
    def from_config(cls, path: str | Path) -> "PlateDrop":
        return cls(_load_config(path))

    # ── Public ────────────────────────────────────────────────────────────────

    def build(self) -> None:
        self._init_genesis()
        self._assemble_scene()
        self._built = True

    def run(self, output_dir: str | Path | None = None) -> list[Path]:
        if not self._built:
            raise RuntimeError("Call build() before run().")

        out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / self.cfg["output"]["dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        prefix  = self.cfg["output"]["prefix"]
        fps     = self.cfg["output"]["fps"]
        sim_cfg = self.cfg["simulation"]
        n_steps = int(sim_cfg["duration_s"] / sim_cfg["dt"])
        dt      = sim_cfg["dt"]

        self._apply_plate_velocity()

        for cam in self.cameras.values():
            cam.start_recording()

        print(f"\n[plate_drop] Simulating {sim_cfg['duration_s']:.1f}s "
              f"({n_steps} steps @ {1/dt:.0f} Hz) …")
        t0 = time.perf_counter()

        for step in range(n_steps):
            self.scene.step()
            for cam in self.cameras.values():
                cam.render()

            if (step + 1) % int(1.0 / dt) == 0:
                elapsed = time.perf_counter() - t0
                sim_t   = (step + 1) * dt
                pos     = self._plate_pos()
                print(f"  t={sim_t:.2f}s  plate=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m"
                      f"  [wall {elapsed:.1f}s]")
                if pos[2] < -0.5:
                    print("  [early stop] plate below floor")
                    break

        saved = []
        for cam_name, cam in self.cameras.items():
            out_path = out_dir / f"{prefix}_{cam_name}.mp4"
            cam.stop_recording(save_to_filename=str(out_path), fps=fps)
            saved.append(out_path)
            print(f"  Saved: {out_path}")

        print(f"\n[plate_drop] Done in {time.perf_counter()-t0:.1f}s. "
              f"{len(saved)} video(s) written.")
        return saved

    # ── Genesis init ──────────────────────────────────────────────────────────

    def _init_genesis(self) -> None:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    # ── Scene assembly ────────────────────────────────────────────────────────

    def _assemble_scene(self) -> None:
        sim_cfg  = self.cfg["simulation"]
        rend_cfg = self.cfg["renderer"]
        cam_cfg  = self.cfg["cameras"]

        renderer, vis_options = self._make_renderer(rend_cfg)

        scene_kwargs = dict(
            sim_options=gs.options.SimOptions(
                dt=sim_cfg["dt"],
                substeps=sim_cfg["substeps"],
                gravity=tuple(sim_cfg["gravity"]),
            ),
            rigid_options=gs.options.RigidOptions(enable_collision=True),
            renderer=renderer,
            show_viewer=False,
        )
        if vis_options is not None:
            scene_kwargs["vis_options"] = vis_options

        self.scene = gs.Scene(**scene_kwargs)

        self._add_floor()
        self._add_room_walls()
        self._add_table()
        self._add_plate()
        self._add_cameras(cam_cfg)

        self.scene.build()

    # ── Renderer ──────────────────────────────────────────────────────────────

    def _make_renderer(self, rend_cfg: dict):
        rtype = rend_cfg.get("type", "Rasterizer")

        if rtype == "RayTracer":
            try:
                lights = []
                lp = rend_cfg.get("light_pos", [0, 0, 2.5])
                lights.append(gs.renderers.SphereLight(
                    pos=tuple(lp),
                    radius=rend_cfg.get("light_radius", 0.3),
                    color=tuple(rend_cfg.get("light_color", [8, 7, 6])),
                ))
                env = None
                if ENV_MAP.exists():
                    env = gs.surfaces.Emission(
                        emissive_texture=gs.textures.ImageTexture(
                            image_path=str(ENV_MAP), encoding="srgb"
                        )
                    )
                return gs.renderers.RayTracer(
                    tracing_depth=rend_cfg.get("tracing_depth", 24),
                    env_surface=env,
                    lights=lights,
                ), None
            except Exception as e:
                print(f"[plate_drop] RayTracer unavailable ({e}), falling back to Rasterizer.",
                      file=sys.stderr)

        # ── Rasterizer ────────────────────────────────────────────────────────
        gs_lights = []
        for spec in rend_cfg.get("lights", []):
            ltype     = spec.get("type", "directional")
            color     = tuple(spec.get("color", [1.0, 1.0, 1.0]))
            intensity = float(spec.get("intensity", 5.0))
            if ltype == "directional":
                gs_lights.append(gs.options.vis.DirectionalLight(
                    dir=tuple(spec["dir"]), color=color, intensity=intensity))
            elif ltype == "point":
                gs_lights.append(gs.options.vis.PointLight(
                    pos=tuple(spec["pos"]), color=color, intensity=intensity))
            elif ltype == "ambient":
                gs_lights.append(gs.options.vis.AmbientLight(
                    color=color, intensity=intensity))

        vis_kw: dict = dict(
            shadow=bool(rend_cfg.get("shadow", True)),
            ambient_light=tuple(rend_cfg.get("ambient_light", [0.3, 0.3, 0.3])),
            background_color=tuple(rend_cfg.get("background_color", [0.1, 0.1, 0.1])),
        )
        if gs_lights:
            vis_kw["lights"] = gs_lights

        return gs.renderers.Rasterizer(), gs.options.VisOptions(**vis_kw)

    # ── Floor ─────────────────────────────────────────────────────────────────

    def _add_floor(self) -> None:
        f = self.cfg["floor"]
        surface = _textured_bsdf(
            color_rgb=f["color_rgb"],
            roughness=f["roughness"],
            diffuse_tex=_tex(TEXTURES_ROOT / "floor" / "floor_diff.jpg"),
            rough_tex  =_tex(TEXTURES_ROOT / "floor" / "floor_rough.jpg", "linear"),
            normal_tex =_tex(TEXTURES_ROOT / "floor" / "floor_nor.jpg",   "linear"),
        )
        self.scene.add_entity(
            gs.morphs.Plane(),
            surface=surface,
            material=gs.materials.Rigid(friction=f["friction"]),
        )

    # ── Room walls + ceiling ──────────────────────────────────────────────────

    def _add_room_walls(self) -> None:
        r   = self.cfg["room"]
        W   = r["width"]        # x extent
        D   = r["depth"]        # y extent
        H   = r["height"]       # ceiling z
        TH  = r["wall_thickness"]
        wc  = r["wall_color_rgb"]
        cc  = r["ceiling_color_rgb"]
        win = r.get("window", {})

        wall_surf    = _solid_bsdf(wc,  roughness=0.75)
        ceil_surf    = _solid_bsdf(cc,  roughness=0.90)

        # ── Four walls ────────────────────────────────────────────────────────
        # South wall  (y = −D/2, observer is behind table)
        self.scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, -D/2 - TH/2, H/2),
                size=(W + 2*TH, TH, H),
                fixed=True,
            ),
            surface=wall_surf,
            material=gs.materials.Rigid(friction=0.8),
        )
        # North wall  (y = +D/2, window side)
        north_surf = wall_surf
        self.scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, D/2 + TH/2, H/2),
                size=(W + 2*TH, TH, H),
                fixed=True,
            ),
            surface=north_surf,
            material=gs.materials.Rigid(friction=0.8),
        )
        # West wall   (x = −W/2)
        self.scene.add_entity(
            gs.morphs.Box(
                pos=(-W/2 - TH/2, 0.0, H/2),
                size=(TH, D, H),
                fixed=True,
            ),
            surface=wall_surf,
            material=gs.materials.Rigid(friction=0.8),
        )
        # East wall   (x = +W/2) — plate falls toward this side
        self.scene.add_entity(
            gs.morphs.Box(
                pos=(W/2 + TH/2, 0.0, H/2),
                size=(TH, D, H),
                fixed=True,
            ),
            surface=wall_surf,
            material=gs.materials.Rigid(friction=0.8),
        )

        # ── Ceiling ───────────────────────────────────────────────────────────
        self.scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, 0.0, H + TH/2),
                size=(W + 2*TH, D + 2*TH, TH),
                fixed=True,
            ),
            surface=ceil_surf,
            material=gs.materials.Rigid(friction=0.8),
        )

        # ── Window — emissive panel on north wall ─────────────────────────────
        if win:
            wx  = float(win.get("pos_x", 0.0))
            wz  = float(win.get("pos_z", H * 0.52))
            ww  = float(win.get("width",  1.2))
            wh  = float(win.get("height", 1.0))
            em  = win.get("emissive", [5.0, 4.8, 4.5])
            window_surf = gs.surfaces.Emission(
                emissive=tuple(em),
            )
            self.scene.add_entity(
                gs.morphs.Box(
                    pos=(wx, D/2 - 0.01, wz),
                    size=(ww, 0.01, wh),
                    fixed=True,
                    collision=False,
                ),
                surface=window_surf,
            )

    # ── Table ─────────────────────────────────────────────────────────────────

    def _add_table(self) -> None:
        t      = self.cfg["table"]
        H      = t["height"]
        TH     = t["thickness"]
        W      = t["width"]
        D      = t["depth"]
        LEG    = t["leg_size"]
        LEG_H  = H - TH

        wood   = _solid_bsdf(t["color_rgb"], roughness=t["roughness"])
        mat    = gs.materials.Rigid(friction=t["friction"])

        # Tabletop
        self.scene.add_entity(
            gs.morphs.Box(pos=(0.0, 0.0, H - TH/2), size=(W, D, TH), fixed=True),
            surface=wood, material=mat,
        )
        # Four legs
        inset = 0.05
        hx = W/2 - inset - LEG/2
        hy = D/2 - inset - LEG/2
        for lx in (hx, -hx):
            for ly in (hy, -hy):
                self.scene.add_entity(
                    gs.morphs.Box(
                        pos=(lx, ly, LEG_H/2), size=(LEG, LEG, LEG_H), fixed=True,
                    ),
                    surface=wood, material=mat,
                )

    # ── Plate ─────────────────────────────────────────────────────────────────

    def _add_plate(self) -> None:
        p  = self.cfg["plate"]
        H  = self.cfg["table"]["height"]

        # plate.obj: local Z in [−0.007, +0.015]
        # Place so plate bottom rim (local Z = −0.007) sits on table surface.
        plate_z = H + 0.007

        mesh_path = (PROJECT_ROOT / p["mesh_path"]).resolve()
        if not mesh_path.exists():
            mesh_path = PLATE_MESH
        if not mesh_path.exists():
            raise FileNotFoundError(f"plate.obj not found. Tried: {mesh_path}")

        ceramic = _solid_bsdf(p["color_rgb"], roughness=p["roughness"], ior=p.get("ior", 1.0))
        mat     = gs.materials.Rigid(
            rho=p["density_kg_m3"],
            friction=p["friction"],
            coup_restitution=p.get("restitution", 0.15),
        )
        self.plate = self.scene.add_entity(
            gs.morphs.Mesh(
                file=str(mesh_path),
                pos=(p["start_x"], p["start_y"], plate_z),
                scale=1.0,
            ),
            surface=ceramic,
            material=mat,
        )

    # ── Cameras ───────────────────────────────────────────────────────────────

    def _add_cameras(self, cam_cfg: dict) -> None:
        res = tuple(cam_cfg["resolution"])
        fov = cam_cfg["fov_deg"]
        for name in ("observer", "closeup", "overhead"):
            spec = cam_cfg[name]
            self.cameras[name] = self.scene.add_camera(
                res=res,
                pos=tuple(spec["pos"]),
                lookat=tuple(spec["lookat"]),
                fov=fov,
                GUI=False,
            )

    # ── Plate initial velocity ────────────────────────────────────────────────

    def _apply_plate_velocity(self) -> None:
        p  = self.cfg["plate"]
        vx = p.get("init_vel_x", 0.0)
        vy = p.get("init_vel_y", 0.0)
        if vx == 0.0 and vy == 0.0:
            return

        try:
            import torch
            n = self.plate.n_dofs
            vel = torch.zeros(1, n, dtype=torch.float32)
            vel[0, 0] = vx
            vel[0, 1] = vy
            self.plate.set_dofs_velocity(vel)
            print(f"[plate_drop] plate velocity: vx={vx:.2f}  vy={vy:.2f} m/s")
        except Exception as e:
            print(f"[plate_drop] Warning: velocity not applied ({e})", file=sys.stderr)

    # ── Position query ────────────────────────────────────────────────────────

    def _plate_pos(self) -> np.ndarray:
        try:
            pos = self.plate.get_pos()
            if hasattr(pos, "cpu"):
                pos = pos.cpu().numpy()
            return np.asarray(pos).reshape(-1)[:3]
        except Exception:
            return np.zeros(3)
