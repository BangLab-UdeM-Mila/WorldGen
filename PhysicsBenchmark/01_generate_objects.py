"""
WorldGen Object Generator  —  Blender Python Script
=====================================================
Compatible with Blender 4.x and 5.x

HOW TO RUN:
  blender --background --python 01_generate_objects.py -- --output ./objects
"""

import bpy
import os
import sys
import math

OUTPUT_DIR = "./objects"
argv = sys.argv
if "--" in argv:
    args = argv[argv.index("--") + 1:]
    for i, a in enumerate(args):
        if a == "--output" and i + 1 < len(args):
            OUTPUT_DIR = args[i + 1]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[WorldGen] Exporting objects to: {os.path.abspath(OUTPUT_DIR)}")


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def apply_all_transforms(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def decimate_mesh(obj, ratio=0.1, max_faces=500):
    mod = obj.modifiers.new(name="Decimate_Col", type='DECIMATE')
    current_faces = len(obj.data.polygons)
    mod.ratio = min(ratio, max_faces / max(current_faces, 1))
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def export_obj(obj, filepath):
    """Export a single object as .obj — works on Blender 4.x and 5.x."""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Blender 5.1 changed the OBJ exporter signature.
    # We try progressively simpler calls until one works.
    exported = False
    for kwargs in [
        dict(filepath=filepath, export_selected_only=True, export_materials=False),
        dict(filepath=filepath, export_selected_only=True),
        dict(filepath=filepath),
    ]:
        try:
            bpy.ops.wm.obj_export(**kwargs)
            exported = True
            break
        except (TypeError, AttributeError):
            continue

    if not exported:
        # Final fallback: legacy exporter (Blender < 3.3)
        bpy.ops.export_scene.obj(filepath=filepath, use_selection=True, use_materials=False)

    print(f"  Exported: {filepath}")


# ── Object 1: UV-Sphere ───────────────────────────────────────────────────────

def make_sphere(name="sphere", radius=0.05):
    """
    Standard UV-sphere, radius 5 cm.
    Physics: smooth Hertzian contact, analytical rolling solution.
    Material: rubber, density ~1100 kg/m3
    """
    clear_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius, segments=64, ring_count=32, location=(0, 0, 0)
    )
    obj = bpy.context.active_object
    obj.name = name
    bpy.ops.object.shade_smooth()
    apply_all_transforms(obj)
    export_obj(obj, os.path.join(OUTPUT_DIR, f"{name}.obj"))

    # Collision mesh: low-res icosphere
    clear_scene()
    bpy.ops.mesh.primitive_ico_sphere_add(radius=radius, subdivisions=2, location=(0, 0, 0))
    col_obj = bpy.context.active_object
    col_obj.name = f"{name}_col"
    apply_all_transforms(col_obj)
    export_obj(col_obj, os.path.join(OUTPUT_DIR, f"{name}_col.obj"))

    print(f"  [Object 1] sphere: radius={radius}m")


# ── Object 2: Beveled Box ─────────────────────────────────────────────────────

def make_beveled_box(name="bevel_box", size=(0.06, 0.04, 0.03), bevel=0.004):
    """
    Rectangular box with beveled edges.
    Dimensions: 6x4x3 cm. Material: wood, density ~700 kg/m3.
    Physics: planar contact, rocking instability, stacking benchmark.
    """
    clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size

    bevel_mod = obj.modifiers.new(name="Bevel", type='BEVEL')
    bevel_mod.width = bevel / max(size)
    bevel_mod.segments = 4
    bevel_mod.limit_method = 'ANGLE'
    bevel_mod.angle_limit = math.radians(30)

    apply_all_transforms(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Bevel")

    export_obj(obj, os.path.join(OUTPUT_DIR, f"{name}.obj"))

    # Collision: plain box
    clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    col_obj = bpy.context.active_object
    col_obj.name = f"{name}_col"
    col_obj.scale = size
    apply_all_transforms(col_obj)
    export_obj(col_obj, os.path.join(OUTPUT_DIR, f"{name}_col.obj"))

    dims = tuple(round(s * 100, 1) for s in size)
    print(f"  [Object 2] {name}: {dims} cm beveled box")


# ── Object 3: L-Bracket ───────────────────────────────────────────────────────

def make_l_bracket(name="l_bracket"):
    """
    L-shaped rigid body. Two arms 8x2x2 cm joined at 90 degrees.
    Material: aluminum, density ~2700 kg/m3.
    Physics: asymmetric inertia tensor — tests URDF ixx/ixy/ixz accuracy.
    Reference: Mahler et al., Dex-Net 2.0, RSS 2017.
    """
    clear_scene()

    arm_len = 0.08
    arm_w   = 0.02
    arm_h   = 0.02

    bpy.ops.mesh.primitive_cube_add(size=1, location=(arm_len/2, 0, 0))
    arm_a = bpy.context.active_object
    arm_a.name = "arm_a"
    arm_a.scale = (arm_len/2, arm_w/2, arm_h/2)
    apply_all_transforms(arm_a)

    bpy.ops.mesh.primitive_cube_add(size=1, location=(arm_w/2, arm_len/2 + arm_w/2, 0))
    arm_b = bpy.context.active_object
    arm_b.name = "arm_b"
    arm_b.scale = (arm_w/2, arm_len/2, arm_h/2)
    apply_all_transforms(arm_b)

    arm_a.select_set(True)
    arm_b.select_set(True)
    bpy.context.view_layer.objects.active = arm_a

    bool_mod = arm_a.modifiers.new(name="Union", type='BOOLEAN')
    bool_mod.operation = 'UNION'
    bool_mod.object = arm_b
    bpy.ops.object.modifier_apply(modifier="Union")

    bpy.ops.object.select_all(action='DESELECT')
    arm_b.select_set(True)
    bpy.ops.object.delete()

    arm_a.name = name
    bpy.ops.object.select_all(action='DESELECT')
    arm_a.select_set(True)
    bpy.context.view_layer.objects.active = arm_a
    export_obj(arm_a, os.path.join(OUTPUT_DIR, f"{name}.obj"))

    # Collision mesh (decimated copy)
    col_obj = arm_a.copy()
    col_obj.data = arm_a.data.copy()
    col_obj.name = f"{name}_col"
    bpy.context.collection.objects.link(col_obj)
    bpy.ops.object.select_all(action='DESELECT')
    col_obj.select_set(True)
    bpy.context.view_layer.objects.active = col_obj
    decimate_mesh(col_obj, ratio=0.3, max_faces=200)
    export_obj(col_obj, os.path.join(OUTPUT_DIR, f"{name}_col.obj"))

    print(f"  [Object 3] {name}: L-bracket 8x2x2 cm, asymmetric inertia")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[WorldGen] Starting object generation...")
    make_sphere(radius=0.05)
    make_beveled_box(size=(0.06, 0.04, 0.03))
    make_l_bracket()
    print("\n[WorldGen] All objects exported successfully.")
    print(f"  Output directory: {os.path.abspath(OUTPUT_DIR)}")
