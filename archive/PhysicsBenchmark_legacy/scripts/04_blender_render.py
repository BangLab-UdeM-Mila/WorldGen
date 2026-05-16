"""
WorldGen Blender Renderer  (Blender 5.x compatible)
====================================================
Reads PyBullet trajectory JSON → renders PNG frame sequence via Cycles
→ assembles MP4 with imageio.

HOW TO RUN:
  Step A — render frames (Blender):
    "D:\APPS\Blender\blender.exe" --background --python scripts\04_blender_render.py -- --logs_dir .\logs --objects_dir .\objects --output_dir .\renders

  Step B — assemble videos (Python):
    python scripts\04b_assemble_video.py --renders_dir .\renders
"""

import bpy
import os, sys, json, math

# ── Parse arguments ───────────────────────────────────────────────────────────
LOGS_DIR    = "./logs"
OBJECTS_DIR = "./objects"
OUTPUT_DIR  = "./renders"
FPS         = 30

argv = sys.argv
if "--" in argv:
    args = argv[argv.index("--") + 1:]
    for i, a in enumerate(args):
        if a == "--logs_dir"    and i+1 < len(args): LOGS_DIR    = args[i+1]
        if a == "--objects_dir" and i+1 < len(args): OBJECTS_DIR = args[i+1]
        if a == "--output_dir"  and i+1 < len(args): OUTPUT_DIR  = args[i+1]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def reset_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for mat in bpy.data.materials: bpy.data.materials.remove(mat)
    for mesh in bpy.data.meshes:   bpy.data.meshes.remove(mesh)


def setup_render(out_dir, samples=64, width=1280, height=720):
    scene = bpy.context.scene
    scene.render.engine        = 'CYCLES'
    scene.render.resolution_x  = width
    scene.render.resolution_y  = height
    scene.render.fps           = FPS
    # PNG sequence output (Blender 5 compatible)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath      = os.path.join(out_dir, "frame_")
    scene.cycles.samples       = samples
    scene.cycles.use_denoising = True
    # Try GPU
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        for d in prefs.devices: d.use = True
        scene.cycles.device = 'GPU'
    except Exception:
        scene.cycles.device = 'CPU'


def add_lighting():
    def sun(name, energy, rx, rz):
        bpy.ops.object.light_add(type='SUN', location=(0,0,5))
        l = bpy.context.active_object
        l.name = name
        l.rotation_euler = (math.radians(rx), 0, math.radians(rz))
        l.data.energy = energy
        l.data.angle  = math.radians(5)
    sun("Key",  8,  45,  30)
    sun("Fill", 2,  60, 210)
    sun("Rim",  4,  20, 150)
    world = bpy.context.scene.world
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.3


def make_mat(name, color, roughness=0.4, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value  = roughness
    b.inputs["Metallic"].default_value   = metallic
    return mat


def import_obj(path, name, mat):
    try:
        bpy.ops.wm.obj_import(filepath=path)
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=path)
    obj = bpy.context.selected_objects[0]
    obj.name = name
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj


def add_camera(loc, target=(0,0,0.1)):
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam
    import mathutils
    d = (target[0]-loc[0], target[1]-loc[1], target[2]-loc[2])
    cam.rotation_euler = mathutils.Vector(d).to_track_quat('-Z','Y').to_euler()
    cam.data.lens = 50
    return cam


def animate(obj, states):
    import mathutils
    scene = bpy.context.scene
    scene.frame_start = 1
    last = 1
    for s in states:
        frame = max(1, int(s['t'] * FPS) + 1)
        obj.location = s['pos']
        obj.keyframe_insert("location", frame=frame)
        qx,qy,qz,qw = s['quat']
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = mathutils.Quaternion((qw,qx,qy,qz))
        obj.keyframe_insert("rotation_quaternion", frame=frame)
        last = frame
    scene.frame_end = last
    print(f"  {len(states)} keyframes, total {last} frames")


# ── Experiment renders ────────────────────────────────────────────────────────

def render_e1(states):
    print("\n[Render E1] Sphere free-fall bounce")
    reset_scene()
    frames_dir = os.path.join(OUTPUT_DIR, "e1_frames")
    os.makedirs(frames_dir, exist_ok=True)
    setup_render(frames_dir, samples=64)
    add_lighting()

    bpy.ops.mesh.primitive_plane_add(size=3, location=(0,0,0))
    plane = bpy.context.active_object
    plane.data.materials.append(make_mat("GroundMat",(0.08,0.08,0.10),0.9,0.0))

    mat  = make_mat("RubberMat",(0.02,0.02,0.02),0.6,0.0)
    path = os.path.abspath(os.path.join(OBJECTS_DIR,"sphere.obj"))
    obj  = import_obj(path,"Sphere",mat)
    animate(obj, states)
    add_camera((0.6,-0.8,0.5),(0,0,0.15))
    bpy.ops.render.render(animation=True)
    print(f"  Frames → {frames_dir}")


def render_e2(states):
    print("\n[Render E2] Box incline sliding")
    reset_scene()
    frames_dir = os.path.join(OUTPUT_DIR, "e2_frames")
    os.makedirs(frames_dir, exist_ok=True)
    setup_render(frames_dir, samples=64)
    add_lighting()

    theta = math.radians(20)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0,0,0))
    incline = bpy.context.active_object
    incline.scale = (0.6,1.0,1.0)
    incline.rotation_euler = (theta,0,0)
    incline.data.materials.append(make_mat("InclMat",(0.35,0.28,0.20),0.85,0.0))

    mat  = make_mat("WoodMat",(0.55,0.35,0.15),0.7,0.0)
    path = os.path.abspath(os.path.join(OBJECTS_DIR,"bevel_box.obj"))
    obj  = import_obj(path,"BevelBox",mat)
    animate(obj, states)
    add_camera((0.8,-1.0,0.5),(0,0,0.1))
    bpy.ops.render.render(animation=True)
    print(f"  Frames → {frames_dir}")


def render_e3(states):
    print("\n[Render E3] L-bracket torque-free rotation")
    reset_scene()
    frames_dir = os.path.join(OUTPUT_DIR, "e3_frames")
    os.makedirs(frames_dir, exist_ok=True)
    setup_render(frames_dir, samples=96)

    # Deep space background
    world = bpy.context.scene.world
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.01,0.01,0.02,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.1

    def sun(name, energy, rx, rz):
        bpy.ops.object.light_add(type='SUN', location=(0,0,5))
        l = bpy.context.active_object
        l.name = name
        l.rotation_euler = (math.radians(rx),0,math.radians(rz))
        l.data.energy = energy
    sun("Key", 10, 40, 30)
    sun("Rim",  5, 20,150)

    mat  = make_mat("AlumMat",(0.72,0.72,0.75),0.15,0.9)
    path = os.path.abspath(os.path.join(OBJECTS_DIR,"l_bracket.obj"))
    obj  = import_obj(path,"LBracket",mat)
    animate(obj, states)
    add_camera((0.3,-0.4,0.25),(0.03,0.03,0.2))
    bpy.ops.render.render(animation=True)
    print(f"  Frames → {frames_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log_path = os.path.join(LOGS_DIR,"benchmark_results.json")
    if not os.path.exists(log_path):
        print(f"ERROR: {log_path} not found — run 03_run_simulation.py first")
        sys.exit(1)

    with open(log_path) as f:
        data = json.load(f)

    if "e1_freefall" in data: render_e1(data["e1_freefall"])
    if "e2_incline"  in data: render_e2(data["e2_incline"])
    if "e3_rotation" in data: render_e3(data["e3_rotation"])

    print(f"\n[Done] Frames saved in {os.path.abspath(OUTPUT_DIR)}/")
    print("  Now run:  python scripts\\04b_assemble_video.py --renders_dir .\\renders")

if __name__ == "__main__":
    main()
