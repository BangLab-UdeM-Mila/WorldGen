"""
WorldGen Physics Benchmark Simulation
======================================
Runs three canonical benchmark experiments in PyBullet.
Outputs: benchmark figures (PNG), data logs (JSON), and simulation videos (MP4).

Usage:
  python 03_run_simulation.py --urdfs_dir ./urdfs --logs_dir ./logs
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import List

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("ERROR: pybullet not found. Run: pip install pybullet")
    sys.exit(1)

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    print("[WARN] imageio not found. Run: pip install imageio[ffmpeg]")
    print("       Videos will not be saved.")
    HAS_IMAGEIO = False

# ── Simulation configuration ──────────────────────────────────────────────────

TIMESTEP    = 1.0 / 240.0
GRAVITY     = (0, 0, -9.81)
SOLVER_ITER = 50

# Camera settings for video rendering
CAM_DISTANCE   = 0.8    # metres from target
CAM_YAW        = 45     # degrees
CAM_PITCH      = -25    # degrees
CAM_TARGET     = [0, 0, 0.1]
IMG_WIDTH      = 640
IMG_HEIGHT     = 480
VIDEO_FPS      = 30
# Record one frame every N simulation steps (240Hz / 8 = 30 fps)
RECORD_EVERY_N = 8


@dataclass
class SimState:
    t:          float
    pos:        List[float]
    quat:       List[float]
    vel:        List[float]
    ang_vel:    List[float]
    ke:         float
    pe:         float
    n_contacts: int


# ── Video recorder ────────────────────────────────────────────────────────────

class VideoRecorder:
    def __init__(self, path, fps=VIDEO_FPS):
        self.path   = path
        self.fps    = fps
        self.frames = []
        self.active = HAS_IMAGEIO

    def capture(self, client):
        if not self.active:
            return
        _, _, rgba, _, _ = p.getCameraImage(
            width=IMG_WIDTH, height=IMG_HEIGHT,
            viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=CAM_TARGET,
                distance=CAM_DISTANCE,
                yaw=CAM_YAW, pitch=CAM_PITCH, roll=0,
                upAxisIndex=2
            ),
            projectionMatrix=p.computeProjectionMatrixFOV(
                fov=60, aspect=IMG_WIDTH/IMG_HEIGHT,
                nearVal=0.01, farVal=10.0
            ),
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=client
        )
        # rgba is (H, W, 4) — drop alpha channel
        frame = np.array(rgba, dtype=np.uint8).reshape(IMG_HEIGHT, IMG_WIDTH, 4)[:, :, :3]
        self.frames.append(frame)

    def save(self):
        if not self.active or not self.frames:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        imageio.mimwrite(self.path, self.frames, fps=self.fps, quality=8)
        print(f"  Video saved: {self.path}  ({len(self.frames)} frames @ {self.fps}fps)")
        self.frames = []


# ── Simulation engine ─────────────────────────────────────────────────────────

class BulletSim:
    def __init__(self):
        # DIRECT mode — headless, but getCameraImage still works
        self.client = p.connect(p.DIRECT)
        p.setPhysicsEngineParameter(numSolverIterations=SOLVER_ITER,
                                    physicsClientId=self.client)
        p.setTimeStep(TIMESTEP, physicsClientId=self.client)
        p.setGravity(*GRAVITY, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.client)

    def reset(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setTimeStep(TIMESTEP, physicsClientId=self.client)
        p.setGravity(*GRAVITY, physicsClientId=self.client)

    def load_plane(self):
        return p.loadURDF("plane.urdf", physicsClientId=self.client)

    def load_object(self, urdf_path, pos=(0,0,0), orn=(0,0,0,1)):
        return p.loadURDF(urdf_path, basePosition=pos,
                          baseOrientation=orn, physicsClientId=self.client)

    def set_dynamics(self, obj_id, friction=0.5, restitution=0.3):
        p.changeDynamics(obj_id, -1,
                         lateralFriction=friction, restitution=restitution,
                         spinningFriction=0.01, rollingFriction=0.001,
                         physicsClientId=self.client)

    def get_state(self, obj_id, t, mass) -> SimState:
        pos, quat = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client)
        vel, ang  = p.getBaseVelocity(obj_id, physicsClientId=self.client)
        ke = 0.5 * mass * sum(v**2 for v in vel)
        pe = mass * 9.81 * pos[2]
        nc = len(p.getContactPoints(obj_id, physicsClientId=self.client))
        return SimState(t=round(t,5), pos=list(pos), quat=list(quat),
                        vel=list(vel), ang_vel=list(ang),
                        ke=round(ke,6), pe=round(pe,6), n_contacts=nc)

    def step(self):
        p.stepSimulation(physicsClientId=self.client)

    def close(self):
        p.disconnect(physicsClientId=self.client)


# ── Experiment E1: Free-fall bounce ───────────────────────────────────────────

def run_e1_freefall(sim, sphere_urdf, logs_dir, duration=2.0):
    print("\n[E1] Free-fall bounce (sphere)...")
    sim.reset()
    sim.load_plane()
    obj  = sim.load_object(sphere_urdf, pos=(0, 0, 0.5))
    sim.set_dynamics(obj, friction=0.8, restitution=0.7)
    mass = p.getDynamicsInfo(obj, -1, physicsClientId=sim.client)[0]

    vid = VideoRecorder(os.path.join(logs_dir, "e1_freefall.mp4"))
    states, t = [], 0.0
    for step_i in range(int(duration / TIMESTEP)):
        sim.step()
        t += TIMESTEP
        states.append(sim.get_state(obj, t, mass))
        if step_i % RECORD_EVERY_N == 0:
            vid.capture(sim.client)

    vid.save()
    plot_e1(states, logs_dir)
    print(f"  {len(states)} timesteps recorded")
    return states


# ── Experiment E2: Inclined-plane sliding ─────────────────────────────────────

def run_e2_incline(sim, box_urdf, logs_dir, duration=3.0):
    print("\n[E2] Incline sliding (bevel_box)...")
    sim.reset()
    theta = np.deg2rad(20)

    plane_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.01],
                                        physicsClientId=sim.client)
    plane_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.01],
                                     rgbaColor=[0.6, 0.6, 0.6, 1],
                                     physicsClientId=sim.client)
    orn = p.getQuaternionFromEuler([theta, 0, 0])
    p.createMultiBody(0, plane_col, plane_vis,
                      basePosition=[0, 0, 0], baseOrientation=orn,
                      physicsClientId=sim.client)

    start_pos = [0, -0.3*np.cos(theta), 0.3*np.sin(theta)+0.05]
    obj  = sim.load_object(box_urdf, pos=start_pos,
                            orn=p.getQuaternionFromEuler([theta, 0, 0]))
    sim.set_dynamics(obj, friction=0.6, restitution=0.3)
    mass = p.getDynamicsInfo(obj, -1, physicsClientId=sim.client)[0]

    vid = VideoRecorder(os.path.join(logs_dir, "e2_incline.mp4"))
    states, t = [], 0.0
    for step_i in range(int(duration / TIMESTEP)):
        sim.step()
        t += TIMESTEP
        states.append(sim.get_state(obj, t, mass))
        if step_i % RECORD_EVERY_N == 0:
            vid.capture(sim.client)

    vid.save()
    plot_e2(states, logs_dir)
    print(f"  {len(states)} timesteps recorded")
    return states


# ── Experiment E3: Torque-free rotation ───────────────────────────────────────

def run_e3_rotation(sim, bracket_urdf, logs_dir, duration=2.0):
    print("\n[E3] Torque-free rotation (l_bracket)...")
    sim.reset()
    p.setGravity(0, 0, 0, physicsClientId=sim.client)
    obj  = sim.load_object(bracket_urdf, pos=(0, 0, 0.2))
    mass = p.getDynamicsInfo(obj, -1, physicsClientId=sim.client)[0]
    p.resetBaseVelocity(obj, linearVelocity=[0,0,0],
                        angularVelocity=[1.0, 0.1, 0.0],
                        physicsClientId=sim.client)

    vid = VideoRecorder(os.path.join(logs_dir, "e3_rotation.mp4"))
    states, t = [], 0.0
    for step_i in range(int(duration / TIMESTEP)):
        sim.step()
        t += TIMESTEP
        states.append(sim.get_state(obj, t, mass))
        if step_i % RECORD_EVERY_N == 0:
            vid.capture(sim.client)

    vid.save()
    plot_e3(states, logs_dir)
    print(f"  {len(states)} timesteps recorded")
    return states


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_e1(states, out_dir):
    t  = [s.t      for s in states]
    z  = [s.pos[2] for s in states]
    ke = [s.ke     for s in states]
    pe = [s.pe     for s in states]
    te = [s.ke+s.pe for s in states]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("E1: Free-Fall Bounce — Sphere", fontsize=14, fontweight='bold')
    axes[0].plot(t, z, color='steelblue', lw=1.5)
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Height z (m)")
    axes[0].set_title("Trajectory"); axes[0].grid(True, alpha=0.4)
    axes[1].plot(t, ke, label="KE", color='tomato', lw=1.2)
    axes[1].plot(t, pe, label="PE", color='seagreen', lw=1.2)
    axes[1].plot(t, te, label="KE+PE", color='k', lw=1.5, ls='--')
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Energy (J)")
    axes[1].set_title("Energy Budget"); axes[1].legend(); axes[1].grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "e1_freefall.png"), dpi=150)
    plt.close()

def plot_e2(states, out_dir):
    t  = [s.t          for s in states]
    vy = [abs(s.vel[1]) for s in states]
    t_arr, vy_arr = np.array(t), np.array(vy)
    mask = t_arr < 1.0
    if mask.sum() > 10:
        coeffs   = np.polyfit(t_arr[mask], vy_arr[mask], 1)
        a_meas   = coeffs[0]
        a_theory = 9.81*(np.sin(np.deg2rad(20))-0.6*np.cos(np.deg2rad(20)))
        label = f"Measured a = {a_meas:.3f} m/s²\nTheory a   = {a_theory:.3f} m/s²"
    else:
        label = "insufficient data"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, vy, color='darkorange', lw=1.5, label=label)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed along incline (m/s)")
    ax.set_title("E2: Incline Sliding — Bevel Box", fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "e2_incline.png"), dpi=150)
    plt.close()

def plot_e3(states, out_dir):
    t  = [s.t           for s in states]
    wx = [s.ang_vel[0]  for s in states]
    wy = [s.ang_vel[1]  for s in states]
    wz = [s.ang_vel[2]  for s in states]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, wx, label="wx", color='crimson',   lw=1.2)
    ax.plot(t, wy, label="wy", color='royalblue', lw=1.2)
    ax.plot(t, wz, label="wz", color='seagreen',  lw=1.2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_title("E3: Torque-Free Rotation — L-Bracket (Dzhanibekov Effect)",
                 fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "e3_rotation.png"), dpi=150)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdfs_dir", default="./urdfs")
    parser.add_argument("--logs_dir",  default="./logs")
    args = parser.parse_args()
    os.makedirs(args.logs_dir, exist_ok=True)

    sphere_urdf  = os.path.join(args.urdfs_dir, "sphere.urdf")
    box_urdf     = os.path.join(args.urdfs_dir, "bevel_box.urdf")
    bracket_urdf = os.path.join(args.urdfs_dir, "l_bracket.urdf")

    sim = BulletSim()
    results = {}

    if os.path.exists(sphere_urdf):
        states = run_e1_freefall(sim, sphere_urdf, args.logs_dir)
        results["e1_freefall"] = [asdict(s) for s in states[::24]]
    if os.path.exists(box_urdf):
        states = run_e2_incline(sim, box_urdf, args.logs_dir)
        results["e2_incline"]  = [asdict(s) for s in states[::24]]
    if os.path.exists(bracket_urdf):
        states = run_e3_rotation(sim, bracket_urdf, args.logs_dir)
        results["e3_rotation"] = [asdict(s) for s in states[::24]]

    sim.close()

    log_path = os.path.join(args.logs_dir, "benchmark_results.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Figures + videos in: {args.logs_dir}/")

if __name__ == "__main__":
    main()
