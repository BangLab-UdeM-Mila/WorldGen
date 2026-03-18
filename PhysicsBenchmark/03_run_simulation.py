"""
WorldGen Physics Benchmark Simulation
======================================
Runs three canonical benchmark experiments in PyBullet to establish
physics-consistency baselines for LLM-generated simulation code.

Experiments
-----------
  E1  Free-fall + bounce    (sphere)      — tests restitution / energy recovery
  E2  Incline sliding       (bevel_box)   — tests friction / Coulomb model
  E3  Torque-induced spin   (l_bracket)   — tests inertia tensor accuracy

Each experiment:
  1. Loads URDF into PyBullet
  2. Steps simulation at 240 Hz
  3. Records state (pos, vel, ang_vel, contact_forces) every step
  4. Saves to logs/<experiment>.json
  5. Generates a matplotlib figure

Usage:
  python3 03_run_simulation.py --urdfs_dir ./urdfs --logs_dir ./logs
"""

import os
import sys
import json
import time
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
    print("ERROR: pybullet not found. Run: pip3 install pybullet")
    sys.exit(1)


# ── Simulation configuration ──────────────────────────────────────────────────

TIMESTEP   = 1.0 / 240.0   # 240 Hz — PyBullet default, good for contact stability
GRAVITY    = (0, 0, -9.81)  # m/s² (standard gravity)
SOLVER_ITER = 50             # PGS constraint solver iterations


@dataclass
class SimState:
    """One timestep of simulation data."""
    t:          float
    pos:        List[float]   # (x, y, z) m
    quat:       List[float]   # (qx, qy, qz, qw)
    vel:        List[float]   # (vx, vy, vz) m/s
    ang_vel:    List[float]   # (wx, wy, wz) rad/s
    ke:         float         # kinetic energy (J)
    pe:         float         # potential energy (J, approx)
    n_contacts: int


# ── Simulation engine ─────────────────────────────────────────────────────────

class BulletSim:
    """
    Thin wrapper around PyBullet for repeatable benchmark experiments.
    Uses DIRECT mode (no GUI) for headless execution.
    Set headless=False to open the GUI for inspection.
    """

    def __init__(self, headless=True):
        self.client = p.connect(p.DIRECT if headless else p.GUI)
        p.setPhysicsEngineParameter(
            numSolverIterations=SOLVER_ITER,
            physicsClientId=self.client
        )
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

    def load_object(self, urdf_path, pos=(0, 0, 0), orn=(0, 0, 0, 1)):
        obj_id = p.loadURDF(
            urdf_path,
            basePosition=pos,
            baseOrientation=orn,
            physicsClientId=self.client
        )
        return obj_id

    def set_dynamics(self, obj_id, friction=0.5, restitution=0.3, spin_friction=0.01):
        p.changeDynamics(
            obj_id, -1,
            lateralFriction=friction,
            restitution=restitution,
            spinningFriction=spin_friction,
            rollingFriction=0.001,
            physicsClientId=self.client
        )

    def get_state(self, obj_id, t, mass) -> SimState:
        pos, quat = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client)
        vel, ang  = p.getBaseVelocity(obj_id, physicsClientId=self.client)
        v_sq   = sum(v**2 for v in vel)
        ke     = 0.5 * mass * v_sq
        pe     = mass * 9.81 * pos[2]
        n_cont = len(p.getContactPoints(obj_id, physicsClientId=self.client))
        return SimState(
            t=round(t, 5), pos=list(pos), quat=list(quat),
            vel=list(vel), ang_vel=list(ang),
            ke=round(ke, 6), pe=round(pe, 6), n_contacts=n_cont
        )

    def step(self):
        p.stepSimulation(physicsClientId=self.client)

    def close(self):
        p.disconnect(physicsClientId=self.client)


# ── Experiment E1: Free-fall bounce (sphere) ──────────────────────────────────

def run_e1_freefall(sim: BulletSim, sphere_urdf: str, duration=2.0) -> List[SimState]:
    """
    Drop a rubber sphere from h=0.5 m onto a flat plane.
    Records height vs time.  Ground truth: z(t) = h - 0.5·g·t² until first contact.

    Expected outcome: exponentially decaying bounce height (restitution < 1).
    Physics check: peak heights should follow h_n = e^(2n) · h_0
                   where e = coefficient of restitution.
    """
    sim.reset()
    sim.load_plane()
    obj = sim.load_object(sphere_urdf, pos=(0, 0, 0.5))
    sim.set_dynamics(obj, friction=0.8, restitution=0.7)

    # Read mass for energy computation
    mass = p.getDynamicsInfo(obj, -1, physicsClientId=sim.client)[0]

    states, t = [], 0.0
    steps = int(duration / TIMESTEP)
    for _ in range(steps):
        sim.step()
        t += TIMESTEP
        states.append(sim.get_state(obj, t, mass))
    return states


# ── Experiment E2: Inclined-plane sliding (bevel_box) ─────────────────────────

def run_e2_incline(sim: BulletSim, box_urdf: str, duration=3.0) -> List[SimState]:
    """
    Place a wooden box on a 20° inclined plane, release from rest.
    Measures sliding velocity vs time.

    Ground truth (no friction): v(t) = g·sin(θ)·t
    With friction:               a = g·(sin(θ) − μ·cos(θ))

    Physics check: measure μ from recorded acceleration and compare
    against the set lateralFriction value.
    """
    sim.reset()

    # Build inclined plane as a thin box tilted 20 degrees
    theta = np.deg2rad(20)
    plane_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.5, 0.5, 0.01],
        physicsClientId=sim.client
    )
    plane_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.5, 0.5, 0.01],
        rgbaColor=[0.6, 0.6, 0.6, 1],
        physicsClientId=sim.client
    )
    orn = p.getQuaternionFromEuler([theta, 0, 0])
    p.createMultiBody(0, plane_col, plane_vis,
                      basePosition=[0, 0, 0],
                      baseOrientation=orn,
                      physicsClientId=sim.client)

    # Object placed at top of incline
    start_pos = [0, -0.3 * np.cos(theta), 0.3 * np.sin(theta) + 0.05]
    obj = sim.load_object(box_urdf, pos=start_pos,
                          orn=p.getQuaternionFromEuler([theta, 0, 0]))
    sim.set_dynamics(obj, friction=0.6, restitution=0.3)
    mass = p.getDynamicsInfo(obj, -1, physicsClientId=sim.client)[0]

    states, t = [], 0.0
    for _ in range(int(duration / TIMESTEP)):
        sim.step()
        t += TIMESTEP
        states.append(sim.get_state(obj, t, mass))
    return states


# ── Experiment E3: Free rotation (l_bracket inertia benchmark) ────────────────

def run_e3_rotation(sim: BulletSim, bracket_urdf: str, duration=2.0) -> List[SimState]:
    """
    Apply an initial angular velocity to the L-bracket and let it spin freely
    (no gravity, no contacts).  Dzhanibekov effect may occur if principal
    inertia axes differ enough.

    Physics check: without external torques, angular momentum L = I·ω is
    conserved.  Verify that |L| is constant over the simulation.
    """
    sim.reset()
    p.setGravity(0, 0, 0, physicsClientId=sim.client)   # torque-free environment

    obj = sim.load_object(bracket_urdf, pos=(0, 0, 1))
    mass = p.getDynamicsInfo(obj, -1, physicsClientId=sim.client)[0]

    # Give an initial spin about the intermediate principal axis (unstable →
    # Dzhanibekov tumbling is physically expected)
    p.resetBaseVelocity(obj,
                        linearVelocity=[0, 0, 0],
                        angularVelocity=[1.0, 0.1, 0.0],   # rad/s
                        physicsClientId=sim.client)

    states, t = [], 0.0
    for _ in range(int(duration / TIMESTEP)):
        sim.step()
        t += TIMESTEP
        states.append(sim.get_state(obj, t, mass))
    return states


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_e1(states: List[SimState], out_dir: str):
    t   = [s.t     for s in states]
    z   = [s.pos[2] for s in states]
    ke  = [s.ke    for s in states]
    pe  = [s.pe    for s in states]
    te  = [s.ke + s.pe for s in states]

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
    path = os.path.join(out_dir, "e1_freefall.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Figure saved: {path}")


def plot_e2(states: List[SimState], out_dir: str):
    t  = [s.t        for s in states]
    vy = [abs(s.vel[1]) for s in states]   # speed along incline direction

    # Estimate acceleration from linear regression on the first 1 second
    t_arr  = np.array(t)
    vy_arr = np.array(vy)
    mask   = t_arr < 1.0
    if mask.sum() > 10:
        coeffs = np.polyfit(t_arr[mask], vy_arr[mask], 1)
        a_meas = coeffs[0]
        a_theory = 9.81 * (np.sin(np.deg2rad(20)) - 0.6 * np.cos(np.deg2rad(20)))
        label = (f"Measured a = {a_meas:.3f} m/s²\n"
                 f"Theory a   = {a_theory:.3f} m/s²")
    else:
        label = "insufficient data"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, vy, color='darkorange', lw=1.5, label=label)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed along incline (m/s)")
    ax.set_title("E2: Incline Sliding — Bevel Box", fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, "e2_incline.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Figure saved: {path}")


def plot_e3(states: List[SimState], out_dir: str):
    t   = [s.t        for s in states]
    wx  = [s.ang_vel[0] for s in states]
    wy  = [s.ang_vel[1] for s in states]
    wz  = [s.ang_vel[2] for s in states]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, wx, label="ωx", color='crimson',  lw=1.2)
    ax.plot(t, wy, label="ωy", color='royalblue', lw=1.2)
    ax.plot(t, wz, label="ωz", color='seagreen',  lw=1.2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_title("E3: Torque-Free Rotation — L-Bracket (Dzhanibekov Effect)",
                 fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, "e3_rotation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Figure saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdfs_dir", default="./urdfs")
    parser.add_argument("--logs_dir",  default="./logs")
    parser.add_argument("--gui",       action="store_true",
                        help="Open PyBullet GUI (requires display)")
    args = parser.parse_args()

    os.makedirs(args.logs_dir, exist_ok=True)
    headless = not args.gui

    sphere_urdf  = os.path.join(args.urdfs_dir, "sphere.urdf")
    box_urdf     = os.path.join(args.urdfs_dir, "bevel_box.urdf")
    bracket_urdf = os.path.join(args.urdfs_dir, "l_bracket.urdf")

    sim = BulletSim(headless=headless)

    results = {}

    # ── E1 ──
    if os.path.exists(sphere_urdf):
        print("\n[E1] Free-fall bounce (sphere)...")
        states = run_e1_freefall(sim, sphere_urdf, duration=2.0)
        plot_e1(states, args.logs_dir)
        results["e1_freefall"] = [asdict(s) for s in states[::24]]  # 10 Hz log
        print(f"  {len(states)} timesteps recorded")
    else:
        print(f"[SKIP E1] {sphere_urdf} not found")

    # ── E2 ──
    if os.path.exists(box_urdf):
        print("\n[E2] Incline sliding (bevel_box)...")
        states = run_e2_incline(sim, box_urdf, duration=3.0)
        plot_e2(states, args.logs_dir)
        results["e2_incline"] = [asdict(s) for s in states[::24]]
        print(f"  {len(states)} timesteps recorded")
    else:
        print(f"[SKIP E2] {box_urdf} not found")

    # ── E3 ──
    if os.path.exists(bracket_urdf):
        print("\n[E3] Torque-free rotation (l_bracket)...")
        states = run_e3_rotation(sim, bracket_urdf, duration=2.0)
        plot_e3(states, args.logs_dir)
        results["e3_rotation"] = [asdict(s) for s in states[::24]]
        print(f"  {len(states)} timesteps recorded")
    else:
        print(f"[SKIP E3] {bracket_urdf} not found")

    sim.close()

    # Save all results to JSON
    log_path = os.path.join(args.logs_dir, "benchmark_results.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Results saved to {log_path}")
    print(f"       Figures in {args.logs_dir}/")


if __name__ == "__main__":
    main()
