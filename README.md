# PyBullet Physics Demos: A Hierarchical Guide

This repository contains a structured series of physics simulations using PyBullet. Below is a detailed breakdown of each level, explaining the physical concepts, the formulas at play, and the exact experimental setups used in the code.

---

## Level 1: Basics of Gravity
**Goal**: Verify Newton's Law of Universal Gravitation locally.

### 1. `basic_freefall.py`
*   **Physical Concept**: **Standard Gravity**.
*   **Formula**: Force $F = mg$, Position $z(t) = z_0 - \frac{1}{2}gt^2$.
*   **Experiment Setup**:
    *   Gravity: $g = -9.81 m/s^2$ (Z-axis).
    *   Object: 1 Sphere dropped from rest at $z=10m$.
*   **Outcome**: The sphere falls, accelerating downwards at a constant rate until it collides with the ground plane.

### 2. `mass_independence.py`
*   **Physical Concept**: **Equivalence Principle**. Acceleration is independent of mass ($a=g$).
*   **Formula**: $ma = mg \implies a = g$.
*   **Experiment Setup**:
    *   **Red Sphere**: Mass = **1 kg**.
    *   **Blue Sphere**: Mass = **10 kg**.
    *   Both dropped simultaneously from $z=5m$.
*   **Outcome**: Despite the 10x mass difference, both spheres hit the ground at the exact same moment, demonstrating that gravity acts equally on all internal mass.

---

## Level 2: Projectile Motion
**Goal**: Analyze 2D motion with independent axes.

### 1. `horizontal_launch.py`
*   **Physical Concept**: **Independence of Motion**.
*   **Formula**: $x(t) = v_x t$, $z(t) = z_0 - \frac{1}{2}gt^2$.
*   **Experiment Setup**:
    *   Initial Velocity: $\vec{v} = [5, 0, 0] m/s$ (Horizontal).
    *   Start Position: $z=5m$.
*   **Outcome**: The sphere travels forward 5 meters for every second it falls, tracing a parabolic arc.

### 2. `angled_launch.py`
*   **Physical Concept**: **Trajectory**.
*   **Formula**: Range $R = \frac{v^2 \sin(2\theta)}{g}$.
*   **Experiment Setup**:
    *   Initial Velocity: $\vec{v} = [5, 0, 8] \approx 9.4 m/s$.
    *   Launch Angle: $\theta \approx 58^\circ$.
*   **Outcome**: The sphere rises to a peak height and falls, covering a specific horizontal distance determined by the launch vector components.

---

## Level 3: Friction
**Goal**: Observe resistive contact forces.

### 1. `flat_sliding.py`
*   **Physical Concept**: **Kinetic Friction**.
*   **Formula**: $F_f = -\mu_k N$.
*   **Experiment Setup**:
    *   **Friction Coefficient** ($\mu$): **0.5**.
    *   Initial Velocity: $10 m/s$.
*   **Outcome**: The box slides but rapidly decelerates due to the opposing friction force ($F_f \approx 0.5 \cdot mg$) until it comes to a stop.

### 2. `incline_sliding.py`
*   **Physical Concept**: **Inclined Plane Dynamics**.
*   **Experiment Setup**:
    *   **Ramp Angle**: $30^\circ$.
    *   **Ramp Friction**: 0.1. **Box Friction**: 0.2.
*   **Outcome**: The gravity component pulling down the slope ($mg \sin 30^\circ$) overcomes the static friction limit, causing the box to slide down the ramp.

---

## Level 4: Restitution (Bounciness)
**Goal**: Model energy dissipation in collisions.

### 1. `bouncing_ball.py`
*   **Physical Concept**: **Coefficient of Restitution ($e$)**.
*   **Formula**: $v_{after} = -e \cdot v_{before}$.
*   **Experiment Setup**:
    *   **Floor Restitution**: 1.0 (Perfectly elastic).
    *   **Ball Restitution**: **0.9** (Loses 10% velocity per bounce).
    *   Drop Height: $5m$.
*   **Outcome**: The ball bounces multiple times. The peak height decreases with each bounce ($h_{n} = 0.9^2 h_{n-1} = 0.81 h_{n-1}$) until it settles.

### 2. `inelastic_collision.py`
*   **Physical Concept**: **Perfectly Inelastic Collision**.
*   **Experiment Setup**:
    *   **Restitution**: **0.0**.
*   **Outcome**: The ball hits the ground and stops instantly. All kinetic energy relative to the floor is dissipated in a single frame.

---

## Level 5: Constraints
**Goal**: Limit degrees of freedom (DOF).

### 1. `simple_pendulum.py`
*   **Physical Concept**: **Pendulum Motion**.
*   **Formula**: Period $T \approx 2\pi\sqrt{L/g}$.
*   **Experiment Setup**:
    *   **Constraint**: Point-to-Point joint connecting a Pivot Body and a Bob Body.
    *   **Length**: 1.0 meter.
*   **Outcome**: The bob is constrained to move on the surface of a sphere (radius 1m) around the pivot. Gravity pulls it down, Tension pulls it up, resulting in oscillatory swing.

---

## Level 6: Angular Momentum
**Goal**: Explore rotational stability.

### 1. `gyroscopic_stability.py`
*   **Physical Concept**: **Gyroscopic Precession**.
*   **Formula**: $\vec{\tau} = \frac{d\vec{L}}{dt}$.
*   **Experiment Setup**:
    *   **Red Top**: $0$ rad/s angular velocity.
    *   **Green Top**: $[0, 50, 0]$ rad/s angular velocity (Spinning on Y-axis).
*   **Outcome**: The Red top falls over immediately due to the gravitational torque. The Green top stays upright, precessing slowly around the vertical axis due to conservation of angular momentum.

---

## Level 7: Aerodynamics
**Goal**: Simulate simplified fluid drag.

### 1. `drag_demo.py`
*   **Physical Concept**: **Linear Damping**.
*   **Formula**: $F_{drag} \propto -v$.
*   **Experiment Setup**:
    *   **Red Sphere**: Damping = 0 (Vacuum physics).
    *   **Blue Sphere**: **Linear Damping = 0.9**.
*   **Outcome**: The Red sphere accelerates constantly at $9.81 m/s^2$. The Blue sphere quickly reaches a (slow) terminal velocity where Drag cancels Gravity.

---

## Level 8: Soft Body
**Goal**: Simulate deformable materials (Flag in Wind).

### 1. `soft_body_cloth.py`
*   **Physical Concept**: **Mass-Spring Network** & **Aerodynamic Force**.
*   **Experiment Setup**:
    *   **Grid**: 15x10 nodes (Horizontal x Vertical).
    *   **Constraints**: Left column pinned (Flagpole). Neighbors linked by Point2Point joints.
    *   **Wind Force**: $\vec{F}_{wind} = [0, 8, 0] + noise$. Applied continuously to all moveable nodes.
*   **Outcome**: The grid behaves like a flag attached to a pole. Gravity pulls it down, while the wind force pushes it sideways, causing it to wave and flutter due to the turbulence (noise).hbors (Springs).
    *   **Parameters**: Node Radius = 0.1, Spacing = 0.3.
    *   **Constraint**: Top row pinned (Mass=0).
*   **Outcome**: The grid behaves like a flexible piece of chainmail cloth, hanging and waving under gravity.

---

## Level 9: Interaction Forces
**Goal**: Side-by-side comparison of material properties.

### 1. `elastic_vs_inelastic.py`
*   **Physical Concept**: **Collision Response**.
*   **Experiment Setup**:
    *   **Green Pair**: Restitution **1.0**.
    *   **Grey Pair**: Restitution **0.0**.
    *   Collision: Head-on or One-way impact at $v=5 m/s$.
*   **Outcome**: The Green pair bounces apart aggressively (conversing energy). The Grey pair collides and "clumps" together (dissipating energy).

---

## Level 10: Springs
**Goal**: Visualize simple harmonic motion.

### 1. `hookes_law.py`
*   **Physical Concept**: **Hooke's Law**.
*   **Formula**: $F = -k(x - x_{rest}) - c v$.
*   **Experiment Setup**:
    *   **Spring Constant ($k$)**: 50.0.
    *   **Damping ($c$)**: 1.0.
    *   **Rest Length**: 3.0 m.
    *   **Start Pos**: Same as anchor (0 extension).
*   **Outcome**: The box drops under gravity, stretching the spring. The restoring force pulls it back up. It oscillates around the equilibrium point with slowly decaying amplitude.

---

## Level 11: Buoyancy
**Goal**: Simulate fluid displacement forces.

### 1. `buoyancy_demo.py`
*   **Physical Concept**: **Archimedes' Principle**.
*   **Formula**: $F_b = \rho_{fluid} g V_{sub}$.
*   **Experiment Setup**:
    *   **Fluid Density**: 10.0.
    *   **Object Mass**: 5.0 (Effective density ~5).
    *   **Water Level**: $z=0$.
    *   **Manual Force**: Applied per frame based on submerged height.
*   **Outcome**: The object falls into the water and floats with approximately 50% of its volume submerged ($Ratio = \rho_{obj}/\rho_{fluid} = 0.5$). It bobs up and down before settling due to water drag.

---

## Level 12: Magnetism
**Goal**: Simulate distance-based field forces.

### 1. `magnetism_demo.py`
*   **Physical Concept**: **Inverse Square Law**.
*   **Formula**: $F = k \frac{1}{r^2}$.
*   **Experiment Setup**:
    *   **Magnetic Strength**: 500.0.
    *   **Configuration**: 8 metallic balls in a circle, 1 central magnet.
*   **Outcome**: All 8 balls are pulled towards the center. As they get closer ($r$ decreases), the force increases quadratically, causing them to accelerate rapidly into the magnet.

---

## Running the Demos
Run any demo by executing the script from the root directory, e.g.:
```bash
python L11_Buoyancy/buoyancy_demo.py
```
At the current moment, the only extra environment that needs to be installed is `pybullet`:
```bash
pip install pybullet
```
