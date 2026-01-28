import pybullet as p
import pybullet_data
import time

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    
    # Plane
    planeId = p.loadURDF("plane.urdf")
    p.changeDynamics(planeId, -1, restitution=0.5)

    # Reset Camera
    p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0, 0, 1])

    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
    
    # --- Pair 1: Elastic Collision (Bouncing) ---
    # Green Spheres
    visElastic = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0, 1, 0, 1])
    
    # Sphere 1 (Moving Right)
    el1 = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col, baseVisualShapeIndex=visElastic, basePosition=[-4, 2, 1])
    p.changeDynamics(el1, -1, restitution=1.0) # Max bounce
    
    # Sphere 2 (Stationary)
    el2 = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col, baseVisualShapeIndex=visElastic, basePosition=[0, 2, 1])
    p.changeDynamics(el2, -1, restitution=1.0)

    # Launch Sphere 1
    p.resetBaseVelocity(el1, linearVelocity=[5, 0, 0])


    # --- Pair 2: Inelastic Collision (Sticking/Dead Stop) ---
    # Grey Spheres
    visInelastic = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0.5, 0.5, 0.5, 1])
    
    # Sphere 3 (Moving Right)
    in1 = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col, baseVisualShapeIndex=visInelastic, basePosition=[-4, -2, 1])
    p.changeDynamics(in1, -1, restitution=0.0) # No bounce
    
    # Sphere 4 (Stationary)
    in2 = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col, baseVisualShapeIndex=visInelastic, basePosition=[0, -2, 1])
    p.changeDynamics(in2, -1, restitution=0.0)
    
    # Launch Sphere 3
    p.resetBaseVelocity(in1, linearVelocity=[5, 0, 0])

    print("Simulating Interaction Forces (Elastic vs Inelastic) for 10 seconds...")
    print("Top Pair (Green): Elastic Collision (High Restitution)")
    print("Bottom Pair (Grey): Inelastic Collision (Zero Restitution)")

    dt = 1./240.
    p.setTimeStep(dt)
    
    for _ in range(240 * 10):
        p.stepSimulation()
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
