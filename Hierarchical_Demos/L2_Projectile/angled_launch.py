import pybullet as p
import pybullet_data
import time

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # Reset Camera
    p.resetDebugVisualizerCamera(cameraDistance=12, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 2])

    
    # Sphere
    startPos = [0, 0, 1]
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[1, 1, 0, 1]) # Yellow
    
    bodyId = p.createMultiBody(baseMass=1,
                               baseCollisionShapeIndex=col,
                               baseVisualShapeIndex=vis,
                               basePosition=startPos)
    
    # Apply initial velocity at an angle (Vx = 5, Vz = 8)
    # Approx 60 degrees launch if we consider vector components
    p.resetBaseVelocity(bodyId, linearVelocity=[5, 0, 8])
    
    print("Simulating Angled Launch for 5 seconds...")
    dt = 1./240.
    p.setTimeStep(dt)
    
    for _ in range(240 * 5):
        p.stepSimulation()
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
