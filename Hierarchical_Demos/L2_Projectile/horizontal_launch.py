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
    p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[5, 0, 2])
    
    # Sphere
    startPos = [0, 0, 5]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0, 1, 0, 1]) # Green
    
    bodyId = p.createMultiBody(baseMass=1,
                               baseCollisionShapeIndex=col,
                               baseVisualShapeIndex=vis,
                               basePosition=startPos,
                               baseOrientation=startOrientation)
    
    # Apply initial horizontal velocity (Vx = 5 m/s)
    p.resetBaseVelocity(bodyId, linearVelocity=[5, 0, 0])
    
    print("Simulating Horizontal Launch for 5 seconds...")
    dt = 1./240.
    p.setTimeStep(dt)
    
    for _ in range(240 * 5):
        p.stepSimulation()
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
