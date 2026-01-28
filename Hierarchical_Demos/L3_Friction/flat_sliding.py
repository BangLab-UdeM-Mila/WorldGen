import pybullet as p
import pybullet_data
import time

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    
    # Reset Camera
    p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[2, 0, 0])
    
    # Plane with friction
    planeId = p.loadURDF("plane.urdf")
    p.changeDynamics(planeId, -1, lateralFriction=0.5)
    
    # Box
    startPos = [0, 0, 0.5]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0, 0, 1, 1])
    
    bodyId = p.createMultiBody(baseMass=1,
                               baseCollisionShapeIndex=col,
                               baseVisualShapeIndex=vis,
                               basePosition=startPos)
    
    # Set Friction for Box
    p.changeDynamics(bodyId, -1, lateralFriction=0.5)
    
    # Initial Push
    p.resetBaseVelocity(bodyId, linearVelocity=[10, 0, 0])
    
    print("Simulating Flat Sliding (Friction Deceleration) for 5 seconds...")
    dt = 1./240.
    p.setTimeStep(dt)
    
    for _ in range(240 * 5):
        p.stepSimulation()
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
