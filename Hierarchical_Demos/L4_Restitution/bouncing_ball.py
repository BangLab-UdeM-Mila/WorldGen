import pybullet as p
import pybullet_data
import time

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    
    # Plane with high restitution
    planeId = p.loadURDF("plane.urdf")
    p.changeDynamics(planeId, -1, restitution=1.0)
    
    # Bouncing Ball
    startPos = [0, 0, 5]
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0, 1, 1, 1])
    
    bodyId = p.createMultiBody(baseMass=1,
                               baseCollisionShapeIndex=col,
                               baseVisualShapeIndex=vis,
                               basePosition=startPos)
    
    # Set high restitution (bounciness)
    p.changeDynamics(bodyId, -1, restitution=0.9)
    
    # Reset Camera
    p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 2])

    print("Simulating Bouncing Ball (High Restitution) for 30 seconds...")
    dt = 1./240.
    p.setTimeStep(dt)
    
    for _ in range(240 * 30):
        p.stepSimulation()
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
