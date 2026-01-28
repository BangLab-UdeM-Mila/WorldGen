import pybullet as p
import pybullet_data
import time

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    
    # Plane with low restitution
    planeId = p.loadURDF("plane.urdf")
    p.changeDynamics(planeId, -1, restitution=0.0)
    
    # Ball
    startPos = [0, 0, 5]
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0.5, 0.5, 0.5, 1]) # Grey
    
    bodyId = p.createMultiBody(baseMass=1,
                               baseCollisionShapeIndex=col,
                               baseVisualShapeIndex=vis,
                               basePosition=startPos)
    
    # Set zero restitution
    p.changeDynamics(bodyId, -1, restitution=0.0)
    
    print("Simulating Inelastic Collision (Zero Restitution) for 5 seconds...")
    dt = 1./240.
    p.setTimeStep(dt)
    
    for _ in range(240 * 5):
        p.stepSimulation()
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
