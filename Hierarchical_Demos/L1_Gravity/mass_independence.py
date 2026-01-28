import pybullet as p
import pybullet_data
import time

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # Sphere 1: Light (1kg), Red
    startPos1 = [-1, 0, 5]
    col1 = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
    vis1 = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[1, 0, 0, 1])
    mass1 = 1
    p.createMultiBody(baseMass=mass1, baseCollisionShapeIndex=col1, baseVisualShapeIndex=vis1, basePosition=startPos1)
    
    # Sphere 2: Heavy (10kg), Blue
    startPos2 = [1, 0, 5]
    col2 = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
    vis2 = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0, 0, 1, 1])
    mass2 = 10
    p.createMultiBody(baseMass=mass2, baseCollisionShapeIndex=col2, baseVisualShapeIndex=vis2, basePosition=startPos2)
    
    print("Simulating Mass Independence (Light vs Heavy) for 5 seconds...")
    dt = 1./240.
    p.setTimeStep(dt)
    
    for _ in range(240 * 5):
        p.stepSimulation()
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
