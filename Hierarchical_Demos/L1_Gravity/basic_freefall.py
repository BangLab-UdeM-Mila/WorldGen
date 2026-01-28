import pybullet as p
import pybullet_data
import time

def main():
    # Connect to GUI
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Set Reset and Gravity
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    
    # Load Plane
    p.loadURDF("plane.urdf")
    
    # Reset Camera for better view
    p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0, 0, 2])

    
    # Create a sphere
    startPos = [0, 0, 5] # Drop from 5 meters
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[1, 0, 0, 1])
    
    mass = 1
    bodyId = p.createMultiBody(baseMass=mass,
                               baseCollisionShapeIndex=col,
                               baseVisualShapeIndex=vis,
                               basePosition=startPos,
                               baseOrientation=startOrientation)
    
    print("Simulating Freefall for 5 seconds...")
    dt = 1./240.
    p.setTimeStep(dt)
    
    # Run Simulation
    for _ in range(240 * 5): # 5 seconds
        p.stepSimulation()
        pos, _ = p.getBasePositionAndOrientation(bodyId)
        # optional: print height to console occasionally
        # if _ % 60 == 0:
        #     print(f"Height: {pos[2]:.2f}")
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
