import pybullet as p
import pybullet_data
import time
import math

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    
    # Create Ramp (using a large thin box)
    rampCol = p.createCollisionShape(p.GEOM_BOX, halfExtents=[5, 5, 0.1])
    rampVis = p.createVisualShape(p.GEOM_BOX, halfExtents=[5, 5, 0.1], rgbaColor=[0.8, 0.8, 0.8, 1])
    
    # Ramp Angle: 30 degrees
    angle = math.radians(30)
    rampOrn = p.getQuaternionFromEuler([0, angle, 0]) 
    
    rampId = p.createMultiBody(baseMass=0, # Static
                               baseCollisionShapeIndex=rampCol,
                               baseVisualShapeIndex=rampVis,
                               basePosition=[0, 0, 2],
                               baseOrientation=rampOrn)
    
    p.changeDynamics(rampId, -1, lateralFriction=0.1) # Low friction to allow sliding

    # Reset Camera
    p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 2])

    
    # Box to Slide
    boxCol = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    boxVis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 1])
    
    # Position box on ramp
    # Simple trig to place it on surface
    # Center of ramp is at 0,0,2. 
    # Let's put it slightly higher up the ramp.
    # Ramp orientation is rotated around Y.
    
    boxStartPos = [-2, 0, 4] # Rough guess, gravity will settle it or it will fall
    boxId = p.createMultiBody(baseMass=1,
                              baseCollisionShapeIndex=boxCol,
                              baseVisualShapeIndex=boxVis,
                              basePosition=boxStartPos)
    
    p.changeDynamics(boxId, -1, lateralFriction=0.2)
    
    print("Simulating Incline Sliding for 5 seconds...")
    dt = 1./240.
    p.setTimeStep(dt)
    
    for _ in range(240 * 5):
        p.stepSimulation()
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
