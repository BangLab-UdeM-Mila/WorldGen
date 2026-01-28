import pybullet as p
import pybullet_data
import time
import math

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    
    # Plane
    p.loadURDF("plane.urdf")
    
    # 1. Non-spinning Top (Falls over)
    # Using a cylinder with small radius at bottom implies inherent instability if not spinning
    # Create a simple "top" shape using a cylinder
    # Upsizing the top for visibility
    topCol = p.createCollisionShape(p.GEOM_CYLINDER, radius=1, height=0.2)
    topVis = p.createVisualShape(p.GEOM_CYLINDER, radius=1, length=0.2, rgbaColor=[1, 0, 0, 1])
    
    # Stand it on its edge (rotated 90 deg around X or Y)
    startOrn = p.getQuaternionFromEuler([math.pi/2, 0, 0])
    
    # Left Top (Static/Low Spin) - Control Group
    p.createMultiBody(baseMass=1,
                      baseCollisionShapeIndex=topCol,
                      baseVisualShapeIndex=topVis,
                      basePosition=[-2, 0, 1],
                      baseOrientation=startOrn)
                      
    # Right Top (High Spin) - Experimental Group
    visSpin = p.createVisualShape(p.GEOM_CYLINDER, radius=1, length=0.2, rgbaColor=[0, 1, 0, 1])
    
    spinningTopId = p.createMultiBody(baseMass=1,
                                      baseCollisionShapeIndex=topCol,
                                      baseVisualShapeIndex=visSpin,
                                      basePosition=[2, 0, 1],
                                      baseOrientation=startOrn)
                                      
    # Apply High Angular Velocity to the Green Top around its local Z axis (which is World Y now)
    # The local Z of the cylinder is the axis of symmetry. 
    # Since we rotated it 90 deg on X, the cylinder's Z axis points along World Y? 
    # Let's verify: Cylinder defaults to Z-up. Rotated 90 on X (roll) -> Z becomes -Y?
    # Let's just spin it along World Y (the axis it's standing on approx).
    
    p.resetBaseVelocity(spinningTopId, angularVelocity=[0, 50, 0])
    
    # Reset Camera
    p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 1])
    
    print("Simulating Gyroscopic Stability for 15 seconds...")
    print("Red: No Spin (Falls)")
    print("Green: High Spin (Stable)")
    
    dt = 1./240.
    p.setTimeStep(dt)
    
    for _ in range(240 * 15):
        p.stepSimulation()
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
