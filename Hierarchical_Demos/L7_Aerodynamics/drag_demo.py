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
    p.resetDebugVisualizerCamera(cameraDistance=15, cameraYaw=0, cameraPitch=-10, cameraTargetPosition=[0, 0, 5])
    
    startPos = [0, 0, 20] # Drop from high up
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=1)
    
    # 1. Vacuum/Low Drag Sphere (Red)
    visRed = p.createVisualShape(p.GEOM_SPHERE, radius=1, rgbaColor=[1, 0, 0, 1])
    id1 = p.createMultiBody(baseMass=1,
                            baseCollisionShapeIndex=col,
                            baseVisualShapeIndex=visRed,
                            basePosition=[-2, 0, 20])
                            
    # 2. High Drag Sphere (Blue)
    visBlue = p.createVisualShape(p.GEOM_SPHERE, radius=1, rgbaColor=[0, 0, 1, 1])
    id2 = p.createMultiBody(baseMass=1,
                            baseCollisionShapeIndex=col,
                            baseVisualShapeIndex=visBlue,
                            basePosition=[2, 0, 20])
                            
    # Apply Linear Damping to Simulate Air Resistance
    # linearDamping=0 is default (vacuum-like/low friction)
    # linearDamping=0.9 is high drag (like a parachute or light balloon)
    p.changeDynamics(id2, -1, linearDamping=0.9)
    
    print("Simulating Aerodynamic Drag for 10 seconds...")
    print("Red: No Drag")
    print("Blue: High Drag (Falls Slower)")
    
    dt = 1./240.
    p.setTimeStep(dt)
    
    for _ in range(240 * 10):
        p.stepSimulation()
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
