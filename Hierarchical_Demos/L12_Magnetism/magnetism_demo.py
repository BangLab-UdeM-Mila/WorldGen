import pybullet as p
import pybullet_data
import time
import math

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    
    p.loadURDF("plane.urdf")
    
    # Reset Camera
    p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-50, cameraTargetPosition=[0, 0, 0])
    
    # 1. Central "Magnet" (Static)
    magnetCol = p.createCollisionShape(p.GEOM_SPHERE, radius=1)
    magnetVis = p.createVisualShape(p.GEOM_SPHERE, radius=1, rgbaColor=[1, 0, 0, 1])
    magnetPos = [0, 0, 0.5]
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=magnetCol, baseVisualShapeIndex=magnetVis, basePosition=magnetPos)
    
    # 2. Metallic Balls (Free moving)
    num_balls = 8
    balls = []
    
    ballCol = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
    ballVis = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[0.5, 0.5, 0.5, 1])
    
    radius_circle = 5.0
    
    for i in range(num_balls):
        angle = (2 * math.pi / num_balls) * i
        x = radius_circle * math.cos(angle)
        y = radius_circle * math.sin(angle)
        
        uid = p.createMultiBody(baseMass=1,
                                baseCollisionShapeIndex=ballCol,
                                baseVisualShapeIndex=ballVis,
                                basePosition=[x, y, 0.5])
        balls.append(uid)
        
    print("Simulating Magnetism (Attraction) for 10 seconds...")
    print("Red Sphere: Magnet")
    print("Grey Spheres: Metallic Balls attracted to Magnet")
    
    magnetic_strength = 500.0
    
    dt = 1./240.
    p.setTimeStep(dt)
    
    for _ in range(240 * 10):
        # Apply Magnetic Force
        for ballId in balls:
            pos, _ = p.getBasePositionAndOrientation(ballId)
            
            # Vector to magnet
            dx = magnetPos[0] - pos[0]
            dy = magnetPos[1] - pos[1]
            dz = magnetPos[2] - pos[2]
            
            dist = (dx**2 + dy**2 + dz**2)**0.5
            
            if dist > 0.1: # Avoid division by zero close up
                # F = k / r^2 direction
                force_mag = magnetic_strength / (dist**2)
                
                # Direction unit vector
                fx = (dx/dist) * force_mag
                fy = (dy/dist) * force_mag
                fz = (dz/dist) * force_mag
                
                p.applyExternalForce(ballId, -1, [fx, fy, fz], pos, p.WORLD_FRAME)
                
                # Draw lines
                if i == 0: # Draw just for one to avoid clutter
                    p.addUserDebugLine(pos, magnetPos, [1, 1, 0], 1, lifeTime=dt)

        p.stepSimulation()
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
