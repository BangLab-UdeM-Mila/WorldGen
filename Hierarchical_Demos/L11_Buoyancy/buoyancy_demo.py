import pybullet as p
import pybullet_data
import time

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    
    p.setGravity(0, 0, -9.81)
    
    # Reset Camera
    # Align with surface (z=0) to see motion clearly
    p.resetDebugVisualizerCamera(cameraDistance=12, cameraYaw=0, cameraPitch=-5, cameraTargetPosition=[0, 0, 0])
    
    # Visual "Water" Plane (Cyan, Semi-Transparent)
    # Surface at z = 0. Deep block (10m deep).
    # Center z = -5. HalfExtent Z = 5. Top = 0.
    waterCol = -1 # No collision
    waterVis = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 10, 5], rgbaColor=[0, 1, 1, 0.4]) # Cyan, semi-transparent
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=waterVis, basePosition=[0, 0, -5]) 
    
    water_level = 0.0
    
    # Cube (Wood-like density)
    cubeHalfExtents = [0.5, 0.5, 0.5]
    startPos = [0, 0, 5]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=cubeHalfExtents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=cubeHalfExtents, rgbaColor=[0.6, 0.4, 0.2, 1])
    
    # Mass = Density * Volume
    # Box Volume = 1 * 1 * 1 = 1 m^3
    # Water Density ~ 1000 kg/m^3
    # Wood Density ~ 700 kg/m^3
    # Let's scale it down for simulation stability:
    # Say water density parameter = 10 (arbitrary units for demo)
    # Object mass = 5
    
    bodyId = p.createMultiBody(baseMass=5,
                               baseCollisionShapeIndex=col,
                               baseVisualShapeIndex=vis,
                               basePosition=startPos)
    
    print("Simulating Buoyancy ('Pseudo-Fluid' Manual Force) for 15 seconds...")
    print("Brown Cube: Falling into Cyan 'water'")
    
    # Simulation Parameters
    fluid_density = 10.0 # Adjust to balance with mass=5 and gravity=9.8
    # Buoyant Force = fluid_density * gravity * Volume_submerged
    # If fully submerged, F_b = 10 * 9.8 * 1 = 98. Weight = 5 * 9.8 = 49.
    # Net force up = 49. It should float up.
    
    drag_coeff = 2.0
    
    dt = 1./240.
    p.setTimeStep(dt)
    
    textId = -1
    
    for _ in range(240 * 15):
        if not p.isConnected():
            break

        # 1. Get position
        try:
            pos, orn = p.getBasePositionAndOrientation(bodyId)
            vel, _ = p.getBaseVelocity(bodyId)
            z_center = pos[2]
        except:
            break
        
        # 2. Calculate Submerged Volume (Simplified)
        # Assuming upright box for simple math.
        # Top of box = z_center + 0.5
        # Bottom of box = z_center - 0.5
        # Water Level = -2.0
        
        bottom = z_center - 0.5
        top = z_center + 0.5
        
        submerged_height = 0
        if bottom < water_level:
            if top < water_level:
                submerged_height = 1.0 # Fully submerged
            else:
                submerged_height = water_level - bottom
                
        status_text = "Status: In Air"
        if submerged_height > 0:
            status_text = "Status: Floating (In Water)"
            # Apply Buoyancy
            # V_sub = 1 * 1 * submerged_height
            v_sub = 1.0 * 1.0 * submerged_height
            f_buoyancy_mag = fluid_density * 9.81 * v_sub
            
            # Apply upwards (0, 0, 1)
            p.applyExternalForce(bodyId, -1, [0, 0, f_buoyancy_mag], pos, p.WORLD_FRAME)
            
            # Apply Water Drag (Linear Damping)
            # F_drag = -k * v
            p.applyExternalForce(bodyId, -1, [-drag_coeff*vel[0], -drag_coeff*vel[1], -drag_coeff*vel[2]], pos, p.WORLD_FRAME)
            
            # Draw line for visual feedback
            p.addUserDebugLine(pos, [pos[0], pos[1], pos[2]+1], [0,0,1], 1, lifeTime=dt)

        # Update Debug Text
        # textId = p.addUserDebugText(status_text, [0, 0, 8], [0, 0, 0], textSize=2, replaceItemUniqueId=textId)
        # Using a fixed position on screen or near object
        textId = p.addUserDebugText(status_text, [pos[0], pos[1], pos[2]+1.5], [0, 0, 0], textSize=1.5, replaceItemUniqueId=textId, lifeTime=dt*2)

        p.stepSimulation()
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
