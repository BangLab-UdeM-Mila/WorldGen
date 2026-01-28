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
    p.resetDebugVisualizerCamera(cameraDistance=12, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 5])
    
    # Create a Grid of Spheres connected by Springs (Soft Body Approximation)
    # Simulating a FLAG blowing in the wind
    rows = 15 # Horizontal (Width of flag)
    cols = 10 # Vertical (Height of flag)
    # Note: "Rows" usually means Y-axis in grid generation logic below, "Cols" means X? 
    # Let's check the generation loop.
    # Generation: i in rows (0..14), j in cols (0..9). 
    # pos = [(i - rows/2)*spacing, (j - cols/2)*spacing, baseHeight]
    # We want a vertical flag.
    # Let's say Row index i corresponds to X (Width). Col index j corresponds to Z (Height).
    
    spacing = 0.3
    baseHeight = 10
    radius = 0.1 
    mass = 0.5 
    
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0, 1, 1, 1])
    
    bodies = []
    
    # Create Grid of Bodies
    # Let's orient it in X-Z plane (Vertical)
    # i: Horizontal (X), j: Vertical (Z)
    for i in range(rows):
        col_bodies = []
        for j in range(cols):
            # i=0 is Left (Flag Pole). i=rows-1 is Right (Free end)
            # j=0 is Bottom. j=cols-1 is Top.
            
            x = i * spacing
            y = 0
            z = baseHeight + j * spacing
            
            # Pin the Left Column (i=0) to act as the Flag Pole
            current_mass = mass
            if i == 0:
                current_mass = 0 # Static anchor
            
            uid = p.createMultiBody(baseMass=current_mass,
                                    baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis,
                                    basePosition=[x, y, z])
            col_bodies.append(uid)
        bodies.append(col_bodies)
        
    print("Creating Constraints (Springs)...")
    for i in range(rows):
        for j in range(cols):
            current_body = bodies[i][j]
            
            # Link to Horizontal neighbor (i+1)
            if i < rows - 1:
                right_body = bodies[i+1][j]
                p.createConstraint(parentBodyUniqueId=current_body,
                                   parentLinkIndex=-1,
                                   childBodyUniqueId=right_body,
                                   childLinkIndex=-1,
                                   jointType=p.JOINT_POINT2POINT,
                                   jointAxis=[0, 0, 0],
                                   parentFramePosition=[spacing/2.0, 0, 0],
                                   childFramePosition=[-spacing/2.0, 0, 0])

            # Link to Vertical neighbor (j+1)
            if j < cols - 1:
                up_body = bodies[i][j+1]
                p.createConstraint(parentBodyUniqueId=current_body,
                                   parentLinkIndex=-1,
                                   childBodyUniqueId=up_body,
                                   childLinkIndex=-1,
                                   jointType=p.JOINT_POINT2POINT,
                                   jointAxis=[0, 0, 0],
                                   parentFramePosition=[0, 0, spacing/2.0],
                                   childFramePosition=[0, 0, -spacing/2.0])
                                   
    print("Simulating Flag in Wind for 30 seconds...")
    print("Wind Force: Constant [0, 5, 0] (Blowing in +Y direction) + Random Turbulence")
    
    dt = 1./240.
    p.setTimeStep(dt)
    
    # Wind Parameters
    wind_base = [0, 8, 0] # Blowing "into" the screen or sideways
    import random
    
    textId = -1
    
    for _ in range(240 * 30):
        if not p.isConnected():
            break
            
        # Apply Wind Force to all non-static components
        # Add some turbulence
        turbulence = random.uniform(-2, 2)
        wind_f = [wind_base[0], wind_base[1] + turbulence, wind_base[2] + turbulence/2]
        
        for i in range(rows):
            for j in range(cols):
                if i > 0: # Don't apply force to the static pole (waste of computation, though physically valid since mass=0/inv_mass=0 ignores it)
                    body_uid = bodies[i][j]
                    curr_pos, _ = p.getBasePositionAndOrientation(body_uid)
                    p.applyExternalForce(body_uid, -1, wind_f, curr_pos, p.WORLD_FRAME)
        
        p.stepSimulation()
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
