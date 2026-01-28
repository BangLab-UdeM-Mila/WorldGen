import pybullet as p
import pybullet_data
import time

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    
    # Plane
    p.loadURDF("plane.urdf")
    
    # Reset Camera
    p.resetDebugVisualizerCamera(cameraDistance=7, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0, 0, 2])
    
    # Create a Cube attached to a fixed point via a spring
    # Start at the anchor position (z=6) to demonstrate "drop" and oscillation
    startPos = [0, 0, 6]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0, 1, 0, 1])
    
    bodyId = p.createMultiBody(baseMass=1,
                               baseCollisionShapeIndex=col,
                               baseVisualShapeIndex=vis,
                               basePosition=startPos)
                               
    # Fixed Anchor Point (Static Sphere for visual reference)
    anchorVis = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])
    anchorId = p.createMultiBody(baseMass=0,
                                 baseVisualShapeIndex=anchorVis,
                                 basePosition=[0, 0, 6])
                                 
    # Create Spring (Joint with stiffness/damping)
    # Using a Slider joint or generic 6DOF joint can simulate a spring, but PyBullet also supports specific spring parameters on joints.
    # A simpler way for a "hanging" spring is a Point2Point constraint with Error Reduction Parameter (ERP) and Constraint Force Mixing (CFM)
    # OR explicit spring force application.
    # Here let's use a 6DOF constraint which allows setting linear stiffness (spring).
    
    constraintId = p.createConstraint(parentBodyUniqueId=anchorId,
                                      parentLinkIndex=-1,
                                      childBodyUniqueId=bodyId,
                                      childLinkIndex=-1,
                                      jointType=p.JOINT_POINT2POINT,
                                      jointAxis=[0, 0, 0],
                                      parentFramePosition=[0, 0, 0],
                                      childFramePosition=[0, 0, 3]) # Initial offset matching distance
                                      
    # To make it springy, we allow the constraint to be "soft"
    # p.changeConstraint(constraintId, maxForce=100, erp=0.1) 
    # Actually, PyBullet's "Soft Body" dynamics are different. For rigid bodies, we can simulate a spring force manually or use Soft constraints.
    # Manual force application is often more educational for "Hooke's Law".
    
    # Let's remove the physical constraint and apply Force = -k * x manually to demonstrate the math.
    p.removeConstraint(constraintId)
    
    # Re-create visually only (optional, but let's just stick to the physics math)
    # We will simulate: F_spring = -k * (current_length - rest_length)
    # And F_damp = -c * velocity
    
    k = 50.0 # Spring Constant
    c = 1.0  # Damping Constant
    rest_length = 3.0
    anchor_pos = [0, 0, 6]
    
    print("Simulating Hooke's Law (Spring Force) for 15 seconds...")
    print("Green Box: Attached to Red Anchor by invisible spring")
    print("Watch it oscillate!")
    
    dt = 1./240.
    p.setTimeStep(dt)
    
    for _ in range(240 * 15):
        # 1. Get current position and velocity
        pos, orn = p.getBasePositionAndOrientation(bodyId)
        vel, ang_vel = p.getBaseVelocity(bodyId)
        
        # 2. Calculate Spring Force
        # Vector from body to anchor
        dx = anchor_pos[0] - pos[0]
        dy = anchor_pos[1] - pos[1]
        dz = anchor_pos[2] - pos[2]
        
        curr_length = (dx**2 + dy**2 + dz**2)**0.5
        
        # Unit vector direction
        if curr_length == 0: dir = [0,0,0]
        else: dir = [dx/curr_length, dy/curr_length, dz/curr_length]
        
        # Hooke's Law: F = k * (extension)
        # Extension = curr_length - rest_length
        # Force pulls towards anchor if extended
        extension = curr_length - rest_length
        f_spring_mag = k * extension
        
        f_spring = [dir[0]*f_spring_mag, dir[1]*f_spring_mag, dir[2]*f_spring_mag]
        
        # 3. Calculate Damping Force
        # F_damp = -c * vel
        f_damp = [-c * vel[0], -c * vel[1], -c * vel[2]]
        
        # 4. Apply Total Force
        f_total = [f_spring[0] + f_damp[0], f_spring[1] + f_damp[1], f_spring[2] + f_damp[2]]
        
        p.applyExternalForce(bodyId, -1, f_total, pos, p.WORLD_FRAME)
        
        # Visualize the spring line
        p.addUserDebugLine(pos, anchor_pos, [1, 1, 0], 1, lifeTime=dt*2)
        
        p.stepSimulation()
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
