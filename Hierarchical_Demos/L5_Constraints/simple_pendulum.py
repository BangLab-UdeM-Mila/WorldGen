import pybullet as p
import pybullet_data
import time

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    
    # Static body for the pivot point
    colCube = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
    visCube = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[0, 0, 0, 1])
    pivotId = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=colCube,
                                baseVisualShapeIndex=visCube,
                                basePosition=[0, 0, 5])
    
    # Dynamic body for the bob
    colSphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25)
    visSphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.25, rgbaColor=[1, 0, 0, 1])
    bobId = p.createMultiBody(baseMass=1,
                              baseCollisionShapeIndex=colSphere,
                              baseVisualShapeIndex=visSphere,
                              basePosition=[1, 0, 5]) 
    
    # Constraint (Hinge/Point2Point)
    # Pivot joint at [0,0,5] (which is [0,0,0] in pivot body frame, and [-1,0,0] in bob body frame)
    # Let's use p.createConstraint
    
    constraintId = p.createConstraint(parentBodyUniqueId=pivotId,
                                      parentLinkIndex=-1,
                                      childBodyUniqueId=bobId,
                                      childLinkIndex=-1,
                                      jointType=p.JOINT_POINT2POINT,
                                      jointAxis=[0, 1, 0],
                                      parentFramePosition=[0, 0, 0],
                                      childFramePosition=[-1, 0, 0])
                           
    # Reset Camera
    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-10, cameraTargetPosition=[0, 0, 4])

    print("Simulating Simple Pendulum for 30 seconds...")
    dt = 1./240.
    p.setTimeStep(dt)
    
    # Store the line ID to update it
    lineId = -1
    pivotPos = [0, 0, 5]

    for _ in range(240 * 30):
        p.stepSimulation()
        
        # Get bob position
        bobPos, _ = p.getBasePositionAndOrientation(bobId)
        
        # Draw/Update visual line representing the string
        if lineId == -1:
            lineId = p.addUserDebugLine(pivotPos, bobPos, lineColorRGB=[0, 0, 1], lineWidth=2)
        else:
            lineId = p.addUserDebugLine(pivotPos, bobPos, lineColorRGB=[0, 0, 1], lineWidth=2, replaceItemUniqueId=lineId)
            
        time.sleep(dt)
        
    p.disconnect()

if __name__ == "__main__":
    main()
