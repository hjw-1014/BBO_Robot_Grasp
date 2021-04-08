import math
import time
import copy
import numpy as np
import pybullet as p
import pybullet_data as pd

grasper_height = 0.105

client = p.connect(p.GUI)
p.setPhysicsEngineParameter(solverResidualThreshold=0)
p.setGravity(0, 0, -9.8, physicsClientId=client)
p.setAdditionalSearchPath(pd.getDataPath())
p.loadURDF('plane.urdf')

legos = list()
legos.append(p.loadURDF('lego/lego.urdf', [0.6, 0., 0.25]))
legos.append(p.loadURDF('lego/lego.urdf', [0.4, 0., 0.25]))
legos.append(p.loadURDF('lego/lego.urdf', [0.8, 0., 0.25]))
p.loadURDF('sphere_small.urdf', [0.6, 0.1, 0.25])
p.loadURDF('sphere_small.urdf', [0.6, -0.1, 0.25])
orn = p.getQuaternionFromEuler([math.pi, 0., 0.])
panda = p.loadURDF('./panda_visual_arm_grasper.udrf', [0, 0, 0], orn, useFixedBase=False)

for i in range(p.getNumJoints(panda)):
    print(p.getJointInfo(panda, i))

width = 128
height = 128
cameraRandom = 0
look = [0.6, 0., 0.]
distance = 1.4
pitch = -85
yaw = 270
roll = 0
view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
fov = 18. + cameraRandom * np.random.uniform(-2, 2)
aspect = width / height
near = 0.1
far = 1.6
took_photo = False

state = -1
state_t = 0
cur_state = 0
states = [-1, 0, 1, 2, 3, 4, 5]
state_durations = [1., 1., 1., 1., 1., 1., 1.]
pre_pos = None
pre_orn = None

while True:

    state_t += 1./240
    if state_t > state_durations[cur_state]:
        cur_state += 1
        if cur_state >= len(states):
            cur_state = 0
        state_t = 0
        state = states[cur_state]

    if state == 1:
        obj_pos, obj_orn = p.getBasePositionAndOrientation(legos[0])
        target_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.2]
        obj_eul = p.getEulerFromQuaternion(obj_orn)
        target_orn = p.getQuaternionFromEuler([math.pi, 0., obj_eul[2]])
        pre_pos = target_pos
        pre_orn = target_orn
        target_clo = 0.04
    elif state == 2:
        obj_pos, obj_orn = p.getBasePositionAndOrientation(legos[0])
        target_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + grasper_height]
        obj_eul = p.getEulerFromQuaternion(obj_orn)
        target_orn = p.getQuaternionFromEuler([math.pi, 0., obj_eul[2]])
        pre_pos = target_pos
        pre_orn = target_orn
        target_clo = 0.04
    elif state == 3:
        obj_pos, obj_orn = p.getBasePositionAndOrientation(legos[0])
        target_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + grasper_height]
        obj_eul = p.getEulerFromQuaternion(obj_orn)
        target_orn = p.getQuaternionFromEuler([math.pi, 0., obj_eul[2]])
        pre_pos = target_pos
        pre_orn = target_orn
        target_clo = 0.005
    elif state == 4:
        target_pos = [pre_pos[0], pre_pos[1], pre_pos[2] + 0.5]
        target_orn = pre_orn
        target_clo = 0.005
    else:
        if state == -1:
            target_pos = [0.0, 0.0, 0.5]
            target_orn = p.getQuaternionFromEuler([math.pi, 0., 0.])
        elif state == 0:
            target_pos = [0.0, 0.0, 0.5]
            target_orn = p.getQuaternionFromEuler([math.pi, 0., 0.])
            if took_photo is False:
                took_photo = True
                proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
                p.getCameraImage(width, height, view_matrix, proj_matrix)
        elif state == 5:
            target_pos = [pre_pos[0], pre_pos[1], pre_pos[2] + 0.5]
            target_orn = pre_orn
            took_photo = False
        else:
            raise RuntimeError()
        target_clo = 0.04

    joint_poses = p.calculateInverseKinematics(panda, 6, target_pos, target_orn,
                                               maxNumIterations=20)
    for j in range(7):
        p.setJointMotorControl2(panda, j, p.POSITION_CONTROL, joint_poses[j], force=1000)

    for j in [7, 8]:
        p.setJointMotorControl2(panda, j, p.POSITION_CONTROL, target_clo, force=50)

    p.stepSimulation()

    # if state == 4:
    #     print(p.getContactPoints(panda, legos[0])[0][8],
    #           p.getContactPoints(panda, legos[0])[1][8])
    time.sleep(1/240.)
