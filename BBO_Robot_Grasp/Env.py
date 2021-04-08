import gym
import numpy as np
#import torch
import pybullet as p
import pybullet_data
import os, inspect
import math as m
import time
#from gym.utils import seeding
import matplotlib.pyplot as plt
import scipy
from own_policy import Own_policy

from mushroom_rl.utils import spaces
from mushroom_rl.environments import MDPInfo, Environment

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)


class GraspEnv(Environment):
    metadata = {'render.modes': ['human', 'rgb_array']}
    initial_positions = {
        'w_x': 0.0, 'x_y': 0.0, 'y_z': 0.0,
        'z_y': 0.0, 'y_p': 0.0, 'p_r': 0.0,
        'hand': 0.0, 'hand_finger_left': 0.0, 'hand_finger_right': 0.0,
    }

    def __init__(self, physicsClientId=1, render=True):
        print('=====init======')
        self.showSimulation =False
        self.ll, self.ul, self.jr, self.rs =[], [], [], []
        self._states = [0, 1, 2, 3]
        self.current_t = 0
        self.state = 1
        self._state = np.array([0])
        self.waittimer = 10
        self.boxId = None
        self.robotId = None
        self.tableId = None
        self.planeId = None
        self.rob_start_pos = [-0.5, 0, 1.]
        self.rob_start_orn = p.getQuaternionFromEuler([m.pi, 0, 0])
        self.obj_start_pos = [-0.5, 0, 0.73175]
        self.obj_top = [-0.5, 0, 0.8384935]
        self.obj_start_orn = p.getQuaternionFromEuler([0, 0, m.pi/2])
        self.end_eff_idx = 6
        self.robot_left_finger_idx = 7
        self.robot_right_finger_idx = 8
        self._use_IK = 0
        self._control_eu_or_quat = 0
        self.joint_action_space = 7
        self._joint_name_to_ids = {}
        self._renders = render
        self.urdfroot = currentdir
        self._physics_client_id = physicsClientId
        self._p = p
        self.first_ep = True
        self.grasp_attemp_pos=[]
        #print("urdrfoot:",self.urdfroot)
        #self.seed()

        # TODO
        gamma = 1.
        horizon = 300
        high = np.array([-0.25, 0.1, 0.91])
        low = np.array([-0.75, -0.1, 0.91])
        self._max_u = 0.37
        self._min_u = 0.30
        #observation_space = np.array([0, 0, 0])
        #action_space = self.sample_action()

        observation_space = spaces.Box(low=low, high=high)
        action_space = spaces.Box(low=np.array([-self._max_u]),
                                  high=np.array([self._min_u]))
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

        if self._renders:
            self._physics_client_id = p.connect(p.SHARED_MEMORY)
            if self.showSimulation:
                if self._physics_client_id < 0:
                    self._physics_client_id = p.connect(p.GUI)
                else:
                    p.connect(p.DIRECT)
            else:
                 p.connect(p.DIRECT)

        # TODO
        # self.planeId = p.loadURDF(pybullet_data.getDataPath() + "/plane.urdf")
        # #
        # # # Place table
        # self.tableId = p.loadURDF(pybullet_data.getDataPath() + '/table/table.urdf')
        # #
        # # # Place robot
        # #self.robotId = p.loadURDF(self.urdfroot + "/robot/panda_visual_arm_grasper.urdf",
        #                           #self.rob_start_pos, self.rob_start_orn)
        # self.robotId = p.loadURDF(currentdir + "/environment-master/panda_robot_grasper/panda_visual_arm_grasper.udrf",
        #                            self.rob_start_pos, self.rob_start_orn)
        # #
        # # # Place ycb object
        # self.boxId = p.loadURDF(self.urdfroot + '/003_cracker_box/model_normalized.urdf', self.obj_start_pos, self.obj_start_orn)


        #p.resetDebugVisualizerCamera(1.5, 245, -56, [-0.5, 0, 0.61])

        # Add user debug parameters, comment back for debug slider use
        """self.control_prismatic_joint1 = p.addUserDebugParameter('control_prismatic_joint1', -.5, .5, 0)  # 0
        self.control_prismatic_joint2 = p.addUserDebugParameter('control_prismatic_joint2', -.5, .5, 0)  # 1
        self.control_prismatic_joint3 = p.addUserDebugParameter('control_prismatic_joint3', -.8, .8, 0)  # 2
        self.control_revolute_joint1 = p.addUserDebugParameter('control_revolute_joint1', -m.pi, m.pi, 0)  # 3
        self.control_revolute_joint2 = p.addUserDebugParameter('control_revolute_joint2', -m.pi, m.pi, 0)  # 4
        self.control_revolute_joint3 = p.addUserDebugParameter('control_revolute_joint3', -m.pi, m.pi, 0)  # 5
        self.force_gripper_fingers = p.addUserDebugParameter('Force_gripper_fingers', 0, .05, 0)  # 7, 8"""

    def reset(self, mode='human', state=np.array([0])):
        if self.first_ep:
            self.first_ep=False
            p.resetSimulation()
            p.setPhysicsEngineParameter(numSolverIterations=150)
            p.setTimeStep(1/240.)
            p.setGravity(0, 0, -9.8)
            # Place plane
            self.planeId = p.loadURDF(pybullet_data.getDataPath() + "/plane.urdf")
            # Place table
            self.tableId = p.loadURDF(pybullet_data.getDataPath() + '/table/table.urdf')
            # Place robot
            self.robotId = p.loadURDF(self.urdfroot + "/robot/panda_visual_arm_grasper.urdf",
                                     self.rob_start_pos, self.rob_start_orn)
            #self.robotId = p.loadURDF(currentdir + "/environment-master/panda_robot_grasper/panda_visual_arm_grasper.udrf",
                                      #self.rob_start_pos, self.rob_start_orn)

            # Place ycb object
            self.boxId = p.loadURDF(self.urdfroot + '/003_cracker_box/model_normalized.urdf', self.obj_start_pos, self.obj_start_orn)
            p.stepSimulation()


            #Comment back for debug slider use
            #self.debug_gui()
            #p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) #Remove for camera output

            ## reset joints to home position
            # num_joints_robot = p.getNumJoints(self.robotId, physicsClientId=self._physics_client_id)
            # idx = 0
            # for i in range(num_joints_robot):
            #     joint_info = p.getJointInfo(self.robotId, i, physicsClientId=self._physics_client_id)
            #     joint_name = joint_info[1].decode("UTF-8")
            #     joint_type = joint_info[2]
            #
            #     if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
            #         assert joint_name in self.initial_positions.keys()
            #
            #         self._joint_name_to_ids[joint_name] = i
            #
            #         p.resetJointState(self.robotId, i, self.initial_positions[joint_name],
            #                           physicsClientId=self._physics_client_id)
            #         p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL,
            #                                 targetPosition=self.initial_positions[joint_name],
            #                                 positionGain=0.2, velocityGain=1.0,
            #                                 physicsClientId=self._physics_client_id)
            #
            #         idx += 1

            #Configure camera view TODO
            width = 48
            height = 48
            camera_random = 0
            look = [-0.5, 0, 0.91]
            distance = 1.5
            pitch = -56
            yaw = 245
            roll = 0
            view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
            fov = 20. + camera_random * np.random.uniform(-2, 2)
            aspect = width / height
            near = 0.2
            far = 10
            p.computeProjectionMatrixFOV(fov, aspect, near, far)
        else:
            self.reset_env()

        self._state = state
        return self._state  # np.array([0])

    def reset_env(self):

        p.resetBasePositionAndOrientation(self.robotId, self.rob_start_pos, self.rob_start_orn)
        p.resetBasePositionAndOrientation(self.boxId, self.obj_start_pos, self.obj_start_orn)

    def square(self, ls):
        for x in ls:
            x = x**2
        return ls

    def get_reward(self, action, fix_z=True):
        gsp = self.check_grasp()  # 0 or 1
        reward = gsp if (self.grasp_attemp_pos[2]>=0.9) else 0
        return reward


    def step(self, action):
        '''
            :param action: list, action
                        action[0]: numpy.ndarray
                        action[1]: numpy.ndarray
                        action[2]: numpy.ndarray
            :return: tuple, (self._state, reward, termination, {})
        '''

        # TODO: sample action from multivate gausian policy
        rewards = []
        actions = []
        rewards_sum = 0

        self.time_steps = 1 / 50.
        time_steps = self.time_steps
        self.sim_time = 1  # change the episodes number
        sim_time = self.sim_time
        self.states = [0, 1, 2, 3]
        states = self.states

        cur_state = 0
        state_durations = [0.5, 0.5, 1, 1]
        times = 0

        reward = 0
        termination = False
        if int(self._state)==121:
            self.grasp_attemp_pos=action[0]
        if int(self._state) > 238:
            reward = self.get_reward(action)
            termination = True
            self._state = np.array([0])
            self.reset_env()

        targetPosition = action[0]
        targetOrientation = p.getQuaternionFromEuler(action[1])

        joint_poses = p.calculateInverseKinematics(self.robotId, self.end_eff_idx, targetPosition, targetOrientation, maxNumIterations=20)

        for j in range(7):
            p.setJointMotorControl2(self.robotId, j, p.POSITION_CONTROL, joint_poses[j], force=500)

        p.setJointMotorControl2(self.robotId, self.robot_left_finger_idx, p.POSITION_CONTROL, float(action[2]),
                                force=100)
        p.setJointMotorControl2(self.robotId, self.robot_right_finger_idx, p.POSITION_CONTROL, float(action[2]),
                                force=100)
        p.stepSimulation()
        time.sleep(1/240.)

        self._state = self._state + 1
        #print('self._state', self._state)
        #print('reward', reward)

        return self._state, reward, termination, {}

        # while times < sim_time:
        #     time_steps += self.time_steps
        #     if time_steps > state_durations[cur_state]:  # check state transition
        #         if cur_state == 3:
        #             times += 1
        #         cur_state += 1
        #         if cur_state >= len(states):
        #             cur_state = 0
        #             self.reset_env() # reset the position and orientation of the object and gripper
        #             #TODO: Replace best pos/orn with their sampled values
        #             best_pos = np.array((-0.5, 0., 0.91))
        #             best_orn = np.array((m.pi, 0, 0))
        #             weights = [best_pos, best_orn]
        #             pol.set_weights(weights)
        #         time_steps = 0
        #
        #     print("Current state:", cur_state)
        #     current_action = pol.draw_action(cur_state)
        #     joint_poses = p.calculateInverseKinematics(self.robotId, self.end_eff_idx, current_action[0], current_action[1], maxNumIterations=20)
        #
        #     for j in range(7):
        #         p.setJointMotorControl2(self.robotId, j, p.POSITION_CONTROL, joint_poses[j], force=500)
        #
        #     p.setJointMotorControl2(self.robotId, self.robot_left_finger_idx, p.POSITION_CONTROL, current_action[2], force=100)
        #     p.setJointMotorControl2(self.robotId, self.robot_right_finger_idx, p.POSITION_CONTROL, current_action[2], force=100)
        #
        #     p.stepSimulation()
        #
        #     # check if the gripper grasp the object successfully
        #     if  cur_state == 3:
        #         reward = self.check_grasp()
        #         print('time_steps | reward: ', reward)
        #         rewards.append(reward)
        #
        #     # check if termination
        #     if  cur_state == 3 and times == sim_time:
        #         rewards_sum = np.sum(rewards)
        #         print('time_steps | sum_reward: ', rewards_sum)
        #         break

    def check_grasp(self):
        '''
            return: 0. means failed to grasp, 1. means successful grasp.
        '''

        object_pos, _ = p.getBasePositionAndOrientation(self.boxId)
        if object_pos[2] > 0.8:
            return 1.
        return 0.

    def plot_distribution(self, policy):
        x = np.arange(0, 1000, 1)
        plt.hist(policy, bins=1000, density=True)
        plt.title('Gausian policy')
        plt.ylabel('Probability')
        plt.show()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    # def compute_distance(self):
    #     rob_obs, obj_obs = self.get_observation()
    #     return np.abs(rob_obs-obj_obs)

    # def _reward(self):
    #     """Calculates the reward for the episode.
    #
    #     The reward is 1 if one of the objects is above height .2 at the end of the
    #     episode.
    #     """
    #     reward = 0
    #     self._graspSuccess = 0
    #     for uid in self.ycbid:
    #         pos, _ = p.getBasePositionAndOrientation(uid)
    #         # If any block is above height, provide reward.
    #         if pos[2] > 1.0:
    #             self._graspSuccess += 1
    #             reward = 1
    #             break
    #     return reward

    def apply_action(self, action):
        # print('p.getJointState(self.robotId, 2)', p.getJointState(self.robotId, 2))
        """p.setJointMotorControl2(self.robotId,
                                j,
                                p.POSITION_CONTROL,
                                targetPosition=action,
                                physicsClientId=self._physics_client_id)"""

        joint_poses = p.calculateInverseKinematics(self.robotId, self.end_eff_idx, action[0], action[1],
                                                   maxNumIterations=20)

        for j in range(7):
            p.setJointMotorControl2(self.robotId, j, p.POSITION_CONTROL, joint_poses[j], force=500)

        p.setJointMotorControl2(self.robotId, self.robot_left_finger_idx, p.POSITION_CONTROL, action[2], force=100)
        p.setJointMotorControl2(self.robotId, self.robot_right_finger_idx, p.POSITION_CONTROL, action[2], force=100)

    # def get_observation(self):
    #     # TODO: return actual current state
    #
    #     robot_observation = []
    #     #observation_lim = []
    #     state = p.getLinkState(self.robotId, self.end_eff_idx)  # getLinkState returns worldposition and worldorientation
    #                                                             # end effector position is the initial positon of the robot!
    #     pos = state[0]
    #     orn = state[1]
    #
    #     robot_observation.extend(list(pos))
    #     obj_observation = []
    #     obj_observation.extend(self.obj_pos)
    #     print("rob pos", pos)
    #     print("obj pos", self.obj_pos)
    #
    #     return pos[2], self.obj_pos[2]  # robot_observation(z-axis), obj_observation(z-axis)
    #


    # def debug_gui(self):
    #
    #     print("Debug gui called")
    #     num_of_joints = p.getNumJoints(self.robotId)
    #     for joint_number in range(num_of_joints):
    #         joint_info = p.getJointInfo(self.robotId, joint_number)
    #         print(joint_info)
    #         print(joint_info[0], ':', joint_info[1])
    #     user_gripper_fingers = p.readUserDebugParameter(self.force_gripper_fingers)  # 7, 8
    #
    #     user_prismatic_joint1 = p.readUserDebugParameter(self.control_prismatic_joint1)  # 0
    #     user_prismatic_joint2 = p.readUserDebugParameter(self.control_prismatic_joint2)  # 1
    #     user_prismatic_joint3 = p.readUserDebugParameter(self.control_prismatic_joint3)  # 2
    #
    #     user_revolute_joint1 = p.readUserDebugParameter(self.control_revolute_joint1)  # 3
    #     user_revolute_joint2 = p.readUserDebugParameter(self.control_revolute_joint2)  # 4
    #     user_revolute_joint3 = p.readUserDebugParameter(self.control_revolute_joint3)  # 5
    #
    #     # Set the control of the fingers.
    #     p.setJointMotorControl2(self.robotId, 7, p.POSITION_CONTROL, targetPosition=user_gripper_fingers)
    #     p.setJointMotorControl2(self.robotId, 8, p.POSITION_CONTROL, targetPosition=user_gripper_fingers)
    #
    #     # Set the control of 6 joints.
    #     p.setJointMotorControl2(self.robotId, 3, p.POSITION_CONTROL, targetPosition=user_revolute_joint1)
    #     p.setJointMotorControl2(self.robotId, 4, p.POSITION_CONTROL, targetPosition=user_revolute_joint2)
    #     p.setJointMotorControl2(self.robotId, 5, p.POSITION_CONTROL, targetPosition=user_revolute_joint3)
    #
    #     p.setJointMotorControl2(self.robotId, 0, p.POSITION_CONTROL, targetPosition=user_prismatic_joint1)
    #     p.setJointMotorControl2(self.robotId, 1, p.POSITION_CONTROL, targetPosition=user_prismatic_joint2)
    #     p.setJointMotorControl2(self.robotId, 2, p.POSITION_CONTROL, targetPosition=user_prismatic_joint3)


    # def get_joint_ranges(self):
    #
    #     lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []
    #
    #     for joint_name in self._joint_name_to_ids.keys():
    #         jointInfo = p.getJointInfo(self.robotId, self._joint_name_to_ids[joint_name], physicsClientId=self._physics_client_id)
    #
    #         ll, ul = jointInfo[8:10]
    #         jr = ul - ll
    #         # For simplicity, assume resting state == initial state
    #         rp = self.initial_positions[joint_name]
    #         lower_limits.append(ll)
    #         upper_limits.append(ul)
    #         joint_ranges.append(jr)
    #         rest_poses.append(rp)
    #
    #     return lower_limits, upper_limits, joint_ranges, rest_poses

# env = GraspEnv()
# env.get_reward()
# pol = env.gausian_policy(mean=0, cov=.1)
# env.plot_distribution(pol)
# while True:
#     env.reset()
#     #p.stepSimulation()
#     time.sleep(1./240)

# env = GraspEnv()
# env.sample_action()
