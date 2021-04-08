from mushroom_rl.policy.deterministic_policy import ParametricPolicy
import numpy as np
import pybullet as p
import math as m

class Own_policy(ParametricPolicy):

    weights=[]

    def reset(self):
        pass


    def draw_action(self, state):
        """
        Sample an action in ``state`` using the policy.

        Args:
            state (np.ndarray): the state where the agent is.

        Returns:
            The action sampled from the policy. # list

        """
        gripper_position, gripper_orientation, gripper_close = self.generate_grasp_trajectory(state)
        action = []
        action.append(gripper_position)
        action.append(gripper_orientation)
        action.append(np.array(gripper_close))
        return action


    def set_weights(self, weights, fixed_height=False):
        """
        Setter.

        Args:
            weights (np.ndarray),（1, 6）: the vector of the new weights to be used by the policy.
            fixed_height(boolean): fix goal position to optimal height position
            fixed_orientation(boolean): fix the goal orientation to optimal orientation

        return:

        """
        if fixed_height:
            weights[2]=0.91
        self.weights=weights

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.
        """
        return self.weights

    def weights_size(self):
        """
        Property.

        Returns:
             The size of the policy weights.

        """
        # weights = np.array([0])
        # gripper_positions, gripper_orientations, gripper_closeds = self.set_weights(weights)
        #
        return len(weights)

    def generate_grasp_trajectory(self, state):
        '''
            param state: int, the current state which the goal should be computed for
            return:
                gripper_position: np.ndarray (3,)
                gripper_orientation: np.ndarray (3,)
                gripper_close: np.ndarray (960, )
        '''
        state_duration = [60, 60, 60, 60]  # Number of calls of step function for each state
        grasp_ori=self.weights[3:6] if (self.weights.size==6 or self.weights.size==7) else [m.pi, 0, 0]
        gripper_closed = self.weights[-1] if (self.weights.size==4 or self.weights.size==7) else 0.03

        # State 4: Lift up object
        if state>state_duration[0]+state_duration[1]+state_duration[2]:
            return [self.weights[0], self.weights[1], 1.0], grasp_ori, gripper_closed
        # State 3: Close gripper on object
        elif state > state_duration[0] + state_duration[1]:
            return self.weights[0:3], grasp_ori, gripper_closed
        # State 2: Move gripper towards object but keep gripper open
        elif state > state_duration[0]:
            return self.weights[0:3], grasp_ori, gripper_closed+0.007
        # State 1: Open gripper over object1
        return [self.weights[0], self.weights[1], 1.01], grasp_ori, gripper_closed+0.007
