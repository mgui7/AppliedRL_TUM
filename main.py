# IMPORT LIBS
# basic gym tools
import gym
from gym.utils import seeding
from gym import spaces, logger
import elements
import pyglet
import time
from rl import Q_LEARNING


# generate ramdom seed 
import random
# calculate the sin and cos function (also pi)
import math
import numpy as np
# to control the time delay
import time

# Size of screen
SCREEN_W, SCREEN_H = 600,600

# Size of Robot
ROBOT_W = 40.0
ROBOT_H = 40.0

# The episodes to control the main function
MAX_EPISODES = 10
MAX_EP_STEPS = 200

# The start / end position of robot
ROBOT_START_POSITION = (SCREEN_W / 2, ROBOT_H / 2)
ROBOT_END_POSITION = (SCREEN_W / 2, SCREEN_H - ROBOT_H)

# Size of Arm 
ARM_W = 10.0
ARM_LEN = 150

# Size of robot
ROBOT_W = 40.0
ROBOT_H = 40.0

# Path of robot
PATH = np.load('maps/long_map_300_sin.npy',encoding = "latin1")[20:]


# ENV CLASS
# create a environment 
class MainEnv(gym.Env):
    # meta data, reference is the source code of gym example
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 1
    }
    def __init__(self):
        """ Initial function
        initial the viewer, state, seed, and so on. 
        """
        self.viewer = None
        self.state = None
        self.seed()

    def seed(self,seed=None):
        """ Create a random function 
        Example for using : 
            self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions=(1,1,1)):
        """ Control the robot
        This function reads the action which is provided by RL Alg and update the states of robot and robot's arm.
        The reward is also calculated at same time.
        Args:
            actions: tuple and the size is 3.
        Returns:
            self.state: tuple and the size is 4 | provide the state of the robot to the Alg.
            reward: double/int (should be discussed)
            done: bool, It indicates whether the movement has been stopped.
        """
        # Unpack the states
        x, y, angle_arm1, angle_arm2 = self.state

        # Update angles
        i, j, k       = actions
        delta_angular =  2 * math.pi / 360
        if j == 1 & k == 1:
            y          = y + 1
            angle_arm1 = angle_arm1 + delta_angular
            angle_arm2 = angle_arm2 + delta_angular
        elif j == 1 & k == -1:
            y          = y + 1
            angle_arm1 = angle_arm1 + delta_angular
            angle_arm2 = angle_arm2 - delta_angular
        elif j == -1 & k == 1:
            y          = y + 1
            angle_arm1 = angle_arm1 - delta_angular
            angle_arm2 = angle_arm2 + delta_angular
        elif j == -1 & k == -1:
            y          = y + 1
            angle_arm1 = angle_arm1 - delta_angular
            angle_arm2 = angle_arm2 - delta_angular

        # Update location of the robot
        x = PATH[round(y)][0] # Currently we use the predefined path-map

        # Pack the state
        self.state = (x, y, angle_arm1, angle_arm2)

        # Update reward 
        reward = 0
        print(x,y)
        # Update done
        done =  int(y) == 560
        done = bool(done)

        return self.state ,reward, done

    def reset(self):
        """ Reset the env after creating the environment
        Returns:
            self.state
        """
        self.state = (0,0,0,0)
        self.steps_beyond_done = None
        return self.state

    def render(self, mode='human'):
        """ Display function
        After updating the variable self.state, 
        render function will read the latest state value and update the screen.
        Returns:
            self.viewer.render(): update function
        """
        if self.viewer is None:
            # draw the elements by rendering module
            from gym.envs.classic_control import rendering

            # create a viewer
            self.viewer = rendering.Viewer(SCREEN_W, SCREEN_H)

            # Create a trackline 
            # track = elements.TrackLine(self.viewer)
            # track.draw_line_track()

            # Create a start and end point 
            start_end_points = elements.StartEndPoint(self.viewer)
            start_end_points.draw_points()

            # Create a score1 
            self.score1 = rendering.make_circle(10)
            self.score1_trans = rendering.Transform(translation=(150, 450))
            self.score1.add_attr(self.score1_trans)
            self.score1.set_color(255 / 255, 192 / 255, 0)
            self.viewer.add_geom(self.score1)

            # Create a score2
            self.score2 = rendering.make_circle(10)
            self.score2_trans = rendering.Transform(translation=(100, 150))
            self.score2.add_attr(self.score2_trans)
            self.score2.set_color(255 / 255, 192 / 255, 0)
            self.viewer.add_geom(self.score2)

            # Create a score3
            self.score3 = rendering.make_circle(10)
            self.score3_trans = rendering.Transform(translation=(450, 200))
            self.score3.add_attr(self.score3_trans)
            self.score3.set_color(255 / 255, 192 / 255, 0)
            self.viewer.add_geom(self.score3)

            # Create a robot
            l, r, t, b = -ROBOT_W / 2, ROBOT_W / 2, ROBOT_H / 2, -ROBOT_H / 2 # l is left, r is right, t is top and b is bottom
            robot = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)]) # draw and fill the polygon
            self.robot_trans = rendering.Transform() # add the Transform to the robot 
            robot.add_attr(self.robot_trans) 
            self.viewer.add_geom(robot) # add the robot into the viewer

            # Create the first arm
            l, r, t, b = -ARM_W / 2, ARM_W / 2, ARM_LEN - ARM_W / 2, -ARM_W / 2
            arm1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            arm1.set_color(.8, .6, .4)
            self.arm1_trans = rendering.Transform()

            # The arm1 and robot move together
            arm1.add_attr(self.arm1_trans)
            arm1.add_attr(self.robot_trans)
            self.viewer.add_geom(arm1)

            # Create the second arm
            l, r, t, b = -ARM_W / 2, ARM_W / 2, ARM_LEN - ARM_W / 2 * 2, -ARM_W / 2
            arm2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            arm2.set_color(.8, .6, .4)
            self.arm2_trans = rendering.Transform(translation=(0, ARM_LEN - ARM_W / 2)) # Attention the initial translation

            # arm2 moves with the arm1 and the robot.
            arm2.add_attr(self.arm2_trans)
            arm2.add_attr(self.arm1_trans)
            arm2.add_attr(self.robot_trans)
            self.viewer.add_geom(arm2)

            # add the joint1 
            self.joint1 = rendering.make_circle(ARM_W / 2)
            self.joint1.add_attr(self.arm1_trans)
            self.joint1.add_attr(self.robot_trans)
            self.joint1.set_color(.5, .5, .8)
            self.viewer.add_geom(self.joint1)

            # add the joint2 
            self.joint2 = rendering.make_circle(ARM_W / 2)
            self.joint2.set_color(.5, .5, .8)
            self.joint2_trans = rendering.Transform(translation=(0, ARM_LEN-(ARM_W / 2)))
            self.joint2.add_attr(self.joint2_trans)
            self.joint2.add_attr(self.arm1_trans)
            self.joint2.add_attr(self.robot_trans)
            self.viewer.add_geom(self.joint2)

            # add the joint3 
            # self.joint3 = rendering.make_circle(ARM_W / 2)
            # self.joint3_trans = rendering.Transform(translation=(0, ARM_LEN - ARM_W))
            # self.joint3.add_attr(self.arm2_trans)
            # self.joint3.add_attr(self.joint3_trans)
            # self.joint3.set_color(.5, .5, .8)
            # self.viewer.add_geom(self.joint3)

        if self.state is None:return None



        # Unpack the states from self.state
        x      = self.state[0]
        y      = self.state[1]
        angle1 = self.state[2]
        angle2 = self.state[3]

        # Refresh the screen
        self.robot_trans.set_translation(x, y)
        self.arm1_trans.set_rotation(angle1)
        self.arm2_trans.set_rotation(angle2)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    
    def close(self):
        """ Close the viewer 
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None


rl = Q_LEARNING()

if __name__ == "__main__":
    env = MainEnv()
    for _ in range(MAX_EPISODES):
        state = env.reset()
        for _ in range(MAX_EP_STEPS):

            # Refresh env and output image
            env.render()

            # Choosing the action based on Q_learning
            action = rl.choose_action(state)

            # rl takes action and acquire the next state and the corresponding reward
            new_state, reward, done = env.step(action)

            # rl learns from this state transition
            rl.learn(state, action, reward, new_state)

            # begin from new state
            state = new_state


            if done:
                input("")
                env.close()
                break
