# IMPORT LIBS
# basic gym tools
import gym
from gym.utils import seeding
from gym import spaces, logger

# generate ramdom seed 
import random
# calculate the sin and cos function (also pi)
import math
import numpy as np
# to control the time delay
import time

# ENV CLASS
# create a environment 
class MainEnv(gym,Env):
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

    def step(self, actions):
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

        return self.state ,reward, done