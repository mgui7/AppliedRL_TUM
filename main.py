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


SCREEN_W, SCREEN_H = 600,600

MAX_EPISODES = 10
MAX_EP_STEPS = 200


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

        # update reward 
        reward = 0

        # update done
        done = False

        return self.state ,reward, done

    def reset(self):
        """ Reset the env after creating the environment
        Returns:
            self.state
        """
        self.state = (0,0,0)
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

        if self.state is None:return None
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    
    def close(self):
        """ Close the viewer 
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    
    env = MainEnv()

    for _ in range(MAX_EPISODES):
        state = env.reset()
        for _ in range(MAX_EP_STEPS):
            # Refresh env and output image
            env.render()
            # robot move 
            new_state, reward, done = env.step(None)
            # robot finish
            if done:
                input("")
                env.close()
                break