import numpy as np
import pyglet
from pyglet.window import Window
import pandas as pd
import random
import math
import time
# Load different Rewards
import reward_calculate as R

from main import Viewer
from main import MeinEnv
from main import discrete

# You can run this directly, then read those introductions and README.md :)

# Input: Action
action = [1,1,1]
# 1. Robot base direction:  1 or 0 or -1
# 2. Arm1 direction: 1 or 0 or -1
# 3. Arm2 direction: 1 or 0 or -1

# Output: State
# for example
# [570.0, 200.00000000000165, 45, 45, (1, 1, 1)]
# 1. x postion of robot base
# 2. y position 
# 3. Angle of arm1
# 4. Angle of arm2
# 5. Status of scores: 1 means it still exists, 0 means it have been collected. 

def main():
    # Initial Env Class
    env = MeinEnv()
    
    # try 10 times 
    for i in range(10):
        # Reset all states
        s = env.reset()
        # Start to run
        while True:
            # display
            env.render()

            # This is sample step
            # To reduce the number of states, we can use this loop to sample the states.
            # you can delete this for loop if you can handle many states.
            for _ in range(45):
                s_prime, reward, done = env.step(action)
                env.render()
                if done:break

            # We also create a function name "discrete"
            # You can test the different after using this function
            print("Original s_prime: ",s_prime)
            print("After discrete function: ",discrete(s_prime))

            if done:
                env.close()
                break

if __name__ == '__main__':
    main()

