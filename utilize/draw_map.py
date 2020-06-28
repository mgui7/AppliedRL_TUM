import gym
import random
import math
import numpy as np
from gym.utils import seeding
from gym import spaces, logger
import time
import matplotlib.pyplot as plt
# 屏幕分辨率
SCREEN_W = 600
SCREEN_H = 600

# 机械臂的宽和高
ARM_W = 10.0
ARM_LEN = 150

# 机器人的尺寸
ROBOT_W = 40.0
ROBOT_H = 40.0

ROBOT_START_POSITION = (SCREEN_W/2,ROBOT_H/2)
ROBOT_END_POSITION = (SCREEN_W/2,SCREEN_H - ROBOT_H)
# Y的偏移量
AX_OFFSET = SCREEN_W/2.0


# 速度
delta_y = 1

# 
MAX_EPISODES = 10
MAX_EP_STEPS = 200

# 开始点坐标 和 颜色
START_POINT_POSITION = (20,20)
START_POINT_COLOR = (255/255,0,0)

# 结束点坐标 和 颜色
END_POINT_POSITION = (90,90)
END_POINT_COLOR  = (0,255/255,0)

# 四个分数点的坐标
SCORE1_POINT_POSITION = (1,2)
SCORE2_POINT_POSITION = (1,2)
SCORE3_POINT_POSITION = (1,2)
SCORE4_POINT_POSITION = (1,2)

# 分数点的颜色
SCORE_POINT_COLOR = (255/255,192/255,0)

# 环境类
class MainEnv(gym.Env):

    def __init__(self):
        pass

    def _robot_movement(self,x,y,init_x,ini_y):
        y = y+1
        amplitude = 150 
        x = amplitude * np.sin(2*math.pi*(y-ini_y)/560) + init_x
        x = round(x)
        x = 300
        return x,y



    def step(self):
        self.x,self.y = self._robot_movement(self.x,self.y,ROBOT_START_POSITION[0],ROBOT_START_POSITION[1])
        return int(self.x),int(self.y)
    
    def reset(self):
        self.x = int(ROBOT_START_POSITION[0])
        self.y = int(ROBOT_START_POSITION[1])
        return self.x,self.y 



env = MainEnv()
x,y = env.reset()
map1 = np.zeros([600,2])
map2 = np.zeros([601,601])
map3 = np.zeros([580,2])
for i in range(20,581):
    map1[i,0] = x
    map1[i,1] = y
    map3[i-20,0] = x
    map3[i-20,0] = y
    # print(x,y)
    map2[x,y] = int(1)
    x,y = env.step()
    if y== 600:
        break

np.save("square_map_line.npy",map2)
np.save("long_map_line.npy",map1)
np.save('clipped_trajectory_line',map3)
