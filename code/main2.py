import gym
import elements
import math
import numpy as np
from rl import Q_LEARNING
from gym import spaces, logger
import matplotlib.pyplot as plt
import reward_calculate as R
import time

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

RENDER_PERIOD = elements.RENDER_PERIOD
ANGLE_STEPS   = elements.ANGLE_STEPS
MAX_EPISODES = 150
MAX_EP_STEPS = RENDER_PERIOD * 50

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

# Load trajectory from file
# PATH = np.load('code\maps\long_map_300_sin.npy',encoding = "latin1")[21:579]
PATH = np.load('maps\long_map_300_sin.npy',encoding = "latin1")[21:579]
PATH = [PATH[i] for i in range(len(PATH)) if i % 20 == 0]
DEG = [(720/ANGLE_STEPS) * math.pi/360 * i for i in range(ANGLE_STEPS)]
TARGET_STATE = [1,1,1]
TARGET_POINTS = [[150,450],[100,150],[450,200]]

# 环境类
class Show(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 1
    }

    def __init__(self):
        """
        初始化函数，需要定义 初始动作，初始状态
        """
        self.viewer = None

    def display(self,old_state_index,new_state_index,residue,mode='human'):
        """屏幕更新函数
        
        After updating the variable self.state, render function will read the latest state value and update the screen.
        """

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            # 创建一个窗口展示
            self.viewer = rendering.Viewer(SCREEN_W, SCREEN_H)
            # 静态 轨迹线
            track = elements.TrackLine(self.viewer)
            track.draw_line_track()
            # 静态 初始点
            start_end_points = elements.StartEndPoint(self.viewer)
            start_end_points.draw_points()

            self.score1 = rendering.make_circle(10)
            self.score1_trans = rendering.Transform(translation=(150,450))
            self.score1.add_attr(self.score1_trans)
            self.score1.set_color(255/255,192/255,0)
            self.viewer.add_geom(self.score1)

            self.score2 = rendering.make_circle(10)
            self.score2_trans = rendering.Transform(translation=(100,150))
            self.score2.add_attr(self.score2_trans)
            self.score2.set_color(255/255,192/255,0)
            self.viewer.add_geom(self.score2)

            self.score3 = rendering.make_circle(40)
            self.score3_trans = rendering.Transform(translation=(450,200))
            self.score3.add_attr(self.score3_trans)
            self.score3.set_color(255/255,192/255,0)
            self.viewer.add_geom(self.score3)


            # 创建一个机器人
            l,r,t,b = -ROBOT_W/2, ROBOT_W/2, ROBOT_H/2, -ROBOT_H/2
            robot = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            print((l,b), (l,t), (r,t), (r,b))
            self.robot_trans = rendering.Transform()
            robot.add_attr(self.robot_trans)
            self.viewer.add_geom(robot)

            # 创建 第一个手臂
            l,r,t,b = -ARM_W/2,ARM_W/2,ARM_LEN-ARM_W/2,-ARM_W/2
            arm1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            print((l,b), (l,t), (r,t), (r,b))
            arm1.set_color(.8,.6,.4)
            self.arm1_trans = rendering.Transform()
            
            arm1.add_attr(self.arm1_trans)
            arm1.add_attr(self.robot_trans)
            self.viewer.add_geom(arm1)

            # 创建 第二个手臂
            l,r,t,b = -ARM_W/2,ARM_W/2,ARM_LEN-ARM_W/2*2,-ARM_W/2
            arm2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            print((l,b), (l,t), (r,t), (r,b))
            arm2.set_color(.8,.6,.4)
            self.arm2_trans = rendering.Transform(translation=(0,ARM_LEN-ARM_W/2))
            
            arm2.add_attr(self.arm2_trans)
            arm2.add_attr(self.arm1_trans)
            arm2.add_attr(self.robot_trans)
            self.viewer.add_geom(arm2)

            self.joint = rendering.make_circle(ARM_W/2)
            self.joint.add_attr(self.arm1_trans)
            self.joint.add_attr(self.robot_trans)
            self.joint.set_color(.5,.5,.8)
            self.viewer.add_geom(self.joint)

        i,j,k = old_state_index
        old_pos  = PATH[i]
        old_ang1 = DEG[j]
        old_ang2 = DEG[k]

        _i,_j,_k = new_state_index
        new_pos  = PATH[_i]
        if   j == 7 and _j == 0: new_ang1 = DEG[_j] + math.pi * 2
        elif j == 0 and _j == 7: new_ang1 = DEG[_j] - math.pi * 2
        else: new_ang1 = DEG[_j]

        if   k == 7 and _k == 0: new_ang2 = DEG[_k] + math.pi * 2
        elif k == 0 and _k == 7: new_ang2 = DEG[_k] - math.pi * 2
        else: new_ang2 = DEG[_k]

        # 更新 self.state 中的坐标
        i1 = (new_pos[0] - old_pos[0])  /RENDER_PERIOD * residue + old_pos[0]
        i2 = (new_pos[1] - old_pos[1])  /RENDER_PERIOD * residue + old_pos[1]
        j = (new_ang1 - old_ang1)/RENDER_PERIOD * residue + old_ang1
        k = (new_ang2 - old_ang2)/RENDER_PERIOD * residue + old_ang2

        self.robot_trans.set_translation(i1,i2)
        self.arm1_trans.set_rotation(j)
        self.arm2_trans.set_rotation(k)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

def step(state, actions):
    global TARGET_STATE
    # PREDICTING next state
    # Based on action，update state
    # Take concrete states from self.state
    base_location_idx,angle1_idx,angle2_idx = state

    # Take actions
    i,j,k = actions

    # Update location
    base_location_idx += i
    # Assuring positive val
    base_location_idx = max(base_location_idx,0)
    # Update the angle
    angle1_idx += j
    if   angle1_idx >= ANGLE_STEPS : angle1_idx -= ANGLE_STEPS
    elif angle1_idx <= -1: angle1_idx += ANGLE_STEPS

    angle2_idx += k
    if   angle2_idx >= ANGLE_STEPS : angle2_idx -= ANGLE_STEPS
    elif angle2_idx <= -1: angle2_idx += ANGLE_STEPS

    # Pack state
    new_state = (base_location_idx,angle1_idx,angle2_idx)

    # Update reward
    x,y = PATH[base_location_idx]
    joint_position = (x - ARM_LEN * np.sin(DEG[angle1_idx]),y + ARM_LEN * np.cos(DEG[angle1_idx]))
    finger_position =(joint_position[0] - ARM_LEN * np.sin(DEG[angle2_idx]+DEG[angle1_idx]),
                      joint_position[1] + ARM_LEN * np.cos(DEG[angle2_idx]+DEG[angle1_idx]))


    # reward = 0
    reward, TARGET_STATE = R.reward1(finger_position,TARGET_STATE)

    # Update done
    done = (base_location_idx == len(PATH) - 1)
    print('Base location: ',x,y)

    return new_state ,reward, done

rl = Q_LEARNING()

if __name__ == "__main__":
    show = Show()
    for _ in range(MAX_EPISODES):

        # reset the state
        new_state_index = (0,0,0)
        old_state_index = (0,0,0)

        if _ > 0: print('EPISODE',_,'\nMean Reward:', round(np.mean(tmp)), 'Last Reward:', round(np.mean(tmp[-5:])))

        for i in range(MAX_EP_STEPS):

            # 指挥部，进行上一步汇总和学习，并且生成下一步状态
            if i % RENDER_PERIOD == 0:

                # Turning new_index to old_index
                old_state_index = new_state_index

                # Choosing the action based on Q_learning
                action = rl.choose_action(old_state_index)
 
                # rl takes action and acquire the next state and the corresponding reward 
                new_state_index, reward, done = step(old_state_index,action)

                # rl learns from this state transition
                rl.learn(old_state_index, action, reward, new_state_index)

            show.display(old_state_index,new_state_index,i%RENDER_PERIOD)

            if done: break
