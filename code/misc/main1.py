import gym
import random
import elements
import math
import numpy as np
import reward_calculate

from gym.utils import seeding
from gym import spaces, logger

from rl import Q_LEARNING
rl = Q_LEARNING()

import time

DO_PLOT = True

# 屏幕分辨率
SCREEN_W = 600
SCREEN_H = 600

# 机械臂的宽和高
ARM_W = 10.0
ARM_LEN = 150

# 机器人的尺寸
ROBOT_W = 40.0
ROBOT_H = 40.0

ROBOT_START_POSITION = (SCREEN_W / 2, ROBOT_H / 2)
ROBOT_END_POSITION = (SCREEN_W / 2, SCREEN_H - ROBOT_H)
# Y的偏移量
AX_OFFSET = SCREEN_W / 2.0

# 速度
delta_y = 1
V_ROBOT = 10

# sample 间隔
SAM_STEP = 45

#
MAX_EPISODES = 500
MAX_EP_STEPS = 20000

# 开始点坐标 和 颜色
START_POINT_POSITION = (20, 20)
START_POINT_COLOR = (255 / 255, 0, 0)

# 结束点坐标 和 颜色
END_POINT_POSITION = (90, 90)
END_POINT_COLOR = (0, 255 / 255, 0)

# 四个分数点的坐标
PATH = np.load('maps/long_map_300_sin.npy',encoding = "latin1")

# 角速度
delta_angular =  math.pi / 360 

# 分数点的颜色
SCORE_POINT_COLOR = (255 / 255, 192 / 255, 0)

#奖励
bonus = (0,0,0)
reward = (0,0,0)
weight = (1,1,1)

# 环境类
class MainEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
    }

    def __init__(self):
        """
        初始化函数，需要定义 初始动作，初始状态
        """
        self.trigger = 0

        self.viewer = None
        self.state  = None
        
        self.seed()

        self.action_space_robot = spaces.Discrete(3)  # Set with 8 elements {0, 1 ,2}
        self.action_space_arm1  = spaces.Discrete(3)  # Set with 8 elements {0, 1 ,2}
        self.action_space_arm2  = spaces.Discrete(3)  # Set with 8 elements {0, 1 ,2}
        self.observation_space  = spaces.Discrete(8)  # Set with 8 elements {0, 1, 2, ..., 7}

        self.finger_position = (0,0)

    def seed(self, seed=None):
        """
        产生随机数的函数，使用范例：self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _robot_movement(self, y, init_x, ini_y):
        """机器人底座行为控制函数。
        机器人会朝着y的方向前进，因此x会随着y的偏移而偏移
        但因为初始坐标不是（0，0），而是（一半屏幕，半个机器人高度），
        因此需要对初始y做一个偏移量，让机器人从y_new=0开始计算x，而不是y_init=xx.
        """
        #amplitude = 150
        #period = 100
        #x = amplitude * np.sin((y - ini_y) / period) + init_x
        
        return x

    def _bound_detection(self):
        """检测是否获得获得点位。

        """
        return True

    def action_update(self, action, x, y, angle_arm1, angle_arm2):

        # 读取动作
        robot_move , arm1_rotate, arm2_rotate       = action
        robot_move = 1
        # 更新机器人的坐标
        y = max(y + V_ROBOT * robot_move/SAM_STEP,20)
        x = PATH[round(y)][0]

        angle_arm1 = angle_arm1 + arm1_rotate * delta_angular
        angle_arm2 = angle_arm2 + arm2_rotate * delta_angular


        # 根据动作更新状态
        # if arm1_rotate == 1 & arm2_rotate == 1:
        #     angle_arm1 = angle_arm1 + delta_angular * delta_y
        #     angle_arm2 = angle_arm2 + delta_angular * delta_y
        # elif arm1_rotate == 1 & arm2_rotate == -1:
        #     angle_arm1 = angle_arm1 + delta_angular * delta_y
        #     angle_arm2 = angle_arm2 - delta_angular * delta_y
        # elif arm1_rotate == -1 & arm2_rotate == 1:
        #     angle_arm1 = angle_arm1 - delta_angular * delta_y
        #     angle_arm2 = angle_arm2 + delta_angular * delta_y
        # elif arm1_rotate == -1 & arm2_rotate == -1:
        #     angle_arm1 = angle_arm1 - delta_angular * delta_y
        #     angle_arm2 = angle_arm2 - delta_angular * delta_y

        # 更新手指坐标
        joint_position = (x - (ARM_LEN-ARM_W) * np.sin(angle_arm1),y + (ARM_LEN-ARM_W) * np.cos(angle_arm1))
        finger_position =(joint_position[0] - (ARM_LEN-ARM_W) * np.sin(angle_arm1 + angle_arm2),joint_position[1] + (ARM_LEN-ARM_W) * np.cos((angle_arm1 + angle_arm2)))
        # ????

        # 更新手指画面
        self.finger_position = finger_position
        return x, y, angle_arm1, angle_arm2,finger_position

    def reward_bonus(self,finger_position,weight1,weight2,weight3):
        
        bonus1 = 0
        bonus2 = 0
        bonus3 = 0

        reward1 = (-math.sqrt(math.pow((finger_position[0] - 450), 2) + math.pow((finger_position[1] - 200), 2)))
        #reward1 = -np.linalg.norm((finger_position[0],finger_position[1])-(450,200))
        #reward1 = -np.sum(np.abs((finger_position[0], finger_position[1]) - (450, 200)))
        reward2 = (-math.sqrt(math.pow((finger_position[0] - 150), 2) + math.pow((finger_position[1] - 450), 2)))
        #reward2 = -np.linalg.norm((finger_position[0],finger_position[1])-(150,450))
        #reward2 = -np.sum(np.abs((finger_position[0], finger_position[1]) - (150, 450)))
        reward3 = (-math.sqrt(math.pow((finger_position[0] - 100), 2) + math.pow((finger_position[1] - 150), 2)))
        #reward3 = np.linalg.norm((finger_position[0],finger_position[1])-(100,150))
        #reward3 = -np.sum(np.abs((finger_position[0], finger_position[1]) - (100, 150)))

        if abs(reward1) < 10:
            weight1 = 0
            print('wow\n')
        if (weight1 == 0):
            bonus1  = 1000

        if abs(reward2) < 10:
            weight2 = 0
        if (weight2 == 0):
            bonus2 = 1000

        if abs(reward3) < 10:
            weight3 = 0
        if (weight3 == 0):
            bonus3 = 1000

        reward_total = reward1 * weight1 + reward2 * weight2 + reward3 * weight3
        bonus_total  = bonus1 + bonus2 + bonus3

        self.reward = reward_total # 暂时用不到，只是为了计算整个生命周期的总reward

        return (reward_total + bonus_total)

    def step(self, action):
        """重要的迭代更新函数。

        Args:
            action: 机器人和机器手臂的动作

        return:
            self.state:
            done: 是否已经完成迭代
        """

        # 提取所有状态 
        x, y, angle_arm1, angle_arm2, (weight1,weight2,weight3,weight4) = self.state


        # 根据动作，更新位置，手臂角度，手指坐标
        x, y, angle_arm1, angle_arm2,finger_position = self.action_update(action, x, y, angle_arm1, angle_arm2)

        # 更新奖励
        reward = self.reward_bonus(finger_position,weight1, weight2, weight3)

        # 把状态打包
        weight_set = (weight1,weight2,weight3,weight4) # 重新打包 weight -> weight_set
        self.state = (x, y, angle_arm1, angle_arm2,weight_set)

        # Update done
        done =  y >= ROBOT_END_POSITION[1]
        done = bool(done)

        return self.state, reward, done

    def reset(self):
        """初始化环境
        state[0]: robot position x
        state[1]: robot position y
        state[2]: angle of arm1
        state[3]: angle of arm2
        state[4]: state of scores. 1 means exist, 0 means dispear.
        """
        self.finger_position = (0,0)
        self.reward = 0 # 如果想要积累全部reward，可以使用这个变量

        weight_set = (1,1,1,1) # 初始所有score都在，值为1
        self.state = (300, 20, 0, 0, weight_set)
        self.steps_beyond_done = None

        return self.state

    def render(self, mode='human'):
        """屏幕更新函数

        After updating the variable self.state, render function will read the latest state value and update the screen.
        """

        world_width = 100
        scale = SCREEN_W / world_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            # 创建一个窗口展示
            self.viewer = rendering.Viewer(SCREEN_W, SCREEN_H)
            # 静态 轨迹线
            # track = elements.TrackLine(self.viewer)
            # track.draw_line_track()
            # 静态 初始点
            start_end_points = elements.StartEndPoint(self.viewer)
            start_end_points.draw_points()

            self.score1       = rendering.make_circle(10)
            self.score1_trans = rendering.Transform(translation=(150, 450))
            self.score1.add_attr(self.score1_trans)
            self.score1.set_color(255 / 255, 192 / 255, 0)
            self.viewer.add_geom(self.score1)

            self.score2       = rendering.make_circle(10)
            self.score2_trans = rendering.Transform(translation=(100, 150))
            self.score2.add_attr(self.score2_trans)
            self.score2.set_color(255 / 255, 192 / 255, 0)
            self.viewer.add_geom(self.score2)

            self.score3       = rendering.make_circle(10)
            self.score3_trans = rendering.Transform(translation=(450, 200))
            self.score3.add_attr(self.score3_trans)
            self.score3.set_color(255 / 255, 192 / 255, 0)
            self.viewer.add_geom(self.score3)

            self.score4       = rendering.make_circle(10)
            self.score4_trans = rendering.Transform(translation=(400, 350))
            self.score4.add_attr(self.score4_trans)
            self.score4.set_color(255 / 255, 192 / 255, 0)
            self.viewer.add_geom(self.score4)

            # 创建一个机器人
            l, r, t, b       = -ROBOT_W / 2, ROBOT_W / 2, ROBOT_H / 2, -ROBOT_H / 2
            robot            = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.robot_trans = rendering.Transform()
            robot.add_attr(self.robot_trans)
            self.viewer.add_geom(robot)

            # 创建 第一个手臂
            l, r, t, b = -ARM_W / 2, ARM_W / 2, ARM_LEN - ARM_W / 2, -ARM_W / 2
            arm1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            arm1.set_color(.8, .6, .4)
            self.arm1_trans = rendering.Transform()

            arm1.add_attr(self.arm1_trans)
            arm1.add_attr(self.robot_trans)
            self.viewer.add_geom(arm1)

            # 创建 第二个手臂
            l, r, t, b = -ARM_W / 2, ARM_W / 2, ARM_LEN - ARM_W / 2 * 2, -ARM_W / 2
            arm2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            arm2.set_color(.8, .6, .4)
            self.arm2_trans = rendering.Transform(translation=(0, ARM_LEN - ARM_W / 2))

            arm2.add_attr(self.arm2_trans)
            arm2.add_attr(self.arm1_trans)
            arm2.add_attr(self.robot_trans)
            self.viewer.add_geom(arm2)

            self.joint = rendering.make_circle(ARM_W / 2)
            self.joint.add_attr(self.arm1_trans)
            self.joint.add_attr(self.robot_trans)
            self.joint.set_color(.5, .5, .8)
            self.viewer.add_geom(self.joint)

            self.joint1 = rendering.make_circle(ARM_W / 2)
            self.joint1_trans = rendering.Transform(translation=(0, ARM_LEN - ARM_W / 2))
            self.joint1.add_attr(self.joint1_trans)
            self.joint1.add_attr(self.arm1_trans)
            self.joint1.add_attr(self.robot_trans)
            self.joint1.set_color(.1, .1, .8)
            self.viewer.add_geom(self.joint1)

            # Draw the position of finger
            finger = rendering.make_circle(5)
            finger.set_color(255 / 255, 192 / 255, 0)
            self.finger_trans = rendering.Transform()
            finger.add_attr(self.finger_trans)
            self.viewer.add_geom(finger)

            #Draw Trajectory
            ys = np.linspace(20, 585, 200)
            xs = 300 * np.sin((ys - 20) / 90) + 300
            trajectory = list(zip(xs, ys))

            track = rendering.make_polyline(trajectory)
            track.set_color(80 / 255, 80 / 255, 80/255)
            track.set_linewidth(4)
            self.viewer.add_geom(track)

        if self.state is None: return None

        # 旋转以（0，0）点为轴心，因此在最开始绘制的时候，需要考虑进去
        # arm = self._arm_geom
        # l,r,t,b = -ARM_W/2,ARM_W/2,ARM_LEN-ARM_W/2,-ARM_W/2
        # arm.v = [(l,b), (l,t), (r,t), (r,b)] # redraw the pole

        # 读取状态，更新坐标
        x      = self.state[0]
        y      = self.state[1]
        angle1 = self.state[2]
        angle2 = self.state[3]
        weights= self.state[4]

        # Refresh the location of the robot and robot's arms
        self.robot_trans.set_translation(x, y)
        self.arm1_trans.set_rotation(angle1)
        self.arm2_trans.set_rotation(angle2)

        # render the finger position
        self.finger_trans.set_translation(self.finger_position[0],self.finger_position[1])

        # Refresh the viewer (screen)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        """close the viewer
        """
        self.reward = 0 # 重新计算reward
        if self.viewer: 
            self.viewer.close()
            self.viewer = None
        return self.reward # 将reward输出，以便后续分析

def discretization(s):
    x,y,angle_arm1,angle_arm2,weight_set = s
    
    delta_angular =  math.pi / 360

    x_r  = int(x) # Read from PATH
    y_r  = round(y) # Calculated by env class
    a1_r = 0 if round((angle_arm1/delta_angular)%360) == 360 else round((angle_arm2/delta_angular)%360/SAM_STEP)
    a2_r = 0 if round((angle_arm1/delta_angular)%360) == 360 else round((angle_arm2/delta_angular)%360/SAM_STEP)
    # for testing 
    # print(x,y,round((angle_arm1/delta_angular)%360/SAM_STEP),round((angle_arm1/delta_angular)%360))
    return (x_r,y_r,a1_r,a2_r,weight_set)


if __name__ == "__main__":
    env = MainEnv()
    for _ in range(MAX_EPISODES):
        # initial the env 
        s = env.reset()
        print("----- initial state : ", s, "----\n")

        while True:
            
            # Take an action
            action = rl.choose_action(s)

            # Observation and render the movement
            for _ in range(SAM_STEP): 
                s_prime, reward, done = env.step(action)
                if DO_PLOT:
                    env.render()
            # Prepare to update the Q-table
            s_prime = discretization(s_prime)

            print("s_prime ", s_prime)
            print("s ", s)
            print("R ",reward, "\n")

            # Update the Q-table
            rl.learn(s, action, reward, s_prime)

            # Reset the s
            s = s_prime
            
            if done: 
                env.close()
                break


