# general imports
import numpy as np
import logging
import os
from collections import deque
import pandas as pd
# Box2D imports
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape,
                      revoluteJointDef, contactListener)

# open AI imports
import gym
from gym import spaces

import rl

# define constants
FPS = 50
SCALE = 30.0

# defining the shapes
STATION_SHAPE = [(-75, -3), (75, -3), (0, 147)]

LANDER_SHAPE = [
    (16.9840, 14.9840), (-27.0160, 14.9840), (-46.0160, -6.0160),
    (-6.0160, -41.0160), (32.9840, -42.0160), (33.9840, -6.0160)
    ]

LANDER_LEGS_SHAPE = [
        [(-31.0160, 36.9840), (-65.0160, 27.9840), (-46.0160, -6.0160)],
        [(17.9840, 35.9840),  (33.9840, -6.0160),  (51.9840, 28.9840)],
        [(53.9840, -82.0160),  (32.9840, -42.0160),  (-6.0160, -41.0160)],
        [(-66.0160, -82.0160), (-6.0160, -41.0160), (-45.0160, -43.0160)],
        [(-65.0160, 27.9840),  (-45.0160, -43.0160),  (-46.0160, -6.0160)],
        [(-45.0160, -43.0160),  (-6.0160, -41.0160),  (-46.0160, -6.0160)],
        [(51.9840, 28.9840), (33.9840, -6.0160), (32.9840, -42.0160)]
        ]

# Position constants
DISPLAY_X = 1600
DISPLAY_Y = 700

STATION_X = DISPLAY_X - 100
STATION_Y = DISPLAY_Y / SCALE

LANDER_INIT_X = 150
LANDER_INIT_Y = 650

# object constants
LANDER_FRICTION = 0.05
ASTEROID_RADIUS = 0.5

# enviroment constatnts
GRAVITY = - 10      # gravity of the mars
LANDER_POWER = 3    # defines the force of the engine
LANDER_SPEED = 100  # defines the lander speed

# level depending consts
MOVING_ASTEROIDS = False
OBSTICALS_ACTIVATED = True
REWARD = 50

# set category bits for collision detection
CB_LANDER = 0x0001
CB_SURFACE = 0x0010
CB_OBSTACLES = 0x0100
CB_STATION = 0x1000
GI_LANDER = -1
GI_SURFACE = 2
GI_OBSTACLES = 3
GI_STATION = 4


class ContactDetector(contactListener):
    '''
    Class for detecting and handle contacts between the different bodys.
    To handle the different contact classes we split the objects in 4 groups:
    (1) Satelite
    (2) Surface
    (3) Asteroid + blackhole
    (4) Station
    For the different contacts we get different rewards. We are mainly
    interested in contacts between 1 and the other objects.
    '''
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        '''
        Detects and handels collisions between the different objects
        '''
        # if the lander touches something the game is over
        filterA = contact.fixtureA.filterData
        filterB = contact.fixtureB.filterData

        if((filterA.groupIndex == GI_LANDER and
            filterB.groupIndex == GI_SURFACE)
            or (filterA.groupIndex == GI_SURFACE and
                filterB.groupIndex == GI_LANDER)):
            # contact lander surface
            self.env.done = True
            lander_position = self.env.lander.worldCenter.x
            station_position = self.env.station.worldCenter.x
            reward = 0.5 * REWARD - (station_position - lander_position)
            self.env.reward = reward
            self.env.last_rewards.append(reward)

        elif((filterA.groupIndex == GI_LANDER and
              filterB.groupIndex == GI_OBSTACLES)
                or (filterA.groupIndex == GI_OBSTACLES and
                    filterB.groupIndex == GI_LANDER)):
            # contact lander obstacle
            self.env.done = True
            self.env.reward = -REWARD
            self.env.last_rewards.append(-REWARD)

        elif((filterA.groupIndex == GI_LANDER and
              filterB.groupIndex == GI_STATION)
                or (filterA.groupIndex == GI_STATION and
                    filterB.groupIndex == GI_LANDER)):
            # contact lander station --> The aim
            self.env.done = True
            if self.env.reward == 0:
                self.env.last_rewards.append(REWARD)
            self.env.reward = REWARD

    def EndContact(self, contact):
        pass

# ############################################################################
# ##########                       LANDER                          ###########
# ############################################################################


class MarsLander(gym.Env):
    '''
    mars landing an Open ai test environment.
    About this environment:
    The goal is to land at the base station with the satelite,
    on the way to the
    station the satelite must avoid obstacles like black holes or asteroids.
    The possible actions of the satelite are discrete,
    either it activates the motors (action=1)
    or it deactivates them (action=0).
    This environment consists of several levels.
    At level 1 there are no obstacles,
    at level 2 there are obstacles and at level 3 there are obstacles
    that always appear on different places and move around the env.

    for level 1 is the observation space:
    s[0] x position of the satelite
    s[1] y position of the saddle rivet
    s[2] x position of the base station
    s[3] y position of the base station

    for level 2 and three the positions of the obstacles are transferred to the
    observations of 1
    s[4] x position of the asteroid
    s[5] y position of the asteroid
    s[6] x position of the black hole_1
    s[7] y position of the black hole_1
    s[8] x position of the black hole_2
    s[9] y position of the black hole_2

    The reward is sparse:
    (1) For hitting the space station the agent gets +50
    (2) For hitting an obstacle he gets -50
    (3) For crushing in the ground he gets 25 - distance to the lander

    Important for usage:
    For setting the level of dificulty init the nviroment like this :
    TODO README:
    env = gym.make('mars_lander-v0', level=1)
    where level is 1, 2 or 3
    '''
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self, level=1):
        # enviroment dificulty
        self.set_level(level)
        self.viewer = None

        # image paths for rendering

        # Path of this python script + image_group_1 subfolder
        img_dir = os.path.join(os.path.dirname(__file__), 'img_group_1')

        self.satelit_pic_path = os.path.join(img_dir, 'satellite.png')
        self.station_pic_path = os.path.join(img_dir, 'station_1.png')
        self.asteroid_pic_path = os.path.join(img_dir, 'asteroid.png')
        self.blackhole_pic_path = os.path.join(img_dir, 'blackhole.png')
        self.rotation_counter = None

        # init BOX2D
        self.mars = Box2D.b2World(gravity=(0, GRAVITY))
        self.mars.allowSleeping = False
        self.surface = None
        self.lander = None

        self.last_rewards = deque(maxlen=10)
        self.reset_counter = 0
        self.solved_problem = {}

        logging.basicConfig(level=logging.INFO)

        self.OBSERVATIONS = 4
        if OBSTICALS_ACTIVATED:
            self.OBSERVATIONS += 6

        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(self.OBSERVATIONS,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.reset()

    def set_level(self, level=1):
        '''
        This method sets the values for the different difficulty levels
        '''
        global MOVING_ASTEROIDS
        global OBSTICALS_ACTIVATED

        if level == 1:
            MOVING_ASTEROIDS = False
            OBSTICALS_ACTIVATED = False

        elif level == 2:
            MOVING_ASTEROIDS = False
            OBSTICALS_ACTIVATED = True

        elif level == 3:
            MOVING_ASTEROIDS = True
            OBSTICALS_ACTIVATED = True

        else:
            Exception('level must be an int between 1 and 3.')

    def _destroy(self):
        '''
        Method that delets all objects from our map.
        '''
        if (not self.surface):
            return
        self.mars.DestroyBody(self.surface)
        self.surface = None
        self.mars.DestroyBody(self.lander)
        self.lander = None
        for leg in self.legs:
            self.mars.DestroyBody(leg)
        if OBSTICALS_ACTIVATED:
            for element in self.blackhole:
                self.mars.DestroyBody(element)
            for element in self.blackhole_2:
                self.mars.DestroyBody(element)
            self.mars.DestroyBody(self.asteroid)

    # #########################################################################
    # ##########                          RESET                     ###########
    # #########################################################################

    def reset(self):
        '''
        Method that resets the enviroment should be executed
        before every new episode.
        @return : state, reward, done, info
        (True when hitting the space station 10 times in a row)
        '''
        self.reset_counter += 1
        self._destroy()
        self.mars.contactListener = ContactDetector(self)
        self.game_over = False
        self.collision = False
        self.prev_shaping = None
        self.done = False
        self.reward = 0

        X_SCALED = DISPLAY_X/SCALE
        Y_SCALED = DISPLAY_Y/SCALE

        # ####################################################################
        # ##########                   SURFACE                     ###########
        # ####################################################################

        CHUNKS = 11
        height = np.random.uniform(0, Y_SCALED/2, size=(CHUNKS+1,))
        chunk_x = [X_SCALED/(CHUNKS-1)*i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS//2-1]
        self.helipad_x2 = chunk_x[CHUNKS//2+1]
        self.helipad_y = Y_SCALED/5
        height[CHUNKS-4] = self.helipad_y
        height[CHUNKS-3] = self.helipad_y
        height[CHUNKS-2] = self.helipad_y
        height[CHUNKS-1] = self.helipad_y
        height[CHUNKS] = self.helipad_y
        smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1])
                    for i in range(CHUNKS)]

        self.surface = self.mars.CreateStaticBody(shapes=edgeShape
                                                  (vertices=[(0, 0),
                                                   (X_SCALED, 0)]))
        self.sky_polys = []
        self.mars_polys = []
        self.asteroid = []
        self.pos_a = []
        self.pos_b = []
        self.blackhole = []
        self.blackhole_2 = []

        p_max = 0
        p = []
        for i in range(CHUNKS-1):
            p1 = (chunk_x[i], smooth_y[i])         # edge point 1
            p2 = (chunk_x[i+1], smooth_y[i+1])     # edge point 2
            self.surface.CreateEdgeFixture(
                vertices=[p1, p2],
                density=0,
                friction=0.1,
                categoryBits=CB_LANDER,
                groupIndex=2)
            self.sky_polys.append([p1, p2,
                                  (p2[0], Y_SCALED), (p1[0], Y_SCALED)])
            self.mars_polys.append([p1, p2, (p2[0], 0), (p1[0], 0)])
            p_max = max(p1[1], p2[1])
            p.append(p_max)

        # ##################################################################
        # ##########              OBSTACLES                      ###########
        # ##################################################################
        if OBSTICALS_ACTIVATED:
            if MOVING_ASTEROIDS:
                x_asteroid = np.random.randint(LANDER_INIT_X, DISPLAY_X)
                y_asteroid = np.random.randint(max(p)*SCALE, DISPLAY_Y)
                x_blackhole = np.random.randint(LANDER_INIT_X, DISPLAY_X)
                y_blackhole = np.random.randint((max(p)+3) * SCALE, DISPLAY_Y)
                x_blackhole_2 = np.random.randint(LANDER_INIT_X, DISPLAY_X)
                y_blackhole_2 = np.random.randint((max(p)+3) * SCALE,
                                                  DISPLAY_Y)
            else:
                x_asteroid = 500
                y_asteroid = 300
                x_blackhole = 800
                y_blackhole = 500
                x_blackhole_2 = 1050
                y_blackhole_2 = 450

            # #################################################################
            # ##########                Asteroid                   ############
            # #################################################################

            if (not MOVING_ASTEROIDS):
                self.pos_a = [(x_asteroid, y_asteroid)]
                self.asteroid = self.mars.CreateStaticBody(
                    position=(x_asteroid / SCALE, y_asteroid / SCALE),
                    fixtures=fixtureDef(
                        shape=circleShape(radius=ASTEROID_RADIUS/SCALE,
                                          pos=(0, 0)),
                        density=0,
                        friction=0.1,
                        categoryBits=CB_OBSTACLES,
                        groupIndex=GI_OBSTACLES
                        )
                )

            else:
                self.pos_a = [(x_asteroid, y_asteroid)]
                self.asteroid = self.mars.CreateDynamicBody(
                    position=(x_asteroid / SCALE, y_asteroid / SCALE),
                    fixtures=fixtureDef(
                        shape=circleShape(radius=ASTEROID_RADIUS/SCALE,
                                          pos=(0, 0)),
                        density=0,
                        friction=10.1,
                        categoryBits=CB_OBSTACLES,
                        groupIndex=GI_OBSTACLES
                    )
                )

                asteriod_impulse = (np.random.uniform(-10, 10),
                                    np.random.uniform(-10, 0))
                asteroid_position = self.asteroid.worldCenter
                self.asteroid.ApplyLinearImpulse(asteriod_impulse,
                                                 asteroid_position, True)

            # #################################################################
            # ##########             Blackhole                      ###########
            # #################################################################
            while (abs(x_asteroid - x_blackhole) < 3*SCALE and
                   abs(y_asteroid - y_blackhole) < 3*SCALE):
                x_blackhole = np.random.randint(LANDER_INIT_X, DISPLAY_X)

            self.pos_b = [(x_blackhole, y_blackhole)]

            for i in range(4):
                bla = self.mars.CreateStaticBody(
                    position=(x_blackhole / SCALE, y_blackhole / SCALE),
                    fixtures=fixtureDef(
                        shape=circleShape(radius=(2+i*0.5) / SCALE,
                                          pos=(0, 0)),
                        density=0,
                        friction=0.1,
                        groupIndex=GI_OBSTACLES,
                        categoryBits=CB_OBSTACLES
                    )
                )
                self.blackhole.append(bla)

            for i in range(4):
                bla = self.mars.CreateStaticBody(
                    position=(x_blackhole_2 / SCALE, y_blackhole_2 / SCALE),
                    fixtures=fixtureDef(
                        shape=circleShape(radius=(2+i*0.5) / SCALE,
                                          pos=(0, 0)),
                        density=0,
                        friction=0.1,
                        groupIndex=GI_OBSTACLES,
                        categoryBits=CB_OBSTACLES
                    )
                )
                self.blackhole_2.append(bla)

        # #####################################################################
        # ##########                     Station                    ###########
        # #####################################################################
        self.station = self.mars.CreateStaticBody(
            position=(STATION_X / SCALE, Y_SCALED/5),
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE / 2, y / SCALE / 2)
                                             for x, y in STATION_SHAPE]),
                density=5.0,
                friction=0.1,
                groupIndex=GI_STATION,
                categoryBits=CB_STATION
            )
        )
        self.station.color1 = (0.5, 0.5, 0.5)

        # init random lander start position
        x_lander = np.random.randint(20, DISPLAY_X/2)
        y_lander = np.random.randint(DISPLAY_Y/2, DISPLAY_Y-20)

        # #####################################################################
        # ##########                    LANDER                      ###########
        # #####################################################################
        self.lander = self.mars.CreateDynamicBody(
            # position=(LANDER_INIT_X/SCALE, LANDER_INIT_Y/SCALE),
            position=(x_lander/SCALE, y_lander/SCALE),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE / 2, y / SCALE / 2)
                                             for x, y in LANDER_SHAPE]),
                density=5.0,
                friction=LANDER_FRICTION,
                categoryBits=CB_LANDER,
                restitution=0.0,
                groupIndex=GI_LANDER
                )
            )

        # this should prevent the satelit from rotating
        self.lander.angular = 0.0
        self.lander.angularDamping = 1000

        # set color of the lander
        self.lander.color1 = (0.8, 0.8, 0.8)
        self.lander.color2 = (0.8, 0.8, 0.8)

        # set const velocity
        impulse = (LANDER_SPEED, 0)
        position = self.lander.worldCenter
        self.lander.ApplyLinearImpulse(impulse, position, True)

        # satelite
        self.legs = []
        for i in range(7):
            leg = self.mars.CreateDynamicBody(
                position=(x_lander/SCALE, y_lander/SCALE),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[(x / SCALE / 2, y / SCALE / 2)
                                       for x, y in LANDER_LEGS_SHAPE[i]]),
                    density=5.0,
                    friction=0.1,
                    restitution=0.2,
                    categoryBits=CB_LANDER,
                    # maskBits=0x001,
                    groupIndex=GI_LANDER
                )
            )
            leg.ground_contact = False
            leg.color1 = (0.8, 0.8, 0.8)
            leg.color2 = (0.8, 0.8, 0.8)

            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=100,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
                )
            leg.joint = self.mars.CreateJoint(rjd)

            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs
        self.drawlist.append(self.station)

        # init action lets decide the env
        a = self.action_space.sample()

        return self.step(a)

    # #########################################################################
    # ##########                       STEP/PHYSICS                 ###########
    # #########################################################################

    def step(self, p_action):
        '''
        Step method executes a single step in the enviroment.
        @param p_action : action that should be executed 0 or 1.
        @return : state, reward, done, info (True when hitting the space
        station 10 times in a row)
        '''
        # this parameter defines the real applied power
        power = float(p_action * LANDER_POWER)

        impulse = (0, power)
        position = self.lander.worldCenter
        self.lander.ApplyLinearImpulse(impulse, position, True)

        self.mars.Step(1.0/FPS, 6*30, 2*30)
        self.lander.angle = 0.0

        # define the states
        state = [
            self.lander.worldCenter.x,
            self.lander.worldCenter.y,
            STATION_X/SCALE,
            STATION_Y/SCALE
            ]
        if OBSTICALS_ACTIVATED:
            state = [
                self.lander.worldCenter.x,
                self.lander.worldCenter.y,
                STATION_X/SCALE,
                STATION_Y/SCALE,
                self.asteroid.worldCenter.x,
                self.asteroid.worldCenter.y,
                self.blackhole[3].worldCenter.x,
                self.blackhole[3].worldCenter.y,
                self.blackhole_2[3].worldCenter.x,
                self.blackhole_2[3].worldCenter.y
                ]

        if OBSTICALS_ACTIVATED:
            assert len(state) == 10

        else:
            assert len(state) == 4

        if (self.lander.worldCenter.y > (DISPLAY_Y/SCALE)):
            self.done = True
            self.reward = -REWARD
            self.last_rewards.append(-REWARD)

        if (self.lander.worldCenter.x > (DISPLAY_X/SCALE)):
            self.done = True
            self.reward = -REWARD
            self.last_rewards.append(-REWARD)

        if (sum(self.last_rewards) >= (10 * REWARD)):
            print('\x1b[6;30;42m' + 'Success! You solved the mars' +
                  'landing problem within {} '.format(self.reset_counter +
                                                      'episodes') + '\x1b[0m')
            self.solved_problem = True

        return np.array(state), self.reward, self.done, self.solved_problem

    # #########################################################################
    # ##########                       RENDERING                     ##########
    # #########################################################################
    def render(self, mode='human'):
        '''
        Rendering method. displays the enviroment so
        you can see what is happening.
        '''
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(DISPLAY_X, DISPLAY_Y)
            self.viewer.set_bounds(0, DISPLAY_X/SCALE, 0, DISPLAY_Y/SCALE)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0.7, 0.7, 0.7))

        for s in self.mars_polys:
            self.viewer.draw_polygon(s, color=(1, 0.3, 0))

        # if we activate the obstacles we want to see the blackholes
        # and asteroids
        if OBSTICALS_ACTIVATED:
            # added to handle the rotations
            if self.rotation_counter is None:
                self.rotation_counter = 0
            elif self.rotation_counter > 100:
                self.rotation_counter -= 100
            self.rotation_counter += 0.01

            # redner blackhole 1
            self.black_1_pic = rendering.Image(self.blackhole_pic_path, 3, 3)
            self.black_1_pic.set_color(0.5, 0.5, 0.5)
            black_1_trans = rendering.Transform()
            black_1_trans.set_translation(self.blackhole[0].worldCenter.x,
                                          self.blackhole[0].worldCenter.y)
            black_1_trans.set_rotation(-self.rotation_counter)
            self.black_1_pic.add_attr(black_1_trans)
            self.viewer.add_onetime(self.black_1_pic)

            # render blackhole 2
            self.black_2_pic = rendering.Image(self.blackhole_pic_path, 3, 3)
            self.black_2_pic.set_color(0.5, 0.5, 0.5)
            black_2_trans = rendering.Transform()
            black_2_trans.set_translation(self.blackhole_2[0].worldCenter.x,
                                          self.blackhole_2[0].worldCenter.y)
            black_2_trans.set_rotation(self.rotation_counter)
            self.black_2_pic.add_attr(black_2_trans)
            self.viewer.add_onetime(self.black_2_pic)

            # render asteroid
            self.ast_img = rendering.Image(self.asteroid_pic_path, 2, 2)
            self.ast_img.set_color(0.5, 0.5, 0.5)
            ast_trans = rendering.Transform()
            ast_trans.set_translation(self.asteroid.worldCenter.x,
                                      self.asteroid.worldCenter.y)
            ast_trans.set_rotation(self.rotation_counter)
            self.ast_img.add_attr(ast_trans)
            self.viewer.add_onetime(self.ast_img)

        # render satelite
        self.sat_img = rendering.Image(self.satelit_pic_path, 2.3, 2.3)
        self.sat_img.set_color(0.5, 0.5, 0.5)
        imgtrans = rendering.Transform()
        imgtrans.set_translation(self.lander.worldCenter.x,
                                 self.lander.worldCenter.y)
        self.sat_img.add_attr(imgtrans)
        self.viewer.add_onetime(self.sat_img)

        # render space station
        self.station_img = rendering.Image(self.station_pic_path, 4, 4)
        self.station_img. set_color(0.5, 0.5, 0.5)
        stationtrans = rendering.Transform()
        stationtrans.set_translation(self.station.worldCenter.x,
                                     (self.station.worldCenter.y + 0.45))
        self.station_img.add_attr(stationtrans)
        self.viewer.add_onetime(self.station_img)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

q_learning = rl.Q_LEARNING()
# q_learning.load_csv('q_table.csv')

if __name__ == "__main__":
    
    
    record_table = pd.DataFrame(columns=('episode','mean_reward','total_reward','final_reward','solved'))
    env = MarsLander()
# 10:04
    for episode in range(1000):
        
        obs = env.reset()
        obs = obs[0]
        
        total_reward = []
        
        while True:
            # env.render()
            action = q_learning.choose_action(obs)

            new_state, reward, done, solved_problem = env.step(action)
            total_reward.append(reward)
            q_learning.learn(obs,action,reward,new_state)

            obs = new_state

            # print(obs, reward, done, solved_problem)
            # input()
            if done: break
    
        record_table = record_table.append([{'episode':episode,
                    'mean_reward':np.mean(total_reward),
                    'total_reward':np.sum(total_reward),
                    "final_reward":reward,
                    'solved':solved_problem}], ignore_index=True)
        print(obs, reward, done, solved_problem)
    
    record_table.to_csv("mars_lander_result2.csv",mode="a",index=False,sep=',')
    q_learning.save_csv('q_table2.csv')
