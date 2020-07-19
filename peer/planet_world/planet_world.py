import sys, math
import numpy as np

import Box2D
from Box2D import b2Vec2
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import pandas as pd
import gym
from gym import spaces
from gym.envs.classic_control import rendering
import rl

# Render graphics
SCALE = 30.0  # Scaling factor
VIEWPORT_W = 900
VIEWPORT_H = 900
FORWARD = [(-10, -10), (-10.5, -11.5), (-9.5, -11.5)]
LEFT = [(-8.5, -12), (-10, -11.5), (-10, -12.5)]
RIGHT = [(-11, -12), (-9.5, -12.5), (-9.5, -11.5)]
G = 6.67408e-11  # Newtonian constant of gravitation
TRACK_HEIGHT = 200  # Desired track height
# Planet defination
PLANET_R = 60 / SCALE
PLANET_X = 0
PLANET_Y = 0
PLANET_MASS = 10e11

# Orbitor defination
ORBITOR_W = 10
ORBITOR_H = 7
ORBITOR_SHAPE = [(+ORBITOR_W, +ORBITOR_H), (-ORBITOR_W, +ORBITOR_H),
                 (-ORBITOR_W, -ORBITOR_H), (ORBITOR_W, -ORBITOR_H)]
# Define Engine power
MAIN_ENGINE_POWER = 1.0
SIDE_ENGINE_POWER = 0.1
# Position of the side engines
SIDE_ENGINE_HEIGHT = 5
SIDE_ENGINE_AWAY = 10

class PlanetWorld(gym.Env):
  def _destroy(self):
    if not self.planet: return
    self.world.DestroyBody(self.planet)
    self.planet = None
    self.world.DestroyBody(self.orbitor)
    self.orbitor = None

  def _findAngle(self, x, y):
    if x >= 0:
      theta = math.atan(y / x)
      if y < 0:
        theta = theta + 2 * math.pi
    else:
      theta = math.atan(y / x) + math.pi
    return theta

  def __init__(self):
    FPS = 60
    self.PRINT_ONCE = True
    self.demo = False  # If True, the space craft is set to have a proper initial position and speed. The simulator runs with out termination
    self.MAX_FRAME = 5000  # maximum frame per each episode
    self.REWARD_FRAME = 1.  # maximum reward per frame
    self.HEIGHT_TOL = 0.05  # Default height error tolerence
    self.ECC_TOL = 0.05  # Default eccentricity tolerence
    self.world = Box2D.b2World(gravity=(0, -0))  # No linear gravity in empty space
    self.planet = None
    self.viewer = None
    self.success = False
    self.action_space = spaces.Discrete(4)
    self.observation_space = spaces.Box(-VIEWPORT_W, VIEWPORT_W, shape=(5,))
    self.act = 0
    self.reset()

  def reset(self,
            FPS=60,
            DEMO=False,
            INITIAL_X=-VIEWPORT_W / SCALE * 0.4,
            INITIAL_Y=0.,
            INITIAL_VX=0.0,
            INITIAL_VY=1.0,
            INITIAL_ANGLE=0.0,
            INITIAL_ANGLE_VEL=0.0):
    # Initial pose and velocity of the orbitor
    self.FPS = FPS
    self.initial_x = INITIAL_X
    self.initial_y = INITIAL_Y
    self.initial_vx = INITIAL_VX
    self.initial_vy = INITIAL_VY
    self.init_angle = INITIAL_ANGLE
    self.init_angle_vel = INITIAL_ANGLE_VEL
    self.success = False
    self._destroy()
    self.traj = []
    self.PAST_LENGTH = round(self.FPS) * 2  # Number of frames keeped in recorder
    self.height_error = [1 for x in range(self.PAST_LENGTH)]  # Past reward recroder
    # Create planet
    self.planet = self.world.CreateStaticBody(
      shapes=circleShape(pos=(PLANET_X, PLANET_Y), radius=PLANET_R),
      angle=0.0,
      fixedRotation=True
    )
    self.planet.CreateCircleFixture(pos=(PLANET_X, PLANET_Y), radius=PLANET_R)
    self.planet.color1 = (0.85, 0.85, 0.85)
    self.planet.color2 = (0.85, 0.85, 0.85)
    # Create orbitor
    self.orbitor = self.world.CreateDynamicBody(
      position=(self.initial_x, self.initial_y),
      angle=self.init_angle,
      fixtures=fixtureDef(
        shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in ORBITOR_SHAPE]),
        density=300.0,
        friction=0.1
      )
    )
    self.orbitor.color1 = (0.3, 0.3, 0.8)
    self.orbitor.color2 = (0.3, 0.3, 0.8)
    self.orbitor.angularVelocity = self.init_angle_vel
    self.orbitor.linearVelocity = b2Vec2(self.initial_vx, self.initial_vy)
    self.drawlist = [self.planet, self.orbitor]
    # Convert initial state into polar coordinate
    distance = math.sqrt(pow(self.initial_x, 2) + pow(self.initial_y, 2))
    theta = self._findAngle(self.initial_x, self.initial_y)
    vel_result = math.sqrt(pow(self.initial_vx, 2) + pow(self.initial_vy, 2))
    vel_tan = self.initial_vx * math.sin(theta) + self.initial_vy * math.cos(theta)
    vel_nor = self.initial_vy * math.sin(theta) + self.initial_vx * math.cos(theta)
    orbitor_angle = (self.init_angle + math.pi / 2) % (2 * math.pi)  # Align axises the and to [0, 2*pi]
    state = [
      distance,
      (theta - orbitor_angle) % (2 * math.pi),
      vel_tan,
      vel_nor,
      INITIAL_ANGLE_VEL
    ]
    self.pose_now = self.orbitor.position - self.planet.position
    return state

  def render(self, mode="human"):
    action = self.act
    pose = self.pose_now
    # Initialize the viewer
    if self.viewer is None:
      self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
      self.viewer.set_bounds(-VIEWPORT_W / SCALE / 2, VIEWPORT_W / SCALE / 2, -VIEWPORT_H / SCALE / 2,
                             VIEWPORT_H / SCALE / 2)
    # Render the background
    self.viewer.draw_polygon(
      [(VIEWPORT_W / SCALE / 2, VIEWPORT_H / SCALE / 2), (-VIEWPORT_W / SCALE / 2, VIEWPORT_H / SCALE / 2),
       (-VIEWPORT_W / SCALE / 2, -VIEWPORT_H / SCALE / 2), (VIEWPORT_W / SCALE / 2, -VIEWPORT_H / SCALE / 2)],
      color=(0.0, 0.0, 0.0))
    # Render target track
    self.viewer.draw_circle(radius=TRACK_HEIGHT / SCALE, res=30, filled=False, color=(1.0, 1.0, 1.0))
    # Draw Planet and orbitor
    for obj in self.drawlist:
      for fixture in obj.fixtures:
        trans = fixture.body.transform
        if type(fixture.shape) is circleShape:
          t = rendering.Transform(translation=trans * fixture.shape.pos)
          self.viewer.draw_circle(fixture.shape.radius, 20, color=obj.color1).add_attr(t)
          self.viewer.draw_circle(fixture.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
        else:
          path = [trans * v for v in fixture.shape.vertices]
          self.viewer.draw_polygon(path, color=obj.color1)
          path.append(path[0])
          self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
    # Action indicator
    if action == 1:
      self.viewer.draw_polygon(FORWARD, color=(1.0, 1.0, 1.0))
    if action == 2:
      self.viewer.draw_polygon(RIGHT, color=(1.0, 1.0, 1.0))
    if action == 3:
      self.viewer.draw_polygon(LEFT, color=(1.0, 1.0, 1.0))
    # Orientation indicator
    rot = self.orbitor.angle + math.pi / 2
    pointer = [(-6, -12), (2 * math.cos(rot) - 6, 2 * math.sin(rot) - 12),
               (2 * math.cos(rot) - 6 + 0.5 * math.cos(rot + 2.5), 2 * math.sin(rot) - 12 + 0.5 * math.sin(rot + 2.5)),
               (2 * math.cos(rot) - 6, 2 * math.sin(rot) - 12),
               (2 * math.cos(rot) - 6 + 0.5 * math.cos(rot - 2.5), 2 * math.sin(rot) - 12 + 0.5 * math.sin(rot - 2.5))]
    self.viewer.draw_polyline(pointer, color=(1.0, 1.0, 1.0), linewidth=3)
    self.traj.append((pose.x, pose.y))
    self.viewer.draw_polyline(self.traj, color=(1.0, 1.0, 0), linewidth=1)
    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None

  def step(self, action):
    self.act = action
    # action = 0 No action taken
    # action = 1 Main engine fire
    # action = 2,3 Side engine right/left
    # Engines
    pos = (self.orbitor.position - self.planet.position)
    vel = self.orbitor.linearVelocity
    tip = (math.sin(self.orbitor.angle), math.cos(self.orbitor.angle))
    side = (-tip[1], tip[0])
    if self.success == True:
      action = 0  # Agent takes no action after successfully entering the orbit
    if action == 1:
      # Main engine
      a = b2Vec2(tip[0] * MAIN_ENGINE_POWER, -tip[1] * MAIN_ENGINE_POWER) * 60  # defalut setting under 60FPS
      impulse_pos = (self.orbitor.position[0], self.orbitor.position[1])
      self.orbitor.ApplyLinearImpulse(-a / self.FPS, impulse_pos, True)
    # Orientation engines
    if action in [2, 3]:
      if action == 2:
        direction = -1
      else:
        direction = 1
      ox = side[0] * (direction * SIDE_ENGINE_AWAY / SCALE)
      oy = -side[1] * (direction * SIDE_ENGINE_AWAY / SCALE)
      impulse_pos = (self.orbitor.position[0] + ox - tip[0] * SIDE_ENGINE_AWAY / SCALE,
                     self.orbitor.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE)
      a = b2Vec2(-ox * SIDE_ENGINE_POWER, -oy * SIDE_ENGINE_POWER) * 60
      self.orbitor.ApplyLinearImpulse(a / self.FPS, impulse_pos, True)
    # Apply gravity
    distance = math.sqrt(pow(pos.x, 2) + pow(pos.y, 2))
    accelleration = G * PLANET_MASS / pow(distance, 2)
    a_x = pos.x / distance * accelleration
    a_y = pos.y / distance * accelleration
    a = b2Vec2(-a_x, -a_y)  # Gravity of the planet is a force pointing to its center
    impulse_pos = self.orbitor.GetWorldPoint(localPoint=(0.0, 0.0))
    self.orbitor.ApplyForce(a * self.orbitor.mass, impulse_pos, True)

    # Transform into polar coordinate system
    # angular coordinate belongs to [0, 2*pi]
    theta = self._findAngle(pos.x, pos.y)
    vel_result = math.sqrt(pow(vel.x, 2) + pow(vel.y, 2))
    vel_tan = vel.x * math.sin(theta) + vel.y * math.cos(theta)
    vel_nor = vel.y * math.sin(theta) + vel.x * math.cos(theta)

    orbitor_angle = (self.orbitor.angle + math.pi / 2) % (2 * math.pi)  # Align axises the and to [0, 2*pi]
    state = [
      distance,  # radial coordinate
      (theta - orbitor_angle) % (2 * math.pi),  # difference between orientation angle and angular coordinate
      vel_tan,  # tangential velocity
      vel_nor,  # normal velocity
      self.orbitor.angularVelocity,  # spin speed
    ]
    assert len(state) == 5

    # High interation number to aquire a fine simulation
    self.world.Step(1.0 / self.FPS, 10, 10)

    reward = 0
    # Calculate squre of eccentricity
    w = pow(distance * pow(vel_result, 2) / G / PLANET_MASS - 1, 2) * pow(vel_tan / vel_result, 2) + pow(
      vel_nor / vel_result, 2)
    if (1 - abs(TRACK_HEIGHT - distance * SCALE) / TRACK_HEIGHT) > self.HEIGHT_TOL:
      reward = self.REWARD_FRAME * (1 - abs(TRACK_HEIGHT - distance * SCALE) / TRACK_HEIGHT)
    else:
      reward = 10 * self.REWARD_FRAME * (1 - abs(TRACK_HEIGHT - distance * SCALE) / TRACK_HEIGHT)
    if not self.demo:
      done = True
      if action == 1:
        reward -= MAIN_ENGINE_POWER / 5  # less fuel spent is better
      elif action in [2, 3]:
        reward -= SIDE_ENGINE_POWER
      # Terminate when the orbitor is on track or task fails
      # Conditions of task failing:
      # Main engine can't resist the gravity in drop zone
      if distance < 3:
        reward -= 100
        print("Crashed")
        self.reset()
      elif distance > 2 * TRACK_HEIGHT / SCALE:
        reward -= 100
        print("Escaped")
        self.reset()
      elif abs(self.orbitor.angularVelocity) > math.pi / 4:
        reward -= 100
        print("Spining too fast")
        self.reset()
      # Judge whether the orbitor is on track
      elif max(self.height_error) < self.HEIGHT_TOL and math.sqrt(w) < self.ECC_TOL:
        if self.PRINT_ONCE:  # Print information for only once
          print("Success")
          print("Eccentricity of orbit: " + str(math.sqrt(w)))
          print("Average height error: " + str(np.mean(self.height_error)))
          self.PRINT_ONCE = False
        reward += 1000
        self.success = True
        # self.reset()
      else:
        self.height_error.pop(0)
        self.height_error.append(abs(TRACK_HEIGHT - distance * SCALE) / TRACK_HEIGHT)
        done = False
    else:
      done = False
    self.pose_now = pos

    return state, reward, self.success, done

q_learning = rl.Q_LEARNING()

if __name__ == '__main__':
  
    record_table = pd.DataFrame(columns=('episode','mean_reward','total_reward','final_reward'))

    env = PlanetWorld()
# 17:53 - 21: 00 = 100 123
    for episode in range(300):
      print(episode)
      total_reward = []
      state = env.reset()

      while True:

        # env.render()

        action = q_learning.choose_action(state)

        new_state, reward, success, done = env.step(action)
        total_reward.append(reward)

        q_learning.learn(state, action,reward,new_state)

        state = new_state

        # print(state,reward, success,done)

        if done: break
        
      record_table = record_table.append([{'episode':episode,'mean_reward':np.mean(total_reward),'total_reward':np.sum(total_reward),"final_reward":reward}], ignore_index=True)
      
    record_table.to_csv( "planet_world_result.csv",mode="a",index=False,sep=',')




