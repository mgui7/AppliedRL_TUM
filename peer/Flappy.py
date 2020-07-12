# import pygame
import random
import numpy as np
import gym
import os
from gym import spaces
from gym.envs.classic_control import rendering

screen_width = 400
screen_height = 600
# gameDisplay = pygame.display.set_mode((screen_width, screen_height))
# pygame.display.set_caption("Flappy")
ground_height = 0.87 * screen_height
fps = 300
dt = 0

class Bird:
  y = 200.0
  v = 0.0
  a = 0.0
  g = 0.05
  b = -0.2

  def __init__(self):
    pass

  def update_position(self, boost_is_active):
    if not boost_is_active:
      self.a = self.g
    else:
      self.a = self.b

    self.v = self.v + self.a
    self.y = self.y + self.v

    if self.y < 0:
      self.y = 0

    if self.y == screen_height:
      # prevent bird from getting stuck at top of the screen
      self.v = 0
    # self.y = screen_height

    # we should not get into if statement, safety measure for observe() in game
    if self.v > 1:
      self.v = 1
    if self.v < -1:
      self.v = -1

    if self.y > ground_height - 50:
      # check if bird would hit the ground
      self.v = 0
      self.y = ground_height
      # print("velocity: ", self.v)
      return True
    else:
      return False

class PipePair:

  def __init__(self, offset_x):
    # random.seed(1)
    self.x = screen_width * 0.5 + offset_x
    self.pair_gap = 0.25 * screen_height
    random_height_variation = random.randint(-50, 50)

    # img_path = os.path.join(os.path.dirname(__file__), 'Images_Flappy')
    # self.lower_pipeImg = pygame.image.load(os.path.join(img_path, 'pipe.png'))
    # self.lower_pipeImg = pygame.transform.scale(self.lower_pipeImg, (150, 335))
    # self.upper_pipeImg = pygame.transform.flip(self.lower_pipeImg, False, True)
    self.y_lower = 0.6 * screen_height + random_height_variation
    self.y_upper = self.y_lower - self.pair_gap - 355

  def check_collision(self, x, y):
    if (x + 40 > self.x and x < self.x + 140) and (y > self.y_lower - 25 or y < self.y_upper + 325):
      return True
    return False

  def update_position(self):
    self.x = self.x - 0.25 * 5
    if self.x < -150:
      # if self.x < -350:
      self.x = screen_width * 1.2
      random_height_variation = random.randint(-50, 50)
      self.y_lower = 0.6 * screen_height + random_height_variation
      self.y_upper = self.y_lower - self.pair_gap - 355

  # y position must be in range [
  def set_position(self, x, y):
    self.x = x
    self.y_lower = y
    self.y_upper = self.y_lower - self.pair_gap - 355

class Game:

  def __init__(self, v=-1, rel_x=-1, rel_y=-1):
    self.boost_time = 0
    self.b = Bird()
    self.pp1 = PipePair(0)
    self.pp2 = PipePair(screen_width / 2)
    self.pp3 = PipePair(screen_width)
    # self.pp2 = PipePair(screen_width *2/3)
    # self.pp3 = PipePair(screen_width *4/3)
    self.pipes = list()
    self.pipes.append(self.pp1)
    self.pipes.append(self.pp2)
    self.pipes.append(self.pp3)
    self.reward = 0
    self.game_over = False
    # self.clock = pygame.time.Clock()
    self.viewer = None

    # if input is valid then set the state
    if v >= 0:
      self.set_state(v, rel_x, rel_y)

  def update(self):
    if self.boost_time > 0:
      check = self.b.update_position(True)
      self.boost_time = self.boost_time - 1
    else:
      check = self.b.update_position(False)

    for pair in self.pipes:
      pair.update_position()

    if self.b.y >= screen_height:
      self.boost_time = 0

    for pair in self.pipes:
      check = check or pair.check_collision(0.33 * screen_width, self.b.y)
    if check:
      self.game_over = True

  def boost(self):
    if self.boost_time == 0:
      self.b.v = 0.0
    if not self.b.y == screen_height:  # if the bird is already on the top of screen do not allow boosts
      self.boost_time = 12

  def observe(self):
    x_pipe = screen_width * 2
    y_pipe = 0

    for pipe in self.pipes:
      if pipe.x > 0.33 * screen_width - 80:
        if pipe.x < x_pipe:
          x_pipe = pipe.x
          y_pipe = pipe.y_lower - pipe.pair_gap / 2

    bird_velocity = self.b.v * 5 + 10
    rel_x = x_pipe - 0.33 * screen_width
    rel_x = min(rel_x, 600)
    rel_x = max(rel_x, -60)

    rel_y = y_pipe - self.b.y
    rel_y = min(rel_y, 280)
    rel_y = max(rel_y, -280)
    bird_velocity = min(bird_velocity, 10)
    bird_velocity = max(bird_velocity, 0)

    self.reward = 15

    if self.boost_time > 0:
      boost = 1
    else:
      boost = 0

    ret = [boost, int(bird_velocity), int(rel_x / 10) + 6, int((rel_y + 280) / 7)]

    return tuple(ret)

  def set_state(self, bird_velocity, rel_x, rel_y):
    self.b.v = 0.22 * bird_velocity - 1.3
    self.b.y = screen_height / 2  # due to relative coordinates the y position of bird can be arbitrary
    rel_x = rel_x * 30
    rel_y = rel_y * 30
    self.pp1.set_position(50 + rel_x, self.b.y + rel_y)
    self.pp2.set_position(50 + rel_x + screen_width / 2, self.b.y + rel_y)
    self.pp3.set_position(50 + rel_x + screen_width, self.b.y + rel_y)

  def action(self, action):
    if action == 1:
      self.boost()
    self.update()

  def evaluate(self):
    if self.game_over:
      return self.reward - 1000
    return self.reward

  def is_done(self):
    if self.game_over:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return True
    return False

  def view(self):
    crashed = False
    # for event in pygame.event.get():
    #	if event.type == pygame.QUIT:
    #		crashed = True
    #	elif event.type == pygame.KEYDOWN:
    #		self.boost()

    # gameDisplay.fill((255, 255, 255))
    # for pipe in self.pipes:
    #	gameDisplay.blit(pipe.upper_pipeImg, (pipe.x, pipe.y_upper))
    #	gameDisplay.blit(pipe.lower_pipeImg, (pipe.x, pipe.y_lower))

    # self.b.birdImage = rendering.Transform()
    # self.b.birdImage.set_translation(0.33*screen_width, self.b.y)
    # self.viewer.add_onetime(self.b.bidrImage)
    # gameDisplay.blit(self.b.birdImage, (0.33 * screen_width, self.b.y))
    # pygame.display.update()
    # self.clock.tick(fps)
    # print(fps)
    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.Viewer(screen_width, screen_height)
      self.viewer.set_bounds(0, screen_width, 0, screen_height)
      img_path = os.path.join(os.path.dirname(__file__), 'Images_Flappy')
      img_path = os.path.join(img_path, 'bird.png')
      self.image_b = rendering.Image(img_path, 50, 35)
      self.image_b.set_color(1, 1, 1)
      self.imgtrans_b = rendering.Transform()
      self.image_b.add_attr(self.imgtrans_b)

      img_path = os.path.join(os.path.dirname(__file__), 'Images_Flappy')
      img_path = os.path.join(img_path, 'pipe.png')
      self.images_pipes = list()
      self.imagetrans_pipes = list()
      for pipepair in self.pipes:
        image_pipe = rendering.Image(img_path, 150, 335)
        image_pipe.set_color(1, 1, 1)
        imgtrans_pipe = rendering.Transform()
        image_pipe.add_attr(imgtrans_pipe)

        image_pipe2 = rendering.Image(img_path, 150, 335)
        image_pipe2.set_color(1, 1, 1)
        imgtrans_pipe2 = rendering.Transform()
        image_pipe2.add_attr(imgtrans_pipe2)

        self.images_pipes.append((image_pipe, image_pipe2))
        self.imagetrans_pipes.append((imgtrans_pipe, imgtrans_pipe2))

    # self.birdImage = pygame.image.load(os.path.join(img_path, 'bird.png'))
    # self.birdImage = rendering.Image(self.img_path, 50, 35)
    # self.birdImage = pygame.transform.scale(self.birdImage, (50, 35))
    # self.imgtrans = rendering.Transform()

    i = 0
    for pipepair in self.pipes:
      x = pipepair.x
      y_lower = pipepair.y_lower
      y_upper = pipepair.y_upper

      self.imagetrans_pipes[i][0].set_translation(x + 75, screen_height - y_upper - 167)
      self.imagetrans_pipes[i][0].set_rotation(3.143)
      self.imagetrans_pipes[i][1].set_translation(x + 75, screen_height - y_lower - 167)
      self.viewer.add_onetime(self.images_pipes[i][0])
      self.viewer.add_onetime(self.images_pipes[i][1])

      i += 1

    # self.imgtrans_p1.set_translation(0.33*screen_width,screen_height-self.b.y)
    # self.viewer.add_onetime(self.image_p1)

    self.imgtrans_b.set_translation(int(0.33 * screen_width) + 50, screen_height - self.b.y - 17.5)
    # self.imgtrans_b.set_translation(0+25,screen_height-0-17.5)
    self.viewer.add_onetime(self.image_b)
    return self.viewer.render(return_rgb_array=True)

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None

class FlappyEnv(gym.Env):

  def __init__(self):
    self.flappy = Game()
    self.action_space = spaces.Discrete(2)
    self.observation_space = spaces.Box(np.array([0, 0, 0, 0]), np.array([1, 26, 30, 80]), dtype=np.int)

  def reset(self):
    del self.flappy
    self.flappy = Game()
    obs = self.flappy.observe()
    return obs

  def step(self, action):
    self.flappy.action(action)
    obs = self.flappy.observe()
    reward = self.flappy.evaluate()
    done = self.flappy.is_done()
    return obs, reward, done, 0

  def render(self, mode=""):
    self.flappy.view()

  def close(self):
    self.flappy.close()


if __name__ == "__main__":
  g = Game()
  while True:
    g.update()
    g.view()
    g.observe()
