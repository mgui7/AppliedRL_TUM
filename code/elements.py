"""
这个module存储着固定的，不会变化的背景图案，例如起点和终点。
"""


import pyglet
from gym.envs.classic_control import rendering

SCREEN_W = 600
SCREEN_H = 600

START_POINT_POSITION = (20,20)
START_POINT_COLOR = (255/255,0,0)

END_POINT_POSITION = (90,90)
END_POINT_COLOR  = (0,255/255,0)

SCORE1_POINT_POSITION = (1,2)
SCORE2_POINT_POSITION = (1,2)
SCORE3_POINT_POSITION = (1,2)
SCORE4_POINT_POSITION = (1,2)

SCORE_POINT_COLOR = (255/255,192/255,0)

p1 = (150,150)
p2 = (450,450)
p3 = (150,450)
p4 = (450,150)

RENDER_PERIOD = 5
ANGLE_STEPS = 8

class TrackLine():
    def __init__(self,viewer):
        self.viewer = viewer

    def draw_line_track(self):
        self.track = rendering.Line((SCREEN_W/2,20), (SCREEN_W/2,SCREEN_H-20))
        self.track.set_color(0,0,0)
        self.viewer.add_geom(self.track)

    
class StartEndPoint():
    def __init__(self,viewer):
        self.start_position = START_POINT_POSITION
        self.end_position = END_POINT_POSITION
        self.viewer = viewer

    def draw_points(self):
        start = rendering.make_circle(10,10)
        start_transform = rendering.Transform(translation=(300,20))
        start.add_attr(start_transform)
        start.set_color(START_POINT_COLOR[0],START_POINT_COLOR[1],START_POINT_COLOR[2])
        self.viewer.add_geom(start)

        end = rendering.make_circle(10,10)
        end_transform = rendering.Transform(translation=(300,580))
        end.add_attr(end_transform)
        end.set_color(END_POINT_COLOR[0],END_POINT_COLOR[1],END_POINT_COLOR[2])
        self.viewer.add_geom(end)



