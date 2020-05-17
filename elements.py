import pyglet
from gym.envs.classic_control import rendering

"""
Introduction of this modules

This module draws some static elements, start point and target, or each trajectory on the map. 
Attention, the score points are not attracted by this module because they will disappear after 
the end of arm2 reach the score point. 

"""

# Size of screen
SCREEN_W = 600
SCREEN_H = 600

# The color of start and end point
START_POINT_COLOR = (255/255,0,0)
END_POINT_COLOR  = (0,255/255,0)

class TrackLine():
    """ The trajectory of movement of robot.
    """
    def __init__(self,viewer):
        """
        Args:
            viewer: the current viewer of the program. A element should be attached on this viewer.
        """
        self.viewer = viewer

    def draw_line_track(self):
        self.track = rendering.Line((SCREEN_W/2,20), (SCREEN_W/2,SCREEN_H-20))
        self.track.set_color(0,0,0)
        self.viewer.add_geom(self.track)

    
class StartEndPoint():
    """ The start and end point on the map
    """
    def __init__(self,viewer):
        self.viewer = viewer

    def draw_points(self):
        start = rendering.make_circle(10,10)
        start_transform = rendering.Transform(translation=(300,20))
        start.add_attr(start_transform)
        start.set_color(START_POINT_COLOR[0],START_POINT_COLOR[1],START_POINT_COLOR[2])
        self.viewer.add_geom(start)

        end = rendering.make_circle(10,10)
        end_transform = rendering.Transform(translation=(300,560))
        end.add_attr(end_transform)
        end.set_color(END_POINT_COLOR[0],END_POINT_COLOR[1],END_POINT_COLOR[2])
        self.viewer.add_geom(end)