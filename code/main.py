import numpy as np
import pyglet
from pyglet.window import Window
import pandas as pd
import random
import math
import time
import reward_calculate as R

from rl import Q_LEARNING
rl = Q_LEARNING()


PATH = np.load('code/maps/long_map_300_sin.npy',encoding = "latin1")[0:581]

SAM_STEP = 45
MAX_EPISODES = 500
MAX_BATCH = 2

DO_PLOT = False
DO_RECORD = True

DELTA_ANGLE = 1
ARM_LEN = 150
V = 20

class Viewer(pyglet.window.Window):
    """ Viewer Class to display movement of robot.
    """

    def __init__(self,state):
        self.state = state
        # Create a window and size of window is predefined.
        super(Viewer, self).__init__(width=600, height=600, resizable=False, caption='Arm', vsync=False)

        # Color of Bg but is displaced by image here.
        # pyglet.gl.glClearColor(1, 1, 1, 1)

        # save the elements into batch, prepare to draw
        self.batch = pyglet.graphics.Batch()

        # Read elements
        pyglet.resource.path=[r"resources"]
        pyglet.resource.reindex()
        robot = pyglet.resource.image("robot.png")
        arm = pyglet.resource.image("arm.png")
        score = pyglet.resource.image("score.png")
        bg = pyglet.resource.image("light_map.png")

        self.center_image(bg)
        self.bg = pyglet.sprite.Sprite(img=bg, x=bg.width/2, y=bg.height/2, batch=self.batch)

        # Based on Anchor，read initial states
        self.center_image(robot)
        self.robot = pyglet.sprite.Sprite(img=robot, x=self.state[0], y=self.state[1], batch=self.batch)

        self.change_image_arm(arm)
        self.arm1 = pyglet.sprite.Sprite(img=arm, x=self.state[0], y=self.state[1], batch=self.batch)
        self.arm2 = pyglet.sprite.Sprite(img=arm, x=self.state[0], y=self.state[1]+arm.height-arm.width, batch=self.batch)

        self.change_image_arm(score)
        self.score1 = pyglet.sprite.Sprite(img=score, x=150, y=450, batch=self.batch)
        self.score2 = pyglet.sprite.Sprite(img=score, x=100, y=150, batch=self.batch)
        self.score3 = pyglet.sprite.Sprite(img=score, x=450, y=200, batch=self.batch)

        # show labels
        # title 
        self.title = pyglet.text.Label('ARL Object Group 5',
            font_name='consoles',
            font_size=14,
            x=300, y=600-10,
            anchor_x='center', anchor_y='center',batch=self.batch,color=(0,0,0,255))
        # Reward label
        self.reward_label = pyglet.text.Label("Reward: " + '%d'%(0),
            font_name='consoles',
            font_size=10,
            x=10, y=600-10,
            anchor_x='left', anchor_y='center',batch=self.batch,color=(0,0,0,255))
        # Robot Position Label
        self.position_label = pyglet.text.Label("Position: " + "(0,0)",
            font_name='consoles',
            font_size=10,
            x=10, y=600-30,
            anchor_x='left', anchor_y='center',batch=self.batch,color=(0,0,0,255))
        self.angle_label = pyglet.text.Label("Angles: " + "(0,0)",
            font_name='consoles',
            font_size=10,
            x=10, y=600-50,
            anchor_x='left', anchor_y='center',batch=self.batch,color=(0,0,0,255))



    def _update_robot(self,state):
        # Update location of robot
        self.robot.x = max(state[0],0)
        self.robot.y = max(state[1],0)

        # Update location of arm1
        self.arm1.x = self.robot.x
        self.arm1.y = self.robot.y
        self.arm1.rotation = state[2]


        # Update location of arm2
        self.arm2.x = self.arm1.x + (self.arm1.height-self.arm1.width) * math.sin(self.arm1.rotation*math.pi/180)
        self.arm2.y = self.arm1.y + (self.arm1.height-self.arm1.width) * math.cos(self.arm1.rotation*math.pi/180)
        self.arm2.rotation = state[3] 

    def _scores_detection(self,state):
        s1,s2,s3 = state[4]
        if not s1:
            self.score1.visible = False
        if not s2:
            self.score2.visible = False
        if not s3:
            self.score3.visible = False


    def _update_label(self,reward):
        self.reward_label.text= "Reward: " + '%f'%(reward)
        self.position_label.text = "Position: " + '({},{})'.format(round(int(self.robot.x)),round(self.robot.y)) 
        self.angle_label.text = "Angles: " + "({},{})".format(round(self.arm1.rotation,2),round(self.arm2.rotation,2))
        

    def render(self,state,reward):
        self._update_robot(state)  
        self._scores_detection(state)
        self._update_label(reward)
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()       
        self.batch.draw()
        self.robot.draw()
        self.arm1.draw()
        self.arm2.draw()
        self.score1.draw()
        self.score2.draw()
        self.score3.draw()
        self.title.draw()
        self.reward_label.draw()
        self.position_label.draw()
        self.angle_label.draw()
        
    def center_image(self,image):
        image.anchor_x = image.width/2
        image.anchor_y = image.height/2
    
    def change_image_arm(self,image):
        image.anchor_x = image.width/2
        image.anchor_y = image.width/2


class MeinEnv(object):
    """ MeinEnv class
    This Class creates an environment.
    """

    viewer = None
    def __init__(self):
        pass

    def update_state(self,action):
        # Unzip state 
        x,y,angle_arm1,angle_arm2,weight_set = self.state 
        # Unzip action
        robot_move, arm1_rotate, arm2_rotate = action
        
        # Update x and y w.r.t. action 

        y = min(580,max(y + V*robot_move/SAM_STEP,20))
        x = PATH[round(y)][0]


        angle_arm1 = angle_arm1 + DELTA_ANGLE * arm1_rotate
        angle_arm2 = angle_arm2 + DELTA_ANGLE * arm2_rotate

        # Zip state 
        self.state = [x,y,self._reset_degree(angle_arm1),self._reset_degree(angle_arm2),weight_set]

        return None,None

    def step(self, action):
        
        # Update state 
        self.update_state(action)

        x,y        = self.state[:2]
        ang1, ang2 = self.state[2:4]
        joint_position = (x + ARM_LEN * np.sin(math.pi*ang1/180), y + ARM_LEN * np.cos(math.pi*ang1/180))
        finger_position = (joint_position[0] + ARM_LEN * np.sin(math.pi*ang2/180),
                        joint_position[1] + ARM_LEN * np.cos(math.pi*ang2/180))

        # Update reward
        self.reward, self.state[4] = R.reward2(finger_position,self.state[4])
        # reward, self.state[4] = R.reward_test(finger_position,self.state[4])

        # Update done
        done = bool(self.state[1] >= 580)

        return self.state, self.reward, done

    def _reset_degree(self,degree):
        """ Degree is in 0 ~ 360
        Args:
            degree: double angle of arms
        Return:
            degree: angle after rescale.
        """
        if degree >= 360: return degree - 360
        elif degree < 0: return degree + 360
        return degree

    def render(self):
        # Initial a viewer
        if self.viewer is None:
            self.viewer = Viewer(self.state)

        # If viewer is created, it should be updated.
        self.viewer.render(self.state,self.reward)

    def reset(self):
        self.viewer = None
        # Reset the state 
        x = PATH[20][0]
        y = 20
        angle_arm1 = 0
        angle_arm2 = 0
        weight_set = [1,1,1]
        # Zip all states
        self.state  = [x,y,angle_arm1,angle_arm2,weight_set]
        # Reset reward
        self.reward = 0
        return self.state
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        


def discrete(s):
    """ The states from Env are not satisfied the condition.
    It have to be converted to certain int states.
    
    Args:
        s: tuple, states
    returns:
        tuple: converted states
    """
    
    x,y,angle_arm1,angle_arm2,weight_set = s
    x_r  = int(x)
    y_r  = int(round(y))
    
    # if angle_arm1 == 360: angle_arm1 = 0 
    # else: angle_arm1 = round(angle_arm1/45)
    # if angle_arm2 == 360: angle_arm2 = 0 
    # else: angle_arm2 = round(angle_arm2/45)

    if angle_arm1 == 360: angle_arm1 = 0 
    else: angle_arm1 = round(angle_arm1)
    if angle_arm2 == 360: angle_arm2 = 0 
    else: angle_arm2 = round(angle_arm2)

    return [x_r,y_r,int(angle_arm1),int(angle_arm2),tuple(weight_set)]


def record(pd_frame,batch,episode,s):
    """ By Record function,  all states will be saved in pd_frame.
    And pd_frame is waiting for exporting.
    Note: pd_frame = pd_frame.append is important. Otherwise pd_frame would be None.
    
    Args:
        pd_frame: contains all state
        episode: string label of pd_frame
        s_prime: tuple last state
        s: tuple new state
        reward: int reward
        action: int number 
    
    Returns:
        pd_frame
    """

    pd_frame = pd_frame.append([{'batch':batch,
                        'term':episode,
                        's1':s[4][0], 
                        's2':s[4][1],
                        's3':s[4][2]}], ignore_index=True)

    return pd_frame

if __name__ == '__main__':
    file_name = 'code/misc/' + time.strftime("%m%d%H%M", time.localtime()) + '_' + str(MAX_EPISODES*MAX_BATCH) + '.csv'
    # file_name = 'misc/06271618_10000.csv'
    rl.load_csv(file_name)

    # Initial record table 
    record_table = pd.DataFrame(columns=('batch','term','s1','s2','s3'))
    data_name = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

    for k in range(MAX_BATCH):

        # Initial env
        env = MeinEnv()

        for i in range(MAX_EPISODES):
            # Update episode label for each term

            # reset all state 
            s = env.reset()
            print("---------- "+str(i)+" term "+"----------")

            for cycle in range(200):

                # choose an action from Q-table
                action = rl.choose_action(discrete(s))

                # if plot window
                if DO_PLOT: env.render()
                
                # Jump into sample period
                max_reward = -1000
                for _ in range(SAM_STEP):
                    s_prime, reward, done = env.step(action)
                    # record maximum reward
                    if reward > 5000: max_reward = reward
                    if DO_PLOT: env.render()
                    if done:break

                # Learning and update table
                rl.learn(discrete(s), action, max(reward,max_reward), discrete(s_prime))
                # print("s_prime: ", discrete(s_prime),"|s: ",discrete(s),"|R: ",reward,"|Action: ",action)


                # Update states
                s = discrete(s_prime)
                
                if done:
                    print("s_prime: ", discrete(s_prime),"|s: ",discrete(s),"|R: ",reward,"\n")
                    # print("Done in",cycle,'Iterations')
                    # pd_frame is exported as file if done is true.
                   
                    env.close()
                    break
            
            # Record states, action and reward
            if DO_RECORD: record_table = record(record_table,k,str(i),s)
    rl.save_csv(file_name)
    if DO_RECORD: record_table.to_csv( "code/misc/"+data_name+ ".csv",mode="a",index=False,sep=',')
