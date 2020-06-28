import math
import numpy as np

target_points = [[150,450],[100,150],[450,200]]
end_point = [20,580]
threshold = 500

def reward1(finger_pos,state):
    res = 0
    x, y = finger_pos
    state = list(state)
    if sum(state) > 0:
        for i in range(3):
            if state[i] == 1:
                x_score, y_score = target_points[i]
                if not isClose(x,y,x_score,y_score):
                    # Normal procedure
                    res += -math.sqrt(math.pow((x - x_score), 2) + math.pow((y - y_score), 2))
                else:
                    # The finger touches the cargo. Cancel the state index
                    res += 10000
                    state[i] = 0
    else:
        x_score,y_score = end_point
        res += -math.sqrt(math.pow((x - x_score), 2) + math.pow((y - y_score), 2))
    return res, tuple(state)

def reward2(finger_pos,state):
    res = 0
    x, y = finger_pos
    for i in range(3):
        x_score, y_score = target_points[i]
        if state[i]: res += -np.sum(np.abs((x,y)-(x_score,y_score)))
    return res

def reward3(finger_pos,state):
    res = 0
    x, y = finger_pos
    state = list(state)
    if sum(state) > 0:
        for i in range(3):
            if state[i] == 1:
                x_score, y_score = target_points[i]
                if not isClose(x,y,x_score,y_score):
                    # Normal procedure
                    res += math.exp(-(math.sqrt(math.pow((x - x_score), 2) + math.pow((y - y_score), 2)))*0.0001) - 1.5
                else:
                    # The finger touches the cargo. Cancel the state index
                    res += 10000
                    state[i] = 0
    else:
        x_score,y_score = end_point
        res += math.exp(-(math.sqrt(math.pow((x - x_score), 2) + math.pow((y - y_score), 2)))*0.0001) - 1.5
    return res - sum(state), tuple(state)

def reward_test(finger_pos, state):
    return finger_pos[1] - 1000, tuple(state)

def isClose(x,y,x1,y1):

    return 0 if math.pow((x - x1), 2) + math.pow((y - y1), 2) > threshold else 1
