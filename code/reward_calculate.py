import math
import numpy as np

target_points = [[150,450],[100,150],[450,200]]
end_point = [300,580]
threshold = 500

def reward1(finger_pos,state):
    """[summary]

    Args:
        finger_pos ([type]): [description]
        state ([type]): [description]

    Returns:
        [type]: [description]
    """
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
    """[summary]

    Args:
        finger_pos ([type]): [description]
        state ([type]): [description]

    Returns:
        [type]: [description]
    """
    res = 0
    x, y = finger_pos
    state = list(state)
    if state == [1,1,1]:
        x_score, y_score = target_points[2]
        if not isClose(x,y,x_score,y_score):
            res += math.exp(-(math.sqrt(math.pow((x - x_score), 2) + math.pow((y - y_score), 2)))*0.0001) - 1.5
        else:
            res += 10000
            state[2] = 0
    elif state == [1,1,0]:
        x_score, y_score = target_points[1]
        if not isClose(x,y,x_score,y_score):
            res += math.exp(-(math.sqrt(math.pow((x - x_score), 2) + math.pow((y - y_score), 2)))*0.0001) - 1.5
        else:
            res += 10000
            state[1] = 0
    elif state == [1,0,0]:
        x_score, y_score = target_points[0]
        if not isClose(x,y,x_score,y_score):
            res += math.exp(-(math.sqrt(math.pow((x - x_score), 2) + math.pow((y - y_score), 2)))*0.0001) - 1.5
        else:
            res += 10000
            state[0] = 0
    elif state == [0,0,0]:
        x_score, y_score = end_point
        if not isClose(x,y,x_score,y_score):
            res += math.exp(-(math.sqrt(math.pow((x - x_score), 2) + math.pow((y - y_score), 2)))*0.0001) - 1.5
        else:
            res += 10000

    return res, tuple(state)

def reward3(finger_pos,state):
    """[summary]

    Args:
        finger_pos ([type]): [description]
        state ([type]): [description]

    Returns:
        [type]: [description]
    """
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

def reward_sparse(finger_pos, state):
    """[summary]

    Args:
        finger_pos ([type]): [description]
        state ([type]): [description]

    Returns:
        [type]: [description]
    """
    state = list(state)
    x,y = finger_pos
    res = 0
    if sum(state) == 0:
        x_score, y_score = end_point
        if isClose(x,y,x_score,y_score): res += 10000
    else:
        for i in range(3):
            if state[i] == 1:
                x_score, y_score = target_points[i]
                if isClose(x,y,x_score,y_score):
                    state[i] = 0
                    res += 10000
                    break

    return res, tuple(state)

def reward_test(finger_pos, state):
    """[summary]

    Args:
        finger_pos ([type]): [description]
        state ([type]): [description]

    Returns:
        [type]: [description]
    """
    return finger_pos[1] - 1000, tuple(state)

def isClose(x,y,x1,y1):
    """[summary]

    Args:
        x ([type]): [description]
        y ([type]): [description]
        x1 ([type]): [description]
        y1 ([type]): [description]

    Returns:
        [type]: [description]
    """
    return 0 if math.pow((x - x1), 2) + math.pow((y - y1), 2) > threshold else 1
