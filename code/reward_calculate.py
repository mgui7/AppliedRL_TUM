import math
import numpy as np

# The cargo locations
target_points = [[150,450],[100,150],[450,200]]
# The final destination location
end_point = [300,580]
# The distance threshold that when the finger get within this range with a cargo,
# it would automatically collect the cargo.
threshold = 500


def reward_Inorder(finger_pos,state):
    """The dense reward, where we set the cargo collecting order
    Args:
        finger_pos (tuple): Finger Position
        state (tuple): Cargo collection state
    Returns:
        reward (float) : The corresponding reward
        state (tuple): Updated cargo collection state
    """
    res = 0
    x, y = finger_pos
    state = list(state)
    if state == [1,1,1]:
        x_score, y_score = target_points[2]
        if not isClose(x,y,x_score,y_score):
            # No cargo collection
            res += math.exp(-(math.sqrt(math.pow((x - x_score), 2) + math.pow((y - y_score), 2)))*0.0001) - 1.5
        else:
            # Cargo collection
            res += 10000
            state[2] = 0
    elif state == [1,1,0]:
        x_score, y_score = target_points[1]
        if not isClose(x,y,x_score,y_score):
            # No cargo collection
            res += math.exp(-(math.sqrt(math.pow((x - x_score), 2) + math.pow((y - y_score), 2)))*0.0001) - 1.5
        else:
            # Cargo collection
            res += 10000
            state[1] = 0
    elif state == [1,0,0]:
        x_score, y_score = target_points[0]
        if not isClose(x,y,x_score,y_score):
            # No cargo collection
            res += math.exp(-(math.sqrt(math.pow((x - x_score), 2) + math.pow((y - y_score), 2)))*0.0001) - 1.5
        else:
            # Cargo collection
            res += 10000
            state[0] = 0
    elif state == [0,0,0]:
        # Proceeding to the destination
        x_score, y_score = end_point
        if not isClose(x,y,x_score,y_score):
            # Not terminal
            res += math.exp(-(math.sqrt(math.pow((x - x_score), 2) + math.pow((y - y_score), 2)))*0.0001) - 1.5
        else:
            # Terminal
            res += 10000

    return res, tuple(state)


def reward_Noorder(finger_pos,state):
    """The dense reward, where we do not set the cargo collecting order
    Args:
        finger_pos (tuple): Finger Position
        state (tuple): Cargo collection state
    Returns:
        reward (float) : The corresponding reward
        state (tuple): Updated cargo collection state
    """
    res = 0
    x, y = finger_pos
    state = list(state)
    if sum(state) > 0:
        for i in range(3):
            if state[i] == 1:
                x_score, y_score = target_points[i]
                if not isClose(x,y,x_score,y_score):
                    # No cargo collection. Normalizing reward
                    res += math.exp(-(math.sqrt(math.pow((x - x_score), 2) + math.pow((y - y_score), 2)))*0.0001) - 1.5
                else:
                    # The finger touches the cargo. Cancel the state index
                    res += 10000
                    state[i] = 0
    else:
        # Proceeding to the destination
        x_score,y_score = end_point
        res += math.exp(-(math.sqrt(math.pow((x - x_score), 2) + math.pow((y - y_score), 2)))*0.0001) - 1.5
    return res - sum(state), tuple(state)


def reward_sparse(finger_pos, state):
    """The dense reward, where we do not set the cargo collecting order
    Args:
        finger_pos (tuple): Finger Position
        state (tuple): Cargo collection state
    Returns:
        reward (float) : The corresponding reward
        state (tuple): Updated cargo collection state
    """
    state = list(state)
    x,y = finger_pos
    res = 0
    if sum(state) == 0:
        # Proceeding to the destination
        x_score, y_score = end_point
        if isClose(x,y,x_score,y_score): res += 10000
    else:
        for i in range(3):
            if state[i] == 1:
                x_score, y_score = target_points[i]
                if isClose(x,y,x_score,y_score):
                    # Cargo collection
                    state[i] = 0
                    res += 10000
                    break

    return res, tuple(state)


def reward_test(finger_pos, state):
    """[Testing reward]
    Args:
        finger_pos (tuple): Finger Position
        state (tuple): Cargo collection state
    Returns:
        reward (float) : The corresponding reward
        state (tuple): Updated cargo collection state
    """
    return finger_pos[1] - 1000, tuple(state)


def isClose(x,y,x1,y1):
    """Determining whether the finger is close enough to a cargo to collect it
    Args:
        x (float)   : finger pos x
        y (float)   : finger pos y
        x1 (float)  : target pos x
        y1 (float)  : target pos y
    Returns:
        (bool): The decision whether the finger collects the cargo
    """
    return 0 if math.pow((x - x1), 2) + math.pow((y - y1), 2) > threshold else 1
