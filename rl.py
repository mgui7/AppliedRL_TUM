import numpy as np
import elements
import pandas as pd

screen_width = elements.SCREEN_W
screen_height = elements.SCREEN_H

class Q_LEARNING(object):
    # def __init__(self, a_dim, s_dim, a_bound,):
    #     pass
    def __init__(self, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
        self.actions = [(i,j,k) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]]
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation = (0,0,0)):
        # Observation: location, ang1, ang2
        self.check_state_if_exist(observation)
        # action select
        if np.random.uniform() < self.epsilon:
            # choose the best action
            state_action = self.q_table.loc[[observation], :]
            # some actions may have the same value, randomly choose from these actions
            tmp = state_action[state_action == np.max(state_action)].columns
            idx = np.random.choice(len(tmp))
            action = tmp[idx]
        else:
            # choose the random action
            idx = np.random.choice(len(self.actions))
            action = self.actions[idx]
        return action

    def learn(self, s, a, r, s_):

        self.check_state_if_exist(s_)
        q_predict = self.q_table.loc[[s], [a]]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[[s_], :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[[s], [a]] += self.lr * (q_target - q_predict)  # update

    def check_state_if_exist(self,state):

        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name  = state,
                )
            )
