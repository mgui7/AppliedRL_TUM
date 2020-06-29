import numpy as np
import elements
import pandas as pd
import os

screen_width = elements.SCREEN_W
screen_height = elements.SCREEN_H
ANGLE_STEPS   = elements.ANGLE_STEPS

class Q_LEARNING(object):

    def __init__(self, learning_rate = 0.1, reward_decay = 0.9, e_greedy = 0.9):
        self.actions = [(i,j,k) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]]
        # self.actions = [(i,j,k) for i in [-1,0,1] for j in [0] for k in [0]]
        # self.actions = [(i,j,k) for i in [1] for j in [-1,0,1] for k in [-1,0,1]]
        self.lr      = learning_rate
        self.gamma   = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def  choose_action(self, observation = (0,0,0,0,[0,0,0])):
        # Observation: location, ang1, ang2
        observation[4] = tuple(observation[4])
        observation = tuple(observation)
        self.check_state_if_exist(observation)
        # action select
        if np.random.uniform() < self.epsilon:
            # choose the best action
            state_action = self.q_table.loc[[observation], :]
            # some actions may have the same value, randomly choose from these actions
            tmp    = state_action.iloc[:,state_action.values[0] == max(state_action.values[0])].columns
            idx    = np.random.choice(len(tmp))
            action = tmp[idx]
        else:
            # choose the random action
            idx    = np.random.choice(len(self.actions))
            action = self.actions[idx]
        return action

    def learn(self, s, a, r, s_):

        s, s_ = tuple(s), tuple(s_)
        self.check_state_if_exist(s_)
        q_predict = self.q_table.loc[[s], [a]]
        if not (s_[1] >= 580 and sum(s_[4]) == 0):
            q_target = r + self.gamma * self.q_table.loc[[s_], :].max(axis = 1)  # next state is not terminal
        else:
            q_target = [r]  # next state is terminal
        self.q_table.loc[[s], [a]] += self.lr * (q_target[0] - q_predict.iloc[0,0])  # update

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

    def save_file(self, name = 'q_table.npy'):
        np.save(name , self.q_table)
        print('ML Process successfully saved into' + name)

    def load_file(self, name = 'q_table.npy'):
        try:
            self.q_table = np.load(name)
            print('ML Process successfully loaded from' + name)
            return True
        except Exception as e:
            print('Error occured while loading NP array')
            return False

    def save_csv(self,name = 'code/misc/q_table.csv'):
        if not os.path.isfile(name):
            print('ML Process successfully saved into ' + name)
            self.q_table.to_csv(name)
        else:
            overwrite = input('WARNING ' + name + ' already exists! Do you wish to overwrite <Y/N>? \n')
            overwrite = overwrite.lower()
            if overwrite == 'n':
                new_filename = input("Type new filename: \n ")
                self.save_csv(new_filename)
            else:
                print('ML Process successfully overwritten into ' + name)
                self.q_table.to_csv(name)

    def load_csv(self, name = 'code/misc/q_table.csv'):
        if os.path.isfile(name):
            self.q_table = pd.read_csv(name , index_col=[0])
            print('ML Process successfully loaded from ' + name)
            def helper(string):
                string = [_ for _ in string]
                for i in range(len(string)):
                    tmp = string[i]
                    tmp = tmp.replace(')','')
                    tmp = tmp.replace('(','')
                    tmp = tmp.split(',')
                    string[i] = [round(float(_)) for _ in tmp]
                return string


            tmp = helper(self.q_table.columns)
            tmp = [tuple(_) for _ in tmp]
            self.q_table.columns = tmp

            tmp = helper(self.q_table.index)
            for i in range(len(tmp)):
                t = tmp[i]
                tmp[i] = (t[0],t[1],t[2],t[3],(t[4],t[5],t[6]))
            self.q_table.index = tmp

            return True
        else:
            print('Error occured while loading CSV, creating new dataset')
            print('---Press ENTER to continue---')
            # input()
            return False
