import numpy as np
import pandas as pd
import os


class Q_LEARNING(object):
    """Main Structure of Q-Learning"""


    def __init__(self, lr = 0.4, reward_decay = 0.9, e_greedy = 0.9):
        """Initializing the Q-Learning Algorithm
        Args:
            lr (float, optional):           [Learning rate]. Defaults to 0.1.
            reward_decay (float, optional): [Reward decay]. Defaults to 0.9.
            e_greedy (float, optional):     [Coeff. for epsilon-greedy algorithm]. Defaults to 0.9.
        """
        # The current action space. Consists of 27 available actions
        self.actions = [(0),(1)]
        # self.actions = [(i,j,k) for i in [1] for j in [-1,0,1] for k in [-1,0,1]]

        self.lr      = lr
        self.gamma   = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


    def  choose_action(self, observation = (0,0,0,0,[0,0,0])):
        """Choosing the action based on the Q-table
        Args:
            observation (tuple, optional): [The current observed state]. Defaults to (0,0,0,0,[0,0,0]).
        Returns:
            action: [The chosen action using epsilon-greedy algorithm]
        """ 
        # Observation: location_x, location_y, angle1, angle2, the cargo states
        observation = tuple(observation)
        self.check_state_if_exist(observation)

        # Action select
        if np.random.uniform() < self.epsilon:
            # Choosing the best action
            state_action = self.q_table.loc[[observation], :]
            # Some actions may have the same value, randomly choose from these actions in this case
            tmp    = state_action.iloc[:,state_action.values[0] == max(state_action.values[0])].columns
            idx    = np.random.choice(len(tmp))
            action = tmp[idx]
        else:
            # Choosing a random action
            idx    = np.random.choice(len(self.actions))
            action = self.actions[idx]
        return action


    def learn(self, s, a, r, s_):
        """Learning from the state to state transition
        Args:
            s (tuple)   : [The former state]
            a (tuple)   : [The chosen action]
            r (float)   : [The consequent reward for this transition process]
            s_ (tuple)  : [The current state]
        """
        s, s_ = tuple(s), tuple(s_)
        self.check_state_if_exist(s_)
        # Locating the predictive value
        q_predict = self.q_table.loc[[s], [a]]

        # Checking if the state is terminal
        # if not (s_[1] >= 580 and sum(s_[4]) == 0):
        #     # Next state is not terminal
        #     q_target = r + self.gamma * self.q_table.loc[[s_], :].max(axis = 1)
        # else:
        #     # Next state is terminal
        #     q_target = [r]

        q_target = r + self.gamma * self.q_table.loc[[s_], :].max(axis = 1)

        # Update
        self.q_table.loc[[s], [a]] += self.lr * (q_target[0] - q_predict.iloc[0,0])


    def check_state_if_exist(self,state):
        """Creating new state if that state is not available in the Q-table
        Args:
            state (tuple): [The state to be created]
        """
        if state not in self.q_table.index:
            # Appending new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0]*len(self.actions),index = self.q_table.columns, name  = state,))


    def save_csv(self,name = 'code/misc/q_table.csv'):
        """Saving the learning progress
        Args:
            name (str, optional): [The relative path to save into]. Defaults to 'code/misc/q_table.csv'.
        """
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
        """Loading the learning progress
        Args:
            name (str, optional): The relative path to load from. Defaults to 'code/misc/q_table.csv'.
        Returns:
            (bool): Whether the loading process has been successfully carried out or not
        """
        if os.path.isfile(name):
            self.q_table = pd.read_csv(name , index_col=[0])
            print('ML Process successfully loaded from ' + name)

            def helper(string):
                # Helper to tuplize the column and row integers
                string = [_ for _ in string]
                for i in range(len(string)):
                    tmp = string[i]
                    tmp = tmp.replace(')','')
                    tmp = tmp.replace('(','')
                    tmp = tmp.split(',')
                    string[i] = [round(float(_)) for _ in tmp]
                return string

            tmp = helper(self.q_table.columns)
            # Tuplize the elements in column names
            tmp = [tuple(_) for _ in tmp]
            self.q_table.columns = tmp

            tmp = helper(self.q_table.index)
            # Tuplize the elements in row names
            for i in range(len(tmp)):
                t = tmp[i]
                # Rearrange into the correct format
                tmp[i] = (t[0],t[1],t[2],t[3],(t[4],t[5],t[6]))
            self.q_table.index = tmp

            return True
        else:
            print('Error occurred while loading CSV. \nCreating new data set')
            print('-' * 50)
            return False
