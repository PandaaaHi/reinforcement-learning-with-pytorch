import pandas as pd
import numpy as np

np.random.seed(2)

class QLearningTable:
    def __init__(self, actions, lr=0.01, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, s):
        self.check_state_exist(s)
        
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[s, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        
        return action
    
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        
        q_predict = self.q_table.loc[s, a]
        
        if s_ == 'terminal':
            q_target = r
        else:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
            
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
    
    def check_state_exist(self, s):
        if s not in self.q_table.index:
            self.q_table = pd.concat([
                self.q_table,
                pd.DataFrame(
                    [[0] * len(self.actions)],
                    columns=self.q_table.columns,
                    index=[s]
                )
            ])