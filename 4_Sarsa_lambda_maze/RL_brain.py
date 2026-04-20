import numpy as np
import pandas as pd

np.random.seed(2)

class RL:
    def __init__(self, actions, lr=0.01, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        
    def choose_action(self, s):
        self.check_state_exist(s)
        if np.random.uniform() < self.epsilon:
            a = np.random.choice(self.actions)
        else:
            s_a = self.q_table.loc[s, :]
            a = np.random.choice(s_a[s_a == np.max(s_a)].index)
        return a
    
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
    
class SarsaLambdaTable(RL):
    def __init__(self, actions, lr=0.01, gamma=0.9, epsilon=0.1, lambda_=0.9):
        super().__init__(actions, lr, gamma, epsilon)
        self.eligibility_trace = self.q_table.copy()
        self.lambda_ = lambda_
        
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
            
            self.eligibility_trace = pd.concat([
                self.eligibility_trace,
                pd.DataFrame(
                    [[0] * len(self.actions)],
                    columns=self.eligibility_trace.columns,
                    index=[s]
                )
            ])
        
    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ == 'terminal':
            q_target = r
        else:
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        error = q_target - q_predict
        
        self.eligibility_trace.loc[s, :] = 0
        self.eligibility_trace.loc[s, a] = 1
        
        self.q_table += self.lr * error * self.eligibility_trace
        self.eligibility_trace *= self.gamma * self.lambda_