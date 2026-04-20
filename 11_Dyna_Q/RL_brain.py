import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, lr=0.01, gamma=0.9, epsilon=0.9):
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.table = pd.DataFrame(columns=self.actions, dtype=np.float)

    def choose_action(self, s):
        self.check_state_exist(s)
        if np.random.uniform() < self.epsilon:
            s_a = self.table.loc[s, :]
            a = np.random.choice(s_a[s_a == np.max(s_a)].index)
        else:
            a = np.random.choice(self.actions)
        return a

    def learn(self, s, a, r, s_, done):
        self.check_state_exist(s_)
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * self.table.loc[s_, :].max()
        self.table.loc[s, a] += self.lr * (q_target - self.table.loc[s, a])

    def check_state_exist(self, s):
        if s not in self.table.index:
            # self.table = pd.concat([
            #     self.table,
            #     pd.DataFrame(
            #         [[0] * len(self.actions)],
            #         columns=self.table.columns,
            #         index=[s]
            #     )
            # ])
            self.table = self.table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.table.columns,
                    name=s
                )
            )

class EnvModel:
    def __init__(self, actions):
        self.actions = actions
        self.database = pd.DataFrame(columns=self.actions, dtype=np.object)

    def store_transition(self, s, a, r, s_):
        if s not in self.database.index:
            # self.database = pd.concat([
            #     self.database,
            #     pd.DataFrame(
            #         [[None] * len(self.actions)],
            #         columns=self.database.columns,
            #         index=[s]
            #     )
            # ])
            self.database = self.database.append(
                pd.Series(
                    [None] * len(self.actions),
                    index=self.database.columns,
                    name=s
                )
            )
        self.database.at[s, a] = (r, s_)

    def sample_s_a(self):
        s = np.random.choice(self.database.index)
        a = np.random.choice(self.database.loc[s].dropna().index)
        return s, a

    def get_r_s_(self, s, a):
        r, s_ = self.database.loc[s, a]
        return r, s_