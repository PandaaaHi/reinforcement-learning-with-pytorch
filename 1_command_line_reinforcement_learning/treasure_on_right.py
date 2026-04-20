import numpy as np
import pandas as pd
import time

np.random.seed(2)

NUM_STATES = 6
ACTIONS = ['left', 'right']
NUM_EPISODES = 13
EPSILON = 0.1
GAMMA = 0.9
ALPHA = 0.1

def build_q_table(num_states, actions):
    q_table = pd.DataFrame(
        np.zeros((num_states, len(actions))),
        columns=actions
    )
    return q_table

def choose_action(s, q_table):
    if np.random.uniform() < EPSILON or (q_table.iloc[s, :] == 0).all():
        return np.random.choice(ACTIONS)
    else:
        return q_table.iloc[s, :].idxmax()
    
def interact_with_env(s, a):
    if a == 'right':
        if s == NUM_STATES - 2:
            s_ = 'terminal'
            r = 1
        else:
            s_ = s + 1
            r = 0
    else:
        if s == 0:
            s_ = s
        else:
            s_ = s -1
        r = 0
    return r, s_

def update_env(s, episode, num_steps):
    env = ['-'] * (NUM_STATES - 1) + ['T']
    if s == 'terminal':
        print('\rEpisode %s: total_steps = %s' % (episode, num_steps), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env[s] = 'o'
        env = ''.join(env)
        print('\r{}'.format(env), end='')
        time.sleep(0.3)

def run():
    q_table = build_q_table(NUM_STATES, ACTIONS)
    for episode in range(NUM_EPISODES):
        s = 0
        num_steps = 0
        update_env(s, episode+1, num_steps)
        while s != 'terminal':
            a = choose_action(s, q_table)
            r, s_ = interact_with_env(s, a)
            q_eval = q_table.loc[s, a]
            if s_ == 'terminal':
                q_target = r
            else:
                q_target = r + GAMMA * q_table.iloc[s_, :].max()
            q_table.loc[s, a] += ALPHA * (q_target - q_eval)
            s = s_

            num_steps += 1
            update_env(s, episode+1, num_steps)
    return q_table

if __name__ == '__main__':
    q_table = run()
    print('\r\nQ-table:\n')
    print(q_table)
