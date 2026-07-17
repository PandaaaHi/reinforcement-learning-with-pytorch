from maze_env import Maze
from RL_brain import QLearningTable

NUM_EPISODES = 100

def update():
    for episode in range(NUM_EPISODES):
        s = env.reset()
        print('\repisode {}/{}'.format(episode+1, NUM_EPISODES), end='')
        
        while True:
            env.render()
            a = RL.choose_action(str(s))
            s_, r, done = env.step(a)
            RL.learn(str(s), a, r, str(s_))
            s = s_
            if done:
                break
            
    print('\r                       ', end='')
    print('\rgame over')
    env.destroy()
        
if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
    print('\rQ-table:\n', RL.q_table)