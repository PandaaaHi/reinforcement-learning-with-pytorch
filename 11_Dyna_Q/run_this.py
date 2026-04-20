from maze_env import Maze
from RL_brain import QLearningTable, EnvModel

NUM_EPISODES = 40
NUM_PLANNINGS = 10

def update():
    for _ in range(NUM_EPISODES):
        s = env.reset()
        s = str(s)
        while True:
            env.render()
            a = RL.choose_action(s)
            s_, r, done = env.step(a)
            s_ = str(s_)
            RL.learn(s, a, r, s_, done)

            env_model.store_transition(s, a, r, s_)
            for _ in range(NUM_PLANNINGS):
                ms, ma = env_model.sample_s_a()
                mr, ms_ = env_model.get_r_s_(ms, ma)
                RL.learn(ms, ma, mr, ms_, done)

            s = s_

            if done:
                break
    print('game over')
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env_model = EnvModel(actions=list(range(env.n_actions)))

    env.after(0, update)
    env.mainloop()