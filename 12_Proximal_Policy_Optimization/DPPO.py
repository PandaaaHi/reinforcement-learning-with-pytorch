import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch.multiprocessing as mp

EP_MAX = 1000
EP_LEN = 200
N_WORKER = 4                # parallel workers
GAMMA = 0.9                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0002               # learning rate for critic
MIN_BATCH_SIZE = 64         # minimum batch size for updating PPO
UPDATE_STEP = 10            # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective
GAME = 'Pendulum-v0'
S_DIM, A_DIM = 3, 1         # state and action dimension

class PPO:
    def __init__(self):
        class Actor(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(S_DIM, 200)
                self.mu = nn.Linear(200, A_DIM)
                self.sigma = nn.Linear(200, A_DIM)

                self.relu = nn.ReLU()
                self.tanh = nn.Tanh()
                self.softplus = nn.Softplus()

            def forward(self, x):
                l1 = self.relu(self.l1(x))
                mu = 2 * self.tanh(self.mu(l1))
                sigma = self.softplus(self.sigma(l1))
                normal_dist = torch.distributions.Normal(mu, sigma)
                return normal_dist

        class Critic(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(S_DIM, 100)
                self.v = nn.Linear(100, 1)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                v = self.v(self.relu(self.l1(x)))
                return v
            
        self.device = torch.device('cpu')

        self.pi = Actor().to(self.device)
        self.oldpi = Actor().to(self.device)
        self.critic = Critic().to(self.device)

        self.actor_optimizer = optim.Adam(params=self.pi.parameters(), lr=A_LR)
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=C_LR)

    def update(self):
        while GLOBAL_EP.value < EP_MAX:            
            while not UPDATE_EVENT.wait(timeout=1.0):
                if GLOBAL_EP.value >= EP_MAX:
                    return
            
            self.oldpi.load_state_dict(self.pi.state_dict())

            data = [QUEUE.get() for _ in range(QUEUE.qsize())]
            data = np.vstack(data)
            s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
            s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
            a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
            r = torch.as_tensor(r, dtype=torch.float32, device=self.device)

            adv = (r - self.critic(s)).detach()
            for _ in range(UPDATE_STEP):
                pi = self.pi(s)
                oldpi = self.oldpi(s)
                ratio = pi.log_prob(a).exp() / (oldpi.log_prob(a).exp() + 1e-5)
                surr = ratio * adv

                aloss = - torch.mean(torch.minimum(surr, torch.clamp(ratio, 1.-EPSILON, 1.+EPSILON) * adv))
                self.actor_optimizer.zero_grad()
                aloss.backward()
                self.actor_optimizer.step()

            for _ in range(UPDATE_STEP):
                adv = r - self.critic(s)
                closs = torch.mean(torch.square(adv))
                self.critic_optimizer.zero_grad()
                closs.backward()
                self.critic_optimizer.step()

            UPDATE_EVENT.clear()
            with GLOBAL_UPDATE_COUNTER.get_lock():
                GLOBAL_UPDATE_COUNTER.value = 0
            ROLLING_EVENT.set()

    def choose_action(self, s):
        s = torch.as_tensor(s[np.newaxis, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            normal_dist = self.pi(s)
        a = normal_dist.sample().cpu().numpy()[0]
        return np.clip(a, -2, 2)
    
    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v = self.critic(s)
        return v.cpu().numpy()[0, 0]
    
class DataWorker(mp.Process):
    def __init__(self, wid):
        super().__init__()
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO

    def run(self):
        while GLOBAL_EP.value < EP_MAX:
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []

            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():
                    while not ROLLING_EVENT.wait(timeout=1.0):
                        if GLOBAL_EP.value >= EP_MAX:
                            return
                    buffer_s, buffer_a, buffer_r = [], [], []
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)
                s = s_
                ep_r += r

                with GLOBAL_UPDATE_COUNTER.get_lock():
                    GLOBAL_UPDATE_COUNTER.value += 1
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER.value >= MIN_BATCH_SIZE:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))
                    if GLOBAL_UPDATE_COUNTER.value >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()
                        UPDATE_EVENT.set()

            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)

            with GLOBAL_EP.get_lock():
                GLOBAL_EP.value += 1

            print('{0:.1f}%'.format(GLOBAL_EP.value/EP_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r,)

class UpdateWorker(mp.Process):
    def __init__(self):
        super().__init__()
        self.ppo = GLOBAL_PPO

    def run(self):
        self.ppo.update()

if __name__ == '__main__':
    GLOBAL_PPO = PPO()
    GLOBAL_PPO.pi.share_memory()
    GLOBAL_PPO.oldpi.share_memory()
    GLOBAL_PPO.critic.share_memory()
    UPDATE_EVENT, ROLLING_EVENT = mp.Event(), mp.Event()
    UPDATE_EVENT.clear()
    ROLLING_EVENT.set()
    workers = [DataWorker(wid=i) for i in range(N_WORKER)]
    workers.append(UpdateWorker())

    GLOBAL_UPDATE_COUNTER = mp.Value('i', 0)
    GLOBAL_EP = mp.Value('i', 0)
    GLOBAL_RUNNING_R = mp.Manager().list()
    QUEUE = mp.Queue()

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode')
    plt.ylabel('Moving reward')
    # plt.ion()
    plt.show()
    env = gym.make('Pendulum-v0')
    while True:
        s = env.reset()
        for t in range(300):
            env.render()
            s = env.step(GLOBAL_PPO.choose_action(s))[0]