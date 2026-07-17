import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from shared_optim import SharedRMSprop

GAME = 'CartPole-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = mp.cpu_count()
MAX_GLOBAL_EP = 1000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
DEVICE = 'cpu'

env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n

class ACNet:
    def __init__(self, scope, globalAC=None, actor_optimizer=None, critic_optimizer=None):
        class Actor(nn.Module):
            def __init__(self):
                super().__init__()
                self.l_a = nn.Linear(N_S, 200)
                self.a_prob = nn.Linear(200, N_A)
                self.relu6 = nn.ReLU6()
                self.softmax = nn.Softmax(dim=1)
                self._init_weights()

            def _init_weights(self):
                nn.init.normal_(self.l_a.weight, 0, 0.1)
                nn.init.normal_(self.a_prob.weight, 0, 0.1)

            def forward(self, x):
                a_prob = self.softmax(self.a_prob(self.relu6(self.l_a(x))))
                return a_prob
            
        class Critic(nn.Module):
            def __init__(self):
                super().__init__()
                self.l_c = nn.Linear(N_S, 100)
                self.v = nn.Linear(100, 1)
                self.relu6 = nn.ReLU6()
                self._init_weights()

            def _init_weights(self):
                nn.init.normal_(self.l_c.weight, 0, 0.1)
                nn.init.normal_(self.v.weight, 0, 0.1)

            def forward(self, x):
                v = self.v(self.relu6(self.l_c(x)))
                return v
           
        self.device = torch.device(DEVICE)
        
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)

        if scope != GLOBAL_NET_SCOPE:
            self.globalAC = globalAC
            self.actor_optimizer = actor_optimizer
            self.critic_optimizer = critic_optimizer

    def actor_loss(self, exp_v):
        return torch.mean(-exp_v)

    def critic_loss(self, td):
        return torch.mean(torch.square(td))
    
    def update_global(self, s, a, v_target):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.long, device=self.device)
        v_target = torch.as_tensor(v_target, dtype=torch.float32, device=self.device)

        a_prob = self.actor(s)
        log_prob = torch.sum(torch.log(a_prob + 1e-5) * torch.nn.functional.one_hot(a, N_A), dim=1, keepdim=True)
        v = self.critic(s)
        td = v_target - v
        exp_v = log_prob * td.detach()
        entropy = - torch.sum(a_prob * torch.log(a_prob + 1e-5), dim=1, keepdim=True)
        exp_v = ENTROPY_BETA * entropy + exp_v

        a_loss = self.actor_loss(exp_v)
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        for local_param, global_param in zip(self.actor.parameters(), self.globalAC.actor.parameters()):
            global_param.grad = local_param.grad
        self.actor_optimizer.step()

        c_loss = self.critic_loss(td)
        self.critic_optimizer.zero_grad()
        c_loss.backward()
        for local_param, global_param in zip(self.critic.parameters(), self.globalAC.critic.parameters()):
            global_param.grad = local_param.grad
        self.critic_optimizer.step()

    def pull_global(self):
        self.actor.load_state_dict(self.globalAC.actor.state_dict())
        self.critic.load_state_dict(self.globalAC.critic.state_dict())

    def choose_action(self, s):
        s = torch.as_tensor(s[np.newaxis, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prob_weights = self.actor(s).cpu().numpy()
        a = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return a
    
class Worker(mp.Process):
    def __init__(self, name, globalAC, actor_optimizer, critic_optimizer, global_ep, global_ep_r, res_queue):
        super().__init__()
        self.name = name
        self.AC = ACNet(name, globalAC, actor_optimizer, critic_optimizer)
        self.global_ep = global_ep
        self.global_ep_r = global_ep_r
        self.res_queue = res_queue
        
        self.env = gym.make(GAME).unwrapped
        self.device = torch.device(DEVICE)

    def run(self):
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        while self.global_ep.value < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                s_ = np.array(s_).flatten()
                
                if done:
                    r = -5

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.AC.critic(torch.as_tensor(s_, dtype=torch.float32, device=self.device).view(1, -1))
                        v_s_ = v_s_.item()
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    self.AC.update_global(buffer_s, buffer_a, buffer_v_target)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    with self.global_ep.get_lock():
                        self.global_ep.value += 1
                    with self.global_ep_r.get_lock():
                        if self.global_ep_r.value == 0.:
                            self.global_ep_r.value = ep_r
                        else:
                            self.global_ep_r.value = self.global_ep_r.value * 0.99 + ep_r * 0.01
                    self.res_queue.put(self.global_ep_r.value)

                    print(
                        self.name,
                        "Ep:", self.global_ep.value,
                        "| Ep_r: %.0f" % self.global_ep_r.value,
                    )
                    
                    break
        self.res_queue.put(None)

if __name__ == '__main__':
    globalAC = ACNet(GLOBAL_NET_SCOPE)
    globalAC.actor.share_memory()
    globalAC.critic.share_memory()

    actor_optimizer = SharedRMSprop(params=globalAC.actor.parameters(), lr=LR_A, alpha=0.9, eps=1e-7)
    critic_optimizer = SharedRMSprop(params=globalAC.critic.parameters(), lr=LR_C, alpha=0.9, eps=1e-7)
    
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    workers = []
    for i in range(N_WORKERS):
        i_name = 'W_%i' % i
        workers.append(Worker(i_name, globalAC, actor_optimizer, critic_optimizer, global_ep, global_ep_r, res_queue))

    worker_threads = []
    for worker in workers:
        worker.start()

    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    for worker in workers:
        worker.join()

    plt.plot(res)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()