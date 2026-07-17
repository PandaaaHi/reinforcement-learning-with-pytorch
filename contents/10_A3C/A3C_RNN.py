import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from shared_optim import SharedRMSprop

GAME = 'Pendulum-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = mp.cpu_count()
MAX_EP_STEP = 200
MAX_GLOBAL_EP = 1500
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
DEVICE = 'cpu'

env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]

class ACNet:
    def __init__(self, scope, globalAC=None, actor_optimizer=None, critic_optimizer=None):
        class Actor(nn.Module):
            def __init__(self):
                super().__init__()
                self.cell_size = 64
                self.l_a = nn.Linear(self.cell_size, 80)
                self.mu = nn.Linear(80, N_A)
                self.sigma = nn.Linear(80, N_A)

                self.relu6 = nn.ReLU6()
                self.tanh = nn.Tanh()
                self.softplus = nn.Softplus()

                self._init_weights()

            def _init_weights(self):
                nn.init.normal_(self.l_a.weight, 0, 0.1)
                nn.init.normal_(self.mu.weight, 0, 0.1)
                nn.init.normal_(self.sigma.weight, 0, 0.1)

            def forward(self, x):
                x = self.relu6(self.l_a(x))
                mu = self.tanh(self.mu(x))
                sigma = self.softplus(self.sigma(x))
                mu, sigma = mu * torch.tensor(A_BOUND[1], dtype=torch.float32, device=x.device), sigma + 1e-4
                normal_dist = torch.distributions.Normal(mu, sigma)
                return normal_dist
            
        class Critic(nn.Module):
            def __init__(self):
                super().__init__()
                self.cell_size = 64
                self.rnn = nn.RNN(N_S, self.cell_size, batch_first=True)
                self.l_c = nn.Linear(self.cell_size, 50)
                self.v = nn.Linear(50, 1)
                self.relu6 = nn.ReLU6()
                self._init_weights()

            def _init_weights(self):
                nn.init.normal_(self.l_c.weight, 0, 0.1)
                nn.init.normal_(self.v.weight, 0, 0.1)

            def forward(self, x, init_state):
                x = torch.unsqueeze(x, dim=0)
                outputs, final_state = self.rnn(x, init_state)
                cell_out = torch.reshape(outputs, (-1, self.cell_size))
                v = self.v(self.relu6(self.l_c(cell_out)))
                return v, cell_out, final_state
            
        self.device = torch.device(DEVICE)
        
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)

        if scope != GLOBAL_NET_SCOPE:
            self.globalAC = globalAC
            self.global_actor_optimizer = actor_optimizer
            self.global_critic_optimizer = critic_optimizer
            # self.local_actor_optimizer = torch.optim.RMSprop(params=self.actor.parameters())
            # self.local_critic_optimizer = torch.optim.RMSprop(params=self.critic.parameters())

        self.action_bound = torch.tensor(np.array(A_BOUND), dtype=torch.float32, device=self.device)

    def actor_loss(self, exp_v):
        return torch.mean(-exp_v)

    def critic_loss(self, td):
        return torch.mean(torch.square(td))
    
    def update_global(self, s, a, v_target, init_state):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        v_target = torch.as_tensor(v_target, dtype=torch.float32, device=self.device)
        init_state = torch.as_tensor(init_state, dtype=torch.float32, device=self.device)

        v, cell_out, _ = self.critic(s, init_state)
        normal_dist = self.actor(cell_out.detach())
        log_prob = normal_dist.log_prob(a)
        td = v_target - v
        exp_v = log_prob * td.detach()

        entropy = normal_dist.entropy()
        exp_v = ENTROPY_BETA * entropy + exp_v

        a_loss = self.actor_loss(exp_v)
        self.global_actor_optimizer.zero_grad()
        a_loss.backward()
        for local_param, global_param in zip(self.actor.parameters(), self.globalAC.actor.parameters()):
            global_param.grad = local_param.grad
        self.global_actor_optimizer.step()

        c_loss = self.critic_loss(td)
        self.global_critic_optimizer.zero_grad()
        c_loss.backward()
        for local_param, global_param in zip(self.critic.parameters(), self.globalAC.critic.parameters()):
            global_param.grad = local_param.grad
        self.global_critic_optimizer.step()

        # a_loss = self.actor_loss(exp_v)
        # self.global_actor_optimizer.zero_grad()
        # self.local_actor_optimizer.zero_grad()
        # a_loss.backward()
        # for local_param, global_param in zip(self.actor.parameters(), self.globalAC.actor.parameters()):
        #     global_param.grad = local_param.grad.clone() # if clone() is used here, then the local net need to be zero_grad()
        # self.global_actor_optimizer.step()

        # c_loss = self.critic_loss(td)
        # self.global_critic_optimizer.zero_grad()
        # self.local_critic_optimizer.zero_grad()
        # c_loss.backward()
        # for local_param, global_param in zip(self.critic.parameters(), self.globalAC.critic.parameters()):
        #     global_param.grad = local_param.grad.clone()
        # self.global_critic_optimizer.step()

    def pull_global(self):
        self.actor.load_state_dict(self.globalAC.actor.state_dict())
        self.critic.load_state_dict(self.globalAC.critic.state_dict())

    def choose_action(self, s, cell_state):
        s = torch.as_tensor(s[np.newaxis, :], dtype=torch.float32, device=self.device).view(1, -1)
        cell_state = torch.as_tensor(cell_state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, cell_out, final_state = self.critic(s, cell_state)
            normal_dist = self.actor(cell_out)
        a = torch.clamp(normal_dist.sample(), self.action_bound[0], self.action_bound[1])
        return a.cpu().numpy().flatten(), final_state.cpu().numpy()
    
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
            rnn_state = np.zeros((1, 1, self.AC.critic.cell_size))
            keep_state = rnn_state.copy()
            for ep_t in range(MAX_EP_STEP):
                # if self.name == 'W_0':
                #     self.env.render()
                a, rnn_state_ = self.AC.choose_action(s, rnn_state)
                s_, r, done, info = self.env.step(a)
                s_ = np.array(s_).flatten()
                done = True if ep_t == MAX_EP_STEP - 1 else False

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r+8)/8)    # normalize

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_, _, _ = self.AC.critic(torch.as_tensor(s_, dtype=torch.float32, device=self.device).view(1, -1), 
                                                    torch.as_tensor(rnn_state_, dtype=torch.float32, device=self.device))
                        v_s_ = v_s_.item()
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    self.AC.update_global(buffer_s, buffer_a, buffer_v_target, keep_state)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()
                    keep_state = rnn_state_.copy()

                s = s_
                rnn_state = rnn_state_
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