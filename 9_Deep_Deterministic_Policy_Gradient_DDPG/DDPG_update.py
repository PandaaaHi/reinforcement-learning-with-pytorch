import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import time

np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'

class DDPG:
    def __init__(self, a_dim, s_dim, a_bound):
        class Actor(nn.Module):
            def __init__(self, a_bound):
                super().__init__()
                self.fc1 = nn.Linear(s_dim, 30)
                self.fc2 = nn.Linear(30, a_dim)
                self.relu = nn.ReLU()
                self.tanh = nn.Tanh()
                self.register_buffer('a_bound', torch.tensor(a_bound, dtype=torch.float32))
                self._init_weight()

            def _init_weight(self):
                nn.init.normal_(self.fc1.weight, 0, 0.3)
                nn.init.constant_(self.fc1.bias, 0.1)
                nn.init.normal_(self.fc2.weight, 0, 0.3)
                nn.init.constant_(self.fc2.bias, 0.1)

            def forward(self, s):
                s = self.relu(self.fc1(s))
                a = self.tanh(self.fc2(s))
                return a * self.a_bound
        
        class Critic(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(s_dim, 30)
                self.fc2 = nn.Linear(a_dim, 30)
                self.fc3 = nn.Linear(30, 1)
                self.bias = nn.Parameter(torch.zeros(30))
                self.relu = nn.ReLU()
                self._init_weight()

            def _init_weight(self):
                nn.init.normal_(self.fc1.weight, 0, 0.1)
                nn.init.normal_(self.fc2.weight, 0, 0.1)
                nn.init.normal_(self.fc3.weight, 0, 0.1)
                nn.init.constant_(self.bias, 0.1)

            def forward(self, s, a):
                h = self.relu(self.fc1(s) + self.fc2(a) + self.bias)
                q = self.fc3(h)
                return q

        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.actor_eval = Actor(self.a_bound).to(self.device)
        self.actor_target = Actor(self.a_bound).to(self.device)
        
        self.critic_eval = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)

        self.actor_optimizer = optim.Adam(params=self.actor_eval.parameters(), lr=LR_A)

        self.critic_loss = nn.MSELoss()
        self.critic_optimizer = optim.Adam(params=self.critic_eval.parameters(), lr=LR_C)
    
    def actor_loss(self, q):
        return - torch.mean(q)

    def choose_action(self, s):
        s = torch.as_tensor(s[np.newaxis, :], dtype=torch.float32, device=self.device)
        a = self.actor_eval(s).detach().cpu().numpy()
        return a.flatten()
    
    def soft_replace(self):
        at_params = self.actor_target.parameters()
        ct_params = self.critic_target.parameters()
        ae_params = self.actor_eval.parameters()
        ce_params = self.critic_eval.parameters()

        for t, e in zip(at_params, ae_params):
            t.data.copy_((1 - TAU) * t.data + TAU * e.data)

        for t, e in zip(ct_params, ce_params):
            t.data.copy_((1 - TAU) * t.data + TAU * e.data)

    def learn(self):
        self.soft_replace()

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.as_tensor(bt[:, :self.s_dim], dtype=torch.float32, device=self.device)
        ba = torch.as_tensor(bt[:, self.s_dim: self.s_dim + self.a_dim], dtype=torch.float32, device=self.device)
        br = torch.as_tensor(bt[:, -self.s_dim - 1: -self.s_dim], dtype=torch.float32, device=self.device)
        bs_ = torch.as_tensor(bt[:, -self.s_dim:], dtype=torch.float32, device=self.device)

        a = self.actor_eval(bs)
        q = self.critic_eval(bs, a)
        actor_loss = self.actor_loss(q)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            a_ = self.actor_target(bs_)
            q_ = self.critic_target(bs_, a_)
        q_target = br + GAMMA * q_
        q_eval = self.critic_eval(bs, ba)
        critic_loss = self.critic_loss(q_target, q_eval)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)