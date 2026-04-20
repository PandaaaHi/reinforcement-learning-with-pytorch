import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

np.random.seed(2)
torch.manual_seed(2)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2)

class Actor:
    def __init__(self, n_features, action_bound, lr=0.0001):
        class ActorNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(n_features, 30)
                self.mu = nn.Linear(30, 1)
                self.sigma = nn.Linear(30, 1)
                self.relu = nn.ReLU()
                self.tanh = nn.Tanh()
                self.softplus = nn.Softplus()
                self._init_weights()

            def _init_weights(self):
                nn.init.normal_(self.fc.weight, 0, 0.1)
                nn.init.constant_(self.fc.bias, 0.1)
                nn.init.normal_(self.mu.weight, 0, 0.1)
                nn.init.constant_(self.mu.bias, 0.1)
                nn.init.normal_(self.sigma.weight, 0, 0.1)
                nn.init.constant_(self.sigma.bias, 1.0)

            def forward(self, x):
                x = self.relu(self.fc(x))
                mu = self.tanh(self.mu(x))
                mu = torch.squeeze(mu*2)
                sigma = self.softplus(self.sigma(x))
                sigma = torch.squeeze(sigma+0.1)
                normal_dist = torch.distributions.Normal(mu, sigma)
                return normal_dist

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.action_bound = torch.tensor(np.array(action_bound), dtype=torch.float32).to(self.device)
        self.net = ActorNet().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def loss(self, log_prob, td, entropy):
        exp_v = torch.mean(log_prob * td + 0.01 * entropy)
        return -exp_v

    def learn(self, s, a, td):
        s = torch.as_tensor(s[np.newaxis, :], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        td = torch.as_tensor(td, dtype=torch.float32, device=self.device)

        normal_dist = self.net(s)
        log_prob = normal_dist.log_prob(a)
        entropy = normal_dist.entropy()

        loss = self.loss(log_prob, td, entropy)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return -loss.item()

    def choose_action(self, s):
        s = torch.as_tensor(s[np.newaxis, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            normal_dist = self.net(s)
        a = torch.clamp(normal_dist.sample(), self.action_bound[0], self.action_bound[1])
        return a.cpu().numpy()
    
class Critic:
    def __init__(self, n_features, lr=0.01):
        class CriticNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(n_features, 30)
                self.fc2 = nn.Linear(30, 1)
                self.relu = nn.ReLU()
                self._init_weights()
            
            def _init_weights(self):
                nn.init.normal_(self.fc1.weight, 0, 0.1)
                nn.init.constant_(self.fc1.bias, 0.1)
                nn.init.normal_(self.fc2.weight, 0, 0.1)
                nn.init.constant_(self.fc2.bias, 0.1)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                return self.fc2(x)
            
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.net = CriticNet().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def loss(self, td_error):
        return torch.mean(torch.square(td_error))

    def learn(self, s, r, s_):
        s = torch.as_tensor(s[np.newaxis, :], dtype=torch.float32, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s_ = torch.as_tensor(s_[np.newaxis, :], dtype=torch.float32, device=self.device)

        v = self.net(s)
        v_ = self.net(s_).detach()
        td_error = (r + GAMMA * v_ - v).squeeze()

        loss = self.loss(td_error)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_error.detach().cpu().numpy()
    
OUTPUT_GRAPH = False
MAX_EPISODE = 1000
MAX_EP_STEPS = 200
DISPLAY_REWARD_THRESHOLD = -100
RENDER = False
GAMMA = 0.9
LR_A = 0.001
LR_C = 0.01

env = gym.make('Pendulum-v0')
env.seed(1)
env = env.unwrapped

N_S = env.observation_space.shape[0]
A_BOUND = env.action_space.high

actor = Actor(n_features=N_S, lr=LR_A, action_bound=[-A_BOUND, A_BOUND])
critic = Critic(n_features=N_S, lr=LR_C)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    ep_rs = []
    while True:
        # if RENDER:
        env.render()
        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)
        r /= 10

        td_error = critic.learn(s, r, s_)
        actor.learn(s, a, td_error)

        s = s_
        t += 1
        ep_rs.append(r)
        if t > MAX_EP_STEPS:
            ep_rs_sum = sum(ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break