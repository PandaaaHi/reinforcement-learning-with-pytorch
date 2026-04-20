import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

np.random.seed(2)
torch.manual_seed(2)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2)

OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

class Actor:
    def __init__(self, n_features, n_actions, lr=0.001):
        class ActorNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(n_features, 20)
                self.fc2 = nn.Linear(20, n_actions)
                self.relu = nn.ReLU()
                self.softmax = nn.Softmax()
                self._init_weights()

            def _init_weights(self):
                nn.init.normal_(self.fc1.weight, 0, 0.1)
                nn.init.constant_(self.fc1.bias, 0.1)
                nn.init.normal_(self.fc2.weight, 0, 0.1)
                nn.init.constant_(self.fc2.bias, 0.1)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                return self.softmax(self.fc2(x))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = ActorNet().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def loss(self, log_prob, td):
        exp_v = torch.mean(log_prob * td)
        return -exp_v

    def learn(self, s, a, td):
        s = torch.tensor(s[np.newaxis, :], dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        td = torch.tensor(td, dtype=torch.float32).to(self.device)

        log_prob = torch.log(self.net(s)[0, a.item()])

        loss = self.loss(log_prob, td)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return -loss.item()
    
    def choose_action(self, s):
        s = torch.tensor(s[np.newaxis, :], dtype=torch.float32).to(self.device)
        probs = self.net(s).detach().cpu().numpy()
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
    
class Critic:
    def __init__(self, n_features, lr=0.01):
        class CriticNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(n_features, 20)
                self.fc2 = nn.Linear(20, 1)
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
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = CriticNet().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def loss(self, td_error):
        return torch.mean(torch.square(td_error))

    def learn(self, s, r, s_):
        s = torch.tensor(s[np.newaxis, :], dtype=torch.float32).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        s_ = torch.tensor(s_[np.newaxis, :], dtype=torch.float32).to(self.device)

        v = self.net(s)
        v_ = self.net(s_).detach()
        td_error = r + GAMMA * v_ - v

        loss = self.loss(td_error)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_error.detach().cpu().numpy()
    
actor = Actor(n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(n_features=N_F, lr=LR_C)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)
        actor.learn(s, a, td_error)

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break