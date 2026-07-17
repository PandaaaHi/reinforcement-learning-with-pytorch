import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

class CuriosityNet:
    def __init__(
        self,
        n_a,
        n_s,
        lr=0.01,
        gamma=0.98,
        epsilon=0.95,
        replace_target_iter=300,
        memory_size=10000,
        batch_size=128,
    ):
        class DynamicNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.dyn_l = nn.Linear(n_s + 1, 32)
                self.dyn_s_ = nn.Linear(32, n_s)
                self.relu = nn.ReLU()

            def forward(self, s, a, s_):
                float_a = (a.to(dtype=torch.float32)).unsqueeze(dim=1)
                sa = torch.concat((s, float_a), dim=1)
                encoded_s_ = s_
                dyn_l = self.relu(self.dyn_l(sa))
                dyn_s_ = self.dyn_s_(dyn_l)
                squared_diff = torch.sum(torch.square(encoded_s_ - dyn_s_), dim=1)
                return squared_diff
            
        class DQN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(n_s, 128)
                self.fc2 = nn.Linear(128, n_a)
                self.relu = nn.ReLU()

            def forward(self, s):
                return self.fc2(self.relu(self.fc1(s)))

        self.n_a = n_a
        self.n_s = n_s
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.memory_counter = 0

        self.memory = np.zeros((self.memory_size, n_s * 2 + 2))

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.dyn = DynamicNet().to(self.device)
        self.dqn_eval = DQN().to(self.device)
        self.dqn_target = DQN().to(self.device)

        self.dyn_optim = optim.RMSprop(params=self.dyn.parameters(), lr=self.lr)
        self.dqn_optim = optim.RMSprop(params=self.dqn_eval.parameters(), lr=self.lr)

    def store_transition(self, s, a, r, s_):
        transition  = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, s):
        s = torch.tensor(s[np.newaxis, :], dtype=torch.float32, device=self.device)
        if np.random.uniform() < self.epsilon:
            with torch.no_grad():
                av = self.dqn_eval(s).cpu().numpy()
            a = np.argmax(av)
        else:
            a = np.random.randint(0, self.n_a)
        return a
    
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.dqn_target.load_state_dict(self.dqn_eval.state_dict())

        top = self.memory_size if self.memory_counter > self.memory_size else self.memory_counter
        sample_index = np.random.choice(top, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        s = torch.tensor(batch_memory[:, :self.n_s], dtype=torch.float32, device=self.device)
        a = torch.tensor(batch_memory[:, self.n_s], dtype=torch.long, device=self.device)
        r = torch.tensor(batch_memory[:, self.n_s+1], dtype=torch.float32, device=self.device)
        s_ = torch.tensor(batch_memory[:, -self.n_s:], dtype=torch.float32, device=self.device)

        curiosity = self.dyn(s, a, s_).detach()
        r = r + curiosity
        
        q_eval = self.dqn_eval(s)
        q_next = self.dqn_target(s_).detach()
        q_wrt_a = q_eval.gather(1, a.unsqueeze(1)).squeeze(1)
        q_target = r + self.gamma * q_next.max(dim=1)[0]

        dqn_loss = torch.mean(torch.square(q_target - q_wrt_a))
        self.dqn_optim.zero_grad()
        dqn_loss.backward()
        self.dqn_optim.step()

        if self.learn_step_counter % 1000 == 0:
            squared_diff = self.dyn(s, a, s_)
            dyn_loss = torch.mean(squared_diff)
            self.dyn_optim.zero_grad()
            dyn_loss.backward()
            self.dyn_optim.step()

        self.learn_step_counter += 1

env = gym.make('MountainCar-v0')
env = env.unwrapped

dqn = CuriosityNet(n_a=3, n_s=2, lr=0.01)
ep_steps = []
for epi in range(200):
    s = env.reset()
    steps = 0
    while True:
        env.render()
        a = dqn.choose_action(s)
        s_, r, done, info = env.step(a)
        dqn.store_transition(s, a, r, s_)
        dqn.learn()
        if done:
            print('Epi: ', epi, "| steps: ", steps)
            ep_steps.append(steps)
            break
        s = s_
        steps += 1

plt.plot(ep_steps)
plt.ylabel("steps")
plt.xlabel("episode")
plt.show()