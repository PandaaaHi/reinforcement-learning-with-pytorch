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
        gamma=0.95,
        epsilon=1.,
        replace_target_iter=300,
        memory_size=10000,
        batch_size=128,
    ):
        class Predictor(nn.Module):
            def __init__(self, s_encode_size):
                super().__init__()
                self.s_encode_size = s_encode_size
                self.net = nn.Linear(n_s, 128)
                self.out = nn.Linear(128, self.s_encode_size)
                self.relu = nn.ReLU()

            def forward(self, s_, rand_encode_s_):
                net = self.relu(self.net(s_))
                out = self.out(net)
                ri = torch.sum(torch.square(rand_encode_s_ - out), dim=1)
                return ri
            
        class RandomNet(nn.Module):
            def __init__(self, s_encode_size):
                super().__init__()
                self.s_encode_size = s_encode_size
                self.net = nn.Linear(n_s, self.s_encode_size)
            
            def forward(self, s_):
                return self.net(s_)
            
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
        self.s_encode_size = 1000

        self.learn_step_counter = 0
        self.memory_counter = 0

        self.memory = np.zeros((self.memory_size, n_s * 2 + 2))

        self.device = torch.device('cpu')

        self.predictor = Predictor(self.s_encode_size).to(self.device)
        self.rand_net = RandomNet(self.s_encode_size).to(self.device)
        self.dqn_eval = DQN().to(self.device)
        self.dqn_target = DQN().to(self.device)

        self.predictor_optim = optim.RMSprop(params=self.predictor.parameters(), lr=self.lr)
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

        rand_encode_s_ = self.rand_net(s_).detach()
        ri = self.predictor(s_, rand_encode_s_)
        r = r + ri.detach()

        q_eval = self.dqn_eval(s)
        q_next = self.dqn_target(s_).detach()
        q_wrt_a = q_eval.gather(1, a.unsqueeze(1)).squeeze(1)
        q_target = r + self.gamma * q_next.max(dim=1)[0]

        dqn_loss = torch.mean(torch.square(q_target - q_wrt_a))
        self.dqn_optim.zero_grad()
        dqn_loss.backward()
        self.dqn_optim.step()

        if self.learn_step_counter % 100 == 0:
            predictor_loss = torch.mean(ri)
            self.predictor_optim.zero_grad()
            predictor_loss.backward()
            self.predictor_optim.step()

        self.learn_step_counter += 1

env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(1)

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