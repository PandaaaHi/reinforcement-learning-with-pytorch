import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gym

np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][0]        # choose the method for optimization

class PPO:
    def __init__(self):
        class Actor(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(S_DIM, 100)
                self.mu = nn.Linear(100, A_DIM)
                self.sigma = nn.Linear(100, A_DIM)

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
            
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.pi = Actor().to(self.device)
        self.oldpi = Actor().to(self.device)
        self.critic = Critic().to(self.device)

        self.actor_optimizer = optim.Adam(params=self.pi.parameters(), lr=A_LR)
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=C_LR)

    def update(self, s, a, r):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)

        self.oldpi.load_state_dict(self.pi.state_dict())
        
        adv = (r - self.critic(s)).detach()

        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                pi = self.pi(s)
                oldpi = self.oldpi(s)
                ratio = pi.log_prob(a).exp() / (oldpi.log_prob(a).exp() + 1e-5)
                surr = ratio * adv

                tflam = METHOD['lam']
                kl = torch.distributions.kl_divergence(oldpi, pi)
                kl_mean = torch.mean(kl).detach().numpy()
                
                aloss = - torch.mean(surr - tflam * kl)
                self.actor_optimizer.zero_grad()
                aloss.backward()
                self.actor_optimizer.step()

                if kl_mean > 4 * METHOD['kl_target']:
                    break

            if kl_mean < METHOD['kl_target'] / 1.5:
                METHOD['lam'] /= 2
            elif kl_mean > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)
        else:
            for _ in range(A_UPDATE_STEPS):
                pi = self.pi(s)
                oldpi = self.oldpi(s)
                ratio = pi.log_prob(a).exp() / (oldpi.log_prob(a).exp() + 1e-5)
                surr = ratio * adv

                aloss = - torch.mean(torch.minimum(surr, torch.clamp(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon']) * adv))
                self.actor_optimizer.zero_grad()
                aloss.backward()
                self.actor_optimizer.step()

        for _ in range(C_UPDATE_STEPS):
            adv = r - self.critic(s)
            closs = torch.mean(torch.square(adv))
            self.critic_optimizer.zero_grad()
            closs.backward()
            self.critic_optimizer.step()

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
    
env = gym.make('Pendulum-v0').unwrapped
ppo = PPO()
all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()