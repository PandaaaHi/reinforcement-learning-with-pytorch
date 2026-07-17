import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

class Net(nn.Module):
    def __init__(self, n_features, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc2 = nn.Linear(10, n_actions)
        self.tanh = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, 0, 0.3)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.normal_(self.fc2.weight, 0, 0.3)
        nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        return self.fc2(x)
    
class PolicyGradient:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.95,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(self.n_features, self.n_actions).to(self.device)

        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=self.lr)

    def my_loss(self, logits, labels, vt):
        neg_log_prob = self.loss(logits, labels)
        return torch.mean(neg_log_prob * vt)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        prob_weights = torch.softmax(self.net(torch.FloatTensor(s).to(self.device)), dim=1).detach().cpu().numpy()
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action
    
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        ep_obs = torch.tensor(np.vstack(self.ep_obs), dtype=torch.float32).to(self.device)
        ep_as = torch.tensor(self.ep_as, dtype=torch.long).to(self.device)
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        ep_rs_norm = torch.tensor(discounted_ep_rs_norm, dtype=torch.float32).to(self.device)

        loss = self.my_loss(self.net(ep_obs), ep_as, ep_rs_norm)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)

        return discounted_ep_rs