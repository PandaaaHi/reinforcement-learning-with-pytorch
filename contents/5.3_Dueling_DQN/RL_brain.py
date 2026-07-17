import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

class Net(nn.Module):
    def __init__(self, n_states, n_actions, dueling=True):
        super().__init__()
        self.fc1 = nn.Linear(n_states, 20)
        self.fc2 = nn.Linear(20, n_actions)
        self.fc2_v = nn.Linear(20, 1)
        self.fc2_a = nn.Linear(20, n_actions)
        self.relu = nn.ReLU()
        self.dueling = dueling
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, 0, 0.3)
        nn.init.constant_(self.fc1.bias, 0.1)
        
        nn.init.normal_(self.fc2.weight, 0, 0.3)
        nn.init.constant_(self.fc2.bias, 0.1)
        
        nn.init.normal_(self.fc2_v.weight, 0, 0.3)
        nn.init.constant_(self.fc2_v.bias, 0.1)
        nn.init.normal_(self.fc2_a.weight, 0, 0.3)
        nn.init.constant_(self.fc2_a.bias, 0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        if self.dueling:
            v = self.fc2_v(x)
            a = self.fc2_a(x)
            x = v + (a - torch.mean(a, dim=1, keepdim=True))
        else:
            x = self.fc2(x)

        return x
    
class DuelingDQN:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.001,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=500,
        batch_size=32,
        e_greedy_increment=None,
        dueling=True,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_greedy_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else e_greedy
        self.dueling = dueling

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.dueling:
            self.eval_net = Net(self.n_features, self.n_actions, dueling=True).to(self.device)
            self.target_net = Net(self.n_features, self.n_actions, dueling=True).to(self.device)
        else:
            self.eval_net = Net(self.n_features, self.n_actions, dueling=False).to(self.device)
            self.target_net = Net(self.n_features, self.n_actions, dueling=False).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        self.memory_counter = 0

        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(
            self.eval_net.parameters(),
            lr=self.lr,
            alpha=0.9,
            eps=1e-10,
            momentum=0,
            centered=False
        )

        self.learn_step_counter = 0
        self.cost_his = []

    def choose_action(self, s):
        s = s[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            v_a = self.eval_net(torch.FloatTensor(s).to(self.device)).detach().cpu().numpy()
            a = np.argmax(v_a, axis=1)[0]
        else:
            a = np.random.randint(0, self.n_actions)
        return a
    
    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = np.hstack((s, [a, r], s_))
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('target_params_replaced, step: ', self.learn_step_counter)

        batch_index = np.random.choice(self.memory_size, self.batch_size)
        batch_memory = self.memory[batch_index, :]
        s = torch.tensor(batch_memory[:, :self.n_features], dtype=torch.float32).to(self.device)
        a = torch.tensor(batch_memory[:, self.n_features], dtype=torch.long).to(self.device)
        r = torch.tensor(batch_memory[:, self.n_features + 1], dtype=torch.float32).to(self.device)
        s_ = torch.tensor(batch_memory[:, -self.n_features:], dtype=torch.float32).to(self.device)

        q_eval = self.eval_net(s)
        q_eval_wrt_a = q_eval.gather(1, a.unsqueeze(1)).squeeze(1)
        q_next = self.target_net(s_).detach()
        q_target = r + self.gamma * q_next.max(dim=1)[0]

        loss = self.loss(q_target, q_eval_wrt_a)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.item())

        self.learn_step_counter += 1
        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.epsilon_max else self.epsilon_max