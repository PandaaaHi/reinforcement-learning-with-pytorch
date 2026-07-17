import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_states, 20)
        self.fc2 = nn.Linear(20, n_actions)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.3)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.3)
        nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
class DoubleDQN():
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.005,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=3000,
        batch_size=32,
        e_greedy_increment=None,
        double_q=True,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.double_q = double_q
        self.learn_step_counter = 0
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.eval_net = Net(self.n_features, self.n_actions).to(self.device)
        self.target_net = Net(self.n_features, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)

        self.memory = np.zeros((self.memory_size, self.n_features * 2 +2))

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]
        with torch.no_grad():
            action_value = self.eval_net(torch.FloatTensor(s).to(self.device)).cpu().numpy()
        action = np.argmax(action_value, axis=1)[0]

        if not hasattr(self, 'q'):
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(action_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:
            action = np.random.randint(0, self.n_actions)

        return action
    
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('target_params_replaced, step: ', self.learn_step_counter)

        if not hasattr(self, 'memory_counter') or self.memory_counter < self.batch_size:
            return
        
        if self.memory_counter > self.memory_size:
            batch_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            batch_index = np.random.choice(self.memory_counter, size=self.batch_size)

        s = torch.tensor(self.memory[batch_index, :self.n_features], dtype=torch.float32).to(self.device)
        a = torch.tensor(self.memory[batch_index, self.n_features], dtype=torch.long).to(self.device)
        r = torch.tensor(self.memory[batch_index, self.n_features+1], dtype=torch.float32).to(self.device)
        s_ = torch.tensor(self.memory[batch_index, -self.n_features:], dtype=torch.float32).to(self.device)

        q_next = self.target_net(s_).detach()
        q_eval = self.eval_net(s)
        q_eval_wrt_a = q_eval.gather(1, a.unsqueeze(1)).squeeze(1)

        if self.double_q:
            q_eval_next = self.eval_net(s_).detach()
            q_target = r + self.gamma * q_next.gather(1, q_eval_next.argmax(dim=1, keepdim=True)).squeeze(1)
        else:
            q_target = r + self.gamma * q_next.max(dim=1)[0]

        loss = self.loss(q_eval_wrt_a, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1