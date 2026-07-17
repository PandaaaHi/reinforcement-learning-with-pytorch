import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_states, 20)
        self.fc2 = nn.Linear(20, n_actions)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, 0, 0.3)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.normal_(self.fc2.weight, 0, 0.3)
        nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
class DQNPrioritizedReplay:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.005,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=500,
        memory_size=10000,
        batch_size=32,
        e_greedy_increment=None,
        prioritized=True,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = self.epsilon_max if e_greedy_increment is None else 0
        self.e_greedy_increment = e_greedy_increment
        self.prioritized = prioritized

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.target_net = Net(self.n_features, self.n_actions).to(self.device)
        self.eval_net = Net(self.n_features, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        if self.prioritized:
            self.memory = Memory(self.memory_size)
        else:
            self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)

    def my_mse_loss(self, target, pred, weight=1.0):
        squared_diff = (pred - target) ** 2
        return (squared_diff * weight).mean()

    def choose_action(self, s):
        s = s[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_value = self.eval_net(torch.FloatTensor(s).to(self.device)).detach().cpu().numpy()
            action = np.argmax(action_value, axis=1)[0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        if self.prioritized:
            self.memory.store(transition)
        else:
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def learn(self):        
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('target_params_replaced, step: ', self.learn_step_counter)

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            batch_index = np.random.choice(self.memory_size, self.batch_size)
            batch_memory = self.memory[batch_index, :]
        
        s = torch.tensor(batch_memory[:, :self.n_features], dtype=torch.float32).to(self.device)
        a = torch.tensor(batch_memory[:, self.n_features], dtype=torch.long).to(self.device)
        r = torch.tensor(batch_memory[:, self.n_features + 1], dtype=torch.float32).to(self.device)
        s_ = torch.tensor(batch_memory[:, -self.n_features:], dtype=torch.float32).to(self.device)

        q_next = self.target_net(s_).detach()
        q_eval = self.eval_net(s)

        q_target = r + self.gamma * q_next.max(dim=1)[0]
        q_eval_wrt_a = q_eval.gather(1, a.unsqueeze(1)).squeeze(1)

        if self.prioritized:
            ISWeights_tensor = torch.tensor(ISWeights, dtype=torch.float32).to(self.device)
            loss = self.my_mse_loss(q_target, q_eval_wrt_a, ISWeights_tensor)
            abs_errors = torch.abs(q_target - q_eval_wrt_a).detach().cpu().numpy()
            self.memory.batch_update(tree_idx, abs_errors)
        else:
            loss = self.my_mse_loss(q_target, q_eval_wrt_a)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1