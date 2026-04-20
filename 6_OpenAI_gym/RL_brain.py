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
        self.fc1 = nn.Linear(n_states, 10)
        self.fc2 = nn.Linear(10, n_actions)
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
    
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=2000,
            batch_size=32,
            e_greedy_increment=0.001,
        ):
        self.n_actions = n_actions
        self.n_states = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.e_greedy_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learn_step_counter = 0
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.eval_net = Net(self.n_states, self.n_actions).to(self.device)
        self.target_net = Net(self.n_states, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
        
        self.memory = np.zeros((self.memory_size, self.n_states * 2 + 2))
        self.cost_hist = []
        
    def choose_action(self, state):
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            state_tensor = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                action_value = self.eval_net(state_tensor).cpu().numpy()
            action = np.argmax(action_value, axis=1)[0]
        else:
            action = np.random.randint(0, self.n_actions)
            
        return action
    
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
            
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        
    def learn(self):
        if not hasattr(self, 'memory_counter') or self.memory_counter < self.batch_size:
            return
        
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            
        # batch_memory = self.memory[sample_index, :]
        # s = torch.FloatTensor(batch_memory[:, :self.n_states]).to(self.device)
        # s_ = torch.FloatTensor(batch_memory[:, -self.n_states:]).to(self.device)
        
        # with torch.no_grad():
        #     q_next = self.target_net(s_).cpu().numpy()
        # q_eval = self.eval_net(s).detach().cpu().numpy()
        # q_target = q_eval.copy()
        
        # batch_index = np.arange(self.batch_size, dtype=np.int32)
        # eval_act_index = batch_memory[:, self.n_states].astype(int)
        # reward = batch_memory[:, self.n_states + 1]
        
        # q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        # q_target_tensor = torch.FloatTensor(q_target).to(self.device)
        # q_eval_tensor = self.eval_net(s)
        
        batch_memory = self.memory[sample_index, :]
        s = torch.tensor(batch_memory[:, :self.n_states], dtype=torch.float32).to(self.device)
        a = torch.tensor(batch_memory[:, self.n_states], dtype=torch.long).to(self.device)
        r = torch.tensor(batch_memory[:, self.n_states + 1], dtype=torch.float32).to(self.device)
        s_ = torch.tensor(batch_memory[:, -self.n_states:], dtype=torch.float32).to(self.device)
        
        q_eval = self.eval_net(s)
        q_next = self.target_net(s_).detach()
        q_eval_wrt_a = q_eval.gather(1, a.unsqueeze(1)).squeeze(1)
        q_target = r + self.gamma * q_next.max(dim=1)[0]
        
        loss = self.loss_fn(q_eval_wrt_a, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.cost = loss.item()
        self.cost_hist.append(self.cost)
        
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('target_params_replaced, step: ', self.learn_step_counter)
        
        self.learn_step_counter += 1
        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_hist)), self.cost_hist)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()