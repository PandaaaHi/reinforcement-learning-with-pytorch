## Reinforcement-Learning-with-PyTorch
Inspired by [MorvanZhou](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow), we implement a set of classic reinforcement learning algorithms (Q-Learning, SARSA, DQN, DDPG, A3C, PPO, etc) with PyTorch. To adapt to legacy runtime environments, it is recommended to create a virtual environment based on Python 3.6 using conda.

### Requirements
To run the code properly, need to install the following certain packages:
- gym=0.16
- matplotlib=3.3.4
- numpy=1.19.5
- pandas=1.1.5
- torch=1.10.2

### Getting Started
To train an AI model with a certain algorithm, e.g., DQN, run the following script from the ``contents/5_Deep_Q_Network`` directory:
```
python run_this.py
```