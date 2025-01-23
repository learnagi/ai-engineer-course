---
title: "强化学习基础"
slug: "reinforcement-learning"
sequence: 3
description: "掌握强化学习的核心概念、算法原理和实践应用"
is_published: true
estimated_minutes: 120
language: "zh-CN"
---

![Reinforcement Learning](images/reinforcement-learning-header.png)
*强化学习是实现智能决策的关键技术*

# 强化学习基础

## 学习目标
完成本模块学习后，你将能够：
- 理解强化学习的基本概念和原理
- 掌握经典的强化学习算法
- 实现简单的强化学习系统
- 应用强化学习解决实际问题

## 先修知识
- Python编程基础
- 概率统计基础
- 机器学习基础概念
- 基本的算法知识

## 1. 强化学习概述

### 1.1 基本概念
```python
import numpy as np
import gym
import matplotlib.pyplot as plt

# 强化学习的核心要素
class RLEnvironment:
    def __init__(self):
        """环境初始化"""
        self.states = range(10)  # 状态空间
        self.actions = range(4)  # 动作空间
        self.current_state = 0
        
    def step(self, action):
        """执行动作，返回新状态和奖励"""
        next_state = min(self.current_state + action, len(self.states)-1)
        reward = 1 if next_state == len(self.states)-1 else 0
        done = (next_state == len(self.states)-1)
        self.current_state = next_state
        return next_state, reward, done
    
    def reset(self):
        """重置环境"""
        self.current_state = 0
        return self.current_state
```

### 1.2 马尔可夫决策过程
强化学习问题通常被建模为马尔可夫决策过程(MDP)，包含以下要素：
- 状态(State)：环境的当前状况
- 动作(Action)：智能体可以采取的行为
- 奖励(Reward)：环境对动作的反馈
- 状态转移：动作导致的状态改变
- 策略(Policy)：决定在每个状态下采取什么动作

```python
class MDP:
    def __init__(self, n_states, n_actions, gamma=0.99):
        """初始化MDP"""
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma  # 折扣因子
        
        # 初始化转移概率和奖励
        self.transitions = np.zeros((n_states, n_actions, n_states))
        self.rewards = np.zeros((n_states, n_actions, n_states))
        
    def set_transition(self, state, action, next_state, prob, reward):
        """设置转移概率和奖励"""
        self.transitions[state, action, next_state] = prob
        self.rewards[state, action, next_state] = reward
```

## 2. 经典强化学习算法

### 2.1 Q-Learning
```python
class QLearning:
    def __init__(self, n_states, n_actions, learning_rate=0.1, gamma=0.99):
        """初始化Q-Learning算法"""
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = gamma
        
    def choose_action(self, state, epsilon=0.1):
        """ε-贪婪策略选择动作"""
        if np.random.random() < epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        """Q-Learning更新规则"""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        # Q-Learning更新公式
        new_value = (1 - self.lr) * old_value + \
                   self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

def train_q_learning(env, agent, episodes=1000):
    """训练Q-Learning智能体"""
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.choose_action(state)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 更新Q值
            agent.learn(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
        
        rewards_history.append(total_reward)
        
    return rewards_history
```

### 2.2 SARSA
```python
class SARSA:
    def __init__(self, n_states, n_actions, learning_rate=0.1, gamma=0.99):
        """初始化SARSA算法"""
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = gamma
    
    def choose_action(self, state, epsilon=0.1):
        """ε-贪婪策略选择动作"""
        if np.random.random() < epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, next_action):
        """SARSA更新规则"""
        old_value = self.q_table[state, action]
        next_value = self.q_table[next_state, next_action]
        
        # SARSA更新公式
        new_value = (1 - self.lr) * old_value + \
                   self.lr * (reward + self.gamma * next_value)
        self.q_table[state, action] = new_value
```

### 2.3 深度Q网络(DQN)
```python
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        """初始化DQN网络"""
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        """初始化DQN智能体"""
        self.state_size = state_size
        self.action_size = action_size
        
        # 创建Q网络和目标网络
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters())
        self.memory = []
        
    def choose_action(self, state, epsilon=0.1):
        """选择动作"""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def learn(self, batch_size=32):
        """从经验回放中学习"""
        if len(self.memory) < batch_size:
            return
        
        # 采样batch
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for i in batch:
            s, a, r, ns, d = self.memory[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        # 转换为tensor
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失并更新
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

## 3. 策略梯度方法

### 3.1 REINFORCE算法
```python
class REINFORCE:
    def __init__(self, state_size, action_size):
        """初始化REINFORCE算法"""
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.network.parameters())
        self.memory = []
    
    def choose_action(self, state):
        """根据策略选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.network(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self):
        """更新策略网络"""
        R = 0
        policy_loss = []
        returns = []
        
        # 计算回报
        for r in reversed(self.memory):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # 更新策略
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        self.memory = []
        self.saved_log_probs = []
```

## 4. 实战案例：CartPole

### 4.1 环境设置
```python
import gym

def create_cartpole():
    """创建CartPole环境"""
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    return env, state_size, action_size

def run_episode(env, agent, render=False):
    """运行一个回合"""
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        if render:
            env.render()
        
        # 选择动作
        action = agent.choose_action(state)
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        
        # 存储经验
        agent.memory.append((state, action, reward, next_state, done))
        
        # 更新状态
        state = next_state
        total_reward += reward
        
        # 学习
        agent.learn()
    
    return total_reward

def train_agent(env, agent, episodes=1000):
    """训练智能体"""
    rewards = []
    for episode in range(episodes):
        reward = run_episode(env, agent)
        rewards.append(reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f'Episode {episode}, Average Reward: {avg_reward}')
    
    return rewards
```

### 4.2 可视化与评估
```python
def plot_training_results(rewards):
    """绘制训练结果"""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

def evaluate_agent(env, agent, episodes=100):
    """评估智能体性能"""
    rewards = []
    for _ in range(episodes):
        reward = run_episode(env, agent, render=True)
        rewards.append(reward)
    
    print(f'Average Reward over {episodes} episodes: {np.mean(rewards)}')
    return np.mean(rewards)
```

## 常见问题解答

Q: 如何选择合适的强化学习算法？
A: 根据问题特点选择：
- 如果状态空间小，可以使用Q-Learning或SARSA
- 如果状态空间大，考虑使用DQN
- 如果需要连续动作空间，考虑使用策略梯度方法

Q: 如何处理探索与利用的平衡？
A: 可以使用以下方法：
- ε-贪婪策略
- Boltzmann探索
- UCB（上置信界）
- 参数噪声

Q: 如何提高训练效率？
A: 可以采用以下技巧：
- 使用经验回放
- 优先经验回放
- 目标网络
- 合适的奖励设计

## 扩展阅读
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [Deep Reinforcement Learning](https://arxiv.org/abs/1810.06339)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
