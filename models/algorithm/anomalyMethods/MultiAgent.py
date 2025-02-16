import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# 定义多项式函数
def f(x):
    a = np.array([1, 2, 3])
    b = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    return np.sum(a * x ** 2) + np.sum(b * np.outer(x, x))


# 深度Q网络
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 多智能体环境
class MultiAgentEnv:
    def __init__(self, num_agents, num_dims, action_space_size):
        self.num_agents = num_agents
        self.num_dims = num_dims
        self.action_space_size = action_space_size
        self.state = np.random.randint(0, action_space_size, num_dims)

    def reset(self):
        self.state = np.random.randint(0, self.action_space_size, self.num_dims)
        return self.state

    def step(self, actions):
        for i, action in enumerate(actions):
            self.state[i] = action
        reward = -f(self.state)  # 使用负的多项式函数值作为奖励
        return self.state, reward


def trian_Multi_DQN():
    # 超参数
    num_agents = 3
    num_dims = 3
    action_space_size = 10
    num_episodes = 500
    max_steps_per_episode = 50  # 新增的最大步数限制
    learning_rate = 0.01
    gamma = 0.99
    epsilon = 0.1

    # 初始化环境和智能体
    env = MultiAgentEnv(num_agents, num_dims, action_space_size)
    agents = [DQN(num_dims, action_space_size) for _ in range(num_agents)]
    optimizers = [optim.Adam(agent.parameters(), lr=learning_rate) for agent in agents]

    # 训练过程
    for episode in range(num_episodes):
        state = env.reset()
        for step in range(max_steps_per_episode):
            actions = []
            for i, agent in enumerate(agents):
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, action_space_size - 1)
                else:
                    with torch.no_grad():
                        q_values = agent(torch.FloatTensor(state))
                        action = q_values.argmax().item()
                actions.append(action)

            next_state, reward = env.step(actions)

            # 训练每个智能体的Q网络
            for i, agent in enumerate(agents):
                optimizer = optimizers[i]
                q_values = agent(torch.FloatTensor(state))
                q_value = q_values[actions[i]]

                with torch.no_grad():
                    next_q_values = agent(torch.FloatTensor(next_state))
                    max_next_q_value = next_q_values.max()

                expected_q_value = reward + gamma * max_next_q_value
                loss = (q_value - expected_q_value) ** 2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        print(f"Episode {episode + 1}/{num_episodes}, Reward: {reward}")
    # 测试最终策略
    state = env.reset()
    actions = [agent(torch.FloatTensor(state)).argmax().item() for agent in agents]
    final_state, final_reward = env.step(actions)
    print(f"最优解: {final_state}, 最小函数值: {-final_reward}")


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class VDN:
    def __init__(self, n_agents, state_dim, action_dim):
        self.n_agents = n_agents
        self.q_nets = [QNetwork(state_dim, action_dim) for _ in range(n_agents)]
        self.optimizers = [optim.Adam(q_net.parameters(), lr=0.01) for q_net in self.q_nets]
        self.action_values = np.arange(-10, 11)  # 动作空间

    def select_actions(self, states):
        actions = []
        for i in range(self.n_agents):
            q_values = self.q_nets[i](torch.FloatTensor(states[i:i + 1]))
            # print(f"Q-values for agent {i}: {q_values}")  # Debug信息
            action_index = torch.argmax(q_values).item()
            actions.append(self.action_values[action_index])  # 从动作空间中选择动作
        return actions

    def compute_total_q(self, states, actions):
        total_q = 0
        for i in range(self.n_agents):
            q_values = self.q_nets[i](torch.FloatTensor(states[i:i + 1]))
            action_index = np.where(self.action_values == actions[i])[0][0]
            if action_index >= q_values.shape[1]:
                raise IndexError(f"action_index {action_index} out of bounds for q_values with shape {q_values.shape}")
            total_q += q_values[0, action_index]
        return total_q

    def update(self, states, actions, rewards, next_states):
        total_loss = 0
        total_q = self.compute_total_q(states, actions)
        next_actions = self.select_actions(next_states)
        next_total_q = self.compute_total_q(next_states, next_actions).detach()
        target_q = rewards + next_total_q

        loss = (total_q - target_q) ** 2
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in self.optimizers:
            optimizer.step()

        return loss.item()

    def find_optimal_actions(self, env, episodes=1000):
        for episode in range(episodes):
            states = env.reset()
            for t in range(100):
                actions = self.select_actions(states)
                next_states, _ = env.step(actions)
                states = next_states

        return actions  # 返回训练后的最优动作组合


class PolynomialEnv:
    def __init__(self, n_agents, degree=2):
        self.n_agents = n_agents
        self.degree = degree
        self.x = np.random.randint(-10, 11, size=(n_agents,))
        self.action_space = np.arange(-10, 11)
        self.state_dim = n_agents - 1
        self.reset()

    def reset(self):
        self.x = np.random.randint(-10, 11, size=(self.n_agents,))
        return self.get_states()

    def get_states(self):
        return [np.delete(self.x, i) for i in range(self.n_agents)]

    def step(self, actions):
        self.x = np.array(actions)
        reward = -self.polynomial_function(self.x)
        return self.get_states(), reward

    def polynomial_function(self, x):
        return np.sum([xi ** self.degree for xi in x])


def train_vdn(env, vdn, episodes=200):
    for episode in range(episodes):
        states = env.reset()
        total_reward = 0
        for t in range(100):
            actions = vdn.select_actions(states)
            next_states, reward = env.step(actions)
            vdn.update(states, actions, reward, next_states)
            total_reward += reward
            states = next_states
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    optimal_actions = vdn.find_optimal_actions(env)
    print(f"Optimal Actions: {optimal_actions}")


if __name__ == "__main__":
    n_agents = 3  # 假设我们有三个变量 x1, x2, x3
    env = PolynomialEnv(n_agents)
    vdn = VDN(n_agents, state_dim=n_agents - 1, action_dim=21)  # 状态是其他智能体的动作，动作取值范围从 -10 到 10
    train_vdn(env, vdn)
