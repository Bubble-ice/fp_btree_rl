from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Bernoulli
import numpy as np
from collections import deque

from warp_env import FplanEnvWrap


ROOT_PATH = Path(__file__).parent.parent

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 超参数配置
class Config:
    def __init__(self):
        self.gamma = 0.99  # 折扣因子
        self.lr = 3e-4  # 学习率
        self.epochs = 10  # 每次数据收集后的训练轮数
        self.clip_epsilon = 0.2  # PPO clip参数
        self.entropy_coef = 0.01  # 熵奖励系数
        self.value_coef = 0.5  # 价值函数损失系数
        self.max_grad_norm = 0.5  # 梯度裁剪
        self.batch_size = 32  # 小批量大小
        self.buffer_size = 1024  # 每次收集的transition数量
        self.hidden_size = 32  # 网络隐藏层大小
        self.update_interval = 2000  # 更新间隔(与环境交互的步数)
        self.max_episodes = 2000  # 最大训练episode数


# 策略和价值网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # 策略网络
        self.actor = nn.Linear(hidden_size, 1)

        # 价值网络
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # 动作概率
        action_probs = torch.sigmoid(self.actor(x))

        # 状态价值
        state_values = self.critic(x)

        return action_probs, state_values


# PPO 算法
class PPO:
    def __init__(self, state_dim, config):
        self.config = config
        self.policy = ActorCritic(state_dim, config.hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)

        # 旧策略用于计算重要性采样比率
        self.old_policy = ActorCritic(state_dim, config.hidden_size).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        # 经验回放缓冲区
        self.buffer = deque(maxlen=self.config.buffer_size)

    def select_action(self, state) -> tuple[bool, float]:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs, _ = self.old_policy(state)

        m = Bernoulli(action_probs)
        action = m.sample()
        log_prob = m.log_prob(action)

        return bool(action.item()), log_prob.item()

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append((state, action, reward, next_state, done, log_prob))

    def update(self):
        if len(self.buffer) < self.config.buffer_size:
            return

        # 准备数据
        states = torch.FloatTensor(np.array([t[0] for t in self.buffer])).to(device)
        actions = (
            torch.FloatTensor(np.array([t[1] for t in self.buffer]))
            .unsqueeze(1)
            .to(device)
        )
        rewards = (
            torch.FloatTensor(np.array([t[2] for t in self.buffer]))
            .unsqueeze(1)
            .to(device)
        )
        next_states = torch.FloatTensor(np.array([t[3] for t in self.buffer])).to(
            device
        )
        dones = (
            torch.FloatTensor(np.array([t[4] for t in self.buffer]))
            .unsqueeze(1)
            .to(device)
        )
        old_log_probs = (
            torch.FloatTensor(np.array([t[5] for t in self.buffer]))
            .unsqueeze(1)
            .to(device)
        )

        # 计算GAE和回报
        with torch.no_grad():
            _, values = self.old_policy(states)
            _, next_values = self.old_policy(next_states)

            # TD误差
            deltas = rewards + self.config.gamma * next_values * (1 - dones) - values

            # 计算GAE
            advantages = torch.zeros_like(rewards).to(device)
            advantage = 0
            for i in reversed(range(len(deltas))):
                advantage = deltas[i] + self.config.gamma * advantage * (1 - dones[i])
                advantages[i] = advantage

            # 计算回报
            returns = advantages + values

        # 更新旧策略
        self.old_policy.load_state_dict(self.policy.state_dict())

        # 训练多个epoch
        for _ in range(self.config.epochs):
            # 随机打乱数据
            indices = np.arange(self.config.buffer_size)
            np.random.shuffle(indices)

            # 小批量训练
            for start in range(0, self.config.buffer_size, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 计算新策略的概率和值
                action_probs, values = self.policy(batch_states)
                m = Bernoulli(action_probs)
                log_probs = m.log_prob(batch_actions)

                # 计算重要性采样比率
                ratios = torch.exp(log_probs - batch_old_log_probs)

                # 计算策略损失
                surr1 = ratios * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratios,
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon,
                    )
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # 计算价值函数损失
                value_loss = F.mse_loss(values, batch_returns)

                # 计算熵奖励
                entropy = m.entropy().mean()

                # 总损失
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                # 梯度下降
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

        # 清空缓冲区
        self.buffer.clear()


# 训练函数
def train(env: FplanEnvWrap, agent: PPO, config: Config):
    episode_rewards = []

    for episode in range(config.max_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # 收集数据
            for _ in range(config.update_interval):
                action, log_prob = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)

                agent.store_transition(
                    state, action, reward, next_state, done, log_prob
                )

                state = next_state
                episode_reward += reward

                if done:
                    break
            print("update model...")
            st = time.time()
            # 更新策略
            agent.update()
            print("time:", time.time() - st)

        episode_rewards.append(episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward}")

        # 每100轮保存一次模型
        if episode % 100 == 0:
            torch.save(agent.policy.state_dict(), f"ppo_model_{episode}.pth")

    return episode_rewards


# 主函数
if __name__ == "__main__":
    file_path = ROOT_PATH / "raw_data" / "ami33"
    env = FplanEnvWrap(file_path.__str__(), max_times=2000)

    config = Config()
    agent = PPO(env.state_dim, config)

    rewards = train(env, agent, config)
