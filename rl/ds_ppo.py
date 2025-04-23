import argparse
import csv
import os
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Bernoulli

from fp_btree import FplanEnv

ROOT_PATH = Path(__file__).parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("run in", device)

models_dir = ROOT_PATH / "models"
os.makedirs(models_dir, exist_ok=True)


class Config:
    def __init__(self):
        # 优化后的超参数
        self.gamma = 0.99
        self.lr = 3e-4
        self.epochs = 4  # 减少训练轮数
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.batch_size = 256  # 增大批量大小
        self.buffer_size = 4096  # 增大经验回放缓冲区
        self.hidden_size = 256  # 增大网络容量
        self.update_interval = 4096  # 与buffer_size对齐
        self.max_episodes = 2048
        self.gae_lambda = 0.95  # 新增GAE参数


class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, 1)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.net(x)
        return torch.sigmoid(self.actor(x)), self.critic(x)


class PPO:
    def __init__(self, state_dim, config):
        self.config = config
        self.state_dim = state_dim

        self.policy = ActorCritic(state_dim, config.hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.old_policy = ActorCritic(state_dim, config.hidden_size).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        # 预分配GPU内存
        self._init_buffers()

    def load_checkpoint(self, ckpt_path):
        """加载模型检查点"""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")

        state_dict = torch.load(ckpt_path, map_location=device)
        self.policy.load_state_dict(state_dict)
        self.old_policy.load_state_dict(state_dict)
        print(f"Successfully loaded checkpoint from {ckpt_path}")

    def _init_buffers(self):
        buffer_size = self.config.buffer_size
        self.states = torch.zeros((buffer_size, self.state_dim), device=device)
        self.actions = torch.zeros((buffer_size, 1), device=device)
        self.rewards = torch.zeros((buffer_size, 1), device=device)
        self.next_states = torch.zeros((buffer_size, self.state_dim), device=device)
        self.dones = torch.zeros((buffer_size, 1), device=device)
        self.log_probs = torch.zeros((buffer_size, 1), device=device)
        self.ptr = 0
        self.full = False

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        idx = self.ptr % self.config.buffer_size
        self.states[idx] = torch.as_tensor(state, device=device)
        self.actions[idx] = torch.tensor([action], device=device)
        self.rewards[idx] = torch.tensor([reward], device=device)
        self.next_states[idx] = torch.as_tensor(next_state, device=device)
        self.dones[idx] = torch.tensor([done], device=device)
        self.log_probs[idx] = torch.tensor([log_prob], device=device)
        self.ptr += 1
        if self.ptr >= self.config.buffer_size:
            self.full = True
            self.ptr = 0

    def select_action(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_probs, _ = self.old_policy(state_tensor.unsqueeze(0))
        m = Bernoulli(action_probs)
        action = m.sample()
        return action.item() > 0.5, m.log_prob(action).item()

    def compute_gae(self, values, next_values):
        advantages = torch.zeros_like(self.rewards)
        last_gae_lam = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = next_values[t]
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = values[t + 1]
            delta = (
                self.rewards[t]
                + self.config.gamma * next_value * next_non_terminal
                - values[t]
            )
            advantages[t] = last_gae_lam = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * next_non_terminal
                * last_gae_lam
            )
        return advantages

    def update(self):
        if not self.full and self.ptr < self.config.buffer_size:
            return

        with torch.no_grad():
            old_values = self.old_policy(self.states)[1]
            next_values = self.old_policy(self.next_states)[1]
            advantages = self.compute_gae(old_values, next_values)
            returns = advantages + old_values

        self.old_policy.load_state_dict(self.policy.state_dict())

        # 打乱索引
        indices = torch.randperm(self.config.buffer_size, device=device)

        for _ in range(self.config.epochs):
            for start in range(0, self.config.buffer_size, self.config.batch_size):
                batch_idx = indices[start : start + self.config.batch_size]

                batch_states = self.states[batch_idx]
                batch_actions = self.actions[batch_idx]
                batch_old_log_probs = self.log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                action_probs, values = self.policy(batch_states)
                dist = Bernoulli(action_probs)
                log_probs = dist.log_prob(batch_actions)

                ratios = (log_probs - batch_old_log_probs).exp()
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

                value_loss = F.mse_loss(values, batch_returns)

                entropy = dist.entropy().mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

        self.full = False
        self.ptr = 0


def train(env: FplanEnv, agent: PPO, config: Config, start_episode=0):
    episode_rewards = []

    for episode in range(start_episode, config.max_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        while not done:
            # 收集完整缓冲区
            while not agent.full and not done:
                action, log_prob = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.store_transition(
                    state, action, reward, next_state, done, log_prob
                )
                state = next_state
                episode_reward += reward
                step_count += 1

            if agent.full:
                print(f"Updating at episode {episode} step {step_count}...")
                st = time.time()
                agent.update()
                print(f"Update time: {time.time() - st:.2f}s")
                print("env statu:")
                env.show_info()

        episode_rewards.append(episode_reward)
        print(f"Episode {episode} | Reward: {episode_reward} | Steps: {step_count}")

        if episode % 50 == 0:
            model_path = models_dir / f"ppo_{episode}.pth"
            torch.save(agent.policy.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    return episode_rewards


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PPO Training with Resume")
    parser.add_argument(
        "--resume",
        type=int,
        default=None,
        help="Model checkpoint episode number to resume training (e.g. 50 for ppo_50.pth)",
    )
    return parser.parse_args()


def save_rewards(rewards, filename="training_rewards.csv"):
    """保存奖励数据到CSV文件"""
    with open(models_dir / filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward"])
        for i, r in enumerate(rewards):
            writer.writerow([i, r])


if __name__ == "__main__":
    args = parse_args()
    config = Config()

    file_path = ROOT_PATH / "raw_data" / "ami33"
    env = FplanEnv(file_path.__str__(), max_times=config.update_interval)

    agent = PPO(env.state_dim, config)
    if args.resume is not None:
        ckpt_path = models_dir / f"ppo_{args.resume}.pth"
        try:
            agent.load_checkpoint(ckpt_path)
            print(f"Resuming training from episode {args.resume}")
            # 可以在此调整初始episode计数（如果需要）
            # start_episode = args.resume + 1
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            exit(1)
    else:
        print(f"model file ppo_{args.resume}.pth no exit.")

    rewards = train(env, agent, config, args.resume + 1)

    save_rewards(rewards)
