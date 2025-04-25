import argparse
from pathlib import Path
import time
import torch
import numpy as np

# 从原脚本导入必要的类
from ds_ppo import PPO, Config
from fp_btree import FplanEnv

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径配置
ROOT_PATH = Path(__file__).parent.parent
models_dir = ROOT_PATH / "models"


class Tester:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.cost_history = []
        self.step_count = 0
        self.max_steps = 10000  # 最大测试步数
        self.min_cost_delta = 1e-4  # cost变化阈值，小于此值则停止
        self.window_size = 100  # 计算cost变化的窗口大小

    def run_episode(self, render=False):
        """运行一个测试episode"""
        state = self.env.reset()
        done = False
        total_reward = 0
        self.step_count = 0
        self.cost_history = [self.env.get_cost()]  # 记录初始cost

        while not done:
            action, _ = self.agent.select_action(state)
            next_state, reward, done = self.env.step(action)

            # 记录数据
            self.step_count += 1
            total_reward += reward
            state = next_state

            # 定期检查cost变化
            if self.step_count % self.window_size == 0:
                current_cost = self.env.get_cost()
                self.cost_history.append(current_cost)

                # 计算最近几次cost的变化
                if len(self.cost_history) >= 2:
                    cost_delta = abs(self.cost_history[-2] - self.cost_history[-1])
                    if cost_delta < self.min_cost_delta:
                        print(
                            f"Cost变化过小({cost_delta:.6f} < {self.min_cost_delta:.6f})，停止测试"
                        )
                        done = True

                if render:
                    print(f"Step {self.step_count} | Current cost: {current_cost:.4f}")

            # 防止无限循环
            if self.step_count >= self.max_steps:
                print(f"达到最大步数 {self.max_steps}，停止测试")
                done = True

        return total_reward, self.step_count, self.cost_history[-1]


def load_model(env, model_path):
    """加载训练好的模型"""
    config = Config()
    agent = PPO(env.state_dim, config)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found")

    state_dict = torch.load(model_path, map_location=device)
    agent.policy.load_state_dict(state_dict)
    agent.old_policy.load_state_dict(state_dict)

    print(f"Successfully loaded model from {model_path}")
    return agent


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PPO Model Testing")
    parser.add_argument(
        "-m",
        "--model",  # 添加简写 -m
        type=int,
        required=True,
        help="Model checkpoint episode number to test (e.g. 50 for ppo_50.pth)",
    )
    parser.add_argument(
        "-e",
        "--episodes",  # 添加简写 -e
        type=int,
        default=5,
        help="Number of test episodes to run (default: %(default)s)",
    )
    parser.add_argument(
        "-r",
        "--render",  # 添加简写 -r
        action="store_true",
        help="Whether to render the testing process",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化环境和配置
    config = Config()
    file_path = ROOT_PATH / "raw_data" / "ami33"
    env = FplanEnv(file_path.__str__(), max_times=config.max_times)

    # 加载模型
    model_path = models_dir / f"ppo_{args.model}.pth"
    agent = load_model(env, model_path)

    print(f"\n开始测试模型 ppo_{args.model}.pth...")
    print(f"设备: {device}")
    print(f"测试环境: {file_path}")
    print(f"计划运行 {args.episodes} 个测试episode\n")

    # 运行测试
    test_results = []
    for ep in range(args.episodes):
        # 创建测试器
        tester = Tester(env, agent)
        print(f"\n=== Episode {ep + 1}/{args.episodes} ===")
        start_time = time.time()

        reward, steps, final_cost = tester.run_episode(render=args.render)

        duration = time.time() - start_time
        print(
            f"结果: Reward={reward:.2f} | Steps={steps} | Final Cost={final_cost:.4f} | Time={duration:.2f}s"
        )

        test_results.append(
            {
                "episode": ep + 1,
                "reward": reward,
                "steps": steps,
                "final_cost": final_cost,
                "time": duration,
            }
        )

    # 打印汇总统计
    print("\n=== 测试结果汇总 ===")
    print(f"测试的模型: ppo_{args.model}.pth")
    print(f"总测试episodes: {len(test_results)}")

    avg_reward = np.mean([r["reward"] for r in test_results])
    avg_steps = np.mean([r["steps"] for r in test_results])
    avg_cost = np.mean([r["final_cost"] for r in test_results])
    avg_time = np.mean([r["time"] for r in test_results])

    print(f"平均Reward: {avg_reward:.2f}")
    print(f"平均Steps: {avg_steps:.1f}")
    print(f"平均Final Cost: {avg_cost:.4f}")
    print(f"平均Time/episode: {avg_time:.2f}s")


if __name__ == "__main__":
    main()
