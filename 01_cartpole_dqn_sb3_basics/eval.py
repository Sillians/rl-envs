from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


"""Evaluate a trained CartPole DQN model."""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--model-path", default="artifacts/best_model/best_model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.with_suffix(".zip").exists() and not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path} or {model_path}.zip")

    render_mode = "human" if args.render else None
    env = gym.make(args.env_id, render_mode=render_mode)
    env = Monitor(env)
    env.reset(seed=args.seed)

    model = DQN.load(str(model_path))
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.episodes,
        deterministic=True,
    )

    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
