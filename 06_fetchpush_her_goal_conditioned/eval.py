"""Evaluate a trained HER model on FetchPush-v3."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium_robotics
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, TD3

ALGOS = {
    "sac": SAC,
    "td3": TD3,
}

gym.register_envs(gymnasium_robotics)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=ALGOS.keys(), default="sac")
    parser.add_argument("--env-id", default="FetchPush-v3")
    parser.add_argument("--model-path", default="artifacts/best_model/best_model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def evaluate_success_rate(model, env: gym.Env, episodes: int, seed: int) -> tuple[float, float]:
    episode_rewards = []
    success_flags = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        success = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if isinstance(info, dict) and "is_success" in info:
                success = max(success, float(info["is_success"]))
            done = bool(terminated or truncated)

        episode_rewards.append(total_reward)
        success_flags.append(success)

    return float(np.mean(episode_rewards)), float(np.mean(success_flags))


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.with_suffix(".zip").exists() and not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path} or {model_path}.zip")

    render_mode = "human" if args.render else None
    env = gym.make(args.env_id, render_mode=render_mode)

    model = ALGOS[args.algo].load(str(model_path), env=env)
    mean_reward, success_rate = evaluate_success_rate(model, env, args.episodes, args.seed)

    print(f"Algo: {args.algo}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Success rate: {success_rate * 100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
