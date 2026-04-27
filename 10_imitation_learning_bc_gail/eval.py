"""Evaluate BC/DAgger learner policy or expert PPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["bc", "dagger", "expert"], default="bc")
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--model-path", default="artifacts/bc_policy.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def evaluate_bc(model: BCPolicy, env_id: str, episodes: int, seed: int, render: bool) -> tuple[float, float]:
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            with torch.no_grad():
                logits = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = int(torch.argmax(logits, dim=1).item())
            obs, reward, terminated, truncated, _info = env.step(action)
            ep_return += float(reward)
            done = bool(terminated or truncated)

        returns.append(ep_return)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)

    if args.mode == "expert":
        if not model_path.exists() and not model_path.with_suffix(".zip").exists():
            raise FileNotFoundError(f"Expert checkpoint not found: {model_path}")

        env = gym.make(args.env_id, render_mode="human" if args.render else None)
        expert = PPO.load(str(model_path), env=env)
        mean_reward, std_reward = evaluate_policy(expert, env, n_eval_episodes=args.episodes)
        env.close()
    else:
        if not model_path.exists():
            raise FileNotFoundError(f"Learner checkpoint not found: {model_path}")

        payload = torch.load(model_path, map_location="cpu")
        obs_dim = int(payload["obs_dim"])
        action_dim = int(payload["action_dim"])

        model = BCPolicy(obs_dim=obs_dim, action_dim=action_dim)
        model.load_state_dict(payload["state_dict"])
        model.eval()

        mean_reward, std_reward = evaluate_bc(
            model,
            env_id=args.env_id,
            episodes=args.episodes,
            seed=args.seed,
            render=args.render,
        )

    print(f"Mode: {args.mode}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")


if __name__ == "__main__":
    main()
