"""Evaluate a trained Atari Breakout DQN model."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Gymnasium 1.0+ recommends explicit registration if auto-discovery fails
gym.register_envs(ale_py)

def make_env(env_id: str, seed: int, render: bool):
    env_kwargs = {"render_mode": "human"} if render else None
    # make_atari_env is an SB3 wrapper that handles specialized Atari preprocessing
    env = make_atari_env(env_id, n_envs=1, seed=seed, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=4)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="ALE/Breakout-v5")
    parser.add_argument("--model-path", default="artifacts/best_model/best_model")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.with_suffix(".zip").exists() and not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path} or {model_path}.zip")

    env = make_env(args.env_id, args.seed, args.render)
    model = DQN.load(str(model_path), env=env)

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
