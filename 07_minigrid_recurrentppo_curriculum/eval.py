"""Evaluate a trained MiniGrid RecurrentPPO model."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from minigrid.wrappers import FlatObsWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def make_env(env_id: str, seed: int, render: bool):
    def _init():
        render_mode = "human" if render else None
        env = gym.make(env_id, render_mode=render_mode)
        env = FlatObsWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return VecMonitor(DummyVecEnv([_init]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--model-path", default="artifacts/recurrentppo_minigrid_final_model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.with_suffix(".zip").exists() and not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path} or {model_path}.zip")

    vec_env = make_env(args.env_id, args.seed, args.render)
    model = RecurrentPPO.load(str(model_path), env=vec_env)

    returns = []
    for episode in range(args.episodes):
        obs = vec_env.reset()
        lstm_states = None
        episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        done = np.array([False])
        total_reward = 0.0

        while not bool(done[0]):
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, rewards, done, _infos = vec_env.step(action)
            total_reward += float(rewards[0])
            episode_starts = done

        returns.append(total_reward)

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    
    print(f"Episodes: {args.episodes}")
    print(f"Mean return: {mean_return:.2f}")
    print(f"Std return: {std_return:.2f}")

    vec_env.close()

if __name__ == "__main__":
    main()
