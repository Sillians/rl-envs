"""Evaluate a trained HalfCheetah SAC model with video recording."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium.wrappers import RecordVideo

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="HalfCheetah-v5")
    parser.add_argument("--model-path", default="artifacts/best_model/best_model")
    parser.add_argument("--vecnorm-path", default="artifacts/vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=5) # Reduced for quicker recording
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--video-dir", default="videos/half_cheetah")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    
    if not model_path.with_suffix(".zip").exists() and not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path} or {model_path}.zip")

    # 1. Create the environment with render_mode="rgb_array"
    # We use a lambda to wrap the env inside make_vec_env
    def make_env():
        env = gym.make(args.env_id, render_mode="rgb_array")
        # Trigger on every episode (x is the episode index)
        env = RecordVideo(env, video_folder=args.video_dir, episode_trigger=lambda x: True)
        return env

    env = make_vec_env(make_env, n_envs=1, seed=args.seed)

    # 2. Handle Normalization
    vecnorm_path = Path(args.vecnorm_path)
    if vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False
        env.norm_reward = False

    # 3. Load and Evaluate
    model = SAC.load(str(model_path), env=env)

    print(f"Recording {args.episodes} episodes to {args.video_dir}...")
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.episodes,
        deterministic=True,
    )

    print(f"\nEpisodes: {args.episodes}")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()