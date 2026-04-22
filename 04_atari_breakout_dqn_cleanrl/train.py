"""Train DQN on Atari Breakout using SB3 wrappers."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Gymnasium 1.0+ recommends explicit registration if auto-discovery fails
gym.register_envs(ale_py)

def make_env(env_id: str, n_envs: int, seed: int):
    # make_atari_env is an SB3 wrapper that handles specialized Atari preprocessing
    env = make_atari_env(env_id, n_envs=n_envs, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="ALE/Breakout-v5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--log-dir", default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(args.env_id, args.n_envs, args.seed)
    eval_env = make_env(args.env_id, 1, args.seed + 100)

    model = DQN(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=50_000,
        batch_size=args.batch_size,
        train_freq=4,
        target_update_interval=10_000,
        tensorboard_log=str(out_dir / "tb"),
        seed=args.seed,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=True)

    final_model_path = out_dir / "dqn_breakout_final_model"
    model.save(str(final_model_path))
    print(f"Saved final model to {final_model_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
