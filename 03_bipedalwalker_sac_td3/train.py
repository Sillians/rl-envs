"""Train SAC or TD3 on BipedalWalker-v3 with Stable-Baselines3."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

ALGOS = {
    "sac": SAC,
    "td3": TD3,
}


def make_env(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=ALGOS.keys(), default="sac") # Algorithm to train (SAC or TD3)
    parser.add_argument("--env-id", default="BipedalWalker-v3") # Environment ID (can be changed to other continuous control environments like "LunarLanderContinuous-v2" or "CarRacing-v0")
    parser.add_argument("--seed", type=int, default=42) # Random seed for reproducibility
    parser.add_argument("--timesteps", type=int, default=500_000) # Total training timesteps (adjust based on convergence speed and computational resources)
    parser.add_argument("--learning-rate", type=float, default=3e-3) # Learning rate for the optimizer (lower can improve stability but slow down learning)
    parser.add_argument("--batch-size", type=int, default=256) # Batch size for training (larger can improve stability but requires more memory)
    parser.add_argument("--buffer-size", type=int, default=1_000_000) # Replay buffer size (large for better performance but more memory usage)
    parser.add_argument("--gamma", type=float, default=0.999) # Discount factor for future rewards
    parser.add_argument("--eval-freq", type=int, default=20_000) # Evaluate every 20k steps (adjust based on episode length and desired evaluation frequency)
    parser.add_argument("--eval-episodes", type=int, default=10) # Number of episodes to average over during evaluation
    parser.add_argument("--log-dir", default="artifacts") # Directory to save logs and models
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(args.env_id, args.seed)
    eval_env = make_env(args.env_id, args.seed + 100)

    model_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        tensorboard_log=str(out_dir / "tb"),
        seed=args.seed,
        verbose=1,
    )

    if args.algo == "td3":
        n_actions = train_env.action_space.shape[-1]
        model_kwargs["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )

    model = ALGOS[args.algo](**model_kwargs)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=True)

    final_model_path = out_dir / f"{args.algo}_final_model"
    model.save(str(final_model_path))
    print(f"Saved final model to {final_model_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
