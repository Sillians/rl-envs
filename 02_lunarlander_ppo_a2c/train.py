"""Train PPO or A2C on LunarLander-v3 with Stable-Baselines3."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

ALGOS = {
    "ppo": PPO,
    "a2c": A2C,
}


def make_env(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=ALGOS.keys(), default="ppo")
    parser.add_argument("--env-id", default="LunarLander-v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--learning-rate", type=float, default=3e-3) # adjusted from default to speed up training for demo purposes; feel free to experiment with this!
    parser.add_argument("--gamma", type=float, default=0.999) # adjusted from default to encourage longer-term planning; feel free to experiment with this! (try 0.95 or 0.98 for more short-sighted behavior)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--log-dir", default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(args.env_id, args.seed)
    eval_env = make_env(args.env_id, args.seed + 100)

    algo_cls = ALGOS[args.algo]
    model = algo_cls(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tensorboard_log=str(out_dir / "tb"),
        seed=args.seed,
        verbose=1,
    )

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
