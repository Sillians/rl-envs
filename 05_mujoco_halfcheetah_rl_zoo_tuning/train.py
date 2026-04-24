"""Train SAC on HalfCheetah-v4 with optional VecNormalize."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

def make_env(env_id: str, n_envs: int, seed: int, normalize: bool):
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed)
    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="HalfCheetah-v5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--log-dir", default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(args.env_id, args.n_envs, args.seed, args.normalize)
    eval_env = make_env(args.env_id, 1, args.seed + 100, args.normalize)

    if isinstance(eval_env, VecNormalize):
        eval_env.training = False
        eval_env.norm_reward = False

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        train_freq=1,
        gradient_steps=1,
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

    model.learn(total_timesteps=args.timesteps, 
                callback=eval_callback, 
                progress_bar=True)

    final_model_path = out_dir / "sac_halfcheetah_final_model"
    model.save(str(final_model_path))
    print(f"Saved final model to {final_model_path}.zip")

    if isinstance(train_env, VecNormalize):
        vec_stats_path = out_dir / "vecnormalize.pkl"
        train_env.save(str(vec_stats_path))
        print(f"Saved VecNormalize stats to {vec_stats_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
