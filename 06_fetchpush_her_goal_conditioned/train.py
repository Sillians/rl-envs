"""Train HER + SAC/TD3 on FetchPush-v3."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium_robotics
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.her import HerReplayBuffer

ALGOS = {
    "sac": SAC,
    "td3": TD3,
}

gym.register_envs(gymnasium_robotics)

def make_env(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=ALGOS.keys(), default="sac")
    parser.add_argument("--env-id", default="FetchPush-v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--n-sampled-goal", type=int, default=4)
    parser.add_argument("--goal-selection-strategy", default="future")
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

    model_kwargs = dict(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": args.n_sampled_goal,
            "goal_selection_strategy": args.goal_selection_strategy,
        },
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

    final_model_path = out_dir / f"{args.algo}_her_final_model"
    model.save(str(final_model_path))
    print(f"Saved final model to {final_model_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
