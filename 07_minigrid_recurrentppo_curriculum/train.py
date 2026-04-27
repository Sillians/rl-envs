"""Train RecurrentPPO on MiniGrid with optional curriculum stages."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

DEFAULT_CURRICULUM = [
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-KeyCorridorS3R3-v0",
]


def make_env(env_id: str, seed: int):
    def _init():
        env = gym.make(env_id)
        env = FlatObsWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def make_vec_env(env_id: str, seed: int):
    return VecMonitor(DummyVecEnv([make_env(env_id, seed)]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps-per-stage", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument(
        "--curriculum-envs",
        default=",".join(DEFAULT_CURRICULUM),
        help="Comma-separated env IDs used only when --curriculum is set.",
    )
    parser.add_argument("--log-dir", default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_ids = [args.env_id]
    if args.curriculum:
        env_ids = [env_id.strip() for env_id in args.curriculum_envs.split(",") if env_id.strip()]

    model = None

    for stage_idx, env_id in enumerate(env_ids):
        vec_env = make_vec_env(env_id, args.seed + stage_idx)

        if model is None:
            model = RecurrentPPO(
                policy="MlpLstmPolicy",
                env=vec_env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                tensorboard_log=str(out_dir / "tb"),
                seed=args.seed,
                verbose=1,
            )
        else:
            model.set_env(vec_env)

        print(f"Training stage {stage_idx + 1}/{len(env_ids)} on {env_id}")
        model.learn(
            total_timesteps=args.timesteps_per_stage,
            reset_num_timesteps=(stage_idx == 0),
            progress_bar=True,
        )
        vec_env.close()

    final_model_path = out_dir / "recurrentppo_minigrid_final_model"
    model.save(str(final_model_path))
    print(f"Saved final model to {final_model_path}.zip")


if __name__ == "__main__":
    main()
