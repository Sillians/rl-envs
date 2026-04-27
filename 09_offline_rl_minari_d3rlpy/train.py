"""Train offline RL with Minari + d3rlpy (CQL, IQL, or BC)."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import d3rlpy
from d3rlpy.algos import BCConfig, CQLConfig, DiscreteBCConfig, DiscreteCQLConfig, IQLConfig
from d3rlpy.datasets import get_minari
from gymnasium.spaces import Discrete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("--dataset-id", default="door-cloned-v1") # Alternative: D4RL/door/cloned-v2
    parser.add_argument("--dataset-id", default="D4RL/door/cloned-v2")
    parser.add_argument("--algo", choices=["cql", "iql", "bc"], default="cql")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-steps", type=int, default=200_000)
    parser.add_argument("--n-steps-per-epoch", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="mps") # Use "cuda" if you have an NVIDIA GPU, "mps" for Apple Silicon, or "cpu" otherwise.
    parser.add_argument("--log-dir", default="artifacts")
    return parser.parse_args()


def build_algo(algo_name: str, env, learning_rate: float, device: str):
    is_discrete = isinstance(env.action_space, Discrete)

    if algo_name == "bc":
        if is_discrete:
            return DiscreteBCConfig(learning_rate=learning_rate).create(device=device)
        return BCConfig(learning_rate=learning_rate).create(device=device)

    if algo_name == "cql":
        if is_discrete:
            return DiscreteCQLConfig(learning_rate=learning_rate).create(device=device)
        return CQLConfig(
            actor_learning_rate=learning_rate,
            critic_learning_rate=learning_rate,
            temp_learning_rate=learning_rate,
        ).create(device=device)

    if algo_name == "iql":
        if is_discrete:
            raise ValueError("IQL in this starter script supports only continuous action spaces.")
        return IQLConfig(
            actor_learning_rate=learning_rate,
            critic_learning_rate=learning_rate,
            value_learning_rate=learning_rate,
        ).create(device=device)

    raise ValueError(f"Unsupported algo: {algo_name}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Minari dataset: {args.dataset_id}")
    dataset, env = get_minari(args.dataset_id)

    d3rlpy.seed(args.seed)
    env.reset(seed=args.seed)

    algo = build_algo(args.algo, env, args.learning_rate, args.device)

    algo.fit(
        dataset,
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        experiment_name=f"{args.algo}_{args.dataset_id}",
        with_timestamp=False,
    )

    model_path = out_dir / f"{args.algo}_{args.dataset_id}.d3"
    algo.save(str(model_path))
    print(f"Saved model to {model_path}")

    env.close()


if __name__ == "__main__":
    main()
