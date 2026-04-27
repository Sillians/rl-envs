"""Evaluate a trained d3rlpy model on the corresponding Minari environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import d3rlpy
import numpy as np
from d3rlpy.datasets import get_minari
from gymnasium.spaces import Discrete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("--dataset-id", default="door-cloned-v1")
    parser.add_argument("--dataset-id", default="D4RL/door/cloned-v2") # Alternative: D4RL/door/cloned-v2
    parser.add_argument("--model-path", default="artifacts/cql_door-cloned-v2.d3")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    # parser.add_argument("--render-modes", default="human") # See https://www.gymlibrary.dev/api/env/#render-modes for available render modes.
    return parser.parse_args()


def batchify_observation(obs):
    if isinstance(obs, dict):
        # Keep key order deterministic for goal-conditioned dict observations.
        flat = np.concatenate(
            [np.asarray(obs[key], dtype=np.float32).ravel() for key in sorted(obs.keys())],
            axis=0,
        )
        return flat[None, :]

    arr = np.asarray(obs, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    dataset, env = get_minari(args.dataset_id)
    del dataset

    if args.render:
        env = env.unwrapped

    algo = d3rlpy.load_learnable(str(model_path))

    returns = []
    for episode in range(args.episodes):
        obs, _info = env.reset(seed=args.seed + episode)
        done = False
        ep_return = 0.0

        while not done:
            batched_obs = batchify_observation(obs)
            action = algo.predict(batched_obs)[0]

            if isinstance(env.action_space, Discrete):
                action = int(np.asarray(action).squeeze())
            else:
                action = np.asarray(action, dtype=np.float32)

            obs, reward, terminated, truncated, _info = env.step(action)
            ep_return += float(reward)
            done = bool(terminated or truncated)

        returns.append(ep_return)

    print(f"Episodes: {args.episodes}")
    print(f"Mean return: {float(np.mean(returns)):.2f}")
    print(f"Std return: {float(np.std(returns)):.2f}")

    env.close()


if __name__ == "__main__":
    main()
