"""Evaluate a trained RLlib PPO policy on PettingZoo simple_spread."""

from __future__ import annotations

import argparse

import numpy as np
import ray
from pettingzoo.mpe import simple_spread_v3
from ray.rllib.algorithms.algorithm import Algorithm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--parameter-sharing", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = "human" if args.render else None

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    try:
        algo = Algorithm.from_checkpoint(args.checkpoint)
        env = simple_spread_v3.parallel_env(
            N=args.num_agents,
            max_cycles=args.max_cycles,
            continuous_actions=False,
            render_mode=render_mode,
        )

        episode_returns = []
        for episode in range(args.episodes):
            obs, _infos = env.reset(seed=args.seed + episode)
            ep_return = 0.0

            while env.agents:
                action_dict = {}
                for agent_id, agent_obs in obs.items():
                    policy_id = "shared_policy" if args.parameter_sharing else agent_id
                    action_dict[agent_id] = algo.compute_single_action(
                        observation=agent_obs,
                        policy_id=policy_id,
                        explore=False,
                    )

                obs, rewards, _terminations, _truncations, _infos = env.step(action_dict)
                ep_return += float(np.sum(list(rewards.values())))

            episode_returns.append(ep_return)

        print(f"Episodes: {args.episodes}")
        print(f"Mean team return: {float(np.mean(episode_returns)):.2f}")
        print(f"Std team return: {float(np.std(episode_returns)):.2f}")

        env.close()
        algo.stop()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
