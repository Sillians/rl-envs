"""Train multi-agent PPO on PettingZoo simple_spread via RLlib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ray
from pettingzoo.mpe import simple_spread_v3
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

REGISTERED_ENV = "pettingzoo_simple_spread"


def make_parallel_env(env_config: dict):
    return simple_spread_v3.parallel_env(
        N=env_config.get("num_agents", 3),
        max_cycles=env_config.get("max_cycles", 25),
        continuous_actions=False,
    )


def make_rllib_env(env_config: dict):
    return ParallelPettingZooEnv(make_parallel_env(env_config))


def save_checkpoint(algo, checkpoint_root: Path) -> str:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if hasattr(algo, "save_to_path"):
        return str(algo.save_to_path(str(checkpoint_root)))
    return str(algo.save(str(checkpoint_root)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=4_000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--parameter-sharing", action="store_true")
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_config = {
        "num_agents": args.num_agents,
        "max_cycles": args.max_cycles,
    }

    try:
        register_env(REGISTERED_ENV, lambda cfg: make_rllib_env(cfg))
    except Exception:
        pass

    probe_env = make_parallel_env(env_config)
    possible_agents = list(probe_env.possible_agents)
    obs_space = probe_env.observation_space(possible_agents[0])
    act_space = probe_env.action_space(possible_agents[0])
    probe_env.close()

    if args.parameter_sharing:
        policies = {
            "shared_policy": PolicySpec(observation_space=obs_space, action_space=act_space)
        }

        def policy_mapping_fn(agent_id, *_, **__):
            del agent_id
            return "shared_policy"

    else:
        policies = {
            agent_id: PolicySpec(observation_space=obs_space, action_space=act_space)
            for agent_id in possible_agents
        }

        def policy_mapping_fn(agent_id, *_, **__):
            return agent_id

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    try:
        config = (
            PPOConfig()
            .environment(env=REGISTERED_ENV, env_config=env_config)
            .framework("torch")
            .rollouts(num_rollout_workers=args.num_workers)
            .training(
                train_batch_size=args.train_batch_size,
                lr=args.learning_rate,
                gamma=args.gamma,
            )
            .resources(num_gpus=args.num_gpus)
            .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
            .debugging(seed=args.seed)
        )

        algo = config.build()
        try:
            for iteration in range(1, args.iterations + 1):
                result = algo.train()
                reward = result.get("episode_reward_mean", float("nan"))
                print(f"iter={iteration} episode_reward_mean={reward:.3f}")

            checkpoint_path = save_checkpoint(algo, out_dir / "checkpoints")
            print(f"Saved checkpoint: {checkpoint_path}")

            metadata = {
                "num_agents": args.num_agents,
                "max_cycles": args.max_cycles,
                "parameter_sharing": args.parameter_sharing,
                "checkpoint": checkpoint_path,
            }
            metadata_path = out_dir / "run_metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Saved metadata: {metadata_path}")
        finally:
            algo.stop()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
