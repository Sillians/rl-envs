"""Train imitation-style agents without extra libraries (BC or DAgger-lite).

This starter script supports discrete-action environments (e.g., CartPole, LunarLander).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


@dataclass
class TransitionBatch:
    observations: np.ndarray
    actions: np.ndarray


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, observations: np.ndarray, actions: np.ndarray) -> None:
        self.observations = torch.as_tensor(observations, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def __getitem__(self, idx: int):
        return self.observations[idx], self.actions[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["bc", "dagger"], default="bc")
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expert-timesteps", type=int, default=100_000)
    parser.add_argument("--demo-episodes", type=int, default=30)
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--dagger-rounds", type=int, default=5)
    parser.add_argument("--dagger-episodes-per-round", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--log-dir", default="artifacts")
    return parser.parse_args()


def ensure_discrete_env(env: gym.Env) -> int:
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("This starter supports only discrete-action envs.")
    return int(env.action_space.n)


def collect_with_policy(
    env_id: str,
    policy_fn,
    episodes: int,
    seed: int,
) -> TransitionBatch:
    env = gym.make(env_id)
    observations: list[np.ndarray] = []
    actions: list[int] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            action = int(policy_fn(obs))
            observations.append(np.asarray(obs, dtype=np.float32))
            actions.append(action)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)

    env.close()
    return TransitionBatch(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
    )


def train_bc(
    observations: np.ndarray,
    actions: np.ndarray,
    obs_dim: int,
    action_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> BCPolicy:
    torch.manual_seed(seed)
    dataset = NumpyDataset(observations, actions)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BCPolicy(obs_dim=obs_dim, action_dim=action_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        losses = []
        for batch_obs, batch_actions in loader:
            logits = model(batch_obs)
            loss = criterion(logits, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        print(f"epoch={epoch + 1}/{epochs} bc_loss={np.mean(losses):.4f}")

    return model


def policy_from_model(model: BCPolicy):
    def _policy(obs: np.ndarray) -> int:
        with torch.no_grad():
            logits = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(logits, dim=1).item())

    return _policy


def evaluate_bc(model: BCPolicy, env_id: str, episodes: int, seed: int) -> tuple[float, float]:
    env = gym.make(env_id)
    returns = []
    policy = policy_from_model(model)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            ep_return += float(reward)
            done = bool(terminated or truncated)

        returns.append(ep_return)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def save_bc_model(path: Path, model: BCPolicy, env_id: str, obs_dim: int, action_dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "env_id": env_id,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expert_env = gym.make(args.env_id)
    action_dim = ensure_discrete_env(expert_env)
    obs_dim = int(np.prod(expert_env.observation_space.shape))
    expert_env.close()

    expert = PPO(
        policy="MlpPolicy",
        env=gym.make(args.env_id),
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(out_dir / "tb_expert"),
    )
    expert.learn(total_timesteps=args.expert_timesteps, progress_bar=True)

    expert_path = out_dir / "expert_ppo"
    expert.save(str(expert_path))
    print(f"Saved expert policy to {expert_path}.zip")

    expert_policy = lambda obs: expert.predict(obs, deterministic=True)[0]
    data = collect_with_policy(
        env_id=args.env_id,
        policy_fn=expert_policy,
        episodes=args.demo_episodes,
        seed=args.seed + 1,
    )

    bc_model = train_bc(
        observations=data.observations,
        actions=data.actions,
        obs_dim=obs_dim,
        action_dim=action_dim,
        epochs=args.bc_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    if args.mode == "dagger":
        all_obs = [data.observations]
        all_actions = [data.actions]
        learner_policy = policy_from_model(bc_model)

        for round_idx in range(args.dagger_rounds):
            learner_rollouts = collect_with_policy(
                env_id=args.env_id,
                policy_fn=learner_policy,
                episodes=args.dagger_episodes_per_round,
                seed=args.seed + 100 + round_idx,
            )

            # Label learner-visited states using the expert.
            relabeled_actions = []
            for obs in learner_rollouts.observations:
                action, _ = expert.predict(obs, deterministic=True)
                relabeled_actions.append(int(action))

            all_obs.append(learner_rollouts.observations)
            all_actions.append(np.asarray(relabeled_actions, dtype=np.int64))

            merged_obs = np.concatenate(all_obs, axis=0)
            merged_actions = np.concatenate(all_actions, axis=0)
            bc_model = train_bc(
                observations=merged_obs,
                actions=merged_actions,
                obs_dim=obs_dim,
                action_dim=action_dim,
                epochs=max(5, args.bc_epochs // 2),
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed + round_idx + 1,
            )
            learner_policy = policy_from_model(bc_model)
            print(f"Completed DAgger round {round_idx + 1}/{args.dagger_rounds}")

    model_name = "dagger_policy.pt" if args.mode == "dagger" else "bc_policy.pt"
    model_path = out_dir / model_name
    save_bc_model(model_path, bc_model, args.env_id, obs_dim, action_dim)
    print(f"Saved learner policy to {model_path}")

    mean_reward, std_reward = evaluate_bc(
        bc_model,
        env_id=args.env_id,
        episodes=args.eval_episodes,
        seed=args.seed + 200,
    )
    print(f"Mode: {args.mode}")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")

    eval_env = gym.make(args.env_id)
    expert_mean, expert_std = evaluate_policy(expert, eval_env, n_eval_episodes=args.eval_episodes)
    eval_env.close()
    print(f"Expert PPO mean reward: {expert_mean:.2f} +/- {expert_std:.2f}")


if __name__ == "__main__":
    main()
