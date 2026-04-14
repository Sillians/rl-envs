"""Evaluate and visualize a trained LunarLander model."""

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo

ALGOS = {"ppo": PPO, 
         "a2c": A2C}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=ALGOS.keys(), default="ppo")
    parser.add_argument("--env-id", default="LunarLander-v3")
    parser.add_argument("--model-path", default="artifacts/best_model/best_model")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="Show live human render")
    parser.add_argument("--record", action="store_true", help="Save MP4 video of episodes")
    return parser.parse_args()

def plot_rewards(rewards, algo_name):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, marker='o', linestyle='-', color='b')
    plt.axhline(y=200, color='r', linestyle='--', label='Solved Threshold (200)')
    plt.title(f"Lunar Lander Evaluation - {algo_name.upper()}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("evaluation_results.png")
    print("Graph saved as evaluation_results.png")

def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    
    # Setup environment
    render_mode = "rgb_array" if args.record else ("human" if args.render else None)
    env = gym.make(args.env_id, render_mode=render_mode)
    
    if args.record:
        # Save videos to ./videos folder
        env = RecordVideo(env, video_folder="./videos", name_prefix=f"eval-{args.algo}")
    
    env = Monitor(env)
    model = ALGOS[args.algo].load(str(model_path))

    # Evaluate and capture individual episode rewards
    print(f"Evaluating {args.algo} for {args.episodes} episodes...")
    rewards, lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.episodes,
        deterministic=True,
        return_episode_rewards=True, # Critical for plotting
    )

    # Statistics
    import numpy as np
    print(f"\nResults for {args.algo}:")
    print(f"Mean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    
    # Generate visualization
    plot_rewards(rewards, args.algo)
    
    env.close()

if __name__ == "__main__":
    main()