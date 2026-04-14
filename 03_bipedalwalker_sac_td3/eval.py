"""Evaluate and visualize a trained BipedalWalker model."""

import argparse
from pathlib import Path
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo

ALGOS = {"sac": SAC, 
         "td3": TD3}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=ALGOS.keys(), default="sac") # Algorithm to evaluate (SAC or TD3)
    parser.add_argument("--env-id", default="BipedalWalker-v3") # Environment ID (should match the one used during training)
    parser.add_argument("--model-path", default="artifacts/best_model/best_model") # Path to the trained model (without .zip extension)
    parser.add_argument("--episodes", type=int, default=5) # Reduced for quicker evaluation and visualization
    parser.add_argument("--seed", type=int, default=123) # Random seed for reproducibility during evaluation
    parser.add_argument("--render", action="store_true") # Whether to render the environment during evaluation (can be turned on for visual inspection)
    parser.add_argument("--record", action="store_true") # Whether to record videos of the evaluation episodes (saves videos in ./walker_videos/)
    return parser.parse_args()


def plot_gait(joint_data, algo_name):
    """Visualizes the hip and knee joint angles over time."""
    plt.figure(figsize=(12, 6))
    # BipedalWalker Obs Indices: 4: Hip1, 6: Knee1, 8: Hip2, 10: Knee2
    plt.plot(joint_data['hip1'], label='Hip 1 (Front)', alpha=0.8)
    plt.plot(joint_data['knee1'], label='Knee 1 (Front)', alpha=0.8)
    plt.title(f"Joint Angle Gait Analysis - {algo_name.upper()}")
    plt.xlabel("Step")
    plt.ylabel("Angle (Rad)")
    plt.legend()
    plt.grid(True)
    plt.savefig("gait_analysis.png")
    print("Gait plot saved as gait_analysis.png")

def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    
    render_mode = "rgb_array" if args.record else ("human" if args.render else None)
    env = gym.make(args.env_id, render_mode=render_mode)
    
    # if args.record:
    #     env = RecordVideo(env, video_folder="./walker_videos", name_prefix=f"{args.algo}")
    
    if args.record:
        video_folder = "./walker_videos"
        # The lambda function forces it to record every episode (0, 1, 2...)
        env = RecordVideo(
            env, 
            video_folder=video_folder, 
            episode_trigger=lambda x: True, 
            name_prefix=f"{args.algo}"
        )
        print(f"Recording enabled. Videos will be saved to: {video_folder}")
    
    env = Monitor(env)
    model = ALGOS[args.algo].load(str(model_path))

    # Tracking joint angles for the first episode
    joint_tracking = {'hip1': [], 'knee1': []}
    
    obs, _ = env.reset(seed=123)
    for _ in range(500): # Track 500 steps of one gait cycle
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Capture Hip and Knee angles (Indices 4 and 6)
        joint_tracking['hip1'].append(obs[4])
        joint_tracking['knee1'].append(obs[6])
        
        if terminated or truncated:
            break

    # Standard Evaluation
    print(f"Running full evaluation for {args.episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.episodes)

    print(f"\nFinal Results: Mean: {mean_reward:.2f} | Std: {std_reward:.2f}")
    plot_gait(joint_tracking, args.algo)
    env.close()

if __name__ == "__main__":
    main()