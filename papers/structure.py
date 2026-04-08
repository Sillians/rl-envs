import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

observation, info = env.reset(seed=42)
print(f"Initial observation: {observation}")
for _ in range(1000):
    action = env.action_space.sample()  # Take a random action
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    
    if terminated or truncated:
        print("Episode finished. Resetting environment.")
        observation, info = env.reset()
env.close()