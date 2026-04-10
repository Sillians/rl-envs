import gymnasium as gym

def main():
    # Create the CartPole-v1 environment with rendering enabled
    env = gym.make("CartPole-v1", render_mode="human")
    # Print the observation space of the environment (Box with 4 values: cart position, cart velocity, pole angle, pole angular velocity)
    print(f"Observation space: {env.observation_space}") 
    # Print the action space of the environment (Discrete(2) - left or right)
    print(f"Action space: {env.action_space}") 

    # Reset the environment to start a new episode and set a seed for reproducibility
    observation, info = env.reset(seed=42) 
    # observation: what the agent can "see" - cart position, velocity, pole angle, etc.
    # info: extra debugging information (usually not needed for basic learning)
    
    # Print the initial observation after resetting the environment
    print(f"Initial observation: {observation}") 
    # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
    # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    
    # Run a loop for a specified number of iterations
    for _ in range(10000): 
        
        # Choose an action: 0 = push cart left, 1 = push cart right
        action = env.action_space.sample()  # Random action for now - real agents will be smarter!
        
        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action) # Take a step in the environment using the sampled action and receive the results
        print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}") # Print the results of the step
        
        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)
    
        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated: # Check if the episode has ended
            print("Episode finished. Resetting environment.") # Print a message indicating the episode has finished
            observation, info = env.reset() # Reset the environment for the next episode
    env.close() # Close the environment after the loop

if __name__ == "__main__":
    main()



"""Action and observation spaces"""
import gymnasium as gym

# Discrete action space (button presses)
env = gym.make("CartPole-v1")
print(f"Action space: {env.action_space}")  # Discrete(2) - left or right
print(f"Sample action: {env.action_space.sample()}")  # 0 or 1

# Box observation space (continuous values)
print(f"Observation space: {env.observation_space}")  # Box with 4 values
# Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])
print(f"Sample observation: {env.observation_space.sample()}")  # Random valid observation



"""Modifying the environment"""
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, RescaleAction

# start with a complex observation space
env = gym.make("CarRacing-v3")
print(f"Original observation space: {env.observation_space}")  # Box(96, 96, 3) RGB image of the game screen
print(f"Original observation shape: {env.observation_space.shape}")  # (96, 96, 3)

# Wrap it to flatten the observation space
env = FlattenObservation(env)
print(f"Flattened observation space: {env.observation_space}")  # Box(27648,)
print(f"Flattened observation shape: {env.observation_space.shape}")  # (27648,)



