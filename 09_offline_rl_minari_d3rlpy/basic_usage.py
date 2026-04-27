import minari

## load the dataset
dataset = minari.load_dataset('D4RL/door/human-v2', download=True)

print("Observation space:", dataset.observation_space)
print("Action space:", dataset.action_space)
print("Total episodes:", dataset.total_episodes)
print("Total steps:", dataset.total_steps)


## Sampling Episodes
dataset = minari.load_dataset("D4RL/door/human-v2", download=True)
dataset.set_seed(seed=123)

for i in range(5):
    # sample 5 episodes from the dataset
    episodes = dataset.sample_episodes(n_episodes=5)
    # get id's from the sampled episodes
    ids = list(map(lambda ep: ep.id, episodes))
    print(f"EPISODE ID'S SAMPLE {i}: {ids}")
    

## Specific indices can be also provided.
dataset = minari.load_dataset("D4RL/door/human-v2", download=True)
episodes_generator = dataset.iterate_episodes(episode_indices=[1, 2, 0])

for episode in episodes_generator:
    print(f"EPISODE ID {episode.id}")


## the minari.MinariDataset dataset itself is iterable:
dataset = minari.load_dataset("D4RL/door/human-v2", download=True)
for episode in dataset:
    print(f"EPISODE ID {episode.id}")
    

# Filter Episodes
print(f'TOTAL EPISODES ORIGINAL DATASET: {dataset.total_episodes}')

# get episodes with mean reward greater than 2
filter_dataset = dataset.filter_episodes(lambda episode: episode.rewards.mean() > 2)

print(f'TOTAL EPISODES FILTER DATASET: {filter_dataset.total_episodes}')


## Split Dataset
split_datasets = minari.split_dataset(dataset, sizes=[20, 5], seed=123)

print(f'TOTAL EPISODES FIRST SPLIT: {split_datasets[0].total_episodes}')
print(f'TOTAL EPISODES SECOND SPLIT: {split_datasets[1].total_episodes}')


## Recover Environment
env = dataset.recover_environment()

env.reset()
for _ in range(100):
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        env.reset()