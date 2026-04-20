# Recording Agents

- [Recording Agents](https://gymnasium.farama.org/introduction/record_agent/)

## Why Record Your Agent?

Recording agent behavior serves several important purposes in RL development:

- `Visual Understanding`: See exactly what your agent is doing - sometimes a 10-second video reveals issues that hours of staring at reward plots miss.
- `Performance Tracking`: Collect systematic data about episode rewards, lengths, and timing to understand training progress.
- `Debugging`: Identify specific failure modes, unusual behaviors, or environments where your agent struggles.
- `Evaluation`: Compare different training runs, algorithms, or hyperparameters objectively.
- `Communication`: Share results with collaborators, include in papers, or create educational content.


## Best Practices

### For Evaluation

- Record every episode to get complete performance picture
- Use multiple seeds for statistical significance
- Save both videos and numerical data
- Calculate confidence intervals for metrics


### For Training

- Record periodically (every 100-1000 episodes)
- Focus on episode statistics over videos during training
- Use adaptive recording triggers for interesting episodes
- Monitor memory usage for long training hours


### For Analysis

- Create moving averages to smooth noisy learning curves
- Look for patterns in both success and failure episodes
- Compare agent behavior at different stages of training
- Save raw data for later analysis and comparison


Recording agent behavior is an essential skill for RL practitioners. It helps you understand what your agent is actually learning, debug training issues, and communicate results effectively. Start with simple recording setups and gradually add more sophisticated analysis as your projects grow in complexity!