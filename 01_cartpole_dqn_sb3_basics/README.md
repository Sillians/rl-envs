# 01 - CartPole DQN (SB3 Basics)

- [CartPole Docs](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

## Difficulty
Beginner

## Goal
Build your first end-to-end RL pipeline with **Gymnasium + Stable-Baselines3** by solving `CartPole-v1` using DQN.


## Environment and Algorithms

- Environment: `CartPole-v1` (Gymnasium Classic Control)
- Algorithms: DQN (baseline), optional comparison with PPO
- Libraries: Gymnasium, SB3, TensorBoard/W&B


## What You Will Learn

- The episode loop, observations, actions, rewards, and termination
- Replay buffers, target networks, epsilon-greedy exploration
- Basic experiment tracking and evaluation protocol


## Implementation Milestones

1. Create a random-agent baseline and log average reward.
2. Train DQN with default SB3 hyperparameters.
3. Add `EvalCallback` and periodic checkpointing.
4. Tune learning rate, buffer size, and exploration schedule.
5. Compare best checkpoint vs random policy over 100 eval episodes.


## Success Criteria

- Mean eval reward >= 475 over 100 episodes.
- Reproducible results over 3 different seeds.


## Stretch Ideas

- Compare DQN vs PPO sample efficiency.
- Add Optuna for automated hyperparameter search.


## Helper Docs and Blogs

- [Gymnasium Basic Usage](https://gymnasium.farama.org/main/introduction/basic_usage/)
- [CartPole Docs](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [SB3 Quickstart](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html)
- [SB3 DQN Docs](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
- [SB3 Callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)
- [RL Tips and Tricks (SB3)](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
- [DQN Paper (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
- [Spinning Up Intro](https://spinningup.openai.com/en/latest/)
