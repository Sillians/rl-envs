# 04 - Atari Breakout with DQN (Pixels, Wrappers, and Scaling)

- [Arcade Learning Environment](https://ale.farama.org/)
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)


## Difficulty
Intermediate

## Goal
Train an Atari agent from pixels and learn preprocessing/wrapper choices that make or break performance.

## Environment and Algorithms
- Environment: `ALE/Breakout-v5` (Atari)
- Algorithms: DQN (SB3), optional PPO comparison (CleanRL)
- Libraries: Gymnasium Atari, SB3, CleanRL, ALE-py

## What You Will Learn
- Frame stacking, grayscale conversion, sticky actions, and no-op starts
- Why replay buffer settings matter more for image tasks
- Evaluation protocols for Atari-style tasks

## Implementation Milestones
1. Build wrapped Atari environment with frame stacking.
2. Train SB3 DQN baseline and log episodic return.
3. Re-run with different frame skip/reward clipping settings.
4. Replicate with a CleanRL Atari script for comparison.
5. Analyze sample efficiency and variance across seeds.

## Success Criteria
- Agent surpasses random baseline by a wide margin.
- Documented impact of at least 2 wrapper choices.

## Stretch Ideas
- Swap to `ALE/Pong-v5` and test transferability.
- Try `QR-DQN` using SB3 Contrib.

## Helper Docs and Blogs
- [Gymnasium Atari Docs](https://gymnasium.farama.org/environments/atari/)
- [SB3 DQN Docs](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
- [SB3 Atari Wrappers](https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html)
- [CleanRL Documentation](https://docs.cleanrl.dev/)
- [ALE Documentation](https://ale.farama.org/)
- [SB3 Contrib (QR-DQN)](https://sb3-contrib.readthedocs.io/)
- [Human-level Control Through Deep RL](https://www.nature.com/articles/nature14236)
- [Rainbow DQN Paper](https://arxiv.org/abs/1710.02298)
