# 08 - Multi-Agent RL with PettingZoo + RLlib

PettingZoo is a simple, pythonic interface capable of representing general multi-agent reinforcement learning (MARL) problems. PettingZoo includes a wide variety of reference environments, helpful utilities, and tools for creating your own custom environments.

## Difficulty
Advanced

## Goal
Train cooperative/competitive multi-agent policies and benchmark parameter sharing vs independent policies.

## Environment and Algorithms
- Environments: PettingZoo MPE/Pistonball/Holdem variants
- Algorithms: PPO/MAPPO-style setups, independent PPO baselines
- Libraries: PettingZoo, SuperSuit, RLlib, (optional) SB3 for single-policy baselines

## What You Will Learn
- Multi-agent environment APIs (AEC and Parallel)
- Policy mapping and parameter sharing strategies
- Scalable training and distributed rollouts with RLlib

## Implementation Milestones
1. Start with a PettingZoo tutorial environment.
2. Train independent policies and track per-agent rewards.
3. Implement parameter sharing and compare stability.
4. Migrate training to RLlib for distributed sampling.
5. Evaluate generalization with changed agent counts/seeds.

## Success Criteria
- Stable convergence on at least one cooperative environment.
- Quantified tradeoff between shared and independent policies.

## Stretch Ideas
- Add self-play league updates.
- Export trained policy and run tournament evaluation.

## Helper Docs and Blogs
- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [PettingZoo RLlib Tutorials](https://pettingzoo.farama.org/tutorials/rllib/index.html)
- [PettingZoo SB3 Tutorials](https://pettingzoo.farama.org/tutorials/sb3/index.html)
- [SuperSuit GitHub](https://github.com/Farama-Foundation/SuperSuit)
- [RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html)
- [RLlib Key Concepts](https://docs.ray.io/en/latest/rllib/key-concepts.html)
- [RLlib Multi-Agent Envs](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html)
- [MAPPO Paper](https://arxiv.org/abs/2103.01955)
