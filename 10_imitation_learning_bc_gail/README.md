# 10 - Imitation Learning (Behavior Cloning + DAgger)

## Difficulty
Advanced

## Goal
Build imitation learning pipelines from expert trajectories, starting with behavior cloning and then improving robustness with DAgger-style aggregation.

## Environment and Algorithms
- Environments: `CartPole-v1`, `LunarLander-v3` (start simple)
- Algorithms: Behavior Cloning (BC), DAgger
- Libraries: SB3, Gymnasium, PyTorch

## What You Will Learn
- Collecting and cleaning demonstration data
- Covariate shift in pure behavior cloning
- How DAgger reduces compounding errors by relabeling learner states with expert actions

## Implementation Milestones
1. Train an expert PPO policy and record demonstrations.
2. Train BC policy and evaluate compounding errors.
3. Run DAgger rounds and compare against BC.
4. Perform trajectory-quality ablations (expert vs noisy demos).
5. Write a short analysis of sample efficiency and robustness.

## Success Criteria
- DAgger materially improves over BC on harder rollout horizons.
- Reproducible results across multiple random seeds.

## Stretch Ideas
- Add confidence-based expert querying to reduce labeling cost.
- Scale from CartPole to larger discrete-control tasks.

## Helper Docs and Blogs
- [SB3 PPO Docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Behavior Cloning Overview](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [DAgger Paper](https://arxiv.org/abs/1011.0686)
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)
