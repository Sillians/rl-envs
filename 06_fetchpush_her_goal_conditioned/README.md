# 06 - FetchPush with HER (Goal-Conditioned RL)

## Difficulty
Advanced

## Goal
Solve sparse-reward robotics tasks by combining off-policy algorithms with **Hindsight Experience Replay (HER)**.

## Environment and Algorithms
- Environment: `FetchPush-v3` (Gymnasium-Robotics)
- Algorithms: HER + SAC/TD3
- Libraries: Gymnasium-Robotics, Gymnasium, SB3

## What You Will Learn
- Goal-conditioned observations and achieved/desired goals
- Sparse reward handling with HER relabeling
- Replay strategy design and relabeling ratios

## Implementation Milestones
1. Install and verify Gymnasium-Robotics environment.
2. Train a no-HER baseline and measure failure rate.
3. Add HER replay buffer and tune relabeling parameters.
4. Compare SAC+HER and TD3+HER.
5. Evaluate success rate and path quality across seeds.

## Success Criteria
- Success rate > 70% on deterministic evaluation.
- Clear ablation showing HER impact vs no-HER baseline.

## Stretch Ideas
- Move from `FetchReach-v3` to `FetchPush-v3`/`FetchSlide-v3`.
- Explore curriculum over goal distances.

## Helper Docs and Blogs
- [Gymnasium-Robotics Docs](https://robotics.farama.org/)
- [Fetch Environment Docs](https://robotics.farama.org/envs/fetch/)
- [Multi-goal API](https://robotics.farama.org/content/multi-goal_api/)
- [SB3 HER Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/her.html)
- [SB3 SAC Docs](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [SB3 TD3 Docs](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
- [HER Paper](https://arxiv.org/abs/1707.01495)
- [Goal-Conditioned RL Overview](https://spinningup.openai.com/en/latest/)
