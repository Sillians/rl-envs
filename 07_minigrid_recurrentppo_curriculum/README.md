# 07 - MiniGrid RecurrentPPO + Curriculum (Sparse/Partial Observability)

## Difficulty
Advanced

## Goal
Handle sparse rewards and partial observability with memory-enabled policies and curriculum design.

## Environment and Algorithms
- Environment: MiniGrid tasks (`MiniGrid-Empty`, `MiniGrid-DoorKey`, `MiniGrid-KeyCorridor`)
- Algorithms: PPO baseline, RecurrentPPO (SB3-Contrib)
- Libraries: MiniGrid, Gymnasium, SB3, SB3-Contrib

## What You Will Learn
- Why memory matters in partially observable MDPs
- Curriculum scheduling across environment difficulty
- Observation wrappers for symbolic and image-based inputs

## Implementation Milestones
1. Train PPO on an easy MiniGrid task.
2. Switch to harder sparse-reward task and record failure modes.
3. Replace PPO with RecurrentPPO and compare.
4. Implement curriculum from easy -> medium -> hard maps.
5. Report sample efficiency and generalization to unseen seeds.

## Success Criteria
- Recurrent policy outperforms feed-forward PPO on hard tasks.
- Curriculum improves convergence speed and final success rate.

## Stretch Ideas
- Add intrinsic motivation bonus (RND/ICM) with custom rewards.
- Evaluate with random map generation.

## Helper Docs and Blogs
- [MiniGrid Documentation](https://minigrid.farama.org/)
- [MiniGrid Environments](https://minigrid.farama.org/environments/minigrid/index.html)
- [SB3 Contrib (RecurrentPPO)](https://sb3-contrib.readthedocs.io/)
- [SB3 PPO Docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Gymnasium Wrappers](https://gymnasium.farama.org/main/api/wrappers/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [RND Paper](https://arxiv.org/abs/1810.12894)
- [Curriculum Learning (Bengio et al.)](https://dl.acm.org/doi/10.1145/1553374.1553380)
