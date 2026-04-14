# 03 - BipedalWalker SAC/TD3 (Continuous Control)

- [Bipedal Walker](https://gymnasium.farama.org/environments/box2d/bipedal_walker/) 

is a classic continuous control environment where an agent learns to walk using two legs. This project focuses on mastering `Soft Actor-Critic (SAC)` and `Twin Delayed DDPG (TD3)` algorithms to achieve stable walking behavior.

## Difficulty
Intermediate

## Goal
Master continuous-action RL by training and tuning SAC/TD3 on `BipedalWalker-v3`.


## Environment and Algorithms
- Environment: `BipedalWalker-v3` (Box2D continuous control)
- Algorithms: SAC, TD3
- Libraries: Gymnasium, SB3, RL Zoo, Optuna (optional)


## What You Will Learn
- Actor-critic for continuous actions
- Entropy regularization (SAC) vs delayed policy updates (TD3)
- Reward scaling, action noise, and normalization


## Implementation Milestones
1. Train SAC baseline and save best checkpoint.
2. Train TD3 baseline with action noise.
3. Evaluate both over 20+ fixed-seed episodes.
4. Tune hyperparameters for sample efficiency.
5. Compare robustness under changed random seeds.


## Success Criteria
- At least one algorithm reaches stable walking behavior.
- Reproducible metric table across >= 3 seeds.


## Stretch Ideas
- Evaluate on `BipedalWalkerHardcore-v3`.
- Add checkpoint ensemble evaluation.


## How to run
1. Train SAC: `uv run python 03_bipedalwalker_sac_td3/train.py --algo sac`
2. Train TD3: `uv run python 03_bipedalwalker_sac_td3/train.py --algo td3`

3. Evaluate: `uv run python 03_bipedalwalker_sac_td3/eval.py --algo sac --model-path artifacts/sac/best_model/best_model`
4. Evaluate: `uv run python 03_bipedalwalker_sac_td3/eval.py --algo td3 --model-path artifacts/td3/best_model/best_model`

5. Evaluate with video recording: `uv run python 03_bipedalwalker_sac_td3/eval.py --algo sac --model-path artifacts/sac/best_model/best_model --render`
6. Evaluate with video recording: `uv run python 03_bipedalwalker_sac_td3/eval.py --algo td3 --model-path artifacts/td3/best_model/best_model --render`



## Key Insights for BipedalWalker Visualization

1. `Gait Symmetry:` In the `gait_analysis.png` plot, look for periodic sine-wave patterns. If the lines are erratic, the walker is "stumbling." If they are smooth and repeating, your agent has discovered a stable walking gait.

2. `The Hull Observation:` BipedalWalker's observation space includes `10 LIDAR sensors` (indices 14–23). If you want to visualize how the walker "sees" the terrain (especially for the `Hardcore` version), you can plot these LIDAR values as a bar chart to see obstacles appearing in front of it.

3. `Video Recording:` Since the walker moves across a scrolling terrain, using `--record` is much more effective than static screenshots. You can watch the video to see if the walker is "knees-forward" or if it has developed a strange "stiff-legged" hop.


## Helper Docs and Blogs
- [BipedalWalker Docs](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)
- [SB3 SAC Docs](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [SB3 TD3 Docs](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
- [SB3 RL Tips and Tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
- [RL Baselines3 Zoo](https://rl-baselines3-zoo.readthedocs.io/)
- [Soft Actor-Critic Paper](https://arxiv.org/abs/1812.05905)
- [TD3 Paper](https://arxiv.org/abs/1802.09477)
- [Spinning Up Algorithm Guides](https://spinningup.openai.com/en/latest/)
