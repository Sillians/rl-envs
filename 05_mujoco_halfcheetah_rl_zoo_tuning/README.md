# 05 - MuJoCo HalfCheetah with SAC + RL Zoo Tuning

- [Half Cheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)
- [PPO Agent playing HalfCheetah-v3](https://huggingface.co/sb3/ppo-HalfCheetah-v3)

## Difficulty
Intermediate-Advanced

## Goal
Build a reproducible MuJoCo benchmark pipeline and tune SAC with RL Zoo + Optuna.

## Environment and Algorithms
- Environment: `HalfCheetah-v4` (MuJoCo)
- Algorithms: SAC (primary), optional TD3 baseline
- Libraries: Gymnasium MuJoCo, SB3, RL Baselines3 Zoo, Optuna

## What You Will Learn
- MuJoCo benchmarking discipline (seeds, eval splits, deterministic eval)
- Hyperparameter optimization workflows for RL
- Experiment reproducibility and result reporting

## Implementation Milestones
1. Run baseline SAC training with RL Zoo defaults.
2. Add Optuna study for core hyperparameters.
3. Re-train top-3 Optuna trials on fresh seeds.
4. Compare learning curves with confidence intervals.
5. Export best policy and record deterministic rollout videos.

## Success Criteria
- Demonstrable improvement over untuned baseline.
- Reproducible top config across multiple seeds.

## Stretch Ideas
- Use vectorized envs and benchmark wall-clock speed.
- Compare MuJoCo tasks: `Hopper-v4`, `Walker2d-v4`.

## Helper Docs and Blogs
- [Gymnasium MuJoCo Docs](https://gymnasium.farama.org/environments/mujoco/)
- [SB3 SAC Docs](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [RL Baselines3 Zoo Documentation](https://rl-baselines3-zoo.readthedocs.io/)
- [RL Zoo Hyperparameter Tuning](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html)
- [SB3 VecEnvs](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [SAC Paper](https://arxiv.org/abs/1812.05905)
- [Deep RL That Matters](https://arxiv.org/abs/1709.06560)
