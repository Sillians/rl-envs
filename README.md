# Reinforcement Learning Learning Track (Gymnasium + SB3 + SOTA Libraries)

This workspace now contains **10 project folders** arranged from beginner to advanced.
Each project has a detailed `README.md` with goals, milestones, evaluation criteria, and helper docs/blogs.

## Suggested Order

1. `01_cartpole_dqn_sb3_basics`
2. `02_lunarlander_ppo_a2c`
3. `03_bipedalwalker_sac_td3`
4. `04_atari_breakout_dqn_cleanrl`
5. `05_mujoco_halfcheetah_rl_zoo_tuning`
6. `06_fetchpush_her_goal_conditioned`
7. `07_minigrid_recurrentppo_curriculum`
8. `08_pettingzoo_multiagent_parameter_sharing`
9. `09_offline_rl_minari_d3rlpy`
10. `10_imitation_learning_bc_gail`

## Project Map

- **01**: Discrete control fundamentals with DQN (CartPole).
- **02**: Policy gradient methods on Box2D (LunarLander) with PPO/A2C.
- **03**: Continuous control with SAC/TD3 (BipedalWalker).
- **04**: Pixel observations and Atari preprocessing (Breakout).
- **05**: MuJoCo benchmarking + hyperparameter tuning (HalfCheetah).
- **06**: Sparse-reward robotics via HER + goal-conditioned RL (FetchPush).
- **07**: Partial observability + memory/curriculum in MiniGrid.
- **08**: Multi-agent RL with PettingZoo and scaling with RLlib.
- **09**: Offline RL using Minari datasets and d3rlpy (CQL/IQL).
- **10**: Imitation learning pipelines (BC + DAgger).

## Core Libraries Used Across Projects

- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [SB3 Contrib](https://sb3-contrib.readthedocs.io/)
- [Gymnasium-Robotics](https://robotics.farama.org/)
- [MiniGrid](https://minigrid.farama.org/)
- [PettingZoo](https://pettingzoo.farama.org/)
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)
- [Minari](https://minari.farama.org/)
- [d3rlpy](https://d3rlpy.readthedocs.io/)
- [PyTorch](https://pytorch.org/)

## Environment Setup (uv)

1. `cd /Users/user/Projects/rl-envs`
2. `uv lock` (refreshes `uv.lock` when dependencies change)
3. `uv sync` (creates/updates `.venv` and installs from `uv.lock`)
4. Run scripts with `uv run`, for example:
   - `uv run python 01_cartpole_dqn_sb3_basics/train.py`
   - `uv run python 01_cartpole_dqn_sb3_basics/eval.py --model-path 01_cartpole_dqn_sb3_basics/artifacts/best_model/best_model`

## Recommended General Study References

- [Gymnasium Basic Usage](https://gymnasium.farama.org/main/introduction/basic_usage/)
- [SB3 RL Tips and Tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/)
- [David Silver RL Course](https://davidstarsilver.wordpress.com/teaching/)

## Practical Note

Use the single project environment managed by `uv` for all folders and log experiments with TensorBoard or Weights & Biases.
For projects 8-10, expect longer training runs and higher compute requirements.
