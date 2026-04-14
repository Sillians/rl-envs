# 02 - LunarLander PPO/A2C (Policy Gradient Fundamentals)

- [Lunar Lander Docs](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

## Difficulty
Beginner-Intermediate

## Goal
Learn on-policy optimization by training PPO and A2C agents on `LunarLander-v3` and comparing stability/performance.

## Environment and Algorithms

- Environment: `LunarLander-v3` (Box2D)
- Algorithms: PPO, A2C
- Libraries: Gymnasium, SB3, RL Baselines3 Zoo


## What You Will Learn

- On-policy rollout collection and policy/value losses
- Advantage estimation and clipping in PPO
- How entropy and value coefficients affect learning


## Implementation Milestones

1. Train PPO with SB3 defaults; capture training curves.
2. Train A2C with matched timesteps and compare.
3. Add 3-seed experiments for both algorithms.
4. Tune PPO (`n_steps`, `batch_size`, `learning_rate`, `gamma`).
5. Produce a short report with reward variance and failure cases.


## Success Criteria

- PPO achieves consistent positive return across seeds.
- Clear experimental conclusion on PPO vs A2C for this task.


## Stretch Ideas

- Add reward normalization and compare stability.
- Port best settings to `LunarLanderContinuous-v3`.


## How to use it

1. To record a video:

Run with the --record flag. It will create a ./videos directory containing .mp4 files of the lander in action.
```bash
uv run python 02_lunarlander_ppo_a2c/eval.py --model-path 02_lunarlander_ppo_a2c/artifacts/best_model/best_model.zip --record
```

2. To see the reward trend:

The script now automatically generates `evaluation_results.png`. This is useful for checking if your agent consistently clears the `+200 point` "solved" threshold or if it occasionally crashes (large negative spikes).

3. Human Observation:

If you just want to watch it live without saving files:

```bash
uv run python 02_lunarlander_ppo_a2c/eval.py --model-path 02_lunarlander_ppo_a2c/artifacts/best_model/best_model.zip --render
```

## Helper Docs and Blogs

- [LunarLander Docs](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [SB3 PPO Docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [SB3 A2C Docs](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)
- [SB3 Evaluation Helper](https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html)
- [RL Baselines3 Zoo Docs](https://rl-baselines3-zoo.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [A3C Paper (A2C Family)](https://arxiv.org/abs/1602.01783)
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/)
