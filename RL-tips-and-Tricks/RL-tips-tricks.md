# Reinforecement Learning Tips and Tricks

- [Reinforcement Learning Tips and Tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

- [RLVS 2021 - Day 6 - RL in practice: tips & tricks and practical session with stable-baselines3](https://www.youtube.com/watch?v=Ikngt0_DXJg)
- [RL Tips and Tricks](https://araffin.github.io/slides/rlvs-tips-tricks/)


- [Designing and Running Real World RL Experiments. Antonin Raffin](https://www.youtube.com/watch?v=eZ6ZEpCi6D8)
- [Designing and Running Real-World Reinforcement Learning Experiments](https://araffin.github.io/slides/design-real-rl-experiments/)


The aim of this section is to help you run reinforcement learning experiments. It covers general advice about RL (where to start, which algorithm to choose, how to evaluate an algorithm, …), as well as tips and tricks when using a custom environment or implementing an RL algorithm.


## General advice when using Reinforcement Learning

- Start with a simple environment and a simple algorithm. For example, start with CartPole and DQN or PPO. This will help you understand the basics of RL and how to debug your code.

- Use a well-known library like Stable-Baselines3, RLlib, or d3rlpy. These libraries have been tested and optimized, and they provide a lot of useful features like logging, evaluation, and hyperparameter tuning.

- Use a consistent evaluation protocol. For example, evaluate your algorithm every 1000 steps and plot the learning curve. This will help you track the progress of your algorithm and compare it with other algorithms.

- Be patient. RL can be slow to train, especially in complex environments. It may take a long time to see significant improvements in performance, so be prepared to wait and experiment with different hyperparameters and algorithms.

- Use a good random seed. RL algorithms can be sensitive to the choice of random seed, so make sure to use a good random seed and to report it in your experiments. This will help ensure that your results are reproducible and that others can compare their results with yours.



### TL;DR

1. Read about RL and Stable-Baselines3 (SB3)
2. Do quantitative experiments and hyperparameter tuning if needed
3. Evaluate the performance using a separate test environment (remember to check wrappers!)
4. For better performance, increase the training budget


Reinforcement Learning differs from other machine learning methods in several ways. The data used to train the agent is collected through interactions with the environment by the agent itself (as opposed to, for example, supervised learning where you have a fixed dataset). This dependency can lead to a vicious circle: if the agent collects poor quality data (e.g. trajectories with no rewards), it will not improve and will continue to collect bad trajectories.

This factor, among others, explains that results in RL may vary from one run to another (i.e., when only the seed of the pseudo-random generator changes). For this reason, you should always do several runs to obtain quantitative results.

Good results in RL generally depend on finding appropriate hyperparameters. Recent algorithms (PPO, SAC, TD3, DroQ) normally require little hyperparameter tuning, however, don’t expect the default ones to work in every environment.

Therefore, we highly recommend you to take a look at the RL zoo (or the original papers) for tuned hyperparameters. A best practice when you apply RL to a new problem is to do automatic hyperparameter optimization.

When applying RL to a custom problem, you should always normalize the input to the agent (e.g. using `VecNormalize` for PPO/A2C) and look at common preprocessing done on other environments (e.g. for Atari, frame-stack, …).

Finally, if you want to achieve better performance, you can increase the training budget (e.g. number of steps, number of parallel environments, …). This is often the most effective way to improve performance, especially when you are using a well-tuned algorithm and hyperparameters.


### Current Limitations of RL

- `Sample inefficiency:` RL algorithms often require a large number of interactions with the environment to learn effectively, which can be impractical in real-world applications.

- `Reward design:` Designing a reward function that encourages the desired behavior can be challenging, and poorly designed rewards can lead to unintended consequences.

- `Exploration vs. exploitation:` Balancing exploration and exploitation is a fundamental challenge in RL, and finding the right balance can be difficult, especially in complex environments.

- `Stability and convergence:` RL algorithms can be unstable and may not converge to an optimal policy, especially in high-dimensional state spaces or with function approximation.

- `Generalization:` RL agents may struggle to generalize to new environments or tasks that differ from the training environment, which can limit their applicability in real-world scenarios.

- `Partial observability:` Many real-world environments are partially observable, meaning the agent does not have access to the full state of the environment, which can make learning more difficult.

- `Multi-agent interactions:` In environments with multiple agents, the presence of other learning agents can create a non-stationary environment, making it challenging for agents to learn effectively.


`Model-free RL algorithms` (i.e. all the algorithms implemented in SB3) are usually sample inefficient. They require a lot of samples (sometimes millions of interactions) to learn anything useful. That’s why most of the successes in RL were achieved on games or in simulation only. For instance, in this work by `ETH Zurich`, the `ANYmal` robot was trained in simulation only, and then tested in the real world.


As a general advice, to obtain better performances, you should augment the budget of the agent (number of training timesteps). 

In order to achieve the desired behavior, expert knowledge is often required to design an adequate reward function. Finding the right balance between exploration and exploitation is a fundamental challenge in RL, and it can be difficult to achieve, especially in complex environments. RL algorithms can be unstable and may not converge to an optimal policy, especially in high-dimensional state spaces or with function approximation. RL agents may struggle to generalize to new environments or tasks that differ from the training environment, which can limit their applicability in real-world scenarios. Many real-world environments are partially observable, meaning the agent does not have access to the full state of the environment, which can make learning more difficult. In environments with multiple agents, the presence of other learning agents can create a non-stationary environment, making it challenging for agents to learn effectively.

A final limitation of RL is the instability of training. That is, you can observe a huge drop in performance during training. This behavior is particularly present in `DDPG`, that’s why its extension `TD3` tries to tackle that issue. Other methods, such as `TRPO` or `PPO` use a trust region to minimize this problem by avoiding too large updates.



### How to evaluate an RL algorithm?

When evaluating an RL algorithm, it is important to use a separate test environment that is not used during training. This will help you assess the generalization performance of your algorithm. Additionally, you should be careful when using wrappers, as they can affect the evaluation results. For example, if you use a wrapper that normalizes the observations during training, you should also apply the same normalization during evaluation to ensure consistency.

Because most algorithms use exploration noise during training, you need a separate test environment to evaluate the performance of your agent at a given time. It is recommended to periodically evaluate your agent for `n` test episodes (`n` is usually between 5 and 20) and average the reward per episode to have a good estimate.


As some policies are stochastic by default (e.g. A2C or PPO), you should also try to set `deterministic=True` when calling the `.predict()` method, this frequently leads to better performance. Looking at the training curve (episode reward function of the timesteps) is a good proxy but underestimates the agent true performance.

`stability` and `convergence` are common issues in RL, especially in high-dimensional state spaces or with function approximation. To mitigate these issues, you can try using techniques such as experience replay, target networks, or regularization methods. Additionally, you can experiment with different algorithms or hyperparameters to find a more stable learning process.



### Which algorithm should I use?

The choice of algorithm depends on the specific problem you are trying to solve, the characteristics of the environment, and your computational resources. Here are some general guidelines:

- For discrete action spaces, you can start with `DQN` or `PPO`.
- For continuous action spaces, you can start with `PPO`, `SAC`, or `TD3`.
- If you have a large state space and want to use function approximation, you can consider using algorithms like `DDPG`, `SAC`, or `PPO`.
- If you have a partially observable environment, you can consider using algorithms that incorporate memory, such as `RNN`-based architectures or recurrent versions of `PPO`.
- If you have a multi-agent environment, you can consider using algorithms designed for multi-agent settings, such as `MADDPG` or multi-agent `PPO`.



There is no silver bullet in RL, you can choose one or the other depending on your needs and problems. The first distinction comes from your action space, i.e., do you have discrete (e.g. LEFT, RIGHT, …) or continuous actions (ex: go to a certain speed)?

Some algorithms are only tailored for one or the other domain: 
- `DQN` supports only discrete actions, while 
- `SAC` is restricted to continuous actions.


The second difference that will help you decide is whether you can parallelize your training or not. If what matters is the wall clock training time, then you should lean towards A2C and its derivatives (PPO, …). 

To accelerate training, you can also take a look at SBX, which is SB3 + Jax, it has less features than SB3 but can be up to 20x faster than SB3 PyTorch thanks to JIT compilation of the gradient update.


### Discrete Actions

This covers `Discrete`, `MultiDiscrete`, `Binary` and `MultiBinary` spaces

**Discrete Actions - Single Process**
- `DQN` (Deep Q-Network): A value-based algorithm that uses a neural network to approximate the Q-function. It is suitable for environments with discrete action spaces and can be sample efficient, but it may struggle with large state spaces.

DQN is usually slower to train (regarding wall clock time) but is the most sample efficient (because of its replay buffer).

**Discrete Actions - Multiprocessed**

- `PPO` (Proximal Policy Optimization): A policy gradient algorithm that optimizes a surrogate objective function. It is suitable for both discrete and continuous action spaces and can be parallelized across multiple environments, making it faster to train. PPO is usually faster to train (regarding wall clock time) but is less sample efficient than DQN.

- `A2C` (Advantage Actor-Critic): A synchronous version of the A3C algorithm that uses multiple parallel environments to update the policy. It is suitable for both discrete and continuous action spaces and can be faster to train than DQN, but it may be less sample efficient.


### Continuous Actions
This covers `Box` spaces

**Continuous Actions - Single Process**
Current State Of The Art (SOTA) algorithms are `SAC`, `TD3`, `CrossQ` and `TQC`. They are all off-policy algorithms, which means they can be sample efficient, but they may require more careful tuning and may be less stable than on-policy algorithms like `PPO`.

**Continuous Actions - Multiprocessed**

- `PPO` (Proximal Policy Optimization): A policy gradient algorithm that optimizes a surrogate objective function. It is suitable for both discrete and continuous action spaces and can be parallelized across multiple environments, making it faster to train. PPO is usually faster to train (regarding wall clock time) but is less sample efficient than off-policy algorithms like `SAC` or `TD3`.

- `TRPO` (Trust Region Policy Optimization): An on-policy algorithm that optimizes a surrogate objective function with a trust region constraint. It is suitable for both discrete and continuous action spaces and can be parallelized across multiple environments, but it may be slower to train than PPO.

- `A2C` (Advantage Actor-Critic): A synchronous version of the A3C algorithm that uses multiple parallel environments to update the policy. It is suitable for both discrete and continuous action spaces and can be faster to train than `DQN`, but it may be less sample efficient than off-policy algorithms like `SAC` or `TD3`.


**Goal Environment**
- `HER` (Hindsight Experience Replay): An off-policy algorithm that is designed for goal-conditioned environments. It can be used with any off-policy algorithm (e.g., DQN, SAC) and is particularly effective in environments with sparse rewards.

If your environment follows the `GoalEnv` interface (cf `HER`), then you should use `HER` + `(SAC/TD3/DDPG/DQN/QR-DQN/TQC)` depending on the action space.



## Tips and Tricks when creating a custom environment

When creating a custom environment, it is important to follow the `OpenAI Gymnasium API` and to ensure that your environment is compatible with the RL algorithms you want to use. Here are some tips and tricks to keep in mind:

- Make sure to implement the `reset()` and `step()` methods correctly, as these are the core methods that RL algorithms interact with.

- Use the `render()` method to visualize your environment and debug any issues with the state transitions or reward structure.

- Consider using `wrappers` to add additional functionality to your environment, such as `observation normalization`, `action clipping`, or `reward shaping`.

- Test your environment with a simple random agent to ensure that it behaves as expected before training a more complex RL agent.

- Use the `check_env()` function from Gymnasium to validate that your environment follows the API and does not have any common issues that could hinder learning.

- If your environment has a large state space, consider using function approximation (e.g., neural networks) to represent the value function or policy, and ensure that your environment provides sufficient information for the agent to learn effectively.

- If your environment is partially observable, consider using recurrent architectures or memory-based approaches to help the agent learn from past observations and actions.

- If your environment has a multi-agent setting, consider using algorithms designed for multi-agent environments and ensure that your environment properly handles interactions between agents.


Two important things to keep in mind when creating a custom environment are avoiding breaking the Markov assumption and properly handle termination due to a timeout (maximum number of steps in an episode). For example, if there is a time delay between action and observation (e.g. due to wifi communication), you should provide a history of observations as input.

When using a custom environment, it is also important to ensure that it is compatible with the RL algorithms you want to use. For example, if you want to use `HER`, your environment should follow the `GoalEnv` interface. Additionally, you should test your environment with a simple random agent to ensure that it behaves as expected before training a more complex RL agent. Finally, you can use the `check_env()` function from Gymnasium to validate that your environment follows the API and does not have any common issues that could hinder learning.

Termination due to timeout (max number of steps per episode) needs to be handled separately. You should return `truncated = True`. If you are using the gym TimeLimit wrapper, this will be done automatically. 



## Tips and Tricks when implementing an RL algorithm

When implementing an RL algorithm, it is important to follow the original paper and to ensure that your implementation is correct and efficient. Here are some tips and tricks to keep in mind:

- Read the original paper several times to understand the algorithm and its components. Pay attention to the details of the algorithm, such as the loss functions, update rules, and hyperparameters.

- Read existing implementations (if available) to see how others have implemented the algorithm and to get ideas for optimization and debugging.

- Try to have some “sign of life” on toy problems (e.g. CartPole) before applying your implementation to more complex environments. This will help you catch any bugs or issues early on and ensure that your implementation is working correctly.

- Validate the implementation by making it run on harder and harder envs (you can compare results against the RL zoo). You usually need to run hyperparameter optimization for that step. 

- Use a consistent evaluation protocol to track the progress of your algorithm and compare it with other algorithms. For example, evaluate your algorithm every 1000 steps and plot the learning curve.

- Use logging and debugging tools to monitor the training process and identify any issues or bottlenecks in your implementation. This can include tools like TensorBoard, Weights & Biases, or custom logging functions.

- Optimize your implementation for efficiency, especially if you are working with large state spaces or complex environments. This can include techniques like vectorized environments, parallel processing, or using efficient data structures for experience replay buffers.

- Be patient and persistent. Implementing an RL algorithm can be challenging, and it may take time to get it working correctly. Don’t be discouraged by initial failures, and keep iterating and improving your implementation until you achieve the desired performance.



You need to be particularly careful on the shape of the different objects you are manipulating (observations, actions, rewards, …) and on the way you compute the loss function. It is a good practice to print the shape of the different tensors at each step of your implementation to ensure that they are correct. Additionally, you can use debugging tools like `pdb` or `TensorBoard` to visualize the training process and identify any issues with your implementation.

Don’t forget to handle termination due to timeout separately (max number of steps per episode). You should return `truncated = True`. If you are using the gym TimeLimit wrapper, this will be done automatically.
