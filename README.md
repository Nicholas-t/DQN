
<p align="center">
  <img src="https://github.com/Nicholas-t/DQN/blob/main/images/banner.png"/>
</p>


# DQN on the CartPole gym environment

This repository contains an implementation of Deep Q Network using PyTorch specifically to solve the CartPole environment.

## Environment and dependencies

- OS : Ubuntu 20.04.2 LTS
- Python 3.7
- PIP

if you have python version above 3.7, then refer to [this](https://phoenixnap.com/kb/how-to-install-python-3-ubuntu) tutorial on how you can install a specific python version in your ubuntu machine


## Organization

All of the code regarding the DQN is in the `./scripts` folder. The `notebook.ipynb` contains explanations regarding DQN and more context regarding the algorithm of interest.

## How to install ?

You can clone the repository and install the necesarry dependencies

```
git clone https://github.com/Nicholas-t/DQN
cd DQN
python3.7 -m pip install -r requirements.txt
```

## How to run ?

If you have jupyter notebook, you can simply run the cells that included the code.

## Plan

1. Motivation
2. Definitions and  Framework of Reinforcement Learning
3. Markov Decision Process (MDP)
4. Estimation of state value function
  - Monte Carlo methods
  - Temporal Differences
5. Estimation of action value function with Q-value
6. Deep Q Network
  - Simple DQN
  - DQN with experience replay
7. Conclusion

## Overview and results

In our notebook, we first explore the state value function with 2 methods, which are:

1. Monte-Carlo methods
2. Temporal Differences

Then we explored action-state value and try to visualize them with the following plot.

<p align="center">
  <img src="https://github.com/Nicholas-t/DQN/blob/main/images/action-state-value.png"/>
</p>


Then, we implemented DQN with the following architecture

<p align="center">
  <img src="https://github.com/Nicholas-t/DQN/blob/main/images/architecture.png"/>
</p>

And after training a simple DQN model we have the following reward evolution:

<p align="center">
  <img src="https://github.com/Nicholas-t/DQN/blob/main/images/result-0.png"/>
</p>

It didnt converge because we have huge correlation between states which is an issue mentioned by the author of the DQN paper.


> Second, learning directly from consecutive samples is inefficient, due to the strong correlations between the samples; randomizing the samples breaks these correlations and therefore reduces the variance of the updates.



With experience replay, I managed to converge my **DQN** model in less than 150 iterations of training loop.

<p align="center">
  <img src="https://github.com/Nicholas-t/DQN/blob/main/images/result.png"/>
</p>

And here is the an episode of the final model

<p align="center">
  <img src="https://github.com/Nicholas-t/DQN/blob/main/images/cartpole_dqn_with_replay.gif"/>
</p>

After testing with DQN we implemented **Double-DQN** with replay and it seems to be performing better in a sense that it converges after ~110 training iterationin contrast to DQN with replay which converges after about 150 iterations

<p align="center">
  <img src="https://github.com/Nicholas-t/DQN/blob/main/images/result-2.png"/>
</p>

And here is the an episode of the DDQN final model

<p align="center">
  <img src="https://github.com/Nicholas-t/DQN/blob/main/images/cartpole_double_dqn.gif"/>
</p>


# Credits

1. Some codes are inspired from the [Practical Data Science](https://github.com/ritakurban/Practical-Data-Science/blob/master/DQL_CartPole.ipynb) However, I needed to modify the code to make it suitable for my repository architecture

2. [Playing Atari with Deep Reinforcement Learning - Mnih, Kavukcuoglu, Silver, Graves, Antonoglou, Wierstra, Riedmiller](https://arxiv.org/pdf/1312.5602.pdf)

3. [Deepmind RL with David Silver](https://www.deepmind.com/learning-resources/introduction-to-reinforcement-learning-with-david-silver)