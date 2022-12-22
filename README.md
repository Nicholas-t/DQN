
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

In this repository, we implemented DQN with the following architecture

<p align="center">
  <img src="https://github.com/Nicholas-t/DQN/blob/main/images/architecture.png"/>
</p>

With experience replay, I managed converge my model in less than 150 iterations of training loop.

<p align="center">
  <img src="https://github.com/Nicholas-t/DQN/blob/main/images/result.png"/>
</p>

And here is the an episode of the final model

<p align="center">
  <img src="https://github.com/Nicholas-t/DQN/blob/main/images/cartpole_dqn_with_replay.gif"/>
</p>

