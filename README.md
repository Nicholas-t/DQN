# Deep Reinforcement Learning 

Dependencies: 
- Python 3.7 (Tensorflow 1.15 only available at this version)
- PIP
- Install python libraries
```
python3.7 -m pip install numpy pygame stable_baselines3 tensorflow==1.15.0 gym==0.21.0 Box2D stable-baselines3[extra]
```

## Introduction 

Deep reinforcement learning has been a prevelant topic among Researchers and Professionals, it event reaches the mainstream's radar (for the more tech-savy ones) when in 2015, [https://www.deepmind.com/](DeepMind) reaches a milestone for the topic by creating [https://www.deepmind.com/research/highlighted-research/alphago](AlphaGo) and challenges the world's best human Go player from Korea, Lee Sedol, noting that the game Go was supposed to be more complex than the game chess and WON. It was an amazing feat to showcase what the future of the space may hold.

In order to better understand the idea of reinforcement learning, lets take an example of a game of chess. Theoritically speaking, one can create an brute force algorithm that analyze all possible states of the game and takes into account all the possible states that can come after the current state, and take the action that give the most probability of winning at the end. However, this method is very inefficient. In 2021, [John Tromp](https://github.com/tromp/ChessPositionRanking) did an analysis and found that there are about $4.8 x 10^{44}$ possible combinations of legal chess positions in chess. For perspective, here is the full form of that number

$$480000000000000000000000000000000000000000000$$

So for sure, this process is very inefficient. This is where Reinforcement learning comes in handy.

In this repository we will methodicallt explore topics in reinforcement learning, we will first discuss the definitions and the frameworks often used in RL courses you see online, then, we will explore a little bit about what is a Markov Decision Process (MDP), then, we will talk about how we can approximate our value function using Q learning and DQN, additionally, we will explore as well some methods from the [rainbow paper](https://arxiv.org/pdf/1710.02298.pdf) and [policy gradient](https://www.davidsilver.uk/wp-content/uploads/2020/03/pg.pdf)

## Definitions and  Framework of Rinforcement Learning

We firstly define :
- **State space** $S$ to be the space of all the parameters of our environment that can help us to understand our environment better
- **Action space** $A$ to be the space of all possible action we can take in a given environment. 

the classic example that we often see to better understand this notion is for example the atari game Breakout. Our agent (the paddle) will need the state space to be able to let it understand the environment best, therefore $S$ will be the tuple of all the coordinates of bricks that are still not yet hit by the ball, the coordinate of the ball, and the coordinate of the paddle. and the set $A$ is simply a set of 2 actions whether the paddle to go left or right.

![Breakout_game_screenshot.png](attachment:Breakout_game_screenshot.png)
Source : [Wikipedia](https://en.wikipedia.org/wiki/Breakout_(video_game))