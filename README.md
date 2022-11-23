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

<p align="center">
  <img src="https://github.com/Nicholas-t/M2_TSE_DSSS_Deep_Reinforcement_Learning_Notebook/blob/main/images/Breakout_game_screenshot.png" alt="Break ouu game"/>
  <a href="https://en.wikipedia.org/wiki/Breakout_(video_game)">source</a>
</p>



We can now define another common terminology which is the **policy** $\pi$ in the topic. A policy $\pi$ is a mapping from the state space to an action space that simply represents the agent to take an action $a$ from a state $s$ i.e.

$$
\begin{align*}
  \pi \colon S &\to A \\
  s &\mapsto a
\end{align*}
$$
_Remark : the policy function can be deterministic $\pi(s) = a$ or stochastic policy where $\pi(a|s) = \mathbb{P}[A=a | S=s]$_

Let us now define the notion of **value function** which is a function that represents a measure of how good of a state / action is for our agent and is represented by the expected of future rewards

$$
v_{\pi(s)} = \mathbb{E}_\pi[R_{t+1} + R_{t+2} + \dots | S_t = s]
$$

For our application, it is suffices to assume an agent is defined by their policy function $\pi$ and their value function $v$.

Reinforcement learning is the process optimizing an agent's action $a \in A$ with $A$ to be defined as the action space of our environment, and a given state $s\in S$ with $S$ to be the state space of our environment, and by optimizing it's policy $\pi(s, a)$ where $\pi(s, a) = Pr(a=a | s=s)$ the stochastic version of the policy, such that it maximizes the total rewards of the agent. In other words, reinforcement learning is a **maximization problem** where we **optimize** our agent's _policy_ to **maximize** future rewards.

Taking the chess as our example, the state space will contains all the possible legal positions of each pieces in the board. and the action space can be defined as a pair of $(p, l_target)$ where $p \in [King, Queen, Horse, etc]$ is the piece that is to be moved and $l_{target} \in \mathbb{R}^2$  is the location that the piece is being moved to.

An environment where we get an information about the reward after every action an agent take, is called **Dense** rewards, for example, minesweeper, on the other hand, environment such as playing a simple game of chess where we only receive rewards at the end of the game (whether we win or loses), is called **Sparse** rewards, naturally, the denser the rewards, the more efficient our data will be and therefore faster our learning will be.

## Markov Decision Process (MDP)

A state $S_t$ is Markov if and only if the distribution of future state can be fully observed simply by looking at the last state. In other words

$$
\mathbb{P}[S_{t+1}|S_1, \dots S_t] =\mathbb{P}[S_{t+1}| S_t]
$$

A markov process can then be defined by the tuple $(\mathcal{S}, \mathcal{P})$ where $\mathcal{S}$ is a finite set of states, and $\mathcal{P}$ to be the state transition probability matrix such that each element represents $P_{ss'} = \mathbb{P}[S_{t+1} = s' | S_t = s] $

Thus, we can now understand that an Markov Decision Process (MDP) is an extension of a markov process with 3 additional parameters:
- **$\mathcal{A}$** to be the finite set of actions
- **$\mathcal{R}$** to be the reward function defined on the action level s.t.
$$
R^a_s = \mathbb{E}[R_{t+1}|S_t=s, A_t = a]
$$
- **$\gamma$** to be the discount factor of future rewards defined on the $[0, 1] \subset \mathbb{R}$

*Remark: Another consequences of this, is that we dont take the future states with certainty but instead given a current state $s_t$, we assume that $s_{t+1}$ adopts a certain probability distribution. For example, any games that includes rolling a dice contains an element of randomness that originates from the random variable of the dice number.*

It is important now to define what is a **state-value function** and the **action-value** function. While both of them are quite similar in a sense that they are both defined on a given MDP, and both represents some sort of value, there is indeed a slight difference. 
- The **State value** function of and MDP is the expected return from state s when following policy $\pi$ at time $t$

$$
v_\pi(s) = \mathbb{E}_\pi[\overbrace{ \sum_{k=0}^\infty \gamma^k R_{t+k+1}}^\text{total future discounted rewards := $G_t$} | S_t = s]
$$


- where **Action value** function of and MDP is the expected return from state s when following policy $\pi$ **and taking action a** at time $t$.

$$
q_\pi(s, a) = \mathbb{E}_\pi[G_t| S_t = s, A_t = a]
$$

It is worth noting that the relationship between $v_\pi(s)$ and $q_\pi(s, a)$ are the following:

$$
v_\pi(s) = \sum_{a \in A} \pi(a|s) \times q_\pi(s, a)
$$

### Existence theorem

A theorem related to the MDP states that for any MDP:
- $\exists \pi^* s.t. v_{\pi^*}(s) \geq v_{\pi}(s) \forall s \in \mathcal{S}, \forall \pi $ where:
    - $v_{\pi^*}(s) = v^*(s)$
    - $q_{\pi^*}(s, a) = q^*(s, a)$

*Remark: Assuming we observe all of our environment, the $\pi^*$ can be found by a greedy algorithm that iteratively modifies $\pi_{t+1}(s) = \text{argmax}_{a\in\mathcal{A}} q_\pi(s, a)$ until there is no more improvements can be made or a stopping criterion has been attained*

### Estimation of state value function

We can estimate our state value function $v_\pi(s)$ using different methods, in particular we will discuss _Monte-Carlo_(MC) learning, and _Temporal-difference_(TD).


#### Monte-Carlo methods

Ever heard of **"learning by doing"** ? This is the simple idea behind every monte carlo method. With this method, we estimate the state value function as the average returns we get when visiting a particular state. 

In other words, it is an estimation method of our value function by updating the value function $V(s)$ with the empirical mean of the value of the states after each iteration. 

In a more concrete manner, the algorithm works by the following:

1. Initialize $\pi$, $V$, and $R$ nested array of returns history for every possible states $s$
2. Repeat:
    - Generate an episode following $\pi$
    - For every state $s$ in $S$ in the episode:
        - add returns of s in the episodes to the R(s)
        - update V(s) as the average of the R(s)

i.e. note that by law of large numbers, we have that $V(s) \rightarrow v_\pi(s)$

We can also recall that :


$$
\begin{aligned}
\mu_k &= \frac{1}{k} \sum_{j=1}^k x_j \\
&= \frac{1}{k} (x_k + \sum_{j=1}^{k-1} x_j) \\
&= \frac{1}{k} (x_k + (k-1) \mu_{k-1})\\
&= \mu_{k-1} + \frac{1}{k} (x_k - \mu_{k-1}) )\\
\end{aligned}
$$

Thus, this allow us to update the $V(s)$ incrementally by setting $V(s)$ at time t by $V(s) + \frac{1}{n_s} (G^s_t - V(s))$, normally we consider an $\alpha$ to be a learning rate and the new update formulation would be:

$$
V(s_t) = V(s_t) + \alpha (G^s_t - V(s_t))
$$

#### Temporal Differences

Temporal Differences uses the estimated return of the next steps in constrast with the MC methods that uses the discounted rewards of the state it self.

Tn other words at every iteration $V(S_t) = V(S_t) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$ with the following elements:
- $\gamma$ to be the discount factor of future values
- $R_{t+1}$ to be the rewards received at t+1 following the policy $\pi$
- $V(S_{t+1})$ to be the state value of being in the next state at time t+1

**Remark: $R_{t+1} + \gamma V(S_{t+1})$ is often called temporal difference target and it represents the estimated return consisting of future rewards and the value of being in the future state**

This method is the simplest form of TD usually noted as TD(0). The idea is to basically look into the immediate next step, at after being in the state S. 

To take this method further, we can look into the next n-step often also called TD($\lambda$). In other words when we consider n=1, 2, $\infty$:


$$
\begin{aligned}
n=1 & \rightarrow G_t^1 = R_{t+1} + \gamma V(S_{t+1})\\
n=2 & \rightarrow G_t^2 = R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2})\\
\vdots \\
n=\infty & \rightarrow G_t^\infty = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{T-1} R_T \implies MC
\end{aligned}
$$