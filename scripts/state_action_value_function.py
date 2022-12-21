import gym
import matplotlib.pyplot as plt
import numpy as np
import collections
import seaborn as sns

import pandas as pd
import time
import bisect
from scripts.utils import convert_state

def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    def policyFunction(state):
        Action_probabilities = np.ones(num_actions,
                dtype = float) * epsilon / num_actions
        best_action = np.argmax(Q[convert_state(state)])
        
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities
   
    return policyFunction

def getActionFromPolicy(policy, state):
    action_probabilities = policy(state)
    action = np.random.choice(np.arange(
              len(action_probabilities)),
                       p = action_probabilities)
    return action


def qLearning(env, num_episodes, discount_factor = 1.0,
                            alpha = 0.6, epsilon = 0.1):
    Q = collections.defaultdict(lambda: np.zeros(env.action_space.n))
    
    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)
    epsGreedy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n)
    for ith_episode in range(num_episodes):
           
        state = env.reset()
        t = 0
        while True:
            t+=1
            action= getActionFromPolicy(epsGreedy, state)
            
            next_state, reward, done, _ = env.step(action)
   
            episode_rewards[ith_episode] += reward
            episode_lengths[ith_episode] = t
            
            best_next_action = np.argmax(Q[convert_state(next_state)])    
            td_target = reward + discount_factor * Q[convert_state(next_state)][best_next_action]
            td_delta = td_target - Q[convert_state(state)][action]
            Q[convert_state(state)][action] += alpha * td_delta
   
            if done:
                break
            state = next_state
    return Q, (episode_lengths, episode_rewards)



def plot_q(ENVIRONMENT="CartPole-v1", iterations = 50000, title = "Epsilon Greedy policy"):
    env = gym.make(ENVIRONMENT)
    Q, stats = qLearning(env, iterations)    
    ranges = np.arange(-4.8, 4.8, 0.01)
    q_values = np.zeros((2, len(ranges)))
    pole_angs = []
    pole_ang_vels = []
    pos = []
    direction = []
    for (state, value) in Q.items():
        cart_pos, cart_vel, pole_ang, pole_ang_vel = [float(s) for s in state.split(",")]
        pole_angs += [pole_ang]
        pole_ang_vels += [pole_ang_vel]
        pos += [cart_pos]
        if value[0] > value[1]:
            direction += [0]
        else:
            direction += [1]
        bin_x = bisect.bisect_left(ranges, cart_pos)
        for i in range(len(q_values)):
            q_values[i][bin_x] += value[i]

    """
    q_left, q_right = q_values

    plt.figure(figsize=(13, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(pole_angs, pole_ang_vels, c = direction)
    plt.title("")
    
    plt.subplot(1, 2, 2)
    plt.plot(ranges, q_right)
    plt.xlim((-1,1))
    plt.title("Right Action Value")
    """

    plt.figure(figsize=(17, 6))
    
    plt.subplot(1, 3, 1)
    plt.scatter(pole_angs, pole_ang_vels, c = direction)
    plt.xlabel("Pole Angle")
    plt.ylabel("Pole Angle Velocity")
    plt.title("Pole Angle vs Pole Angle Velocity")
    
    plt.subplot(1, 3, 2)
    plt.scatter(pole_angs, pos, c = direction)
    plt.xlabel("Pole Angle")
    plt.ylabel("Pole Position")
    plt.title("Pole Angle vs Position")

    
    plt.subplot(1, 3, 3)
    plt.scatter(pole_ang_vels, pos, c = direction)
    plt.xlabel("Pole Angle Velocity")
    plt.ylabel("Pole Position")
    plt.title("Pole Angle Velocity vs Position")

    plt.show()
    