import gym
import matplotlib.pyplot as plt
import numpy as np
import collections
import seaborn as sns

import pandas as pd
import time
import bisect
from scripts.utils import convert_state


def visualize_cart_distribution(episode: list):
    df = pd.DataFrame({"pole": [s[0] for s in episode]})
    sns.displot(df, x="pole", kind="kde", bw_adjust=0.5)


def play_random_policy_episode(env):
    episode = []
    state = env.reset()
    while True:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def MC(env, discount_factor: float = 1.0, iterations: int = 100):
    vfn = {}
    states_count = {}
    returns = {}
    for i in range(iterations):
        episode = play_random_policy_episode(env)
        G = 0
        viewed_state = []
        for t, (_state, action, reward) in enumerate(episode):
            state = convert_state(_state)
            G = G * discount_factor + reward
            if not state in viewed_state:
                viewed_state += [state]
                returns[state] = G
                states_count[state] = 1
            else:
                returns[state] += G
                states_count[state] += 1
            vfn[state] = returns[state] / states_count[state]
    return vfn


def TD(
    env, discount_factor: float = 1.0, iterations: int = 100, alpha: float = 0.01
) -> list:
    vfn = collections.defaultdict(float)
    for i in range(iterations):
        episode = play_random_policy_episode(env)
        for t, (_state, action, reward) in enumerate(episode[1:]):
            state = convert_state(_state)
            old_state = convert_state(episode[t - 1][0])
            if old_state not in vfn.keys():
                vfn[old_state] = 0
            if state not in vfn.keys():
                vfn[state] = 0
            vfn[old_state] += alpha * (reward + discount_factor * reward - vfn[state])
    return vfn


def plot_value_function(
    env,
    estimator,
    iterations: int = 50000,
    title: str = "Value function",
    discount_factor: float = 1.0,
    alpha: float = 0.01,
):
    vfn = None
    if "Monte-carlo" in title:
        vfn = estimator(env, iterations=iterations, discount_factor=discount_factor)
    else:
        vfn = estimator(
            env, iterations=iterations, discount_factor=discount_factor, alpha=alpha
        )
    ranges = np.arange(-4.8, 4.8, 0.01)

    value_location = np.zeros(len(ranges))
    num_location = np.zeros(len(ranges))

    for (state, value) in vfn.items():
        cart_pos, cart_vel, pole_ang, pole_ang_vel = [
            float(s) for s in state.split(",")
        ]
        bin_x = bisect.bisect_left(ranges, cart_pos)
        num_location[bin_x] += 1
        value_location[bin_x] += value
    plt.plot(ranges, value_location)
    plt.title(title)
    plt.xlim((-1, 1))
    plt.show()
    return value_location


def plot_value_function_mc(
    ENVIRONMENT: str = "CartPole-v1",
    ITERATIONS: int = 2000,
    discount_factor: float = 1.0,
):
    env = gym.make(ENVIRONMENT)
    _ = plot_value_function(
        env,
        MC,
        iterations=ITERATIONS,
        title="Monte-carlo ({} iterations) - env:{}".format(ITERATIONS, ENVIRONMENT),
        discount_factor=discount_factor,
    )


def plot_value_function_td(
    ENVIRONMENT: str = "CartPole-v1",
    ITERATIONS: int = 2000,
    discount_factor: float = 1.0,
    alpha: float = 0.01,
):
    env = gym.make(ENVIRONMENT)
    _ = plot_value_function(
        env,
        TD,
        iterations=ITERATIONS,
        title="Temporal Differences ({} iterations) - env:{}".format(
            ITERATIONS, ENVIRONMENT
        ),
        discount_factor=discount_factor,
        alpha=alpha,
    )
