import torch
from torch.autograd import Variable
from scripts.utils import plot_res
import random
import time


def q_learning(
    env,
    model,
    episodes: int,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    eps_decay: float = 0.99,
    title: str = "DQL",
    verbose: bool = True,
) -> list:
    final = []
    episode_i = 0
    for episode in range(episodes):
        episode_i += 1
        # Reset state
        state = env.reset()
        done = False
        total = 0

        while not done:
            # Tradeoff between exploration and exploitation
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)

            total += reward
            q_values = model.predict(state).tolist()

            if done:
                q_values[action] = reward
                model.update(action, state, q_values)
                break

            q_values_next = model.predict(next_state)

            # Assign values of action 
            q_values[action] = reward + gamma * torch.max(q_values_next).item()
            model.update(action, state, q_values)

            state = next_state
            
        # Decaying epsilon to reduce the amount of exploration after every episodes
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        plot_res(final, title)

        if verbose:
            print("episode: {}, total reward: {}".format(episode_i, total))
    return final


def q_learning_with_experience_replay(
    env,
    model,
    episodes: int,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    eps_decay: float = 0.99,
    batch_size: int = 20,
    title: str = "DQL With Replay",
    verbose: bool = True,
) -> list:
    final = []
    memory = []
    episode_i = 0
    sum_total_replay_time = 0
    for episode in range(episodes):
        episode_i += 1
        # Reset state
        state = env.reset()
        done = False
        total = 0

        while not done:
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()

            # Take action and add reward to total
            next_state, reward, done, _ = env.step(action)

            # Update total and memory
            total += reward
            memory.append((state, action, next_state, reward, done))
            q_values = model.predict(state).tolist()

            if done:
                break

            t0 = time.time()
            # Update network weights using replay memory
            model.replay(memory, batch_size, gamma)
            t1 = time.time()
            sum_total_replay_time += t1 - t0

            state = next_state

        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        plot_res(final, title)

        if verbose:
            print("episode: {}, total reward: {}".format(episode_i, total))
            print("Average replay time:", sum_total_replay_time / episode_i)
    return final

