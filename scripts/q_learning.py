
import torch
from torch.autograd import Variable
from scripts.utils import plot_res
import random
import time

def q_learning(env, model, episodes, gamma=0.9, 
               epsilon=0.3, eps_decay=0.99, title = 'DQL', verbose=True):
    final = []
    episode_i=0
    for episode in range(episodes):
        episode_i+=1
        # Reset state
        state = env.reset()
        done = False
        total = 0
        
        while not done:
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
                model.update(state, q_values)
                break

            q_values_next = model.predict(next_state)
            q_values[action] = reward + gamma * torch.max(q_values_next).item()
            model.update(state, q_values)

            state = next_state
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        plot_res(final, title)
        
        if verbose:
            print("episode: {}, total reward: {}".format(episode_i, total))
    return final


def q_learning_with_experience_replay(env, model, episodes, gamma=0.9, 
               epsilon=0.3, eps_decay=0.99, replay_size=20, 
               title = 'DQL With Replay', verbose=True):
    final = []
    memory = []
    episode_i=0
    sum_total_replay_time=0
    for episode in range(episodes):
        episode_i+=1
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

            t0=time.time()
            # Update network weights using replay memory
            model.replay(memory, replay_size, gamma)
            t1=time.time()
            sum_total_replay_time+=(t1-t0)

            state = next_state
        
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        plot_res(final, title)
        
        if verbose:
            print("episode: {}, total reward: {}".format(episode_i, total))
            print("Average replay time:", sum_total_replay_time/episode_i)
    return final
