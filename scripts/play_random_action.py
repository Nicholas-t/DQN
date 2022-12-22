import gym
from scripts.utils import plot_res


def play_random_action(
    ENVIRONMENT: str, NUM_EPISODES: int, title: str = "Random Strategy"
):
    env = gym.make(ENVIRONMENT)
    final = []
    ep = 0
    for episode in range(NUM_EPISODES):
        ep += 1
        state = env.reset()
        done = False
        total = 0
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            total += reward
            if done:
                break
        # Add to the final reward
        final.append(total)
        plot_res(final, title)
    plot_res(final, title)
    env.close()
