# Standard imports
import sys
import os
import math
import random

# 3rd party imports
import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Discrete, Box
import pygame

# Local imports
from example.breakout.breakout_rl import Game


class BreakoutEnv(gym.Env):
    """ Custom PyGame OpenAI Gym Environment 
    The user has the following discrete actions:
     - 0: Don't move
     - 1: Right
     - 2: Left
     
    The environment will provide the score as the rewards
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
        self, 
        mode = 'agent',
        lives = 3,
        framerate = 60,
        output_size=64
    ):
        self.mode = mode
        self.lives_start = lives
        self.output_size = output_size
        self.framerate = framerate
        self.game = self.init_game()
        self.iteration = 0
        self.iteration_max = 15 * 60 * self.game.framerate  # 15 minutes
        self.init_obs = self.get_state()

        self.action_space = Discrete(3)
        self.observation_space = Box(low=0, high=800, shape=(self.lidar_n_beams * 2, 1), dtype=np.float16)

        self.reward_range = (-1, 0, 1)

    def init_game(self):
        game = Game(
            mode=self.mode,
            lives=self.lives_start
        )
        return game


    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # Step frame

        print(action)

        self.game.step_frame(action)
        self.iteration += 1

        # Gather observation
        observation = self.get_state()

        print(observation)
        # Check stop conditions
        if self.game.lives == 0:
            done = True
        elif self.iteration > self.iteration_max:
            done = True
        else:
            done = False

        # Gather metadata/info
        info = {
            'iteration': self.iteration
        }

        return (observation, reward, done, info)

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.
        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.
        Returns:
            observation (object): the initial observation.
        """
        self.game = self.init_game()
        self.iteration = 0
        observation = self.get_state()
        return observation

    def render(self, mode, render_lidar=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
            return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
            representing RGB values for an x-by-y pixel image, suitable
            for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
            terminal-style text representation. The text can include newlines
            and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
                the list of supported modes. It's recommended to call super()
                in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        if mode == 'human':
            self.game.turn_on_screen()
            self.game.update_screen()
            self.game.render_screen()
        if mode == 'rgb_array':
            return self.get_rgb_array()
    
    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self):
        self.lidar.sync_position(self.game.player)
        ls_radius, ls_collide = self.lidar.scan(
            collide_sprites=self.game.rocks
        )
        array_radius = np.array(ls_radius)
        array_radius = array_radius / (self.lidar_max_radius_pct * self.game.screen_size)
        array_collide = np.array(ls_collide)
        array_state = np.concatenate([array_radius, array_collide])
        array_state = array_state.reshape((len(array_state), 1))
        return array_state
        
    def get_rgb_state(self):
        rgb_array = self.get_rgb_array()
        rgb_array = self.down_sample_rgb_array(rgb_array, self.output_size)
        rgb_array = rgb_array[:, :, 0]
        rgb_array = np.reshape(rgb_array, (self.output_size, self.output_size, 1))
        rgb_array = rgb_array.astype(np.uint8)
        return rgb_array

    def get_rgb_array(self):
        surf = pygame.display.get_surface()
        array = pygame.surfarray.array3d(surf).astype(np.float16)
        array = np.rot90(array)
        array = np.flip(array)
        array = np.fliplr(array)
        array = array.astype(np.uint8)
        return array

    def down_sample_rgb_array(self, array, output_size):
        bin_size = int(self.game.screen_size / output_size)
        array_ds = array.reshape((output_size, bin_size, output_size, bin_size, 3)).max(3).max(1)
        return array_ds
