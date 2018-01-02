import unittest
import numpy as np
import gym
import random

from environment import AtariGymEnvironment

class BaseEnvironmentTest(unittest.TestCase):

    def test_BaseEnvironment(self):

        # Create the environment
        env = AtariGymEnvironment(display=False, game="Breakout-v0")
        # Start a new game
        env.new_game()

        assert type(env.avg_reward_per_episode()) is float
        assert type(env.avg_steps_per_episode()) is float
        assert type(env.episode_number) is int
        assert type(env.episode_step) is int
        assert type(env.global_step) is int
        assert type(env.episode_reward) is float
        assert type(env.global_reward) is float
        assert type(env.max_reward_per_episode) is float


class AtariEnvironmentTest(unittest.TestCase):


    def test_AtariGymEnvironment(self):

        # Create the environment
        # In Skiing, we get instant rewards
        env = AtariGymEnvironment(display=False, game="Skiing-v0")
        # Start a new game
        env.new_game()

        # Tests
        assert env.screen.shape == (84, 84)
        #assert type(env.lives) is float
        assert type(env.screen) is np.ndarray
        assert type(env.screen) is np.ndarray
        assert type(env.num_actions) is int
        assert type(env.legal_actions) is gym.spaces.discrete.Discrete

        # Pre-step
        assert env.episode_number == 1
        assert env.episode_step == 0
        assert env.global_step == 0
        assert env.episode_reward == 0
        assert env.global_reward == 0
        assert env.max_reward_per_episode == 0

        # Step
        env.act(0)

        assert env.episode_number == 1
        assert env.episode_step == 1
        assert env.global_step == 1
        assert env.episode_reward != 0
        assert env.global_reward == env.episode_reward


        # Test output types
        s, r, d = env.new_game()
        assert type(s) is np.ndarray
        assert type(r) is float
        assert type(d) is bool

        s, r, d = env.new_random_game()
        assert type(s) is np.ndarray
        assert type(r) is float
        assert type(d) is bool

        s, r, d = env.act(0)
        assert type(s) is np.ndarray
        assert type(r) is float
        assert type(d) is bool

        s, r, d = env.state
        assert type(s) is np.ndarray
        assert type(r) is float
        assert type(d) is bool


if __name__ == '__main__':
    unittest.main()
