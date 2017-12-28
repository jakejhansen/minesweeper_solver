import unittest
import numpy as np
import gym
import random
import cv2
import argparse
import sys

from environment import AtariGymEnvironment
from replay_memory import ReplayMemory, ScreenHistory

from matplotlib import pyplot as plt


def create_parser():
    #parser = argparse.ArgumentParser(prog="test_memoy.py", description="Train Deep Q-Network for Atari games")
    parser = argparse.ArgumentParser()

    # Parameters network input (screens)
    parser.add_argument('--inputsize', dest="input_size", type=int, default=84, help="screen input size")
    parser.add_argument('--historylength', dest="history_length", type=int, default=4, help="screen input depth")


    # For replay memory
    parser.add_argument('--replaycap', dest="replay_capacity", type=int, default=int(3), help="maximum number of samples in replay memory")
    parser.add_argument('--batchsize', dest="batch_size", type=int, default=2, help="training batch size")


    return parser


class ScreenHistoryTest(unittest.TestCase):
    def set_up(self):
        self.parser = create_parser()

    def test_screen_history(self):

        parser = create_parser()
        params = parser.parse_args()

        assert params.input_size == 84
        assert params.history_length == 4

        # Create the environment
        env = AtariGymEnvironment(display=False, game="Breakout-v0")
        s, _, _ = env.new_game()

        assert s.shape == (84, 84)

        history = ScreenHistory(params)

        assert history.get().shape == (1, 84, 84, 4) # Batch, height, width, channel
        assert history.screens.shape == (4, 84, 84) # channel, height, width

        history.add(s)

        # Since they are floating point arrays, we just check if they are close
        assert np.allclose(s, history.screens[3], rtol=1e-03, atol=1e-08)

        history.add(s+0.5)
        history.add(s+1.0)
        history.add(s+1.5)

        assert np.allclose(s, history.screens[0], rtol=1e-03, atol=1e-08)
        assert np.allclose((s+0.5), history.screens[1], rtol=1e-03, atol=1e-08)
        assert np.allclose((s+1.0), history.screens[2], rtol=1e-03, atol=1e-08)
        assert np.allclose((s+1.5), history.screens[3], rtol=1e-03, atol=1e-08)

        # How to obtain the originals
        test = np.transpose(np.reshape(history.get(), [84, 84, 4]), [2, 0, 1])
        assert np.allclose(s, test[0], rtol=1e-03, atol=1e-08)
        assert np.allclose((s+0.5),  test[1], rtol=1e-03, atol=1e-08)
        assert np.allclose((s+1.0),  test[2], rtol=1e-03, atol=1e-08)
        assert np.allclose((s+1.5),  test[3], rtol=1e-03, atol=1e-08)

class ReplayMemoryTest(unittest.TestCase):
    def set_up(self):
        self.parser = create_parser()

    def test_replay_memory(self):

        parser = create_parser()
        params = parser.parse_args()

        replay_mem1 = ReplayMemory(params.replay_capacity, params.batch_size, 84, 84, "test", 10, False, './output')

        env = AtariGymEnvironment(display=False, game="Breakout-v0")
        s1, r1, d1 = env.new_game()
        s2, r2, d2 = env.act(0)

        replay_mem1.add(0, r2, s2, d2)
        replay_mem1.add(0, r2, s2, d2)
        replay_mem1.add(0, r2, s2, d2)

        print(replay_mem1.counter)
        print(replay_mem1.current)

        replay_mem1.save_memory()
        replay_mem2 = ReplayMemory(params.replay_capacity, params.batch_size, 84, 84, "test", 10, True, './output')

        print(replay_mem2.counter)
        print(replay_mem2.current)

        assert replay_mem2.counter == replay_mem1.counter
        assert replay_mem2.current == replay_mem1.current

        print(replay_mem1.num_examples())
        print(replay_mem2.num_examples())


        

# any member function whose name begins with test in a class deriving from 
# unittest.TestCase be run, and its assertions checked, when unittest.main()

if __name__ == '__main__':
    unittest.main()
