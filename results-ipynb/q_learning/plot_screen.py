import numpy as np
import gym
import random
import cv2
import argparse
import sys

from environment import AtariGymEnvironment
from replay_memory import ReplayMemory, ScreenHistory

from matplotlib import pyplot as plt

# This is just to see the game

games = ["SpaceInvaders-v0", "Skiing-v0", "DemonAttack-v0", "BeamRider-v0", "Pong-v0", "Enduro-v0"]
random_starts = [2, 20, 20, 60, 30, 10]
interpolation = "lanczos"

for i in range(0, len(games)):
    env = AtariGymEnvironment(random_start = random_starts[i], display=False, game=games[i])
    s, _, _ = env.new_random_game()

    # No interpolation in plot show
    plt.imshow(s, cmap='gray')
    #plt.show()
    plt.savefig('interpolation/' + interpolation + games[i] + '.png')

