"""
This module contains class definitions for open ai gym environments.
"""

import os
import collections
import argparse
import random
from datetime import datetime
import time
from functools import reduce

import numpy as np
import tensorflow as tf

from qnetwork import DeepQNetwork, update_target_network
from replay_memory import ReplayMemory, ScreenHistory
from agent import QAgent

import random

def train(params):

    # https://stackoverflow.com/questions/11526975/set-random-seed-programwide-in-python
    # https://stackoverflow.com/questions/30517513/global-seed-for-multiple-numpy-imports
    random.seed(params.seed)
    np.random.seed(params.seed)
    # Must be called before Session
    # https://stackoverflow.com/questions/38469632/tensorflow-non-repeatable-results/40247201#40247201
    tf.set_random_seed(params.seed)

    qagent = QAgent(params)
    if params.is_train:
        qagent.fit()
    elif params.env == 'atari':
        qagent.play()
    elif params.env == 'minesweeper':
        qagent.evaluate_ms()

# View tensorboard with 
# tensorboard --logdir output

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="train.py", description="Train Deep Q-Network for Atari games")

    # Atari ROM, TensorFlow model and output directory
    parser.add_argument('--rom', dest='rom_file', default="./roms/Breakout.bin", type=str, help="path to atari rom (.bin) file")
    parser.add_argument('--model', dest='model_file', type=str, required=False, help="path to TensorFlow model file")
    parser.add_argument('--out', dest='output_dir', type=str, default="./output/", help="output path models and screen captures")

    parser.add_argument('--train', dest="is_train", action="store_true", help="training or only playing")
    parser.add_argument('--randomstart', dest='random_start_wait', type=int, default=30, help="random number of frames to wait at start of episode")
    parser.add_argument('--game', dest='game', type=str, default="DemonAttack-v0", help="The game we play")
    parser.add_argument('--env', dest='env', type=str, default="atari", help="If we want to use atari or minesweeper")


    parser.add_argument('--gpumemory', dest="gpu_memory", type=float, default=0.5, help="The percentage of GPU memory allowed to be used by Tensorflow")

    # Parameters network input (screens)
    parser.add_argument('--inputheight', dest="input_height", type=int, default=84, help="screen input height")
    parser.add_argument('--inputwidth', dest="input_width", type=int, default=84, help="screen input width")    
    parser.add_argument('--historylength', dest="history_length", type=int, default=4, help="Numbe of moves which are repeated in atari")
    parser.add_argument('--mines-min', dest="mines_min", type=int, default=5, help="The number of mines")
    parser.add_argument('--mines-max', dest="mines_max", type=int, default=7, help="The number of mines")
    parser.add_argument('--nchannels', dest="nchannels", type=int, default=4, help="screen input depth")


    parser.add_argument('--network-type', dest='network_type', type=str, default="conv", help="conv|fc")


    # Parameters CNN architecture
    parser.add_argument('--filtersizes', dest="filter_sizes", type=str, default="8,4,3", help="CNN filter sizes")
    parser.add_argument('--filterstrides', dest="filter_strides", type=str, default="4,2,1", help="CNN filter strides")
    parser.add_argument('--numfilters', dest="num_filters", type=str, default="32,64,64", help="CNN number of filters per layer")
    parser.add_argument('--numhidden', dest="num_hidden", type=int, default=512, help="CNN number of neurons in FC layer")
    parser.add_argument('--duelingtype', dest="dueling_type", default=None, type=str, help="Type of dueling enabled")
    # See 
    # http://cs231n.github.io/neural-networks-2/
    parser.add_argument('--bias-init', dest="bias_init", type=float, default=0.01, help="The initial value of the biases")


    # Parameters for training the CNN
    parser.add_argument('--num-iterations', dest="num_iterations", type=int, default=50000000, help="Number of training iterations, i.e., number of passes, each pass using [batch size] number of examples")
    parser.add_argument('--batchsize', dest="batch_size", type=int, default=32, help="training batch size")
    parser.add_argument('--trainfreq', dest="train_freq", type=int, default=4, help="training frequency, default every frame")
    parser.add_argument('--epsilonstep', dest="epsilon_step", type=float, default=1e6, help="epsilon decrease step, linear annealing over iterations")
    parser.add_argument('--learnrate', dest="learning_rate", type=float, default=0.00025, help="optimization learning rate")
    parser.add_argument('--learnratedecay', dest="learning_rate_decay", type=float, default=0.98, help="learning rate decay")
    parser.add_argument('--learnratestep', dest="learning_rate_step", type=float, default=100000, help="learning rate decay step over iterations")
    parser.add_argument('--learnratemin', dest="learning_rate_minimum", type=float, default=0.0001, help="minimum learning rate")
    parser.add_argument('--discount', dest="discount", type=float, default=0.99, help="gamma for future discounted rewards")
    parser.add_argument('--clipdelta', dest="clip_delta", type=bool, default=True, help="clipping of error term in loss function")
    parser.add_argument('--networkupdate', dest="network_update_rate", type=float, default=10000, help="number of steps after which the Q-network is copied for predicting targets")
    parser.add_argument('--batchaccumulator', dest="batch_accumulator", type=str, default="mean", help="batch accumulator in loss function (mean or sum)")

    parser.add_argument('--replaycap', dest="replay_capacity", type=int, default=int(1e6), help="maximum number of samples in replay memory")
    parser.add_argument('--trainstart', dest="train_start", type=int, default=50000, help="start training when replay memory is of this size")

    # Parameters for evaluation of the model
    parser.add_argument('--evalfreq', dest="eval_frequency", type=int, default=250000, help="frequency of model evaluation")
    parser.add_argument('--evaliterations', dest="eval_iterations", type=int, default=125000, help="number of game iterations for each evaluation")
    parser.add_argument('--evalepsilon', dest="eval_epsilon", type=float, default=0.05, help="epsilon random move when evaluating")
    parser.add_argument('--minepsilon', dest="min_epsilon", type=float, default=0.1, help="Lowest epsilon when exploring")
    parser.add_argument('--num-steps', dest="num_steps", type=int, default=5000, help="Number of test steps when playing, each step is an action")
    parser.add_argument('--reward-recent', dest="reward_recent", type=int, default=1000, help="The number of episodes before resetting recent reward")
    parser.add_argument('--num-games', dest="num_games", type=int, default=5000, help="Number of test games to play minesweeper")


    # Parameters for outputting/debugging
    parser.add_argument('--intsummary', dest="interval_summary", type=int, default=200, help="frequency of adding training summaries, currently depending on train_iteration")
    parser.add_argument('--intcheckpoint', dest="interval_checkpoint", type=int, default=10000, help="frequency of saving model checkpoints")
    parser.add_argument('--memorycheckpoint', dest="memory_checkpoint", type=int, default=int(1e5), help="Frequency of saving memory based on addition counter.")
    parser.add_argument('--restore-memory', dest="restore_memory", type=bool, default=False, help="If True, restore replay memory.")
    parser.add_argument('--show', dest="show_game", action="store_true", help="show the Atari game output")
    parser.add_argument('--seed', dest="seed", type=int, default=0, help="The random seed value. Default at 0 means deterministic for all ops in Tensorflow 1.4")


    # Parse command line arguments and run the training process


    parser.set_defaults(game="minesweeper")
    parser.set_defaults(env='minesweeper')
    parser.set_defaults(mines_min=6)
    parser.set_defaults(mines_max=6)

    parser.set_defaults(input_width=6)
    parser.set_defaults(input_height=6)
    parser.set_defaults(history_length=1)
    parser.set_defaults(train_freq=1) # This should be the same as history length
    parser.set_defaults(nchannels=2)

    parser.set_defaults(batch_size=400)
    #parser.set_defaults(restore_memory=True)
    parser.set_defaults(memory_checkpoint=int(5e5))
    parser.set_defaults(train_start=int(8e5)) # Needs to be larger than batch-size, if reloading set to 0.
    #parser.set_defaults(train_start=int(5e4)) # Needs to be larger than batch-size, if reloading set to 0.
    #parser.set_defaults(train_start=int(100))
    parser.set_defaults(replay_capacity=int(1e6))
    parser.set_defaults(interval_checkpoint=int(2e4))

    parser.set_defaults(eval_frequency=20000)
    parser.set_defaults(eval_iterations=1000) # Changed to number of games player in minesweeper
    parser.set_defaults(reward_recent_update=int(1e5))

    parser.set_defaults(discount=0.0)
    #parser.set_defaults(learning_rate=0.00025/4)
    #parser.set_defaults(learning_rate=0.00025)
    parser.set_defaults(learning_rate=0.00004)
    #parser.set_defaults(learning_rate=0.001)
    #parser.set_defaults(learning_rate_step=50000)
    parser.set_defaults(learning_rate_step=20000)
    parser.set_defaults(learning_rate_decay=0.90)
    #parser.set_defaults(learning_rate_minimum=0.00025/4)
    parser.set_defaults(learning_rate_minimum=0.00004)
    parser.set_defaults(network_update_rate=int(1e5))
    parser.set_defaults(min_epsilon=0.1)
    parser.set_defaults(epsilon_step=2.5e5)
    #parser.set_defaults(eval_epsilon=0.001) # For exploration

    parser.set_defaults(network_type='conv')
    #parser.set_defaults(clip_delta=True) # This should be False for minesweeper, it seems
    #parser.set_defaults(dueling_type="mean") # Without this and with fc, the same network as Jacob


    # If we want to play
    parser.set_defaults(num_steps=500) # Number of steps to play atarai
    parser.set_defaults(num_games=10000) # Number of games to play in minesweeper

    #parser.set_defaults(model_file="model-1880000")
    #parser.set_defaults(model_file="model-1680000")
    #parser.set_defaults(model_file="model-1700000")
    #parser.set_defaults(model_file="model-1720000")
    parser.set_defaults(eval_iterations=10000) # For finally testing the best model
    parser.set_defaults(is_train=True) # Note, something is wrong with the play code!!!
    parser.set_defaults(show_game=False)


    params = parser.parse_args()

    train(params)
