"""
This module contains class definitions for open ai gym environments.
"""

import random
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from agent import QAgent

class MinesweeperParameters():
    def __init__(self):
        self.model_file = None
        self.output_dir = "./q_learning/output/"
        self.is_train = False
        #self.randomstart = 
        self.game = 'minesweeper'
        self.env = 'minesweeper'
        self.gpu_memory = 0.5
        self.input_height = 6
        self.input_width = 6
        self.history_length = 1
        self.mines_min = 6
        self.mines_max = 6
        self.nchannels = 2
        self.network_type = 1
        self.dueling_type = None
        self.bias_init = 0.01
        self.num_iterations = 500000000
        self.batch_size = 32
        self.train_freq = 1
        self.epsilon_step = 1e6
        self.learning_rate = 0.00025
        self.learning_rate_decay = 0.98
        self.learning_rate_step = 100000
        self.learning_rate_minimum = 0.0001
        self.discount = 0.0
        self.clip_delta = False
        self.network_update_rate = 100000
        self.batch_accumulator = "mean"
        self.replay_capacity = int(1e6)
        self.train_start = 50000
        self.eval_frequency = 250000
        self.eval_iterations = 125000
        self.eval_epsilon = 0.05
        self.min_epsilon = 0.1
        self.num_steps = 5000
        self.reward_recent_update = 10000
        self.num_games = 5000
        self.interval_summary = 200
        self.interval_checkpoint = 10000
        self.memory_checkpoint = int(1e5)
        self.restore_memory = False
        self.show_game = False
        self.eval_mode = 0
        self.seed = 0

def setup_model(mode = "test"):

    params = MinesweeperParameters()

    if mode == "train": # Train
        print("Training the network which obtained 90.20% win-rate on 6x6 board with 6 mines")
        params.is_train = True
        params.eval_iterations = 1000
        params.eval_frequency=20000
        params.interval_summary=500

        params.memory_checkpoint = int(5e5)
        params.train_start = int(5e4)
        params.replay_capacity = int(1e6)
        params.interval_checkpoint = int(2e4)

        params.batch_size = 400
        params.discount=0.0
        params.learning_rate=0.00025
        params.learning_rate_step=20000
        params.learning_rate_decay=0.90
        params.learning_rate_minimum=0.00025/4
        params.network_update_rate=int(1e5)
        params.min_epsilon=0.1
        params.epsilon_step=2.5e5

        run_model(params)

    elif mode == "train_random_mines": # Train
        print("Training the network which obtained the best win-rate on 6x6 board with random mines")
        params.is_train = True
        params.eval_iterations = 1000
        params.eval_frequency=20000
        params.interval_summary=500

        params.mines_min = 1
        params.mines_max = 12

        params.memory_checkpoint = int(5e5)
        params.train_start = int(5e4)
        params.replay_capacity = int(1e6)
        params.interval_checkpoint = int(2e4)        

        params.batch_size = 400
        params.discount=0.0
        params.learning_rate=0.00025
        params.learning_rate_step=20000
        params.learning_rate_decay=0.90
        params.learning_rate_minimum=0.00025/4
        params.network_update_rate=int(1e5)
        params.min_epsilon=0.1
        params.epsilon_step=2.5e5

        run_model(params)

    elif mode == "test": # Test minesweeper
        print("Test minesweeper model on 6x6 board with 6 mines")
        params.output_dir = './q_learning/output_best/'
        params.eval_iterations = 10000
        params.model_file = 'model-best'
        params.eval_mode = 1

        run_model(params)

    elif mode == "test_random_mines": # Evaluate for a different number of mines
        print("Test minesweeper model on 6x6 board with a random number of mines")
        params.output_dir = './q_learning/output_best/'
        params.eval_iterations = 10000
        params.model_file = 'model-best-random'
        params.eval_mode = 2

        print("\nTesting with best model on random number of mines")
        run_model(params)

        print("\nTesting with best model on board with 6 mines")
        params.model_file = 'model-best'

        tf.reset_default_graph()
        run_model(params)

    elif mode == "play": # Play 10 games
        params.show_game = True
        params.output_dir='./q_learning/output_best'
        params.eval_iterations=10
        params.model_file='model-best'
        params.eval_mode=1

        run_model(params)

# View tensorboard with 
# tensorboard --logdir output

def run_model(params):

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
    elif params.eval_mode == 0:
        qagent.evaluate_mine()
    elif params.eval_mode == 1:
        qagent.test_mine()
    elif params.eval_mode == 2:
        for mines in range(1, 13):
            params.mines_min=mines
            params.mines_max=mines
            print("Mines =", mines)
            qagent.test_mine()
            tf.reset_default_graph()
            qagent = QAgent(params)


if __name__ == "__main__":
    setup_model(2)