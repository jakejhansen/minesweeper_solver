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

def setup_model(mode = 0):

    params = MinesweeperParameters()

    # parser = argparse.ArgumentParser(prog="train.py", description="Train Deep Q-Network for Minesweeper game")

    # # Atari ROM, TensorFlow model and output directory
    # parser.add_argument('--model', dest='model_file', type=str, required=False, help="path to TensorFlow model file")
    # parser.add_argument('--out', dest='output_dir', type=str, default="./q_learning/output/", help="output path models and screen captures")

    # parser.add_argument('--train', dest="is_train", action="store_true", help="training or only playing")
    # #parser.add_argument('--randomstart', dest='random_start_wait', type=int, default=30, help="random number of frames to wait at start of episode")
    # parser.add_argument('--game', dest='game', type=str, default="DemonAttack-v0", help="The game we play")
    # parser.add_argument('--env', dest='env', type=str, default="atari", help="If we want to use atari or minesweeper")

    # parser.add_argument('--gpumemory', dest="gpu_memory", type=float, default=0.5, help="The percentage of GPU memory allowed to be used by Tensorflow")

    # # Parameters network input (screens)
    # parser.add_argument('--inputheight', dest="input_height", type=int, default=84, help="screen input height")
    # parser.add_argument('--inputwidth', dest="input_width", type=int, default=84, help="screen input width")    
    # parser.add_argument('--historylength', dest="history_length", type=int, default=4, help="Numbe of moves which are repeated in atari")
    # parser.add_argument('--mines-min', dest="mines_min", type=int, default=5, help="The number of mines")
    # parser.add_argument('--mines-max', dest="mines_max", type=int, default=7, help="The number of mines")
    # parser.add_argument('--nchannels', dest="nchannels", type=int, default=4, help="screen input depth")

    # parser.add_argument('--network-type', dest='network_type', type=int, default=1, help="Different networks")

    # # Parameters CNN architecture
    # parser.add_argument('--duelingtype', dest="dueling_type", default=None, type=str, help="Type of dueling enabled")
    # # See 
    # # http://cs231n.github.io/neural-networks-2/
    # parser.add_argument('--bias-init', dest="bias_init", type=float, default=0.01, help="The initial value of the biases")

    # # Parameters for training the CNN
    # parser.add_argument('--num-iterations', dest="num_iterations", type=int, default=50000000, help="Number of training iterations, i.e., number of passes, each pass using [batch size] number of examples")
    # parser.add_argument('--batchsize', dest="batch_size", type=int, default=32, help="training batch size")
    # parser.add_argument('--trainfreq', dest="train_freq", type=int, default=4, help="training frequency, default every frame")
    # parser.add_argument('--epsilonstep', dest="epsilon_step", type=float, default=1e6, help="epsilon decrease step, linear annealing over iterations")
    # parser.add_argument('--learnrate', dest="learning_rate", type=float, default=0.00025, help="optimization learning rate")
    # parser.add_argument('--learnratedecay', dest="learning_rate_decay", type=float, default=0.98, help="learning rate decay")
    # parser.add_argument('--learnratestep', dest="learning_rate_step", type=float, default=100000, help="learning rate decay step over iterations")
    # parser.add_argument('--learnratemin', dest="learning_rate_minimum", type=float, default=0.0001, help="minimum learning rate")
    # parser.add_argument('--discount', dest="discount", type=float, default=0.99, help="gamma for future discounted rewards")
    # parser.add_argument('--clipdelta', dest="clip_delta", type=bool, default=True, help="clipping of error term in loss function")
    # parser.add_argument('--networkupdate', dest="network_update_rate", type=float, default=10000, help="number of steps after which the Q-network is copied for predicting targets")
    # parser.add_argument('--batchaccumulator', dest="batch_accumulator", type=str, default="mean", help="batch accumulator in loss function (mean or sum)")

    # parser.add_argument('--replaycap', dest="replay_capacity", type=int, default=int(1e6), help="maximum number of samples in replay memory")
    # parser.add_argument('--trainstart', dest="train_start", type=int, default=50000, help="start training when replay memory is of this size")

    # # Parameters for evaluation of the model
    # parser.add_argument('--evalfreq', dest="eval_frequency", type=int, default=250000, help="frequency of model evaluation")
    # parser.add_argument('--evaliterations', dest="eval_iterations", type=int, default=125000, help="number of games played in each evaluation")
    # parser.add_argument('--evalepsilon', dest="eval_epsilon", type=float, default=0.05, help="epsilon random move when evaluating")
    # parser.add_argument('--minepsilon', dest="min_epsilon", type=float, default=0.1, help="Lowest epsilon when exploring")
    # parser.add_argument('--num-steps', dest="num_steps", type=int, default=5000, help="Number of test steps when playing, each step is an action")
    # parser.add_argument('--reward-recent-update', dest="reward_recent_update", type=int, default=10000, help="The number of episodes before resetting recent reward")
    # parser.add_argument('--num-games', dest="num_games", type=int, default=5000, help="Number of test games to play minesweeper")

    # # Parameters for outputting/debugging
    # parser.add_argument('--intsummary', dest="interval_summary", type=int, default=200, help="frequency of adding training summaries, currently depending on train_iteration")
    # parser.add_argument('--intcheckpoint', dest="interval_checkpoint", type=int, default=10000, help="frequency of saving model checkpoints")
    # parser.add_argument('--memorycheckpoint', dest="memory_checkpoint", type=int, default=int(1e5), help="Frequency of saving memory based on addition counter.")
    # parser.add_argument('--restore-memory', dest="restore_memory", type=bool, default=False, help="If True, restore replay memory.")
    # parser.add_argument('--show', dest="show_game", action="store_true", help="show the Minesweeper game output")
    # parser.add_argument('--eval-mode', dest="eval_mode", 
    #     help="0 = evaluate models in range (only used for selecting the best model), 1 = test model win-rate by playing the game, 2 = win-rate for random mines")
    # parser.add_argument('--seed', dest="seed", type=int, default=0, help="The random seed value. Default at 0 means deterministic for all ops in Tensorflow 1.4")

    # # Parse command line arguments and run the training process

    if mode == 0: # Train
        print("Training the network")
        params.is_train = True
        params.eval_iterations = 1000
        params.eval_frequency=20000
        params.interval_summary=500

        params.discount=0.0
        params.learning_rate=0.00025
        params.learning_rate_step=20000
        params.learning_rate_decay=0.90
        params.learning_rate_minimum=0.00025/4
        params.network_update_rate=int(1e5)
        params.min_epsilon=0.1
        params.epsilon_step=2.5e5

        run_model(params)

    elif mode == 1: # Test minesweeper
        print("Test minesweeper model on 6x6 board with 6 mines")
        params.output_dir = './q_learning/output_best/'
        params.eval_iterations = 10000
        params.model_file = 'model-best'
        params.eval_mode = 1

        run_model(params)

    elif mode == 2: # Evaluate for a different number of mines
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

    elif mode == 3: # Play 10 games
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