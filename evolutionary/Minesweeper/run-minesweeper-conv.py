import argparse
import cProfile
import multiprocessing as mp
import os
import pstats
import time
import gym
import IPython
import numpy as np
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Input, Model, Sequential, clone_model, load_model
from keras.optimizers import Adam

from context import core
from core.strategies import ES
from minesweeper_tk import Minesweeper


def fitnessfun(env, model):
    total_reward = 0
    done = False
    observation = env.reset()
    steps = 0
    while not done and steps < rows*cols-mines:
        # Predict action
        action = model.predict(observation.reshape((1, rows, cols, n_chs)))
        # Mask invalid moves (no need to renormalize when argmaxing)
        mask = env.get_validMoves().flatten()
        action[0, ~mask] = 0
        # Step and get reward
        observation, reward, done, info = env.step(np.argmax(action))
        total_reward += reward
        steps += 1
    win = True if done and reward is 0.9 else False
    return total_reward, {'steps': steps, 'win': win}


def testfun(env, model, episodes):
    total_reward = []
    try:
        for i in range(episodes):
            total_reward.append(0)
            input()
            observation = env.reset()
            done = False
            t = 0
            while not done and t < rows*cols-mines:
                input()
                action = model.predict(observation.reshape((1, rows, cols, n_chs)))
                observation, reward, done, info = env.step(np.argmax(action))
                total_reward[i] += reward
                t += 1
                print('Reward: {: >2.1f}'.format(reward))
    except KeyboardInterrupt:
        raise

    return total_reward

parser = argparse.ArgumentParser()
parser.add_argument('--nwrk', type=int, default=mp.cpu_count())
parser.add_argument('--nags', type=int, default=20)
parser.add_argument('--ngns', type=int, default=10000)
parser.add_argument('--cint', type=int, default=20)
parser.add_argument('--sigm', type=float, default=0.1)
parser.add_argument('--lrte', type=float, default=0.01)
parser.add_argument('--regu', type=float, default=0.001)
parser.add_argument('--size', type=int, default=8)
parser.add_argument('--mine', type=int, default=10)

args = parser.parse_args()

rows = args.size
cols = args.size
mines = args.mine
OUT = 'FULL'

rewards = {"win": 0.9, "loss": -1, "progress": 0.9, "noprogress": -0.3, "YOLO": -0.3}
env = Minesweeper(display=False, OUT=OUT, ROWS=rows, COLS=cols, MINES=mines, rewards=rewards)

n_chs = 10 if OUT is 'FULL' else 2
n_hidden = [200, 200, 200, 200]
n_outputs = rows*cols

# Model
model = Sequential()
# Convs
model.add(Conv2D(15, (5, 5), input_shape=(rows, cols, n_chs), padding='same', activation='relu'))
model.add(Conv2D(35, (3, 3), padding='same', activation='relu'))
# Dense
model.add(Flatten())
model.add(Dense(units=n_hidden[0],
                activation='relu',
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,#l2(reg),
                bias_regularizer=None))#l2(reg)))
# Hidden
for n_units in n_hidden[1:]:
    model.add(Dense(units=n_units,
                    activation='relu',
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    kernel_regularizer=None,#l2(reg),
                    bias_regularizer=None))#l2(reg)))
# Output
model.add(Dense(units=n_outputs,
                activation='softmax',
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None))

model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.summary()

DO_PROFILE = False
save_dir = os.path.split(os.path.realpath(__file__))[0]
if __name__ == '__main__':
    mp.freeze_support()
    e = ES(fun=fitnessfun, model=model, env=env, reg={'L2': args.regu}, population=args.nags, learning_rate=args.lrte, sigma=args.sigm, workers=args.nwrk, save_dir=save_dir)
    e.load_checkpoint()
    
    if DO_PROFILE:
        cProfile.run('e.evolve(args.ngns, print_every=1, plot_every=10)', 'profilingstats')
    
    e.evolve(args.ngns, checkpoint_every=args.cint, plot_every=args.cint)
    
    if DO_PROFILE:
        p = pstats.Stats('profilingstats')
        p.sort_stats('cumulative').print_stats(10)
        p.sort_stats('time').print_stats(10)
    
    #model = load_model('model.h5')
    #env = Minesweeper(display=True, OUT=OUT, ROWS=rows, COLS=cols, MINES=mines, rewards=rewards)
    #fitnessfun(env, model)

"""
model = Sequential()
model.add(Dense(input_shape=(1, n_chs),
                units=n_hidden[0],
                activation='relu',
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros'))
# Hidden
for n_units in n_hidden[1:]:
    model.add(Dense(units=n_units,
                    activation='relu',
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros'))
# Output
model.add(Dense(units=n_outputs,
                activation='softmax',
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
"""

"""
======= PROFILING WITH 1 WORKER =======
Wed Dec  6 10:08:15 2017    profilingstats

         1107747 function calls (1092605 primitive calls) in 125.457 seconds

   Ordered by: cumulative time
   List reduced from 2470 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     20/1    0.000    0.000  125.458  125.458 {built-in method builtins.exec}
        1    0.001    0.001  125.458  125.458 <string>:1(<module>)
        1    0.032    0.032  125.457  125.457 /Users/Jakob/Desktop/minesweeper_solver/evolutionary/CartPole-v1-multi/strategies.py:70(evolve)
       30    0.000    0.000  121.111    4.037 /Users/Jakob/anaconda/lib/python3.6/multiprocessing/pool.py:261(map)
       33    0.000    0.000  121.108    3.670 /Users/Jakob/anaconda/lib/python3.6/threading.py:533(wait)
       33    0.000    0.000  121.108    3.670 /Users/Jakob/anaconda/lib/python3.6/threading.py:263(wait)
       30    0.000    0.000  121.108    4.037 /Users/Jakob/anaconda/lib/python3.6/multiprocessing/pool.py:637(get)
       30    0.000    0.000  121.107    4.037 /Users/Jakob/anaconda/lib/python3.6/multiprocessing/pool.py:634(wait)
      166  121.107    0.730  121.107    0.730 {method 'acquire' of '_thread.lock' objects}
       30    0.038    0.001    2.091    0.070 es-multi-threaded.py:15(fitnessfun)


Wed Dec  6 10:08:15 2017    profilingstats

         1107747 function calls (1092605 primitive calls) in 125.457 seconds

   Ordered by: internal time
   List reduced from 2470 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      166  121.107    0.730  121.107    0.730 {method 'acquire' of '_thread.lock' objects}
     4618    0.432    0.000    0.614    0.000 /Users/Jakob/anaconda/lib/python3.6/site-packages/theano/compile/function_module.py:725(__call__)
       10    0.344    0.034    0.344    0.034 {method 'poll' of 'select.poll' objects}
        4    0.227    0.057    0.227    0.057 {built-in method _tkinter.create}
    22372    0.212    0.000    0.212    0.000 {built-in method numpy.core.multiarray.array}
     2472    0.207    0.000    0.207    0.000 {built-in method numpy.core.multiarray.dot}
    61099    0.123    0.000    0.123    0.000 {built-in method builtins.hasattr}
     4618    0.118    0.000    1.007    0.000 /Users/Jakob/anaconda/lib/python3.6/site-packages/keras/engine/training.py:1209(_predict_loop)
        1    0.101    0.101    0.101    0.101 {method 'acquire' of '_multiprocessing.SemLock' objects}
     4618    0.084    0.000    0.084    0.000 /Users/Jakob/anaconda/lib/python3.6/site-packages/keras/engine/training.py:406(<listcomp>)



======= PROFILING WITH 4 WORKERS =======
Wed Dec  6 10:00:43 2017    profilingstats

         3111894 function calls (3068601 primitive calls) in 211.293 seconds

   Ordered by: cumulative time
   List reduced from 2462 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     27/1    0.001    0.000  211.296  211.296 {built-in method builtins.exec}
        1    0.115    0.115  211.295  211.295 /Users/Jakob/Desktop/minesweeper_solver/evolutionary/CartPole-v1-multi/strategies.py:70(evolve)
      100    0.001    0.000  200.251    2.003 /Users/Jakob/anaconda/lib/python3.6/multiprocessing/pool.py:261(map)
      103    0.001    0.000  200.241    1.944 /Users/Jakob/anaconda/lib/python3.6/threading.py:533(wait)
      100    0.000    0.000  200.240    2.002 /Users/Jakob/anaconda/lib/python3.6/multiprocessing/pool.py:637(get)
      100    0.000    0.000  200.239    2.002 /Users/Jakob/anaconda/lib/python3.6/multiprocessing/pool.py:634(wait)
      103    0.001    0.000  200.239    1.944 /Users/Jakob/anaconda/lib/python3.6/threading.py:263(wait)
      515  200.238    0.389  200.238    0.389 {method 'acquire' of '_thread.lock' objects}
      100    0.122    0.001    5.254    0.053 es-multi-threaded.py:15(fitnessfun)
      100    0.001    0.000    4.544    0.045 /Users/Jakob/Desktop/minesweeper_solver/evolutionary/CartPole-v1-multi/strategies.py:58(plot_progress)


Wed Dec  6 10:00:43 2017    profilingstats

         3111894 function calls (3068601 primitive calls) in 211.293 seconds

   Ordered by: internal time
   List reduced from 2462 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      515  200.238    0.389  200.238    0.389 {method 'acquire' of '_thread.lock' objects}
    15292    1.299    0.000    1.880    0.000 /Users/Jakob/anaconda/lib/python3.6/site-packages/theano/compile/function_module.py:725(__call__)
    67701    0.658    0.000    0.658    0.000 {built-in method numpy.core.multiarray.array}
     7026    0.574    0.000    0.574    0.000 {built-in method numpy.core.multiarray.dot}
       11    0.490    0.045    0.490    0.045 {built-in method _tkinter.create}
    15292    0.368    0.000    3.128    0.000 /Users/Jakob/anaconda/lib/python3.6/site-packages/keras/engine/training.py:1209(_predict_loop)
       10    0.294    0.029    0.294    0.029 {method 'poll' of 'select.poll' objects}
    15292    0.264    0.000    0.264    0.000 /Users/Jakob/anaconda/lib/python3.6/site-packages/keras/engine/training.py:406(<listcomp>)
    15292    0.261    0.000    0.493    0.000 /Users/Jakob/anaconda/lib/python3.6/site-packages/gym/envs/classic_control/cartpole.py:56(_step)
    15292    0.203    0.000    0.248    0.000 /Users/Jakob/anaconda/lib/python3.6/site-packages/keras/engine/training.py:364(_make_batches)
"""
