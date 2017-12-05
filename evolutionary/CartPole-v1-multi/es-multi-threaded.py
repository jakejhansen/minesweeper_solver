import multiprocessing as mp
import time
import gym
import IPython
import numpy as np
from keras.layers import Dense
from keras.models import Input, Model, Sequential, clone_model
from keras.optimizers import Adam
import argparse
from strategies import Evolver


def fitnessfun(env, model):
    total_reward = 0
    done = False
    observation = env.reset()
    while not done:
        action = model.predict(observation.reshape((1,)+observation.shape))
        observation, reward, done, info = env.step(np.argmax(action))
        total_reward += reward
    return total_reward


def testfun(model, env, episodes):
    total_reward = []
    for i in range(episodes):
        total_reward.append(0)
        observation = env.reset()
        done = False
        while not done:
            action = model.predict(observation.reshape((1,)+observation.shape))
            observation, reward, done, info = env.step(np.argmax(action))
            env.render()
            total_reward[i] += reward
    return total_reward

parser = argparse.ArgumentParser()
parser.add_argument('--nwrk', type=int, default=-1)
parser.add_argument('--nags', type=int, default=20)
parser.add_argument('--ngns', type=int, default=1000)
args = parser.parse_args()

env = gym.make('CartPole-v0')

o_shape = env.observation_space.shape
a_shape = env.action_space.n
model = Sequential()
model.add(Dense(input_shape=o_shape, units=32))
model.add(Dense(units=128))
model.add(Dense(units=128))
model.add(Dense(units=a_shape))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
envs_simple = range(args.nags)

if __name__ == '__main__':
    #try:
        mp.freeze_support()
        e = Evolver(fun=fitnessfun, model=model, env=env, population=args.nags, learning_rate=1, sigma=0.1, workers=args.nwrk)
        #e.load_checkpoint()
        #e.evolve(args.ngns, print_every=1, plot_every=10)
        model.load_weights('weights.h5')
        testfun(model, env, 10)
    #except expression as identifier:
    #    pass
