from __future__ import print_function
import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
from keras.models import Sequential, Model, Input
from keras.layers import Dense
from keras.optimizers import Adam
import IPython
from math import sqrt
import time
import gym


def eval_fun_wrap(arg, **kwarg):
    """
    Method that unwraps the self of an object method call and 
    calls the same method again
    """
    return EvolutionStrategy.get_reward(*arg, **kwarg)

"""
input_layer = Input(shape=(5, 1))
layer = Dense(8)(input_layer)
output_layer = Dense(3)(layer)
model_basic = Model(input_layer, output_layer)
model_basic.compile(Adam(), 'mse')


solution = np.array([0.1, -0.4, 0.5])
inp = np.asarray([[1, 2, 3, 4, 5]])
inp = np.expand_dims(inp, -1)

env = gym.make('CartPole-v0')
o_shape = (1,) + env.observation_space.shape
a_shape = env.action_space.n

print(o_shape)
print(a_shape)
model = Sequential()
model.add(Dense(input_shape=o_shape, units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=a_shape, activation='softmax'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

envs = [gym.make('CartPole-v0') for i in range(50)]
"""

def fitness_rank_transform(rewards):
    # Performs the fitness rank transformation used for CMA-ES
    # Reference: Natural Evolution Strategies [2014]
    n = len(rewards)
    sorted_indices = np.argsort(-rewards)
    u = np.zeros(n)
    for k in range(n):
        u[sorted_indices[k]] = np.max([0, np.log(n/2+1)-np.log(k+1)])
    u = u/np.sum(u)-1/n
    return u


def get_reward(weights):
    env = gym.make('CartPole-v0')
    o_shape = (1,) + env.observation_space.shape
    a_shape = env.action_space.n
    model = Sequential()
    model.add(Dense(input_shape=o_shape, units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=a_shape, activation='softmax'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    total_reward = 0.0
    episodes_to_average = 1
    #IPython.embed()
    model.set_weights(weights)
    for episode in range(episodes_to_average):
        done = False
        observation = env.reset()
        while not done:
            action = model.predict(observation.reshape((1,)+o_shape))
            observation, reward, done, _ = env.step(np.argmax(action))
            total_reward += reward
    return total_reward/episodes_to_average


def get_reward_basic(weights):
    global solution, model, inp
    model_basic.set_weights(weights)
    prediction = model_basic.predict(inp)[0]
    # here our best reward is zero
    reward = -np.sum(np.square(solution - prediction))
    #time.sleep(0.5)
    return reward

#IPython.embed()
#observation = env.reset()
#action = model.predict(observation.reshape((1,) + o_shape))
#observation, reward, done, _ = env.step(np.argmax(action))


class EvolutionStrategy(object):
    def __init__(self, population_size=10, sigma=0.1, learning_rate=0.001):
        np.random.seed(0)
        env = gym.make('CartPole-v0')
        o_shape = (1,) + env.observation_space.shape
        a_shape = env.action_space.n
        model = Sequential()
        model = Sequential()
        model.add(Dense(input_shape=o_shape, units=128, activation='relu'))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=a_shape, activation='softmax'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.weights = model.get_weights()
        #self.get_reward = get_reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
        #self.envs = [gym.make('CartPole-v0') for i in range(self.POPULATION_SIZE)]

    def _get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA*i
            weights_try.append(w[index] + jittered)
        return weights_try

    def get_weights(self):
        return self.weights

    """
    @staticmethod
    def get_reward_basic(weights):
        global solution, model, inp
        model.set_weights(weights)
        prediction = model.predict(inp)[0]
        # here our best reward is zero
        reward = -np.sum(np.square(solution - prediction))
        time.sleep(0.5)
        return reward

    @staticmethod
    def get_reward(weights, env):
        total_reward = 0.0
        episodes_to_average = 1
        model.set_weights(weights)
        for episode in range(episodes_to_average):
            done = False
            observation = env.reset()
            while not done:
                action = model.predict(observation.reshape((1,)+o_shape))
                observation, reward, done, _ = env.step(np.argmax(action))
                total_reward += reward
        return total_reward/episodes_to_average
    """

    def run(self, iterations, print_step=10):

            for iteration in range(iterations):

                #if iteration % print_step == 0:
                #    print('iter %d. reward: %f' % (iteration, self.get_reward(self.weights)))

                population = []
                weights_try = []
                rewards = np.zeros(self.POPULATION_SIZE)
                for i in range(self.POPULATION_SIZE):
                    x = []
                    for w in self.weights:                 
                        x.append(np.random.randn(*w.shape))
                    population.append(x)
                    weights_try.append(self._get_weights_try(self.weights, population[i]))

                
                ts = time.time()
                #IPython.embed()
                #print("before")
                #get_reward(weights_try[0])
                rewards = Parallel(n_jobs=1)(delayed(get_reward)(weights_try[i]) for i in range(self.POPULATION_SIZE))
                #print("after second")
                
                #with Parallel(n_jobs=2) as parallel:
                #    rewards = parallel(delayed(get_reward)(weights_try[i], envs[i]) for i in range(self.POPULATION_SIZE))

                t = time.time()-ts
                print('iter %d. reward: %f. time %f' % (iteration, get_reward(self.weights), t))
                # rewards = parallel(delayed(EvolutionStrategy.get_reward)(w) for w in weights_try)
                # rewards = parallel(delayed(eval_fun_wrap)(w) for w in weights_try)
                #parallel(delayed(perm_and_eval_wrapper)(inp) for inp in inputs)
                #for i in range(self.POPULATION_SIZE):
                #    weights_try = self._get_weights_try(self.weights, population[i])
                #    rewards[i] = self.get_reward(weights_try)


                #rewards = (rewards - np.mean(rewards)) / np.std(rewards)
                rewards = fitness_rank_transform(np.array(rewards))

                for index, w in enumerate(self.weights):
                    A = np.array([p[index] for p in population])
                    self.weights[index] = w + self.LEARNING_RATE/(self.POPULATION_SIZE*self.SIGMA) * np.dot(A.T, rewards).T


    def run_org(self, iterations, print_step=10):
        for iteration in range(iterations):

            if iteration % print_step == 0:
                print('iter %d. reward: %f' % (iteration, self.get_reward(self.weights)))

            population = []
            rewards = np.zeros(self.POPULATION_SIZE)
            for i in range(self.POPULATION_SIZE):
                x = []
                for w in self.weights:                 
                    x.append(np.random.randn(*w.shape))
                population.append(x)

            for i in range(self.POPULATION_SIZE):
                weights_try = self._get_weights_try(self.weights, population[i])
                rewards[i] = self.get_reward(weights_try)

            rewards = (rewards - np.mean(rewards)) / np.std(rewards)

            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = w + self.LEARNING_RATE/(self.POPULATION_SIZE*self.SIGMA) * np.dot(A.T, rewards).T
