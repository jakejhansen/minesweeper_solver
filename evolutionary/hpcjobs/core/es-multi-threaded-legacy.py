import multiprocessing as mp
import time
import gym
import IPython
import pickle
import numpy as np
from joblib import Parallel, delayed
from keras.layers import Dense
from keras.models import Input, Model, Sequential, clone_model
from keras.optimizers import Adam
import argparse
import matplotlib.pyplot as plt
import os
import pathlib


def fitness_rank_transform(rewards):
    # Performs the fitness rank transformation used for CMA-ES.
    # Reference: Natural Evolution Strategies [2014]
    n = len(rewards)
    sorted_indices = np.argsort(-rewards)
    u = np.zeros(n)
    for k in range(n):
        u[sorted_indices[k]] = np.max([0, np.log(n/2+1)-np.log(k+1)])
    u = u/np.sum(u)-1/n
    return u


def pickle_save(obj, name, directory=None):
    if directory is None:
        directory = os.getcwd()
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    with open(directory + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(name, directory=None):
    with open(directory + name + '.pkl', 'wb') as f:
        return pickle.load(f)


class Evolver(object):
    def __init__(self, model, envs, learning_rate=0.001, sigma=0.1, workers=mp.cpu_count()):
        self.nWorkers = workers
        self.model = model
        self.envs = envs
        self.weights = self.model.get_weights()
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.population_size = len(self.envs)
        self.results = results = {'generations': [], 'population_rewards': [],
                                  'test_rewards': [], 'time': []}

    def print_progress(self, gen=1, generations=1):
        if self.print_every and (gen % self.print_every == 0 or gen == generations - 1):
            print('Generation {:>4d} | Test reward {: >6.1f} | Mean pop reward {: >6.1f} | Time {:>4.2f} seconds'.format(
                gen, self.results['test_rewards'][-1], np.mean(self.results['population_rewards'][-1]), self.results['time'][-1]))

    def make_checkpoint(self, gen=1, generations=1):
        if self.checkpoint_every and (gen % self.checkpoint_every == 0 or gen == generations - 1):
            self.model.save_weights('weights.h5')
            pickle_save(self.results, 'results')

    def load_checkpoint(self, filename):
        raise NotImplementedError

    def plot_progress(self, gen=1, generations=1):
        if self.plot_every and (gen % self.plot_every == 0 or gen == generations - 1):
            fig = plt.figure()
            plt.plot(self.results['generations'], np.mean(self.results['population_rewards'], 1))
            plt.plot(self.results['generations'], self.results['test_rewards'])
            plt.xlabel('Generation')
            plt.ylabel('Reward')
            plt.legend(['Mean population reward', 'Test reward'])
            plt.tight_layout()
            plt.savefig('progress.pdf')
            plt.close(fig)

    def evolve(self, generations, print_every=0, plot_every=0, checkpoint_every=50):
        self.print_every = print_every
        self.plot_every = plot_every
        self.checkpoint_every = checkpoint_every
        with mp.Pool(self.nWorkers) as p:
            for gen in range(generations):
                t_start = time.time()

                # noise = []
                # weights_try = []
                # rewards = np.zeros(self.population_size)
                # for i in range(self.population_size):
                #     x = []
                #     for w in self.weights:                 
                #         x.append(np.random.randn(*w.shape))
                #     noise.append(x)
                #     weights_try.append(self.permute_weights(noise[i]))

                # Evaluate fitness

                # TODO figure out how to give permuted weights
                # TODO passed arguments have their old name (e.g. 'name' in self.model=name) FIX THIS
                inputs = zip(self.envs, [self.model]*self.population_size, [True]*self.population_size)
                output = p.map(self.fitnessfun, inputs)
                rewards = [t[0] for t in output]
                noise = [t[1] for t in output]
                               
                # [(noise1, reward1), (n2,r2), ...]
                # noise = [noise1, noise2, ...]
                # reward = ...
                # rewards = []
                # for i in range(self.population_size):
                #     self.model.set_weights(weights_try[i])
                #     rewards.append(fitnessfun(self.model, self.envs[i]))
                    
                fitnesses = fitness_rank_transform(np.array(rewards))
                #fitnesses = (rewards - np.mean(rewards))/np.std(rewards)

                #IPython.embed()
                for index, w in enumerate(self.weights):
                    A = np.array([n[index] for n in noise])
                    self.weights[index] = w + self.learning_rate/(self.population_size*self.sigma) * np.dot(A.T, fitnesses).T
                self.model.set_weights(self.weights)

                t = time.time()-t_start

                test_reward = self.fitnessfun((self.envs[0], self.model, False))[0]

                self.results['generations'].append(gen)
                self.results['population_rewards'].append(rewards)
                self.results['test_rewards'].append(test_reward)
                self.results['time'].append(t)
                
                # On cluster, extract plot data using sed like so
                # sed -e 's/.*Reward \(.*\) | Time.*/\1/' deep/evo/CartPole-v1-\(4\)/output_008.txt > plotres.txt
                self.print_progress(gen, generations)
                self.make_checkpoint(gen, generations)
                self.plot_progress(gen, generations)

        self.make_checkpoint()
        return self.results

    def permute_weights(self, p):
        weights = []
        for index, i in enumerate(p):
            jittered = self.sigma*i
            weights.append(self.weights[index] + jittered)
        return weights

    def get_noise(self):
        noise = []
        for w in self.weights:
            noise.append(np.random.randn(*w.shape))
        return noise

    def fitnessfun(self, tup):
        env, model, do_permute = tup
        noise = []
        if do_permute:
            noise = self.get_noise()
            weights = self.permute_weights(noise)
            model.set_weights(weights)
        observation = env.reset()
        o_shape = observation.shape
        total_reward = 0
        done = False
        while not done:
            action = model.predict(observation.reshape((1,)+o_shape))
            observation, reward, done, info = env.step(np.argmax(action))
            total_reward += reward
        return (total_reward, noise)


def testfun(model, env, episodes):
    o_shape = env.observation_space.shape
    total_reward = []
    for i in range(episodes):
        total_reward.append(0)
        observation = env.reset()
        done = False
        while not done:
            action = model.predict(observation.reshape((1,)+o_shape))
            observation, reward, done, info = env.step(np.argmax(action))
            env.render()
            total_reward[i] += reward
    return total_reward

parser = argparse.ArgumentParser()
parser.add_argument('--nwrk', type=int, default=-1)
parser.add_argument('--nags', type=int, default=20)
parser.add_argument('--ngns', type=int, default=1000)
args = parser.parse_args()

envs = [gym.make('CartPole-v0') for i in range(args.nags)]

o_shape = envs[0].observation_space.shape
a_shape = envs[0].action_space.n
model = Sequential()
model.add(Dense(input_shape=o_shape, units=32))
model.add(Dense(units=128))
model.add(Dense(units=128))
model.add(Dense(units=a_shape))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
envs_simple = range(args.nags)

if __name__ == '__main__':
    try:
        mp.freeze_support()
        e = Evolver(model=model, envs=envs, learning_rate=0.01, sigma=0.1, workers=args.nwrk)
        results = e.evolve(args.ngns, print_every=1, plot_every=10)
        model.load_weights('weights.h5')
        testfun(model, envs[0], 10)
    except expression as identifier:
        pass
