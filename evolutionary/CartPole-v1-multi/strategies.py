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


class Evolver(object):
    def __init__(self, fun, model, env, population=20, learning_rate=0.001, sigma=0.1, workers=mp.cpu_count()):
        self.fitnessfun = fun
        self.nWorkers = workers
        self.model = model
        self.env = env
        self.weights = self.model.get_weights()
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.population = population
        self.results = results = {'generations': [], 'population_rewards': [],
                                  'test_rewards': [], 'time': []}

    def print_progress(self, gen=1, generations=1):
        if self.print_every and (gen % self.print_every == 0 or gen == generations - 1):
            print('Generation {:>4d} | Test reward {: >6.1f} | Mean pop reward {: >6.1f} | Time {:>4.2f} seconds'.format(
                gen, self.results['test_rewards'][-1], np.mean(self.results['population_rewards'][-1]), self.results['time'][-1]))

    def make_checkpoint(self, gen=1, generations=1):
        if self.checkpoint_every and (gen % self.checkpoint_every == 0 or gen == generations - 1):
            self.model.save_weights('weights.h5')
            with open('results.pkl', 'wb') as f:
                pickle.dump(self.results, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self):
        self.model.load_weights('weights.h5')
        with open('results.pkl', 'rb') as f:
            self.results = pickle.load(f)

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

                # Evaluate fitness of permuted policies
                inputs = [{'env': env, 'model': model} for env, model in
                          zip([self.env]*self.population, [self.model]*self.population)]
                output = p.map(self.evaluate_fitness, inputs)
                rewards = [d['reward'] for d in output]
                noise = [d['noise'] for d in output]
                               
                # Transform rewards to fitness scores
                fitnesses = fitness_rank_transform(np.array(rewards))  
                #fitnesses = (rewards - np.mean(rewards))/np.std(rewards)

                # Update policy network
                for index, w in enumerate(self.weights):
                    A = np.array([n[index] for n in noise])
                    self.weights[index] = w + self.learning_rate/(self.population*self.sigma) * np.dot(A.T, fitnesses).T
                self.model.set_weights(self.weights)

                # Test current model
                test_reward = self.fitnessfun(self.env, self.model)

                # Save results
                t = time.time()-t_start
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

    def evaluate_fitness(self, d):
        env, model = d['env'], d['model']
        noise = self.get_noise()
        weights = self.permute_weights(noise)
        model.set_weights(weights)
        reward = self.fitnessfun(env, model)
        return {'reward': reward, 'noise': noise}

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
