import argparse
import multiprocessing as mp
import os
import pathlib
import pickle
import time

import gym
import IPython
import matplotlib
matplotlib.use('Agg')  # Allow plotting on servers (e.g. without screen)
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Input, Model, Sequential, clone_model
from keras.optimizers import Adam
from keras.models import load_model, save_model


# TODO: Fitness functions output reward, dictionary_of_other_measures
# TODO: Add fitness_transform method as a property of Strategy to allow using different transforms
# TODO: Add regularizers as functions  e.g. from keras.regularizers (l1, l2, l1_l2)
# TODO: Implement virtual batch normalization (for each generation run some number of timesteps, 
#       compute normalizing statistics for inputs, normalize inputs
# TODO: Implement weight decay regularization 
# DONE: Put method parameters into dictionary
# DONE: Implement antithetic sampling

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


def uniform_transform(x):
    # Scales the input into the range [0,1]
    x = x - np.min(x)
    x = x/np.max(x)
    return x


def normalization_transform(rewards):
    # Normalizes the rewards, i.e. rewards~N(0,1)
    return (rewards - np.mean(rewards))/np.std(rewards)


class Strategy(object):
    '''
    Abstract evolutionary strategy that defines the general behaviour of the class.
    '''
    def __init__(self, fun, model, env, population, workers, save_dir=None):
        self.fitnessfun = fun
        self.model = model
        self.env = env
        self.population = population
        self.workers = workers
        self.weights = self.model.get_weights()
        self.generations = 0
        self.save_dir = save_dir if save_dir is not None else os.getcwd()
        self.results = {'generations': [], 'steps': [], 'mean_pop_rewards': [],
                        'test_rewards': [], 'time': [], 'weight_norm': [],
                        'win_rate': []}
        self.parameters = {'population': self.population, 'workers': self.workers,
                           'save_dir': self.save_dir}

    def print_progress(self, gen=None):
        if self.print_every and (gen is None or gen % self.print_every == 0):
            string = 'Generation {:>6d} | Steps {:>10d} | Test reward {: >7.1f} | Mean pop reward {: >7.1f} | Time {:>7.2f} seconds'.format(
                gen, self.results['steps'][-1], self.results['test_rewards'][-1], np.mean(self.results['mean_pop_rewards'][-1]), self.results['time'][-1])
            print(string)
            with open(os.path.join(self.save_dir, 'out.log'), 'a') as f:
                f.write(string + '\n')

    def print_setting(self):
        max_key_len = max([len(k) for k in self.parameters.keys()])
        max_val_len = max([len(str((v))) for v in self.parameters.values()])
        string = ''
        for k, v in self.parameters.items():
            string += '{:<{wk}} : {:{wv}}\n'.format(k, str(v), wk=max_key_len, wv=max_val_len)
        with open(os.path.join(self.save_dir, 'out.log'), 'a') as f:    
            f.write(string+'\n')

    def make_checkpoint(self, gen, steps, rewards, t_start, p):
        if self.checkpoint_every and (gen is None or gen % self.checkpoint_every == 0):
            
            # for i in range(n_tests):
            #     tup = self.fitnessfun(self.env, self.model)
            #     test_reward += tup[0]
            #     wins += tup[2]

            # Test current model
            n_tests = 30
            inputs = [{'env': env, 'model': model} for env, model in
                      zip([self.env]*n_tests, [self.model]*n_tests)]
            output = p.map(self.evaluate_fitness, inputs)
            test_rewards = [d['reward'] for d in output]
            test_rewards.extend([d['reward_antithetic'] for d in output])
            wins = [d['win'] for d in output]
            wins.extend([d['win_antithetic'] for d in output])
            test_reward = np.mean(test_rewards)
            win_rate = np.sum(wins)/len(wins)

            # Store results
            self.results['generations'].append(gen)
            self.results['steps'].append(steps)
            self.results['test_rewards'].append(test_reward)
            self.results['time'].append(time.time()-t_start)
            self.results['weight_norm'].append(self.get_weight_norms(self.weights))
            self.results['mean_pop_rewards'].append(np.mean(rewards))
            self.results['win_rate'].append(win_rate)
            self.print_progress(gen)

            # Save model
            save_model(self.model, os.path.join(self.save_dir, 'model.h5'))
            with open(os.path.join(self.save_dir, 'model.json'), 'w') as f:
                f.write(self.model.to_json())
            # Save results
            with open(os.path.join(self.save_dir, 'results.pkl'), 'wb') as f:
                pickle.dump(self.results, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self):
        try:
            with open(os.path.join(self.save_dir, 'results.pkl'), 'rb') as f:
                self.results = pickle.load(f)
            self.generations = max(self.results['generations'])
            self.model = load_model(os.path.join(self.save_dir, 'model.h5'))
            print('Loaded checkpoint. Resuming trainig from generation {:d}'.format(self.generations))
        except:
            print("Failed to load checkpoint")
            #raise

    def plot_progress(self, gen=None):
        if self.plot_every and (gen is None or gen % self.plot_every == 0):
            fig = plt.figure()
            plt.plot(self.results['steps'], self.results['mean_pop_rewards'])
            plt.plot(self.results['steps'], self.results['test_rewards'])
            plt.xlabel('Environment steps')
            plt.ylabel('Reward')
            plt.legend(['Mean population reward', 'Test reward'])
            plt.tight_layout()
            plt.grid()
            plt.savefig(os.path.join(self.save_dir, 'progress1.pdf'))
            plt.close(fig)

            fig = plt.figure()
            plt.plot(self.results['steps'], self.results['win_rate'])
            plt.xlabel('Environment steps')
            plt.ylabel('Win rate')
            plt.tight_layout()
            plt.grid()
            plt.savefig(os.path.join(self.save_dir, 'progress2.pdf'))
            plt.close(fig)

            fig = plt.figure(figsize=(4, 8))
            plt.subplot(3, 1, 1)
            plt.plot(self.results['steps'], self.results['mean_pop_rewards'])
            plt.ylabel('Mean population reward')
            plt.grid()
            plt.subplot(3, 1, 2)
            plt.plot(self.results['steps'], self.results['test_rewards'])
            plt.ylabel('Test reward')
            plt.grid()
            plt.subplot(3, 1, 3)
            plt.plot(self.results['steps'], self.results['weight_norm'])
            plt.ylabel('Weight norm')
            plt.xlabel('Environment steps')
            plt.tight_layout()
            plt.grid()
            plt.savefig(os.path.join(self.save_dir, 'progress3.pdf'))
            plt.close(fig)


class ES(Strategy):
    '''
    ES implements a simple evolutionary strategy based on isotropic zero mean Gaussian
    permutations of policy network weights.

    Reference: Evolutionary Strategies as a Scalable Alternative to Reinforcement Learning <https://arxiv.org/abs/1703.03864>
    '''

    def __init__(self, fun, model, env, reg={'L2': 0.001}, population=20, learning_rate=0.001, sigma=0.1, workers=mp.cpu_count(), save_dir=None):
        super(ES, self).__init__(fun=fun, model=model, env=env, population=population, workers=workers, save_dir=save_dir)
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.reg = reg
        self.parameters['learning_rate'] = self.learning_rate
        self.parameters['sigma'] = self.sigma
        self.parameters['reg'] = self.reg
        self.print_setting()

    def evolve(self, generations, checkpoint_every=25, plot_every=0):
        self.print_every = checkpoint_every
        self.plot_every = plot_every if plot_every > checkpoint_every or plot_every==0 else checkpoint_every
        self.checkpoint_every = checkpoint_every
        steps = 0
        with mp.Pool(self.workers) as p:
            t_start = time.time()
            for gen in range(self.generations, generations+self.generations):
                # Evaluate fitness of permuted policies
                inputs = [{'env': env, 'model': model} for env, model in
                          zip([self.env]*self.population, [self.model]*self.population)]
                output = p.map(self.evaluate_fitness, inputs)
                # output = [self.evaluate_fitness(inputs[i]) for i in range(self.population)]
                rewards = [d['reward'] for d in output]
                rewards.extend([d['reward_antithetic'] for d in output])
                steps_list = [d['steps'] for d in output]
                steps_list.extend([d['steps_antithetic'] for d in output])
                steps += np.sum(steps_list)
                noises = [d['noise'] for d in output]
                noises.extend([d['noise_antithetic'] for d in output])
                weight_norms = [d['weight_norms'] for d in output]
                weight_norms.extend([d['weight_norms_antithetic'] for d in output])
                
                # Squash rewards and weight norms into [0,1]
                weight_norms_unif = uniform_transform(weight_norms)
                rewards_unif = uniform_transform(np.array(rewards))
                reg_costs = self.reg['L2'] * weight_norms_unif

                # Compute regularized reward
                rewards_reg = rewards_unif - reg_costs

                # Transform rewards to fitness scores and regularize
                fitnesses = fitness_rank_transform(np.array(rewards_reg))

                # Update policy network
                for index, w in enumerate(self.weights):
                    A = np.array([n[index] for n in noises])
                    self.weights[index] = w + self.learning_rate/(self.population * self.sigma**2) * np.dot(A.T, fitnesses).T
                self.model.set_weights(self.weights)
                
                # Save
                self.make_checkpoint(gen, steps, rewards, t_start, p)
                self.plot_progress(gen)

        # Save
        self.make_checkpoint(gen, steps, rewards, t_start, p)
        self.plot_progress(gen)
        self.generations += generations
        return self.results

    def get_weight_norms(self, weights):
        # Compute different weight norms
        weight_norm = self.reg
        weight_list = []
        for l in weights:
            for w in l.flatten():
                weight_list.append(w)
        #if 'L2' in self.reg.keys():
        #    weight_norm['L2'] = np.linalg.norm(weight_list, 2)
        #if 'L1' in self.reg.keys():
        #    weight_norm['L1'] = np.linalg.norm(weight_list, 1)
        return np.linalg.norm(weight_list, 2)

    def evaluate_fitness(self, d):
        env, model = d['env'], d['model']
        # Get noise
        noise, noise_antithetic = self.get_noise()
        # Permute and evaluate
        weights = self.permute_weights(noise)
        model.set_weights(weights)
        reward, d = self.fitnessfun(env, model)
        weight_norms = self.get_weight_norms(weights)
        # Antithetic permute and evaluate
        weights_antithetic = self.permute_weights(noise_antithetic)
        model.set_weights(weights_antithetic)
        reward_antithetic, d_antithetic = self.fitnessfun(env, model)
        weight_norms_antithetic = self.get_weight_norms(weights_antithetic)
        # Store standard outputs
        out = {'reward': reward,
               'reward_antithetic': reward_antithetic,
               'noise': noise,
               'noise_antithetic': noise_antithetic,
               'weight_norms': weight_norms,
               'weight_norms_antithetic': weight_norms_antithetic}
        # Store additional outputs from fitness function
        for k, v in d.items():
            out[k] = v
        for k, v in d_antithetic.items():
            out[k+'_antithetic'] = v
        return out

    def permute_weights(self, noise):
        weights = []
        for index, n in enumerate(noise):
            weights.append(self.weights[index] + self.sigma * n)
        return weights

    def get_noise(self):
        noise = []
        noise_antithetic = []
        for w in self.model.get_weights():
            r = np.random.randn(*w.shape)
            noise.append(r)
            noise_antithetic.append(-r)
        return (noise, noise_antithetic)


class VES(Strategy):
    '''
    VES implements the variational optimization principle that uses a differentiable
    upper bound on the objective function to compute the gradient of an otherwise non-differentiable 
    function. The analytical gradient bound is then approximated using Monte Carlo.
    Reference: Optimization by Variational Bounding <https://arxiv.org/abs/1212.4507>
    '''
    def __init__(self, fun, model, env, population=20, lr_mu=0.001, lr_sigma=0.01, sigma=0.1, workers=mp.cpu_count()):
        super(VES, self).__init__(fitnessfun=fun, model=model, env=env, population=population, workers=workers)
        self.lr_mu = learning_rate
        self.lr_sigma
        self.sigma = sigma
        self.beta = 2 * np.log(self.sigma)  # parameterized variance
        self.results = {'generations': [], 'mean_pop_rewards': [],
                        'test_rewards': [], 'time': [], 'weight_norm': [],
                        'sigma': []}

    def evolve(self, generations, print_every=0, plot_every=0, checkpoint_every=25):
        pass

    def evaluate_fitness(self, d):
        pass

    def permute_weights(self, p):
        pass

    def get_noise(self):
        pass


class CMAES(Strategy):
    """
    CMAES implements an evolutionary strategy based on the Covariance Matrix Adaptation 
    Evolutionary Strategy (CMA-ES) algorithm.

    Reference: The CMA Evolution Strategy A Tutorial
    """
    def __init__(self):
        pass


class TRCMAES(Strategy):
    """
    TRCMAES implements an evolutionary strategy based on the trust region version of the 
    CMA-ES algorithm.

    Reference: Deriving and Improving CMA-ES with Information Geometric Trust Regions
    """
    def __init__(self):
        pass
