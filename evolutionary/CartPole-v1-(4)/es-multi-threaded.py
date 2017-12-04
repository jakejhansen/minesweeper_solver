import multiprocessing as mp
import time
import gym
import IPython
import numpy as np
from joblib import Parallel, delayed
from keras.layers import Dense
from keras.models import Input, Model, Sequential, clone_model
from keras.optimizers import Adam
import argparse
import matplotlib.pyplot as plt


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
    def __init__(self, model, envs, learning_rate=0.001, sigma=0.1, workers=mp.cpu_count()):
        self.nWorkers = workers
        self.model = model
        self.envs = envs
        self.weights = self.model.get_weights()
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.population_size = len(self.envs)

    def _get_weights_try(self, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.sigma*i
            weights_try.append(self.weights[index] + jittered)
        return weights_try

    def evolve(self, generations, print_every=0, plot_every=0):
        results = {'generations': [], 'population_rewards': [], 'test_rewards': []}      
        with mp.Pool(self.nWorkers) as p:
            for gen in range(generations):
                t_start = time.time()

                noise = []
                weights_try = []
                rewards = np.zeros(self.population_size)
                for i in range(self.population_size):
                    x = []
                    for w in self.weights:                 
                        x.append(np.random.randn(*w.shape))
                    noise.append(x)
                    weights_try.append(self._get_weights_try(noise[i]))

                # Evaluate fitness

                # TODO figure out how to give permuted weights
                # TODO passed arguments have their old name (e.g. 'name' in self.model=name) FIX THIS
                rewards = p.map(self.fitnessfun, zip(self.envs, [self.model]*self.population_size))

                # rewards = []
                # for i in range(self.population_size):
                #     self.model.set_weights(weights_try[i])
                #     rewards.append(fitnessfun(self.model, self.envs[i]))
                    
                #fitnesses = fitness_rank_transform(np.array(rewards))
                fitnesses = (rewards - np.mean(rewards))/np.std(rewards)

                for index, w in enumerate(self.weights):
                    A = np.array([p[index] for p in noise])
                    self.weights[index] = w + self.learning_rate/(self.population_size*self.sigma) * np.dot(A.T, fitnesses).T
                self.model.set_weights(self.weights)

                test_reward = self.fitnessfun((self.envs[0], self.model))
                results['generations'].append(gen)
                results['population_rewards'].append(rewards)
                results['test_rewards'].append(test_reward)
                t = time.time()-t_start

                if print_every and (gen % print_every == 0 or gen == generations - 1):
                    print('Generation {:3d} | Reward {: 4.1f} | Time {:4.2f} seconds'.format(gen, test_reward, t))
                    
                if False: #plot_every and (gen % plot_every == 0 or gen == generations - 1):
                    fig = plt.figure()
                    plt.plot(results['generations'], np.mean(results['population_rewards'], 1))
                    plt.plot(results['generations'], results['test_rewards'])
                    plt.xlabel('Generation')
                    plt.ylabel('Reward')
                    plt.legend(['Mean population reward', 'Test reward'])
                    plt.tight_layout()
                    plt.savefig('progress.pdf')
                    plt.close(fig)
                    
        self.model.save_weights('weights.h5')
        return results

    def fitnessfun(self, tup):
        env, model = tup
        for w in self.weights:
            noise = np.random.randn(*w.shape)
            #print(noise)
        weights_try = self._get_weights_try(noise)
        model.set_weights(weights_try)

        total_reward = 0
        o_shape = env.observation_space.shape
        observation = env.reset()
        done = False
        t = 0
        while not done:
            action = model.predict(observation.reshape((1,)+o_shape))
            observation, reward, done, info = env.step(np.argmax(action))
            # action = env.action_space.sample()
            # observation, reward, done, info = env.step(action)
            t += 1
            total_reward += reward
        return total_reward


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
            # action = env.action_space.sample()
            # observation, reward, done, info = env.step(action)
            total_reward[i] += reward
    return total_reward


def fitnessfun_simple(model):
    #model.set_weights(weights_try)
    solution = np.array([0.1, -0.4, 0.5])
    inp = np.asarray([1, 2, 3, 4, 5])
    #inp = np.expand_dims(inp, 1)
    # Best reward is zero (all others are negative)
    prediction = model.predict(inp.reshape(1, 1, 5))
    reward = -np.sum(np.square(solution - prediction))
    return reward
    

parser = argparse.ArgumentParser()
parser.add_argument('--nwrk', type=int, default=-1)
parser.add_argument('--nags', type=int, default=20)
parser.add_argument('--ngns', type=int, default=100)
args = parser.parse_args()

envs = [gym.make('CartPole-v1') for i in range(args.nags)]

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
    mp.freeze_support()
    e = Evolver(model=model, envs=envs, learning_rate=0.01, sigma=0.1)
    results = e.evolve(args.ngns, print_every=1, plot_every=10)
    #model.load_weights('CartPole-v1-weights.h5')
    #testfun(model, envs[0], 10)
