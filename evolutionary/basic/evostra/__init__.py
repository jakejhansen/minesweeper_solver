from __future__ import print_function
import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
from keras.models import Model, Input
from keras.layers import Dense
from keras.optimizers import Adam
import IPython
from math import sqrt
import time


def eval_fun_wrap(arg, **kwarg):
    """
    Method that unwraps the self of an object method call and 
    calls the same method again
    """
    return EvolutionStrategy.get_reward(*arg, **kwarg)


input_layer = Input(shape=(5, 1))
layer = Dense(8)(input_layer)
output_layer = Dense(3)(layer)
model = Model(input_layer, output_layer)
model.compile(Adam(), 'mse')


solution = np.array([0.1, -0.4, 0.5])
inp = np.asarray([[1, 2, 3, 4, 5]])
inp = np.expand_dims(inp, -1)


class EvolutionStrategy(object):



    def __init__(self, weights, population_size=50, sigma=0.1, learning_rate=0.001):
        np.random.seed(0)
        self.weights = weights
        #self.get_reward = get_reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate

    def _get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA*i
            weights_try.append(w[index] + jittered)
        return weights_try

    def get_weights(self):
        return self.weights

    @staticmethod
    def get_reward(weights):
        global solution, model, inp
        model.set_weights(weights)
        prediction = model.predict(inp)[0]
        # here our best reward is zero
        reward = -np.sum(np.square(solution - prediction))
        time.sleep(0.005)
        return reward

    def run(self, iterations, print_step=10):

        with Parallel(n_jobs=2) as parallel:
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
                
                

                #IPython.embed()
                ts = time.time()
                rewards = parallel(delayed(self.get_reward)(w) for w in weights_try)
                t = time.time()-ts
                print('iter %d. reward: %f. time %f' % (iteration, self.get_reward(self.weights), t))
                # rewards = parallel(delayed(EvolutionStrategy.get_reward)(w) for w in weights_try)
                # rewards = parallel(delayed(eval_fun_wrap)(w) for w in weights_try)
                #parallel(delayed(perm_and_eval_wrapper)(inp) for inp in inputs)
                #for i in range(self.POPULATION_SIZE):
                #    weights_try = self._get_weights_try(self.weights, population[i])
                #    rewards[i] = self.get_reward(weights_try)



                rewards = (rewards - np.mean(rewards)) / np.std(rewards)

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
