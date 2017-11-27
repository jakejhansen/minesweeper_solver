from evostra import EvolutionStrategy
from keras.models import Model, Input
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np

input_layer = Input(shape=(5, 1))
layer = Dense(8)(input_layer)
output_layer = Dense(3)(layer)
model = Model(input_layer, output_layer)
model.compile(Adam(), 'mse')


solution = np.array([0.1, -0.4, 0.5])
inp = np.asarray([[1, 2, 3, 4, 5]])
inp = np.expand_dims(inp, -1)


def get_reward(weights):
    global solution, model, inp
    model.set_weights(weights)
    prediction = model.predict(inp)[0]
    # here our best reward is zero
    reward = -np.sum(np.square(solution - prediction))
    return reward


#es = EvolutionStrategy(model.get_weights(), get_reward, population_size=50, sigma=0.1, learning_rate=0.001)
es = EvolutionStrategy(model.get_weights(), population_size=50, sigma=0.1, learning_rate=0.001)
es.run(1000, print_step=100)
