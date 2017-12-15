# imports
import matplotlib.pyplot as plt
import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense, Reshape, Flatten
#from keras.optimizers import Adam
#from keras.layers.convolutional import Convolution2D

import tensorflow as tf
from tensorflow.python.ops.nn import relu, softmax

#import gym
from minesweeper import Minesweeper
#import tensorflow as tf


def stateConverter(state):
    """ Converts 2d state to one-hot encoded 3d state
        input: state (rows x cols)
        output state3d (row x cols x 10)
    """
    rows, cols = state.shape
    res = np.zeros((rows,cols,10))
    for row in range(rows):
        for col in range(cols):
            field = state[row][col]
            if type(field) == int:
                res[row][col][field-1] = 1
            elif field == 'U':
                res[row][col][8] = 1
            else:
                res[row][col][9] = 1

    assert(np.sum(res) == 100)
    
    return(res)


n = 6

"""
# setup policy network
n_input = n*n*10
n_output = n*n
learning_rate = 0.003
model = Sequential()
model.add(Dense(300, input_dim=n_input, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_output, activation='softmax'))

opt = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=opt)
model.summary()
"""

#################################################################################################

# setup policy network
n_inputs = 1000
n_hidden = 300
n_hidden2 = 200
n_hidden3 = 100
n_outputs = 100

tf.reset_default_graph()

states_pl = tf.placeholder(tf.float32, [None, n_inputs], name='states_pl')
actions_pl = tf.placeholder(tf.int32, [None, n_outputs], name='actions_pl')
advantages_pl = tf.placeholder(tf.float32, [None], name='advantages_pl')
learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate_pl')
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)

l_hidden = tf.layers.dense(inputs=states_pl, units=n_hidden, activation=relu, name='l_hidden')
l_hidden2 = tf.layers.dense(inputs=l_hidden, units=n_hidden2, activation=relu, name='l_hidden2')
l_hidden3 = tf.layers.dense(inputs=l_hidden2, units=n_hidden3, activation=relu, name='l_hidden3')
l_out = tf.layers.dense(inputs=l_hidden3, units=n_outputs, activation=softmax, name='l_out')

# print network
print('states_pl:', states_pl.get_shape())
print('actions_pl:', actions_pl.get_shape())
print('advantages_pl:', advantages_pl.get_shape())
print('l_hidden:', l_hidden.get_shape())
print('l_hidden2:', l_hidden2.get_shape())
print('l_out:', l_out.get_shape())

# define loss and optimizer
responsible_weight = tf.slice(l_out, action_holder,[1])
loss_f = -(tf.log(responsible_weight)*reward_holder)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_pl)
train_f = optimizer.minimize(loss_f)

saver = tf.train.Saver() # we use this later to save the model

#############################################################################################


game = Minesweeper(COLS = n, ROWS = n, MINES = 7, display = True)
state = game.get_state()
state = stateConverter(state)
state = np.reshape(state, (1, n_input))

i = 0
while True:
    old_state = state.copy()
    action = model.predict(state)
    props = model.predict(state)
    action = np.unravel_index(action.argmax(), (n,n))
    s = game.action(action[0], action[1])

    state = s['s']
    reward = s['r']

    state = stateConverter(state)
    state = np.reshape(state, (1, n_input))

    #Training

    y = np.zeros([n_output])
    y[props.argmax()] = 1
    gradients = np.array(y).astype('float32') - props
    gradients *= reward

    Y = props + learning_rate * gradients

    model.train_on_batch(old_state, Y)

    #if input() == "i":
    #    import IPython
    #    IPython.embed()

    print(i)
    i += 1

    if i > 15000:
        input()







"""
while True:

    action = model.predict(state)
    val_act = action[0][action.argmax()] 

    if np.random.rand() < 0.9:
        action = np.unravel_index(action.argmax(), (10,10))
    else:
        action = (np.random.randint(10), np.random.randint(10))

    print(action)

    s = game.action(action[0], action[1])

    state = s['s']
    reward = s['r']

    state = stateConverter(state)
    state = np.reshape(state, (1, 1000))


    import IPython
    IPython.embed()
    
    input()
"""

