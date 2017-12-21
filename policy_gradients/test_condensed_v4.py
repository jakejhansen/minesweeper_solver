# review solution

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.nn import relu, softmax
import gym

import sys
import os
sys.path.append('../')
from minesweeper_tk import Minesweeper


model = "condensed_6x6_v4"
# training settings

epochs = 100000 # number of training batches
batch_size = 200 # number of timesteps in a batch
rollout_limit = 50 # max rollout length
discount_factor = 0 # reward discount factor (gamma), 1.0 = no discount
learning_rate = 0.000002  # you know this by now #0.001, 
                                               #4400: 56% win --> LR: 0.0003
                                               #5600: 69% win --> LR: 0.0001 
                                               #7200: 74% win --> LR: 0.00003
                                               #8400: 77% win --> LR: 0.00001
                                               #9600: 75% win --> LR: 0.000005
                                               #10400: 75% win --> LR: 0.000002 
early_stop_loss = 0 # stop training if loss < early_stop_loss, 0 or False to disable

""" condensed
epochs = 100000 # number of training batches
batch_size = 400 # number of timesteps in a batch
rollout_limit = 50 # max rollout length
discount_factor = 0 # reward discount factor (gamma), 1.0 = no discount
learning_rate = 0.00004  # you know this by now #0.0005
early_stop_loss = 0 # stop training if loss < early_stop_loss, 0 or False to disable
"""

""" 261 epocs to learn 2 specific board (overfit)
epochs = 10000 # number of training batches
batch_size = 200 # number of timesteps in a batch
rollout_limit = 130 # max rollout length
discount_factor = 0 # reward discount factor (gamma), 1.0 = no discount
learning_rate = 0.001 # you know this by now
early_stop_loss = 0 # stop training if loss < early_stop_loss, 0 or False to disable

"""


# setup policy network
n = 6
n_inputs = 6*6*2
n_hidden = 6*6*8
n_hidden2 = 220
n_hidden3 = 220
n_hidden4 = 220
n_outputs = 6*6

dropout = 0

tf.reset_default_graph()

states_pl = tf.placeholder(tf.float32, [None, n_inputs], name='states_pl')
actions_pl = tf.placeholder(tf.int32, [None, 2], name='actions_pl')
advantages_pl = tf.placeholder(tf.float32, [None], name='advantages_pl')
learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate_pl')

input_layer = tf.reshape(states_pl, [-1, n, n, 2])
conv1 = tf.layers.conv2d(inputs=input_layer,filters=18,kernel_size=[5, 5],padding="same", activation=tf.nn.relu)
conv2 = tf.layers.conv2d(inputs=conv1,filters=36,kernel_size=[3, 3],padding="same", activation=tf.nn.relu)
conv2_flat = tf.contrib.layers.flatten(conv2)
l_hidden = tf.layers.dense(inputs=conv2_flat, units=n_hidden, activation=relu, name='l_hidden')
l_hidden2 = tf.layers.dense(inputs=l_hidden, units=n_hidden2, activation=relu, name='l_hidden2')
l_hidden2 = tf.layers.dropout(l_hidden2, rate=dropout)
l_hidden3 = tf.layers.dense(inputs=l_hidden2, units=n_hidden3, activation=relu, name='l_hidden3')
l_hidden3 = tf.layers.dropout(l_hidden3, rate=dropout)
#l_hidden4 = tf.layers.dense(inputs=l_hidden3, units=n_hidden4, activation=relu, name='l_hidden4')
l_hidden3 = tf.layers.dropout(l_hidden3, rate=dropout)
l_out = tf.layers.dense(inputs=l_hidden3, units=n_outputs, activation=softmax, name='l_out')

# print network
print('states_pl:', states_pl.get_shape())
print('actions_pl:', actions_pl.get_shape())
print('advantages_pl:', advantages_pl.get_shape())
print('l_hidden:', l_hidden.get_shape())
print('l_hidden2:', l_hidden2.get_shape())
print('l_hidden3:', l_hidden3.get_shape())
print('l_out:', l_out.get_shape())

# define loss and optimizer
loss_f = -tf.reduce_mean(tf.multiply(tf.log(tf.gather_nd(l_out, actions_pl)), advantages_pl))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_pl, beta1=0.8, beta2=0.92)
train_f = optimizer.minimize(loss_f)

saver = tf.train.Saver() # we use this later to save the model

# test forward pass
from minesweeper_tk import Minesweeper
env = Minesweeper(display=False, ROWS = 6, COLS = 6, MINES = 7, OUT = "CONDENSED", rewards = {"win" : 1, "loss" : -1, "progress" : 0.9, "noprogress" : -0.3, "YOLO" : -0.3})
state = env.stateConverter(env.get_state()).flatten()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    action_probabilities = sess.run(fetches=l_out, feed_dict={states_pl: [state]})
print(action_probabilities)

# helper functions

def get_rollout(sess, env, rollout_limit=None, stochastic=False, seed=None):
    """Generate rollout by iteratively evaluating the current policy on the environment."""
    rollout_limit = rollout_limit
    
    env.reset()
    s = env.stateConverter(env.get_state()).flatten()
    states, actions, rewards = [], [], []
    for i in range(rollout_limit):
        a = get_action(sess, s, stochastic)
        s1, r, done, _ = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s1
        if done: break
    return states, actions, rewards, i+1

def get_action(sess, state, stochastic=False):
    """Choose an action, given a state, with the current policy network."""
    # get action probabilities
    a_prob = sess.run(fetches=l_out, feed_dict={states_pl: np.atleast_2d(state)})
    #valid_moves = env.get_validMoves()
    #a_prob[~valid_moves.flatten().reshape(1,36)] = 0
    #a_prob[a_prob < 0.00001] = 0.000001
    #a_prob / np.sum(a_prob)
    #a_prob = normalize(a_prob, norm = 'l1')
    #if abs(1-np.sum(a_prob)) > 0.01:
    #    a_prob = sess.run(fetches=l_out, feed_dict={states_pl: np.atleast_2d(state)})

    if stochastic:
        # sample action from distribution
        return (np.cumsum(np.asarray(a_prob)) > np.random.rand()).argmax()
    else:
        # select action with highest probability
        return a_prob.argmax()

def get_advantages(rewards, rollout_limit, discount_factor, eps=1e-12):
    """Compute advantages"""
    returns = get_returns(rewards, rollout_limit, discount_factor)
    # standardize columns of returns to get advantages
    advantages = (returns - np.mean(returns, axis=0)) / (np.std(returns, axis=0) + eps)
    # restore original rollout lengths
    advantages = [adv[:len(rewards[i])] for i, adv in enumerate(advantages)]
    return advantages

def get_returns(rewards, rollout_limit, discount_factor):
    """Compute the cumulative discounted rewards, a.k.a. returns."""
    returns = np.zeros((len(rewards), rollout_limit))
    for i, r in enumerate(rewards):
        returns[i, len(r) - 1] = r[-1]
        for j in reversed(range(len(r)-1)):
            returns[i,j] = r[j] + discount_factor * returns[i,j+1]
    return returns



import time


if __name__ == "__main__":
    import time
    display = False
    env = Minesweeper(display=display, ROWS = 6, COLS = 6, MINES = 6, OUT = "CONDENSED", rewards = {"win" : 1, "loss" : -1, "progress" : 0.1, "noprogress" : -0.3, "YOLO": -0.3})

    i = 0
    #start = time.time()

    

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "{}/{}_best.ckpt".format(model,model))
        #view = Viewer(env, custom_render=True)
        games = 0
        moves = 0
        stuck = 0
        won_games = 0
        lost_games = 0
        while games < 10000:
            if games % 500 == 0:
                print(games)

            r = 1
            r_prev = 1
            while True:
                if display:
                    input()
                s = env.stateConverter(env.get_state()).flatten()
                if r < 0: 
                    a = get_action(sess, s, stochastic=True)
                else:
                    a = get_action(sess, s, stochastic=False)
                moves += 1
                r_prev = r
                s, r, done, _ = env.step(a)
                s = s.flatten()
                if display:
                    print("Reward = ", r)
                #print("\nReward = {}".format(r))
                if r == 1:
                    won_games += 1
                if r == -1:
                    lost_games += 1

                if done:
                    games += 1
                    env.reset()
                    moves = 0
                    break
                elif moves >= 30:
                    stuck += 1
                    games += 1
                    env.lost = env.lost + 1
                    env.reset()
                    moves = 0
                    break

        #print(env.lost)
        #print(env.won)
        print("games: {}, won: {}, lost: {}, stuck: {}, win_rate : {:.1f}%".format(games, won_games, lost_games, stuck, won_games/games * 100))
        
        #view.render(close=True, display_gif=True)

        #69% win-rate for 54000 epochs
