# review solution

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.nn import relu, softmax
import gym
from minesweeper_tk import Minesweeper

# setup policy network
n_inputs = 360
n_hidden = 360
n_hidden2 = 150
n_hidden3 = 150
n_hidden4 = 150
n_outputs = 6*6

tf.reset_default_graph()

states_pl = tf.placeholder(tf.float32, [None, n_inputs], name='states_pl')
actions_pl = tf.placeholder(tf.int32, [None, 2], name='actions_pl')
advantages_pl = tf.placeholder(tf.float32, [None], name='advantages_pl')
learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate_pl')

l_hidden = tf.layers.dense(inputs=states_pl, units=n_hidden, activation=relu, name='l_hidden')
l_hidden2 = tf.layers.dense(inputs=l_hidden, units=n_hidden2, activation=relu, name='l_hidden2')
l_hidden3 = tf.layers.dense(inputs=l_hidden2, units=n_hidden3, activation=relu, name='l_hidden3')
l_hidden4 = tf.layers.dense(inputs=l_hidden3, units=n_hidden4, activation=relu, name='l_hidden4')
l_out = tf.layers.dense(inputs=l_hidden4, units=n_outputs, activation=softmax, name='l_out')

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

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_pl)
train_f = optimizer.minimize(loss_f)

saver = tf.train.Saver() # we use this later to save the model

# helper functions

def get_rollout(sess, env, rollout_limit=None, stochastic=False, seed=None):
    """Generate rollout by iteratively evaluating the current policy on the environment."""
    rollout_limit = rollout_limit or env.spec.timestep_limit
    
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
    print(a_prob)
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
    env = Minesweeper(display=True, ROWS = 6, COLS = 6, MINES = 7, rewards = {"win" : 100, "loss" : -10, "progress" : 1, "noprogress" : -3})

    i = 0
    #start = time.time()

    
    r = 1
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "tmp/model.ckpt")
        #view = Viewer(env, custom_render=True)
        while True:
            s = env.stateConverter(env.get_state()).flatten()
            if r < 0: 
                a = get_action(sess, s, stochastic=True)
            else:
                a = get_action(sess, s, stochastic=False)
            print(a)
            s, r, done, _ = env.step(a)
            print("\nReward = {}".format(r))
            if done:
                env.reset()

            input()
        
        for _ in range(500):
            print(_)
            time.sleep(0.5)
        
        #view.render(close=True, display_gif=True)