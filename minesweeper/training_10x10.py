import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.nn import relu, softmax
import gym
import pickle



# training settings

epochs = 100000 # number of training batches
batch_size = 300 # number of timesteps in a batch
rollout_limit = 80 # max rollout length
discount_factor = 0 # reward discount factor (gamma), 1.0 = no discount
learning_rate = 0.000001    # you know this by now
early_stop_loss = 0 # stop training if loss < early_stop_loss, 0 or False to disable

""" 261 epocs to learn 2 specific board (overfit)
epochs = 10000 # number of training batches
batch_size = 200 # number of timesteps in a batch
rollout_limit = 130 # max rollout length
discount_factor = 0 # reward discount factor (gamma), 1.0 = no discount
learning_rate = 0.001 # you know this by now
early_stop_loss = 0 # stop training if loss < early_stop_loss, 0 or False to disable

"""


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

# test forward pass
from minesweeper_tk import Minesweeper
env = Minesweeper(display=False, ROWS = 6, COLS = 6, MINES = 7, rewards = {"win" : 10, "loss" : -1, "progress" : 0.9, "noprogress" : -0.3, "YOLO" : -0.3})
state = env.stateConverter(env.get_state()).flatten()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    action_probabilities = sess.run(fetches=l_out, feed_dict={states_pl: [state]})
print(action_probabilities)

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


# train policy network

try:
    statistics = []
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, "model11/model11.ckpt")
        print('start training')

        #Load last model
        try:
            #stats = pickle.load(open("stats.p", "rb"))
            #epoch_start = stats[len(stats)-1][0]
            epoch_start = 0
        except:
            epoch_start = 0        

        for epoch in range(epoch_start, epochs):
            # generate rollouts until batch_size total timesteps are collected
            states, actions, rewards = [], [], []
            timesteps = 0
            while timesteps < batch_size:
                _rollout_limit = min(rollout_limit, batch_size - timesteps) # limit rollout to match batch_size
                s, a, r, t = get_rollout(sess, env, _rollout_limit, stochastic=True, seed=epoch)            
                states.append(s)
                actions.append(a)
                rewards.append(r)
                timesteps += t
            # compute advantages
            advantages = get_advantages(rewards, rollout_limit, discount_factor)
            # policy gradient update
            loss, _ = sess.run(fetches=[loss_f, train_f], feed_dict={
                states_pl: np.concatenate(states),
                actions_pl: np.column_stack((np.arange(timesteps), np.concatenate(actions))),
                advantages_pl: np.concatenate(advantages),
                learning_rate_pl: learning_rate
            })            
            # validation
            val_rewards = [get_rollout(sess, env, rollout_limit, stochastic=True, seed=(epochs+i))[2] for i in range(10)]
            # store and print training statistics
            mtr = np.mean([np.sum(r) for r in rewards])
            mvr = np.mean([np.sum(r) for r in val_rewards])
            statistics.append([epoch, mtr, mvr, loss])
            if epoch % 20 == 0:
                print('%4d. training reward: %6.2f, validation reward: %6.2f, loss: %7.4f' % (epoch+1, mtr, mvr, loss))
                saver.save(sess, 'model11/model11.ckpt')   

            if early_stop_loss and loss < early_stop_loss: break
        print('done')
        # save session
except KeyboardInterrupt:
    print('Saving Statistics')

    try: 
        stats = pickle.load(open("stats.p", "rb"))
        for i in range(len(statistics)):
            stats.append(statistics[i])
        statistics = stats
    except:
        print("No old model data found, saving into new file")

    #pickle.dump(statistics, open("stats.p", "wb"))
