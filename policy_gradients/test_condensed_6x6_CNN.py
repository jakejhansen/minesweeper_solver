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


model = "condensed_6x6_CNN"

# setup policy network
n = 6
n_inputs = 6*6*2
n_hidden = 6*6*8
n_hidden2 = 220
n_hidden3 = 220
n_outputs = 6*6


tf.reset_default_graph()


states_pl = tf.placeholder(tf.float32, [None, n_inputs], name='states_pl')
actions_pl = tf.placeholder(tf.int32, [None, 2], name='actions_pl')


#Define Network
input_layer = tf.reshape(states_pl, [-1, n, n, 2])
conv1 = tf.layers.conv2d(inputs=input_layer,filters=18,kernel_size=[5, 5],padding="same", activation=tf.nn.relu)
conv2 = tf.layers.conv2d(inputs=conv1,filters=36,kernel_size=[3, 3],padding="same", activation=tf.nn.relu)
conv2_flat = tf.contrib.layers.flatten(conv2)
l_hidden = tf.layers.dense(inputs=conv2_flat, units=n_hidden, activation=relu, name='l_hidden')
l_hidden2 = tf.layers.dense(inputs=l_hidden, units=n_hidden2, activation=relu, name='l_hidden2')
l_hidden3 = tf.layers.dense(inputs=l_hidden2, units=n_hidden3, activation=relu, name='l_hidden3')
l_out = tf.layers.dense(inputs=l_hidden3, units=n_outputs, activation=softmax, name='l_out')


saver = tf.train.Saver() # we use this later to save the model



def get_action(sess, state, stochastic=False):
    """Choose an action, given a state, with the current policy network."""
    # get action probabilities


    a_prob = sess.run(fetches=l_out, feed_dict={states_pl: np.atleast_2d(state)})
    if stochastic:
        # sample action from distribution
        action = (np.cumsum(np.asarray(a_prob)) > np.random.rand()).argmax()
    else:
        # select action with highest probability
        action = a_prob.argmax()

    return(action, a_prob)






if __name__ == "__main__":

    display = False

    #Check to see if user has inputtet the display option
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--display", help= "run with this argument to display the game and see the agent play by pressing enter")
    args = parser.parse_args()
    if args.display:
        display = True

    env = Minesweeper(display=display, ROWS = 6, COLS = 6, MINES = 6, OUT = "CONDENSED", rewards = {"win" : 1, "loss" : -1, "progress" : 0.9, "noprogress" : -0.3, "YOLO": -0.4})


    

    with tf.Session() as sess:
        #Restore Model
        saver = tf.train.Saver()
        saver.restore(sess, "{}/{}.ckpt".format(model,model))

        #Initialize test parameters
        games = 0
        moves = 0
        stuck = 0
        won_games = 0
        lost_games = 0

        #Test for a number of games
        while games < 10000:
            if games % 500 == 0:
                print("Games completed:", games)

            r = 1
            r_prev = 1
            while True:


                s = env.stateConverter(env.get_state()).flatten()

                #If negative reward (stuck), do stochastic sample else don't
                if r < 0: 
                    a, a_prob = get_action(sess, s, stochastic=True)
                else:
                    a, a_prob = get_action(sess, s, stochastic=False)
                moves += 1
                r_prev = r

                if display:
                    input()
                s, r, done, _ = env.step(a)
                s = s.flatten()
                
                if display:
                    if r == 1:
                        print("WIN")
                    elif r == -1:
                        print("LOSS")
                    elif r == 0.9:
                        print("Progress")
                    elif r == -0.3:
                        print("No Progress")
                    elif r == -0.4:
                        print("YOLO MOVE")
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

                #Check if agent is stuck
                elif moves >= 30:
                    stuck += 1
                    games += 1
                    env.lost = env.lost + 1
                    env.reset()
                    moves = 0
                    break

        print("games: {}, won: {}, lost: {}, stuck: {}, win_rate : {:.1f}%".format(games, won_games, lost_games, stuck, won_games/games * 100))
    
