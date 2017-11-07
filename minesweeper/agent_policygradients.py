import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import matplotlib.pyplot as plt
from minesweeper_tk import Minesweeper
import IPython

gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))


tf.reset_default_graph() #Clear the Tensorflow graph.

n = 10
m = n
s_size = n*m*10
a_size = n*m
mines = math.floor((n*m)/20)

myAgent = agent(lr=1e-2, s_size=s_size, a_size=a_size, h_size=8) #Load the agent.
env = Minesweeper(ROWS=n, COLS=m, MINES=mines,rewards = {"win" : 100, "loss" : -100, "progress" : 0, "noprogress" : -1})

total_episodes = 5000 #Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5


# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i = 0
    total_reward = []
    total_lenght = []
        
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while i < total_episodes:
        env.initGame()
        s = env.get_state()
        s = env.stateConverter(s)
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            #Probabilistically pick an action given our network outputs.
            s = np.reshape(s,(1,s_size))
            #IPython.embed()
            a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in: s})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)

            out = env.action(np.unravel_index(a, (n,m)))
            s1,r,d = out["s"],out["r"],out["d"]
            s1 = env.stateConverter(s1)
            ep_history.append([s,a,r,s1])
            s = s1
            

            if d:
                #Update the network.
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2])
                feed_dict={myAgent.reward_holder:ep_history[:,2],myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
                #IPython.embed()
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                
                total_reward.append(running_reward)
                total_lenght.append(j)
                break

        #Update our running tally of scores.
        if i % 100 == 0:
            print(i,np.mean(total_reward[-100:]))
        i += 1

