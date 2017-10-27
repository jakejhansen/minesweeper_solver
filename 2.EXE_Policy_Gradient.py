
# coding: utf-8

# # Policy Gradient
# 
# > <span style="color:gray">
# Created by Jonas Busk ([jbusk@dtu.dk](mailto:jbusk@dtu.dk)).
# </span>
# 
# In this part, we will create an agent that can learn to solve tasks from OpenAI Gym by applying the Policy Gradient method. We will implement the agent with a probabilistic policy, that given a state of the environment, $s$, outputs a probability distribution over available actions, $a$:
# 
# $$
# p_\theta(a|s)
# $$
# 
# Since this is a deep learning course, we will model the policy as a neural network with parameters $\theta$ and train it with gradient descent (now the name 'Policy Gradient' should start to make sense). 
# When the set of available actions is discrete, we can use a network with softmax output. 
# 
# The core idea of training the policy network is simple: *we want to maximize the expected total reward by increasing the probability of good actions and decreasing the probability of bad actions*. 
# 
# The expectation over the (discounted) total reward, $R$, is:
# 
# $$
# \mathbb{E}[R|\theta] = \int p_\theta({\bf a}|{\bf s}) R({\bf a}) d{\bf a} \ ,
# $$
# 
# where ${\bf a} = a_1,\ldots,a_T$, ${\bf s}=s_1,\ldots,s_T$. 
# 
# Then we can use the gradient to maximize the total reward:
# 
# $$
# \begin{align}
# \nabla_\theta \mathbb{E}[R|\theta] &= \nabla_\theta \int p_\theta({\bf a}|{\bf s}) R({\bf a}) \, d{\bf a} \\
# &= \int \nabla_\theta p_\theta({\bf a}|{\bf s}) R({\bf a})  \, d{\bf a} \\
# &= \int p_\theta({\bf a}|{\bf s}) \nabla_\theta \log p_\theta({\bf a}|{\bf s}) R({\bf a}) \, d{\bf a} \\
# &= \mathbb{E}[R({\bf a}) \nabla_\theta \log p_\theta({\bf a}|{\bf s})]
# \end{align}
# $$
# 
# using the identity 
# 
# $$
# \nabla_\theta p_\theta({\bf a}|{\bf s}) = p_\theta({\bf a}|{\bf s}) \nabla_\theta \log p_\theta({\bf a}|{\bf s})
# $$
# 
# to express the gradient as an average over $p_\theta({\bf a},{\bf s})$.
# 
# We cannot evaluate the average over roll-outs analytically, but we have an environment simulator that when supplied with our current policy $p_\theta(a|s)$ can return the sequence of action, states and rewards. This allows us to replace the integral by a Monte Carlo average over $V$ roll-outs:
# 
# $$
# \nabla_\theta \mathbb{E}[R|\theta] \approx \frac{1}{V} \sum_{v=1}^V \nabla_\theta \log p_\theta({\bf a}^{(v)}|{\bf s}^{(v)}) R({\bf a}^{(v)}) \ .
# $$
# 
# In practice, to reduce the variance of the gradient, instead of $R$, we use the adjusted discounted future reward, also known as the *advantage*, $A$:
# 
# $$
# A_t = R_t - b_t \ ,
# $$
# 
# where the *baseline*, $b_t$, is the (discounted) total future reward at timestep $t$ averaged over the $V$ roll-outs:
# 
# $$
# b_t = \frac{1}{V} \sum_{v=1}^V R_t^{(v)} \ .
# $$
# 
# This way we are always encouraging and discouraging roughly half of the performed actions, which gives us the final gradient estimator:
# 
# $$
# \nabla_\theta \mathbb{E}[R|\theta] \approx \frac{1}{V} \sum_{v=1}^V \nabla_\theta \log p_\theta({\bf a}^{(v)}|{\bf s}^{(v)}) A({\bf a}^{(v)})
# $$
# 
# And that's it! Please refer to [this blog post](http://karpathy.github.io/2016/05/31/rl/) by Karpathy for more discussion on the Policy Gradient method.
# 
# --
# 
# *Note: For simple reinforcement learning problems (like the one we will address in this exercise) there are simpler methods that work just fine. However, the Policy Gradient method has been shown to also work well for complex problems with high dimensional inputs and many parameters, where simple methods become inadequate.*

# ## Policy Gradient code

# In[2]:


# imports
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.nn import relu, softmax
import gym
from utils import Viewer
from IPython.display import clear_output
import os.path


# In this lab we will work with the [CartPole-v0](https://gym.openai.com/envs/CartPole-v0) environment. Later you can change the code below to explore other [environments](https://gym.openai.com/envs/) and solve different tasks. 
# 
# *Note: The policy implemented in this notebook is designed to work on environments with a discrete action space. Extending the code to also handle environments with a continuous action space is left as an optional exercise.*

# In[3]:


# create gym environment
env_name = "CartPole-v0"
#env_name = "CartPole-v1"
#env_name = "Acrobot-v1"
#env_name = "MountainCar-v0"
#env_name = "Pendulum-v0" # Continuous
env = gym.make(env_name)


# In[4]:


env.observation_space, env.action_space


# Let us see how the environment looks when we just take random actions.

# In[5]:


# demo the environment
#env.reset() # reset the environment
#view = Viewer(env, custom_render=True) # we use this custom viewer to render the environment inline in the notebook
#for _ in range(200):
#    #view.render()
#    env.render() # uncomment this to use gym's own render function
#    env.step(env.action_space.sample()) # take a random action
##view.render(close=True, display_gif=True) # display the environment inline in the notebook
#env.render(close=True) # uncomment this to use gym's own render function


# Taking random actions does not do a very good job at balancing the pole. Let us now apply the Policy Gradient method described above to solve this task!
# 
# To start with, our policy will be a rather simple neural network with one hidden layer. We can retrieve the shape of the state space (input) and action space (output) from the environment.

# In[6]:


# setup policy network
n_inputs = env.observation_space.shape[0]
print("state size = " + str(n_inputs))

if type(env.action_space) == gym.spaces.discrete.Discrete:
    n_outputs = env.action_space.n
elif type(env.action_space) == gym.spaces.discrete.Continuous:
    n_outputs = env.action_space.shape[0]

print("action size = " + str(n_outputs))

tf.reset_default_graph()

states_pl = tf.placeholder(tf.float32, [None, n_inputs], name='states_pl')
actions_pl = tf.placeholder(tf.int32, [None, 2], name='actions_pl')
advantages_pl = tf.placeholder(tf.float32, [None], name='advantages_pl')
learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate_pl')

l_hidden1 = tf.layers.dense(inputs=states_pl, units=40, activation=relu, name='l_hidden1')
l_hidden2 = tf.layers.dense(inputs=l_hidden1, units=20, activation=relu, name='l_hidden2')
l_hidden3 = tf.layers.dense(inputs=l_hidden2, units=5, activation=relu, name='l_hidden3')

l_out = tf.layers.dense(inputs=l_hidden1, units=n_outputs, activation=softmax, name='l_out')

network_string = "3layer_40_20_5units"

# print network
print('states_pl:', states_pl.get_shape())
print('actions_pl:', actions_pl.get_shape())
print('advantages_pl:', advantages_pl.get_shape())
print('l_hidden1:', l_hidden1.get_shape())
print('l_out:', l_out.get_shape())


# In[7]:


# define loss
policy_grad_loss = -tf.reduce_mean(tf.multiply(tf.log(tf.gather_nd(l_out, actions_pl)), advantages_pl))

# define regularization
reg_scale = 0.0
#regularize = tf.contrib.layers.l2_regularizer(reg_scale)
#params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#reg_loss = sum([regularize(param) for param in params])

loss_f = policy_grad_loss

# define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_pl)
train_f = optimizer.minimize(loss_f)

saver = tf.train.Saver() # we use this later to save the model


# In[8]:


# test forward pass
state = env.reset()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    action_probabilities = sess.run(fetches=l_out, feed_dict={states_pl: [state]})
print(state)
print(action_probabilities)


# *Note: As we create our solution, we will make very few assumptions about the cart-pole environment. We aim to develop a general model for solving reinforcement learning problems, and therefore care little about the specific meaning of the inputs and outputs.*

# In[9]:


# helper functions

def get_rollout(sess, env, rollout_limit=None, stochastic=False, seed=None):
    """Generate rollout by iteratively evaluating the current policy on the environment."""
    rollout_limit = rollout_limit or env.spec.timestep_limit
    env.seed(seed)
    s = env.reset()
    states, actions, rewards = [], [], []
    for i in range(rollout_limit):
        a = get_action(sess, s, stochastic)
        s1, r, done, _ = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s1
        if done: break
    env.seed(None)
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


# In[10]:


# training settings
epochs = 300 # number of training batches
batch_size = 300 # number of timesteps in a batch
discount_factor = 0.9 # reward discount factor (gamma), 1.0 = no discount
rollout_limit = env.spec.timestep_limit # math.ceil(-3/math.log(discount_factor,10)) # max rollout length
learning_rate = 0.002 # you know this by now
early_stop_loss = 0 # stop training if loss < early_stop_loss, 0 or False to disable
max_reward = min(batch_size,rollout_limit)
max_reward_array = np.array([max_reward for i in range(epochs)])

# Figure for plotting during training
fig_during = plt.figure(figsize=(10,6))

# train policy network
try:
    statistics = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('start training')
        for epoch in range(epochs):
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
            val_rewards = [get_rollout(sess, env, rollout_limit, stochastic=False, seed=(epochs+i))[2] for i in range(10)]
            
            # store and print training statistics
            mtr = np.mean([np.sum(r) for r in rewards])
            mvr = np.mean([np.sum(r) for r in val_rewards])
            statistics.append((mtr, mvr, loss))
                
            # Plotting
            # plot training statistics
            stats = np.array(statistics).T
            mean_training_rewards = stats[0]
            mean_validation_rewards = stats[1]
            losses = stats[2]
            
            #print('%4d. training reward: %6.2f, validation reward: %6.2f, loss: %7.4f' % (epoch+1, mtr, mvr, loss))
            
            plt.subplot(211)
            plt.plot(losses, label='loss')
            plt.xlabel('epoch'); plt.ylabel('loss')
            plt.xlim((0, epochs))
            plt.legend(loc=1); plt.grid()
            
            plt.subplot(212)
            plt.plot(mean_training_rewards, label='mean training reward')
            plt.plot(mean_validation_rewards, label='mean validation reward')
            plt.plot(max_reward_array,label="max reward")
            plt.xlabel('epoch'); plt.ylabel('mean reward')
            plt.xlim((0, epochs))
            plt.legend(loc=4); plt.grid()
            plt.tight_layout(); plt.show()
            clear_output(wait=True)
            
            # early stopping
            if early_stop_loss and loss < early_stop_loss:
                break
            
        print('done')
        # save session
        saver.save(sess, 'tmp/model.ckpt')
        # plot training statistics
        statistics = np.array(statistics).T
        mean_training_rewards = statistics[0]
        mean_validation_rewards = statistics[1]
        losses = statistics[2]
        
except KeyboardInterrupt:
    pass    


clear_output(wait=True)

plt.figure(figsize=(10,6))
        
plt.subplot(211)
plt.plot(losses, label='loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xlim((0, len(losses)))
plt.grid()

plt.subplot(212)
plt.plot(mean_training_rewards, label='mean training reward')
plt.plot(mean_validation_rewards, label='mean validation reward')
plt.plot(max_reward_array,label="max reward")
plt.xlabel('epoch')
plt.ylabel('mean reward')
plt.xlim((0, len(mean_validation_rewards)))
plt.legend(loc=4)
plt.grid()

plt.tight_layout(); 
s = env_name + "_e" + str(epochs) + "_bs" + str(batch_size) + "_rl" + str(rollout_limit)
s = s + "_df" + str(discount_factor) + "_lr" + str(learning_rate) + "_esl" + str(early_stop_loss)
s = s + "_reg" + str(reg_scale)+ "_" + network_string

# Save the plot
s_duplicate = s
iterator = 1
while os.path.isfile(s_duplicate+".png"):
    s_duplicate = s + "(" + str(iterator) + ")"
    
plt.savefig(s_duplicate+".png")


# In[18]:


# review solution
with tf.Session() as sess:
    saver.restore(sess, "tmp/model.ckpt")
    s = env.reset()
    # view = Viewer(env, custom_render=True)
    for i in range(500):
        #view.render()
        env.render()
        a = get_action(sess, s, stochastic=False)
        s, r, done, _ = env.step(a)
              
    #view.render(close=True, display_gif=True)
    env.render(close=True)


# ## Exercises
# 
# Now it is your turn! Play around the code above and try to make it learn better and faster.
# 
# Experiment with the:
# 
# * number of timesteps in a batch.
# * max length of rollouts.
# * discount factor.
# * learning rate.
# * number of hidden units and layers.
# 
# 
# ### Exercise 1 
# 
# *Describe any changes you made to the code and why you think they improve the agent. Are you able to get solutions consistently?*
# 
# ### Answer
# 
# All the different training configurations tried have had their training process image saved. These are hosted at https://www.dropbox.com/sh/nojxqf066qmis9q/AADSgGHw8rF8IoS_9vwrPHKla?dl=0.
# 
# 
# #### epochs
# Increasing the number of epochs is the number one change to make to the training loop in order to increase the performance of the learner. More experience generally maps to better performance, given a learner that is able to learn. 
# 
# 
# #### batch_size
# It is clear from the visualization of the cartpole problem that a batch_size of approximately 300 timesteps is sufficient for simulating all aspects of the physics behind the control problem. 
# 
# Depending on the size and complexity of the policy network, reducing the batch_size from the default 1000 to around 300 can significantly reduce the training time. This is due to the simulation also requiring considerable computation compared to the network backprop.
# 
# However, reducing the batch_size too much results in the learner not experiencing the full dynamics of the environment and thus it ends up learning only partially what is needed to e.g. stabilize the cartpole.
# 
# #### discount_factor
# Choosing a discount factor of about 0.9 increases the performance of the learner since the expected future reward decreases the further into the future it lies. This reflects the increasing uncertainty associated with such future rewards.
# 
# #### rollout_limit
# The limit of the length of the rollout sequence has the same qualitative effect as the discount factor since it cuts off the memory of the learner at a certain number of time steps. However, conversely to the discount factor, the cutoff is hard (step function) such that every included reward has discount 1 and every excluded reward has reward 0.
# 
# When the rollout sequence length limit and discount factor are combined, the included rewards are discounted as usual while the exluded rewards are completely ignored.
# 
# In order to avoid too much computation associated with the rollout it makes sense to add a hard limit to the length of the sequence considered. Denoting the rollout sequence length limit by $l$, then $l$ can be determined by setting a threshold, $10^{-x}$, on the minimal wanted value of the last weight used in the weighting of future rewards
# 
# \begin{align}
# \gamma^l &> 10^{-x}\\
# l\log_{10}\gamma &> \log_{10} 10^{-x}\\
# l\log_{10}\gamma &> -x\\
# l &> -\frac{x}{\log_{10}\gamma}\\
# \end{align}
# 
# E.g. with $\gamma=0.9$ and a threshold of $10^{-3}$
# 
# \begin{equation}
# l > -\frac{x}{\log_{10}\gamma} = -\frac{3}{\log_{10} 0.9} = 65.6\\
# \end{equation}
# 
# such that we should choose $l=66$. This is much shorter than the default value used in the above, so a lot of practically useless computations can be saved.
# 
# 
# 
# #### learning_rate
# Increasing the learning rate of the policy network results in faster learning but only until a certain limit where it results in too large steps being taken in the gradient descent of the network. 
# 
# The Adam optimizer helps in adapting the learning rate to values that result in better learning but an initially too large learning rate does in general result in poor learning. 
# 
# Experimenting with different learning rates yields quite good results for a learning rate of 0.002 or 0.001. A learning rate of 0.005 gives oscillatory behaviour in the validation reward during training indicating that the policy network is unable to efficiently learn.
# 
# 
# #### early_stop_loss
# Stoppping early may help prevent the policy network from overfitting the observed state transitions and thus result in better generalization. Stopping when the loss reaches a certain value is one way to do this. 
# 
# However, the loss of the policy network in general is quite noisy making it a bit challenging to select a suitable `early_stop_loss`.
# 
# An alternative stopping criteria that is somewhat more unlikely to occur is to check for the validation reward being the maximally attainable reward. Since the cartpole problem is somewhat easy however, this does actually happen and can save some iterations of the training loop.
# 
# 
# 
# 
# #### Policy network
# Increasing the number of hidden units from 20 to 30 enables the network to better represent a useful policy. The learner thus learns much faster and also gets a better solution in general.
# 
# Increasing the number of hidden layers from 1 to 2, with 20 and 15 units respectively, also increases the performance of the learner. 
# 
# Increasing the number of hidden layers from 2 to 3 with 40, 30 and 5 units respectively, also increases the performance. The learner get maximal reward already after 150 epochs of simulating 300 time steps.
# 
# 
# 
# 
# 
# 

# ### Exercise 2 
# 
# *Consider the following sequence of rewards produced by an agent interacting with an environment for 10 timesteps: [0, 1, 1, 1, 0, 1, 1, 0, 0, 0].*
# 
# * *What is the total reward?*
# * *What is the total future reward in each timestep?*
# * *What is the discounted future reward in each timestep if $\gamma = 0.9$?*
# 
# *[Hint: See previous notebook.]*
# 
# ### Solution
# 
# The **total reward** is simply the sum of the reward at each time step from start, $t=0$ to end $t=T$, so
# 
# $$ R = \sum_{i=1}^T r_i = 1+1+1+1+1 = 5 $$
# 
# The **total future reward** in each time step is computed as the sum of the remaining future rewards
# $$ R_t = \sum_{i=t}^T r_i$$
# such that
# $$ R_1 = R = 5$$
# $$ R_2 = 5 $$
# $$ R_3 = 4$$
# $$ R_4 = 3$$
# $$ R_5 = 3 $$
# $$ R_6 = 2$$
# $$ R_7 = 1$$
# $$ R_8 = 0$$
# $$ R_9 = 0$$
# $$ R_{10} = 0$$
# 
# The **discounted future reward** in each time step with $\gamma = 0.9$ is given as the weighted sum of the remaining future rewards
# 
# $$  R_t = \sum_{i=t}^T \gamma^{i-1}r_i = \gamma^{t-1}r_t + \gamma^t r_2 + \gamma^{t+1} r_3 + \cdots + \gamma^{T-1} r_T$$
# 
# so
# $$ R_1 = 0.9+0.9^2+0.9^3+0.9^5+0.9^6 = 4.5 $$
# $$ R_2 = 1 + 0.9 + 0.9^2 + 0.9^4 + 0.9^5 = 4.6 $$
# $$ R_3 = 3.7 $$
# $$ R_4 = 2.8 $$
# $$ R_5 = 1.8 $$
# $$ R_6 = 1.9 $$
# $$ R_7 = 1 $$
# $$ R_8 = 0 $$
# $$ R_9 = 0 $$
# $$ R_{10} = 0 $$

# ### Exercise 3
# 
# *In the plot of the training and validation mean reward above, you will sometimes see the validation reward starts out lower than the training reward but later they cross. How can you explain this behavior? [Hint: Do we use the policy network in the same way during training and validation?]*
# 
# ### Solution
# As described in "1.Reinforcement_learning.ipynb", if we consider a conditional probability distribution over possible actions given a state, $p(a|s)$, the policy can either be *stochastic*, by sampling an action from the probability distribution:
# 
# $$\pi(s) = a \sim p(a|s)$$
# 
# or *deterministic*, by simply selecting the action with the highest probability:
# 
# $$\pi(s) = argmax_a \, p(a|s)$$
# 
# During training of the reinforcement learner, a stochastic policy is implemented. However, for validation, the policy is deterministic and the action with the highest probability is selected every time. Thus, the performance of the learner will in general be better for the validation since it exploits all of its learned knowledge and takes no "chances" in order to explore as it does during training.

# ### Exercise 4
# 
# *How does the policy gradient method we have used address the exploration-exploitation dilemma (see the previous notebook for definition)?*
# 
# ### Solution
# This question is in logical succession to the one above. 
# 
# The policy gradient method adresses the exploration-exploitation problem by considering the learned policy as  being stochastic during training in order to explore unknown parts of the state-action space. 
# 
# During validation, test and usage it completely turns off exploration by using a deterministic policy, always choosing the action with the highest probability.
# 

# ## Optional exercises:
# 
# * **Explore!** Train a policy for a different [environment](https://gym.openai.com/envs/).
# * **Let's get real!** Modify the code to work on an environment with a continuous action space. 

# ## Exercise from Nielsen
# 
# ### Problem
# It's tempting to use gradient descent to try to learn good values for hyper-parameters such as $\lambda$ and $\eta$. Can you think of an obstacle to using gradient descent to determine $\lambda$. Can you think of an obstacle to using gradient descent to determine $\eta$?
# 
# 
# ### Solution
# A regularized costfunction has the general form
# 
# $$ E_\text{reg}(\mathbf{w},\lambda) = E(\mathbf{w}) + R(\mathbf{w},\lambda),\;\; \lambda,\eta > 0 $$
# 
# where $E(\mathbf{w})$ is an unregularized cost function and $R(\mathbf{w},\lambda)$ is the chosen regularizer which is generally a function of the weights and the regularization parameter $\lambda$. 
# 
# Often $R(\mathbf{w},\lambda)$ has the form of a sum over the weights multiplied by $\lambda$.
# 
# If we set out to minimize $E_\text{reg}(\mathbf{w},\lambda)$ and $R(\mathbf{w},\lambda)$ has this form, it is clear that the optimal value of $\lambda$ is zero, since this results in $R(\mathbf{w},\lambda)=0$.
# 
# The learning rate $\eta$ is in general not part of the cost function. As such it is not possible to obtain a direction along which to change the learning rate from the derivative of the cost function.
# 
