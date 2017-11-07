#!/usr/bin/env python
import math
import numpy as np
import tensorflow as tf
import IPython
import gym
from minesweeper_tk import Minesweeper


class PolicyGradientAgent(object):
	def __init__(self, hparams, sess):
		# initialization
		self._s = sess

		# build the graph
		self._input = tf.placeholder(tf.float32,
			shape=[None, hparams['input_size']])
		hidden1 = tf.contrib.layers.fully_connected(
			inputs=self._input,
			num_outputs=hparams['hidden1_size'],
			activation_fn=tf.nn.relu,
			weights_initializer=tf.random_normal_initializer())
#		hidden2 = tf.contrib.layers.fully_connected(
#			inputs=hidden1,
#			num_outputs=hparams['hidden2_size'],
#			activation_fn=tf.nn.relu,
#			weights_initializer=tf.random_normal_initializer())
#		hidden3 = tf.contrib.layers.fully_connected(
#			inputs=hidden2,
#			num_outputs=hparams['hidden3_size'],
#			activation_fn=tf.nn.relu,
#			weights_initializer=tf.random_normal_initializer())
		logits = tf.contrib.layers.fully_connected(
			inputs=hidden1,
			num_outputs=hparams['num_actions'],
			activation_fn=None)

		# op to sample an action
		self._sample = tf.reshape(tf.multinomial(logits, 1), [])
		# get log probabilities
		log_prob = tf.log(tf.nn.softmax(logits))

		# training part of graph
		self._acts = tf.placeholder(tf.int32)
		self._advantages = tf.placeholder(tf.float32)

		# get log probs of actions from episode
		indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self._acts
		act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

		# surrogate loss
		loss = -tf.reduce_sum(tf.multiply(act_prob, self._advantages))

		# update
		optimizer = tf.train.RMSPropOptimizer(hparams['learning_rate'])
		self._train = optimizer.minimize(loss)


	def act(self, observation):
		# get one action, by sampling
		return self._s.run(self._sample, feed_dict={self._input: [observation]})


	def train_step(self, obs, acts, advantages):
		batch_feed = {self._input: obs, self._acts: acts, self._advantages: advantages}
		self._s.run(self._train, feed_dict=batch_feed)


def policy_rollout(env, agent):
	"""Run one episode."""
	observation, reward, done = env.reset(), 0, False
	obs, acts, rews = [], [], []
	while not done:
		#env.render()
		obs.append(observation)
		action = agent.act(observation)
		observation, reward, done, info = env.step(action)
		acts.append(action)
		rews.append(reward)
		#print("Episode steps: {}".format(len(rews)),end='\r')
	#print("Episode steps: {}, total reward={}".format(len(rews),np.sum(rews)),end='\r')
	return obs, acts, rews


def get_advantages(rewards, discount_factor=0.9, eps=1e-12):
    """Compute advantages"""
    returns = get_returns(rewards, discount_factor)
    # standardize columns of returns to get advantages
    advantages = (returns - np.mean(returns, axis=0)) / (np.std(returns, axis=0) + eps)
    return advantages


def get_returns(rewards, discount_factor=0.9):
    """Compute the cumulative discounted rewards, a.k.a. returns."""
    returns = np.zeros(len(rewards))
    for t in reversed(range(len(rewards)-1)):
        returns[t] = rewards[t] + discount_factor * returns[t+1]
    return returns


def process_rewards(rews):
	"""Rewards -> Advantages for one episode. """

	# total reward: length of episode
	#print("Episode steps: {}, total reward={}, total advantage={}".format(len(rews),np.sum(rews),np.sum([len(rews)] * len(rews))),end='\r')
	#print()
	return [len(rews)] * len(rews)


def main():
	# Minesweeper environment
	n = 10
	m = n
	s_size = n*m*10
	a_size = n*m
	mines = math.floor((n*m)/20)
	rewsd = {"win" : 100, "loss" : -100, "progress" : 1, "noprogress" : -1}
	env = Minesweeper(ROWS=n, COLS=m, MINES=mines,rewards = rewsd,display=False)

	# Cartpole environment
	env = gym.make('CartPole-v0')
	s_size = env.observation_space.shape[0]
	a_size = env.action_space.n

	# hyper parameters
	hparams = {
			'input_size': s_size,
			'hidden1_size': 350,
			'hidden2_size': 250,
			'hidden3_size': 150,
			'num_actions': a_size,
			'learning_rate': 0.01
	}

	# environment params
	eparams = {
			'num_batches': 15,
			'ep_per_batch': 32
	}

	saver = tf.train.Saver()
	with tf.Graph().as_default(), tf.Session() as sess:
		agent = PolicyGradientAgent(hparams, sess)
		sess.run(tf.initialize_all_variables())
		for batch in range(eparams['num_batches']):
			print('=========\nBATCH {}'.format(batch))
			b_obs, b_acts, b_rews = [], [], []
			for epoch in range(eparams['ep_per_batch']):
				obs, acts, rews = policy_rollout(env, agent)
				#print('Episode steps: {}'.format(len(rews)),'\r')
				print('Episode steps: %4d | Mean reward: %6.2f' % (len(rews),np.mean(rews)))
				b_obs.extend(obs)
				b_acts.extend(acts)
				#advantages = get_advantages(rews,discount_factor=1)
				advantages = process_rewards(rews)
				b_rews.extend(advantages)

			# update policy
			# normalize rewards; don't divide by 0
			b_rews = (b_rews - np.mean(b_rews)) / (np.std(b_rews) + 1e-10)
			agent.train_step(b_obs, b_acts, b_rews)
		saver.save(sess, 'tmp/model.ckpt')

	print("Done")

	# Review agent performance
	if env is gym.wrappers.time_limit.TimeLimit:
		with tf.Session() as sess:
			saver.restore(sess, "tmp/model.ckpt")
			s = env.reset()
			# view = Viewer(env, custom_render=True)
			for i in range(500):
				#view.render()
				env.render()
				a = agent.act(s)
				s, r, done, _ = env.step(a)

			#view.render(close=True, display_gif=True)
			env.render(close=False)
	


if __name__ == "__main__":
	main()


