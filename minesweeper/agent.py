import math
import numpy as np
import random
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizer import SGD, Adam, Nadam
from keras.models import load_model


state_dim = 
action_dim = 


class Agent(Object):
	def __init__(self, env=None, model=None, learning_method="policy gradients"):
		self.env = env or Minesweeper()
		self.method = learning_method

		# Model (Keras)
		if network is None:
			self.model = Sequential()
			self.model.add(Dense(20, activation='softmax', input_dim=state_dim))
			self.model.compile(optimizer='rmsprop',
				loss='categorical_crossentropy',
				metrics=['accuracy'])
		else:
			self.model = model


	def get_rollout(rollout_limit=None, stochastic=False):
		"""Generate rollout by iteratively evaluating the current policy on the environment."""
		rollout_limit = rollout_limit
		s = self.env.get_state()
		states, actions, rewards = [], [], []
		for i in range(rollout_limit):
			a = self.get_action(s, stochastic)
			out = self.env.action(a)
			s_next = out["s"]
			r = out["r"]
			states.append(s)
			actions.append(a)
			rewards.append(r)
			s = s_next
			if done: 
				break
		return states, actions, rewards, i+1


	def get_action(state, stochastic=False):
		"""Choose an action, given a state, with the current policy network."""
		# get action probabilities
		a_prob = self.model.predict(state, batch_size=1)
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


	def train_pg(epochs=1, batch_size=100, discount_factor=0.9, rollout_limit=None, learning_rate=0.001, early_stop_loss=0, plot_progress=False):
		print("Policy gradients training started...")
		for epoch in range(epochs):
			# generate rollouts until batch_size total timesteps are collected
			states, actions, rewards = [], [], []
			timesteps = 0
			while timesteps < batch_size:
				_rollout_limit = min(rollout_limit, batch_size - timesteps) # limit rollout to match batch_size
				s, a, r, t = self.get_rollout(_rollout_limit, stochastic=True)
				states.append(s)
				actions.append(a)
				rewards.append(r)
				timesteps += t

			# Compute advantages
			advantages = self.get_advantages(rewards, rollout_limit, discount_factor)

			# Policy gradient update
			self.model.fit(s, a, sample_weight=advantages, epochs=1)

			# validation
			val_rewards = [self.get_rollout(rollout_limit, stochastic=False)]

			# store and print training statistics
			mtr = np.mean([np.sum(r) for r in rewards])
			mvr = np.mean([np.sum(r) for r in val_rewards])
			statistics.append((mtr, mvr, loss))
			print('%4d. training reward: %6.2f, validation reward: %6.2f, loss: %7.4f' % (epoch+1, mtr, mvr, loss))
			
			# Plotting
			if plot_progress:
				# Plotting
				stats = np.array(statistics).T
				mean_training_rewards = stats[0]
				mean_validation_rewards = stats[1]
				losses = stats[2]

				plt.subplot(211)
				plt.plot(losses, label='loss')
				plt.xlabel('epoch'); plt.ylabel('loss')
				plt.xlim((0, epochs))
				plt.legend(loc=1); plt.grid()

				plt.subplot(212)
				plt.plot(mean_training_rewards, label='mean training reward')
				plt.plot(mean_validation_rewards, label='mean validation reward')
				#plt.plot(max_reward_array,label="max reward")
				plt.xlabel('epoch')
				plt.ylabel('mean reward')
				plt.xlim((0, epochs))
				plt.legend(loc=4)
				plt.grid()
				plt.tight_layout()
				plt.show()

			# Early stopping
			if early_stop_loss and loss < early_stop_loss:
				break

			model.save('pg_model.h5')






### Copied from RL notebook

#def get_rollout(sess, env, rollout_limit=None, stochastic=False, seed=None):
#	"""Generate rollout by iteratively evaluating the current policy on the environment."""
#	rollout_limit = rollout_limit or env.spec.timestep_limit
#	env.seed(seed)
#	s = env.reset()
#	states, actions, rewards = [], [], []
#	for i in range(rollout_limit):
#		a = get_action(sess, s, stochastic)
#		s1, r, done, _ = env.step(a)
#		states.append(s)
#		actions.append(a)
#		rewards.append(r)
#		s = s1
#		if done: break
#	env.seed(None)
#	return states, actions, rewards, i+1
#

#def get_action(sess, state, stochastic=False):
#	"""Choose an action, given a state, with the current policy network."""
#	# get action probabilities
#	a_prob = sess.run(fetches=l_out, feed_dict={states_pl: np.atleast_2d(state)})
#	if stochastic:
#		# sample action from distribution
#		return (np.cumsum(np.asarray(a_prob)) > np.random.rand()).argmax()
#	else:
#		# select action with highest probability
#		return a_prob.argmax()
#

#def get_advantages(rewards, rollout_limit, discount_factor, eps=1e-12):
#	"""Compute advantages"""
#	returns = get_returns(rewards, rollout_limit, discount_factor)
#	# standardize columns of returns to get advantages
#	advantages = (returns - np.mean(returns, axis=0)) / (np.std(returns, axis=0) + eps)
#	# restore original rollout lengths
#	advantages = [adv[:len(rewards[i])] for i, adv in enumerate(advantages)]
#	return advantages

#	
#def get_returns(rewards, rollout_limit, discount_factor):
#	"""Compute the cumulative discounted rewards, a.k.a. returns."""
#	returns = np.zeros((len(rewards), rollout_limit))
#	for i, r in enumerate(rewards):
#		returns[i, len(r) - 1] = r[-1]
#		for j in reversed(range(len(r)-1)):
#			returns[i,j] = r[j] + discount_factor * returns[i,j+1]
#	return returns


	