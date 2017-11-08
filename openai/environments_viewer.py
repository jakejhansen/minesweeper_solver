import gym

# LunarLanderContinuous-v2
# BipedalWalker-v2

env = gym.make('BipedalWalker-v2')

n_epsiodes = 20
n_timesteps = 100
for i_episode in range(n_epsiodes):
	observation = env.reset()
	for t in range(n_timesteps):
		env.render()
		print(observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
