import matplotlib.pyplot as plt
import os
import pickle 


try:
    with open(os.path.join('results.pkl'), 'rb') as f:
        results = pickle.load(f)
except:
    print("Failed to load checkpoint")
    raise

fig = plt.figure()
plt.plot(results['steps'], results['mean_pop_rewards'])
plt.plot(results['steps'], results['test_rewards'])
plt.xlabel('Environment steps')
plt.ylabel('Reward')
plt.legend(['Mean population reward', 'Test reward'])
plt.tight_layout()
plt.grid()
plt.savefig(os.path.join('progress1.pdf'))
plt.close(fig)

fig = plt.figure(figsize=(4, 8))
plt.subplot(3, 1, 1)
plt.plot(results['steps'], results['mean_pop_rewards'])
plt.ylabel('Mean population reward')
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(results['steps'], results['test_rewards'])
plt.ylabel('Test reward')
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(results['steps'], results['weight_norm'])
plt.ylabel('Weight norm')
plt.xlabel('Environment steps')
plt.tight_layout()
plt.grid()
plt.savefig(os.path.join('progress2.pdf'))
plt.close(fig)

try:
    fig = plt.figure()
    plt.plot(results['steps'], results['win_rate'])
    plt.xlabel('Environment steps')
    plt.ylabel('Win rate')
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join('progress2.pdf'))
    plt.close(fig)
except KeyError:
    print('No win rate logged')

